"""
==============================================================================
Module: recorder_engine.py
Mô tả : Bộ thu thập dữ liệu video gốc (Clean Raw 1280x720 @ 30 FPS) tự động
        theo cơ chế Zero-Impact HLS Ring-Buffer trên RAM Disk (/dev/shm).
        - Ghim cứng tiến trình nén ffmpeg vào Core 3 (taskset -c 3 nice -n 10)
          để hoàn toàn không ảnh hưởng luồng End-to-End 120 FPS (Core 0-2 + Hailo-8).
        - Khi IDLE: chỉ giữ xoay vòng 6 đoạn .ts (12 giây gần nhất) trên RAM (/dev/shm),
          tốn 0 MB ghi thẻ nhớ.
        - Khi phát hiện Drone (det_n > 0 hoặc CANDIDATE/LOCKED): tự động gom cả
          Pre-roll (8-10s trước khi thấy Drone) + toàn bộ quá trình bám + Post-roll (4s)
          cùng file nhãn gợi ý `labels_metadata.jsonl` lưu xuống `dataset_captures/`.
==============================================================================
"""

from collections import deque
import json
import os
import shutil
import subprocess
import threading
import time
from queue import Empty, Queue

from config import (
    BASE_DIR,
    CAM_DIAG,
    CAM_DIAG_LOCK,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    STATE,
    STOP_EVENT,
    put_latest,
)


AUTO_RECORD_ENABLED = os.getenv("DRONE_TRACKER_AUTO_RECORD", "1").strip().lower() not in {"0", "false", "no"}
RECORD_FPS = int(os.getenv("DRONE_TRACKER_RECORD_FPS", "30"))
RECORD_CRF = int(os.getenv("DRONE_TRACKER_RECORD_CRF", "18"))
HLS_SEGMENT_SEC = 2
HLS_RING_SIZE = 6  # Giữ 12 giây gần nhất trong RAM (/dev/shm)
POST_ROLL_SEC = float(os.getenv("DRONE_TRACKER_POST_ROLL_SEC", "4.0"))
MAX_SESSION_DURATION_SEC = float(os.getenv("DRONE_TRACKER_MAX_CLIP_SEC", "180.0"))
MAX_CAPTURE_STORAGE_GB = float(os.getenv("DRONE_TRACKER_MAX_CAPTURE_GB", "20.0"))
MIN_FREE_DISK_GB = 8.0

CAPTURE_ROOT_DIR = os.getenv(
    "DRONE_TRACKER_CAPTURE_DIR",
    os.path.join(os.path.dirname(BASE_DIR), "dataset_captures"),
)
SHM_RING_DIR = f"/dev/shm/drone_hls_ring_{os.getpid()}"

q_record = Queue(maxsize=2)
_LAST_REC_ENQUEUE_TS = 0.0
_LATEST_DETECTIONS_LOCK = threading.Lock()
_LATEST_DETECTIONS = []


def notify_recorder_frame(timestamp, raw_frame, detections=None):
    """Đẩy tham chiếu khung hình gốc (Clean Raw) kèm detections đồng bộ vào hàng đợi ghi 30 FPS (< 0.005ms)."""
    global _LAST_REC_ENQUEUE_TS
    if not AUTO_RECORD_ENABLED or raw_frame is None:
        return
    if timestamp - _LAST_REC_ENQUEUE_TS < (1.0 / max(RECORD_FPS, 1)):
        return
    _LAST_REC_ENQUEUE_TS = timestamp
    dets_snap = [
        {
            "cx": round(float(d["cx"]), 1),
            "cy": round(float(d["cy"]), 1),
            "w": round(float(d["w"]), 1),
            "h": round(float(d["h"]), 1),
            "conf": round(float(d["conf"]), 4),
            "cls": int(d.get("cls", 0)),
        }
        for d in (detections or [])
    ]
    put_latest(q_record, (timestamp, raw_frame, dets_snap))


class AutoHlsDataRecorder:
    """Bộ ghi hộp đen HLS tự động trên RAM Disk (/dev/shm) + lưu sự kiện sang thẻ nhớ."""

    def __init__(self, stop_event=None):
        self.stop_event = stop_event or STOP_EVENT
        self.proc = None
        self.frame_idx = 0
        self.active_session_dir = None
        self.active_jsonl_fh = None
        self.copied_segments = {}  # seg_name -> last_mtime_copied
        self.session_segments = []
        self.session_start_ts = 0.0
        self.last_active_ts = 0.0
        self.meta_ring = deque(maxlen=RECORD_FPS * (HLS_SEGMENT_SEC * HLS_RING_SIZE + 4))
        self.session_count = 0

    def _ensure_dirs(self):
        os.makedirs(SHM_RING_DIR, exist_ok=True)
        os.makedirs(CAPTURE_ROOT_DIR, exist_ok=True)

    def _enforce_disk_quota(self):
        """Tự động xóa session cũ nhất nếu vượt hạn mức GB hoặc đĩa gần đầy."""
        try:
            usage = shutil.disk_usage(CAPTURE_ROOT_DIR)
            free_gb = usage.free / (1024 ** 3)
            sessions = sorted(
                [
                    os.path.join(CAPTURE_ROOT_DIR, d)
                    for d in os.listdir(CAPTURE_ROOT_DIR)
                    if os.path.isdir(os.path.join(CAPTURE_ROOT_DIR, d)) and d.startswith("flight_")
                ]
            )
            total_bytes = 0
            sizes = {}
            for s in sessions:
                sz = sum(
                    os.path.getsize(os.path.join(r, f))
                    for r, _, files in os.walk(s)
                    for f in files
                )
                sizes[s] = sz
                total_bytes += sz

            while sessions and (
                (total_bytes / (1024 ** 3)) > MAX_CAPTURE_STORAGE_GB or free_gb < MIN_FREE_DISK_GB
            ):
                oldest = sessions.pop(0)
                if oldest == self.active_session_dir:
                    continue
                shutil.rmtree(oldest, ignore_errors=True)
                total_bytes -= sizes.get(oldest, 0)
                free_gb = shutil.disk_usage(CAPTURE_ROOT_DIR).free / (1024 ** 3)
                print(f"[REC] Disk quota guard: Da xoa clip cu {os.path.basename(oldest)}")
        except OSError:
            pass

    def _start_ffmpeg(self):
        self._ensure_dirs()
        m3u8_path = os.path.join(SHM_RING_DIR, "ring.m3u8")
        seg_pattern = os.path.join(SHM_RING_DIR, "seg_%05d.ts")
        cmd = [
            "taskset", "-c", "3",
            "nice", "-n", "10",
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
            "-r", str(RECORD_FPS),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-threads", "1",
            "-crf", str(RECORD_CRF),
            "-g", str(RECORD_FPS * HLS_SEGMENT_SEC),
            "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_SEC),
            "-hls_list_size", str(HLS_RING_SIZE),
            "-hls_flags", "delete_segments+temp_file",
            "-hls_segment_filename", seg_pattern,
            m3u8_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print(
            f"[REC] Auto-HLS Data Collector ON (Core 3 Pinned | {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{RECORD_FPS}fps "
            f"CRF={RECORD_CRF} | RAM Ring={SHM_RING_DIR} -> {CAPTURE_ROOT_DIR})"
        )

    def _build_frame_meta(self, ts, dets):
        stats = STATE.get_stats()
        with CAM_DIAG_LOCK:
            cam_info = {
                "exp_us": round(float(CAM_DIAG["exp_us"]), 0),
                "gain": round(float(CAM_DIAG["gain"]), 2),
                "af_state": int(CAM_DIAG["af_state"]),
            }
        return {
            "frame_idx": self.frame_idx,
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S.") + f"{int((time.time() % 1) * 1000):03d}",
            "mono_ts": round(float(ts), 4),
            "tracker_state": stats["tracker_state"],
            "control_state": stats["control_state"],
            "track_id": stats["track_id"],
            "roi_mode": stats["roi_mode"],
            "ai_fps": stats["ai_fps"],
            "pan_deg": stats["pan_angle"],
            "tilt_deg": stats["tilt_angle"],
            "err_x": stats["error_x"],
            "err_y": stats["error_y"],
            "detections": dets,
            "cam": cam_info,
        }

    def _open_session(self, now_ts, meta):
        self._enforce_disk_quota()
        self.session_count += 1
        stamp_str = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"flight_{stamp_str}_s{self.session_count:03d}"
        self.active_session_dir = os.path.join(CAPTURE_ROOT_DIR, folder_name)
        os.makedirs(self.active_session_dir, exist_ok=True)
        self.copied_segments.clear()
        self.session_segments.clear()
        self.session_start_ts = now_ts

        jsonl_path = os.path.join(self.active_session_dir, "labels_metadata.jsonl")
        self.active_jsonl_fh = open(jsonl_path, "a", encoding="utf-8", buffering=1)

        # Dump pre-roll metadata from ring buffer
        for old_meta in self.meta_ring:
            self.active_jsonl_fh.write(json.dumps(old_meta, ensure_ascii=False) + "\n")

        # Sync existing pre-roll .ts segments from /dev/shm
        self._sync_shm_segments()
        print(f"[REC] >>> AUTO-CAPTURE TRIGGERED: Luu video & nhan vao {self.active_session_dir}")

    def _sync_shm_segments(self):
        if not self.active_session_dir or not os.path.exists(SHM_RING_DIR):
            return
        try:
            ts_files = sorted(
                f for f in os.listdir(SHM_RING_DIR)
                if f.endswith(".ts") and not f.endswith(".tmp")
            )
            for fname in ts_files:
                src = os.path.join(SHM_RING_DIR, fname)
                try:
                    st = os.stat(src)
                except OSError:
                    continue
                if st.st_size == 0:
                    continue
                prev_mtime = self.copied_segments.get(fname)
                if prev_mtime is None or st.st_mtime > prev_mtime:
                    dst = os.path.join(self.active_session_dir, fname)
                    shutil.copy2(src, dst)
                    self.copied_segments[fname] = st.st_mtime
                    if fname not in self.session_segments:
                        self.session_segments.append(fname)
        except OSError:
            pass

    def _close_session(self):
        if not self.active_session_dir:
            return
        self._sync_shm_segments()
        if self.active_jsonl_fh is not None:
            try:
                self.active_jsonl_fh.close()
            except OSError:
                pass
            self.active_jsonl_fh = None

        # Write standard HLS playlist.m3u8 for direct playback / ffmpeg concat
        if self.session_segments:
            m3u8_out = os.path.join(self.active_session_dir, "playlist.m3u8")
            try:
                with open(m3u8_out, "w", encoding="utf-8") as f:
                    f.write("#EXTM3U\n#EXT-X-VERSION:3\n")
                    f.write(f"#EXT-X-TARGETDURATION:{HLS_SEGMENT_SEC + 1}\n")
                    f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
                    for seg in self.session_segments:
                        f.write(f"#EXTINF:{float(HLS_SEGMENT_SEC):.3f},\n{seg}\n")
                    f.write("#EXT-X-ENDLIST\n")
            except OSError:
                pass
        print(
            f"[REC] <<< AUTO-CAPTURE SAVED: {os.path.basename(self.active_session_dir)} "
            f"({len(self.session_segments)} segments .ts + labels_metadata.jsonl)"
        )
        self.active_session_dir = None

    def run(self):
        if not AUTO_RECORD_ENABLED:
            print("[REC] Auto-HLS Data Collector OFF (DRONE_TRACKER_AUTO_RECORD=0)")
            return
        try:
            self._start_ffmpeg()
        except Exception as exc:
            print(f"[REC] Khong khoi dong duoc ffmpeg recorder: {exc}")
            return

        last_sync_check = 0.0
        try:
            while not self.stop_event.is_set():
                try:
                    ts, frame, dets = q_record.get(timeout=0.1)
                except Empty:
                    if self.active_session_dir and (time.monotonic() - self.last_active_ts) > POST_ROLL_SEC:
                        self._close_session()
                    continue

                try:
                    self.proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    break

                self.frame_idx += 1
                meta = self._build_frame_meta(ts, dets)
                is_interesting = (
                    bool(meta["detections"])
                    or meta["tracker_state"] in {"CANDIDATE", "LOCKED"}
                )

                now_mono = time.monotonic()
                if is_interesting:
                    self.last_active_ts = now_mono
                    if self.active_session_dir is None:
                        self._open_session(now_mono, meta)

                if self.active_session_dir is not None:
                    if self.active_jsonl_fh is not None:
                        try:
                            self.active_jsonl_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        except OSError:
                            pass
                    if now_mono - last_sync_check >= 0.5:
                        last_sync_check = now_mono
                        self._sync_shm_segments()
                    if (now_mono - self.last_active_ts) > POST_ROLL_SEC or (
                        (now_mono - self.session_start_ts) >= MAX_SESSION_DURATION_SEC
                    ):
                        self._close_session()
                else:
                    self.meta_ring.append(meta)
        finally:
            if self.proc is not None:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=2.0)
                except Exception:
                    pass
            if self.active_session_dir is not None:
                self._close_session()
            shutil.rmtree(SHM_RING_DIR, ignore_errors=True)


def thread_recorder():
    AutoHlsDataRecorder(stop_event=STOP_EVENT).run()
