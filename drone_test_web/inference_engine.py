"""
Thread-safe Drone Detection & Tracking Inference Engine for Web Testing.

Supports:
- YOLO26s-P2 INT8 ONNX (CPU AVX-512 VNNI) [Default]
- YOLO26s-P2 FP32 ONNX (CPU)
- YOLO26s-P2 Original FP32 PyTorch (CUDA GPU / CPU)
- TrackTrack (custom Anti-UAV drone_tracker.yaml), ByteTrack, BoT-SORT, OC-SORT
- Live MJPEG streaming, browser-compatible video rendering, JSON/CSV tracking export.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Prevent Ultralytics ONNXBackend from invoking pip install 'onnxruntime' when 'onnxruntime-gpu' is installed
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = Path(r"E:\trend_AI")
WEIGHTS_DIR = BASE_DIR / "yolo26s_p2_img640" / "yolo26s_p2_img640" / "weights"
BEST_PT_PATH = WEIGHTS_DIR / "best.pt"
BEST_ONNX_PATH = WEIGHTS_DIR / "best.onnx"
BEST_INT8_ONNX_PATH = WEIGHTS_DIR / "best_int8.onnx"
QUANT_REPORT_PATH = WEIGHTS_DIR / "int8_quantization_report.json"
CUSTOM_TRACKER_PATH = BASE_DIR / "clean_dataset" / "drone_tracker.yaml"

WEB_DIR = Path(__file__).resolve().parent
RUNS_DIR = WEB_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_INFERENCE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], YOLO] = {}
_CODEC_CACHE: tuple[str, str, str, bool] | None = None
_CODEC_LOCK = threading.Lock()


def get_available_models() -> list[dict[str, Any]]:
    """Return list of runnable model backends discovered on disk."""
    models: list[dict[str, Any]] = []

    if BEST_INT8_ONNX_PATH.exists():
        models.append({
            "key": "int8_cpu",
            "label": "YOLO26s-P2 INT8 (ONNX — CPU AVX-512)",
            "short_label": "INT8 ONNX",
            "path": str(BEST_INT8_ONNX_PATH),
            "device": "cpu",
            "imgsz_fixed": 640,
            "size_mb": round(BEST_INT8_ONNX_PATH.stat().st_size / (1024 * 1024), 2),
            "default": True,
        })

    if BEST_ONNX_PATH.exists():
        models.append({
            "key": "fp32_onnx_cpu",
            "label": "YOLO26s-P2 FP32 (ONNX — CPU)",
            "short_label": "FP32 ONNX",
            "path": str(BEST_ONNX_PATH),
            "device": "cpu",
            "imgsz_fixed": 640,
            "size_mb": round(BEST_ONNX_PATH.stat().st_size / (1024 * 1024), 2),
            "default": not BEST_INT8_ONNX_PATH.exists(),
        })

    if BEST_PT_PATH.exists():
        pt_mb = round(BEST_PT_PATH.stat().st_size / (1024 * 1024), 2)
        if torch.cuda.is_available():
            models.append({
                "key": "pt_cuda",
                "label": f"YOLO26s-P2 Original FP32 (PyTorch — {torch.cuda.get_device_name(0)})",
                "short_label": "FP32 CUDA",
                "path": str(BEST_PT_PATH),
                "device": "0",
                "imgsz_fixed": None,
                "size_mb": pt_mb,
                "default": False,
            })
        models.append({
            "key": "pt_cpu",
            "label": "YOLO26s-P2 Original FP32 (PyTorch — CPU)",
            "short_label": "FP32 PT CPU",
            "path": str(BEST_PT_PATH),
            "device": "cpu",
            "imgsz_fixed": None,
            "size_mb": pt_mb,
            "default": False,
        })

    return models


def get_available_trackers() -> list[dict[str, Any]]:
    """Return available Ultralytics tracker configurations."""
    trackers: list[dict[str, Any]] = []
    if CUSTOM_TRACKER_PATH.exists():
        trackers.append({
            "key": "drone_tracktrack",
            "label": "Drone TrackTrack (Anti-UAV Custom)",
            "path": str(CUSTOM_TRACKER_PATH),
            "default": True,
        })
    else:
        trackers.append({
            "key": "tracktrack",
            "label": "TrackTrack (Default)",
            "path": "tracktrack.yaml",
            "default": True,
        })

    trackers.extend([
        {"key": "bytetrack", "label": "ByteTrack", "path": "bytetrack.yaml", "default": False},
        {"key": "botsort", "label": "BoT-SORT", "path": "botsort.yaml", "default": False},
        {"key": "ocsort", "label": "OC-SORT", "path": "ocsort.yaml", "default": False},
    ])
    return trackers


def load_quantization_report() -> dict[str, Any] | None:
    """Load quantization benchmark & accuracy report if generated."""
    if not QUANT_REPORT_PATH.exists():
        return None
    try:
        return json.loads(QUANT_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_hardware_info() -> dict[str, Any]:
    """Return summary of local CPU and GPU devices."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "cpu_threads": os.cpu_count() or 1,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


def detect_best_video_codec() -> tuple[str, str, str, bool]:
    """Probe OpenCV VideoWriter codecs in browser-compatibility priority order."""
    global _CODEC_CACHE
    with _CODEC_LOCK:
        if _CODEC_CACHE is not None:
            return _CODEC_CACHE

        candidates = [
            ("VP80", ".webm", "video/webm", True),
            ("avc1", ".mp4", "video/mp4", True),
            ("H264", ".mp4", "video/mp4", True),
            ("mp4v", ".mp4", "video/mp4", False),
        ]

        tmp_dir = Path(tempfile.mkdtemp(prefix="codec_probe_"))
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            dummy[16:48, 16:48] = (0, 200, 100)
            for fourcc_str, ext, mime, playable in candidates:
                out_file = tmp_dir / f"test_{fourcc_str}{ext}"
                writer = None
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    writer = cv2.VideoWriter(str(out_file), fourcc, 15.0, (64, 64))
                    if not writer.isOpened():
                        continue
                    for _ in range(4):
                        writer.write(dummy)
                    writer.release()
                    writer = None
                    if out_file.exists() and out_file.stat().st_size > 500:
                        _CODEC_CACHE = (fourcc_str, ext, mime, playable)
                        return _CODEC_CACHE
                except Exception:
                    pass
                finally:
                    if writer is not None:
                        try:
                            writer.release()
                        except Exception:
                            pass
                    out_file.unlink(missing_ok=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        _CODEC_CACHE = ("mp4v", ".mp4", "video/mp4", False)
        return _CODEC_CACHE


def _reset_model_tracker(model: YOLO) -> None:
    """Cleanly clear tracker state and TrackTrack hooks attached to a YOLO predictor."""
    pred = getattr(model, "predictor", None)
    if pred is not None:
        if hasattr(pred, "trackers"):
            del pred.trackers
        if hasattr(pred, "vid_path"):
            del pred.vid_path
        if hasattr(pred, "_orig_postprocess"):
            pred.postprocess = pred._orig_postprocess
            del pred._orig_postprocess
        if hasattr(pred, "_hook"):
            try:
                pred._hook.remove()
            except Exception:
                pass
            del pred._hook
        if hasattr(pred, "_feats"):
            pred._feats = None


def _get_or_load_model(model_key: str, mode: str, model_spec: dict[str, Any]) -> YOLO:
    """Load or fetch cached YOLO model separated by (model_key, mode) so callbacks stay isolated."""
    cache_key = (model_key, mode)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = YOLO(model_spec["path"], task="detect")
        _MODEL_CACHE[cache_key] = model
    _reset_model_tracker(model)
    return model


def _track_color_bgr(track_id: int | None) -> tuple[int, int, int]:
    """Deterministic vivid BGR color per track ID using golden-angle HSV spacing."""
    if track_id is None:
        return (0, 190, 255)  # Amber/yellow for pending unconfirmed detections
    hue = int((track_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 235, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 8,
) -> None:
    """Draw a clean dashed rectangle for unconfirmed/pending detections."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in edges:
        dist = float(np.hypot(ex - sx, ey - sy))
        if dist <= 0:
            continue
        steps = max(1, int(dist / dash_len))
        for i in range(0, steps, 2):
            t0 = i / steps
            t1 = min(1.0, (i + 1) / steps)
            p0 = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_corner_ticks(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw tactical corner accents around a bounding box."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    tick = max(6, min(18, int(min(w, h) * 0.28)))
    t = thickness + 1
    cv2.line(img, (x1, y1), (x1 + tick, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + tick), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - tick, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + tick), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + tick, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - tick), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - tick, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - tick), color, t, cv2.LINE_AA)


def draw_annotations(
    frame: np.ndarray,
    dets: list[dict[str, Any]],
    trails: dict[int, deque[tuple[int, int]]],
    config: dict[str, Any],
    hud_text: str,
) -> np.ndarray:
    """Draw bounding boxes, labels, fading trajectory trails, and compact HUD overlay."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale_factor = max(0.55, min(1.3, w / 960.0))
    box_thick = max(1, int(round(2 * scale_factor)))
    font_scale = max(0.45, 0.55 * scale_factor)
    font = cv2.FONT_HERSHEY_SIMPLEX

    show_boxes = bool(config.get("show_boxes", True))
    show_ids = bool(config.get("show_ids", True))
    show_conf = bool(config.get("show_conf", True))
    show_trails = bool(config.get("show_trails", True))
    show_pending = bool(config.get("show_pending", True))
    mode = config.get("mode", "track")

    # 1. Draw trails for confirmed tracks
    if show_trails and mode == "track":
        active_ids = {d["id"] for d in dets if d["id"] is not None}
        for tid in active_ids:
            pts = trails.get(tid)
            if not pts or len(pts) < 2:
                continue
            color = _track_color_bgr(tid)
            pts_list = list(pts)
            n_pts = len(pts_list)
            for idx in range(1, n_pts):
                alpha_ratio = idx / float(n_pts)
                seg_color = tuple(int(c * (0.35 + 0.65 * alpha_ratio)) for c in color)
                seg_thick = max(1, int(round(1 + 2.2 * alpha_ratio * scale_factor)))
                cv2.line(out, pts_list[idx - 1], pts_list[idx], seg_color, seg_thick, cv2.LINE_AA)

    # 2. Draw boxes & labels
    for d in dets:
        confirmed = bool(d["confirmed"])
        tid = d["id"]
        if mode == "track" and not confirmed and not show_pending:
            continue

        x1, y1, x2, y2 = int(round(d["x1"])), int(round(d["y1"])), int(round(d["x2"])), int(round(d["y2"]))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = _track_color_bgr(tid if mode == "track" else 1)

        if show_boxes:
            if mode == "track" and not confirmed:
                _draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness=max(1, box_thick))
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick, cv2.LINE_AA)
                _draw_corner_ticks(out, x1, y1, x2, y2, color, box_thick)

        parts: list[str] = []
        if mode == "track" and show_ids:
            parts.append(f"#{tid}" if tid is not None else "?")
        parts.append(d["cls"])
        if show_conf:
            parts.append(f"{d['conf']:.2f}")

        if parts and show_boxes:
            label = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            pad_x, pad_y = 5, 4
            ly1 = max(0, y1 - th - pad_y * 2)
            ly2 = ly1 + th + pad_y * 2
            if ly2 > y1 and y1 < th + pad_y * 2:
                ly1 = min(h - th - pad_y * 2, y2)
                ly2 = ly1 + th + pad_y * 2
            lx1 = x1
            lx2 = min(w - 1, x1 + tw + pad_x * 2)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            text_color = (15, 18, 24) if sum(color) > 360 else (255, 255, 255)
            cv2.putText(out, label, (lx1 + pad_x, ly2 - pad_y - 1), font, font_scale, text_color, 1, cv2.LINE_AA)

    # 3. Semi-transparent top-left HUD bar
    if hud_text:
        hud_scale = max(0.45, 0.52 * scale_factor)
        (htw, hth), _ = cv2.getTextSize(hud_text, font, hud_scale, 1)
        box_w = min(w - 16, htw + 20)
        box_h = hth + 14
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (12, 17, 28), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
        cv2.rectangle(out, (8, 8), (8 + box_w, 8 + box_h), (56, 189, 248), 1)
        cv2.putText(out, hud_text, (18, 8 + hth + 6), font, hud_scale, (240, 246, 255), 1, cv2.LINE_AA)

    return out


def encode_preview_jpeg(frame: np.ndarray, max_width: int = 1280, quality: int = 80) -> bytes | None:
    """Resize frame if needed and encode as JPEG bytes for MJPEG streaming."""
    h, w = frame.shape[:2]
    view = frame
    if w > max_width:
        scale = max_width / float(w)
        view = cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


@dataclass
class JobState:
    job_id: str
    job_dir: Path
    input_path: Path
    video_meta: dict[str, Any]
    status: str = "uploaded"  # uploaded | queued | running | completed | stopped | error
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source_job_id: str | None = None
    output_path: Path | None = None
    output_mime: str | None = None
    browser_playable: bool = False
    codec_fourcc: str | None = None
    error: str | None = None

    # Live metrics
    current_frame: int = 0
    processed_frames: int = 0
    total_frames: int = 0
    avg_infer_ms: float = 0.0
    last_infer_ms: float = 0.0
    avg_fps: float = 0.0
    eta_seconds: float = 0.0
    active_tracks: int = 0
    unique_track_ids: int = 0
    frames_with_drone: int = 0
    summary: dict[str, Any] = field(default_factory=dict)

    # Streaming & cancellation
    latest_jpeg: bytes | None = None
    frame_seq: int = 0
    frame_cond: threading.Condition = field(default_factory=threading.Condition)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: threading.Thread | None = None

    def to_status_dict(self) -> dict[str, Any]:
        progress = 0.0
        if self.total_frames > 0:
            progress = min(100.0, round(self.current_frame / float(self.total_frames) * 100.0, 1))
        return {
            "job_id": self.job_id,
            "source_job_id": self.source_job_id,
            "created_at": self.created_at,
            "status": self.status,
            "error": self.error,
            "video_meta": self.video_meta,
            "config": self.config,
            "progress_pct": progress,
            "current_frame": self.current_frame,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "avg_infer_ms": round(self.avg_infer_ms, 2),
            "last_infer_ms": round(self.last_infer_ms, 2),
            "avg_fps": round(self.avg_fps, 2),
            "eta_seconds": round(self.eta_seconds, 1),
            "active_tracks": self.active_tracks,
            "unique_track_ids": self.unique_track_ids,
            "frames_with_drone": self.frames_with_drone,
            "output_ready": bool(self.output_path and self.output_path.exists()),
            "output_filename": self.output_path.name if self.output_path else None,
            "output_mime": self.output_mime,
            "browser_playable": self.browser_playable,
            "codec_fourcc": self.codec_fourcc,
            "tracks_json_ready": (self.job_dir / "tracks.json").exists(),
            "tracks_csv_ready": (self.job_dir / "tracks.csv").exists(),
            "summary": self.summary,
        }


class JobManager:
    """Manages uploaded videos, job creation, re-runs, and background inference."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._load_existing_runs()

    def _load_existing_runs(self) -> None:
        """Restore completed/uploaded jobs from runs/ on server startup."""
        if not RUNS_DIR.exists():
            return
        for entry in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not entry.is_dir():
                continue
            meta_file = entry / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                input_name = meta.get("saved_input_name")
                input_path = entry / input_name if input_name else None
                if not input_path or not input_path.exists():
                    continue
                job = JobState(
                    job_id=entry.name,
                    job_dir=entry,
                    input_path=input_path,
                    video_meta=meta["video_meta"],
                    status=meta.get("status", "uploaded"),
                    config=meta.get("config", {}),
                    created_at=meta.get("created_at", ""),
                    source_job_id=meta.get("source_job_id"),
                    total_frames=meta["video_meta"].get("total_frames", 0),
                    summary=meta.get("summary", {}),
                    browser_playable=bool(meta.get("browser_playable", False)),
                    output_mime=meta.get("output_mime"),
                    codec_fourcc=meta.get("codec_fourcc"),
                )
                out_name = meta.get("output_filename")
                if out_name and (entry / out_name).exists():
                    job.output_path = entry / out_name
                if job.status in {"queued", "running"}:
                    job.status = "stopped"
                if job.summary:
                    job.current_frame = job.summary.get("total_video_frames", job.total_frames)
                    job.processed_frames = job.summary.get("processed_frames", 0)
                    job.avg_infer_ms = job.summary.get("avg_infer_ms", 0.0)
                    job.avg_fps = job.summary.get("avg_fps", 0.0)
                    job.unique_track_ids = job.summary.get("unique_track_ids", 0)
                    job.frames_with_drone = job.summary.get("frames_with_drone", 0)
                self._jobs[job.job_id] = job
            except Exception:
                continue

    def _persist_meta(self, job: JobState) -> None:
        payload = {
            "job_id": job.job_id,
            "source_job_id": job.source_job_id,
            "created_at": job.created_at,
            "status": job.status,
            "saved_input_name": job.input_path.name,
            "video_meta": job.video_meta,
            "config": job.config,
            "summary": job.summary,
            "output_filename": job.output_path.name if job.output_path else None,
            "output_mime": job.output_mime,
            "browser_playable": job.browser_playable,
            "codec_fourcc": job.codec_fourcc,
        }
        (job.job_dir / "meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def probe_video(self, video_path: Path, original_filename: str) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Không thể mở file video bằng OpenCV/FFmpeg. Vui lòng thử định dạng MP4/AVI/MOV/MKV/WEBM.")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # Fallback frame count probe if container header lacks frame count
            if total_frames <= 0:
                cnt = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if width == 0 or height == 0:
                        height, width = frame.shape[:2]
                    cnt += 1
                total_frames = cnt

            if width <= 0 or height <= 0:
                raise ValueError("Video không có frame hợp lệ (width/height = 0).")
            if fps <= 0 or fps > 240:
                fps = 30.0
            duration_s = round(total_frames / fps, 2) if fps > 0 else 0.0
            size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
            return {
                "filename": original_filename,
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_s": duration_s,
                "file_size_mb": size_mb,
            }
        finally:
            cap.release()

    def create_uploaded_job(self, file_storage: Any, original_filename: str) -> JobState:
        ext = Path(original_filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            raise ValueError(f"Định dạng '{ext}' không hỗ trợ. Hỗ trợ: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        job_dir = RUNS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / f"input{ext}"
        file_storage.save(str(input_path))

        try:
            video_meta = self.probe_video(input_path, original_filename)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        job = JobState(
            job_id=job_id,
            job_dir=job_dir,
            input_path=input_path,
            video_meta=video_meta,
            status="uploaded",
            total_frames=video_meta["total_frames"],
        )
        # Seed preview thumbnail with first frame
        cap = cv2.VideoCapture(str(input_path))
        ok, first = cap.read()
        cap.release()
        if ok and first is not None:
            job.latest_jpeg = encode_preview_jpeg(first)

        with self._lock:
            self._jobs[job_id] = job
        self._persist_meta(job)
        return job

    def clone_job_for_rerun(self, source_job_id: str) -> JobState:
        """Create a new run folder sharing the same input video so previous outputs stay intact."""
        src = self.get_job(source_job_id)
        if src is None:
            raise KeyError(f"Không tìm thấy job {source_job_id}")
        if not src.input_path.exists():
            raise FileNotFoundError("File video gốc của job này không còn tồn tại.")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        job_dir = RUNS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dst_input = job_dir / src.input_path.name
        try:
            os.link(str(src.input_path), str(dst_input))
        except OSError:
            shutil.copy2(str(src.input_path), str(dst_input))

        job = JobState(
            job_id=job_id,
            job_dir=job_dir,
            input_path=dst_input,
            video_meta=dict(src.video_meta),
            status="uploaded",
            source_job_id=src.job_id,
            total_frames=src.video_meta["total_frames"],
        )
        with self._lock:
            self._jobs[job_id] = job
        self._persist_meta(job)
        return job

    def get_job(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def ensure_preview_jpeg(self, job: JobState) -> bytes | None:
        """Return cached preview JPEG or lazily extract and cache the first frame."""
        if job.latest_jpeg is not None:
            return job.latest_jpeg
        if not job.input_path.exists():
            return None
        cap = cv2.VideoCapture(str(job.input_path))
        ok, first = cap.read()
        cap.release()
        if ok and first is not None:
            jpg = encode_preview_jpeg(first)
            if jpg is not None:
                job.latest_jpeg = jpg
                return jpg
        return None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_status_dict() for j in jobs[:limit]]

    def delete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        if job.status in {"queued", "running"}:
            job.stop_event.set()
            if job.worker_thread and job.worker_thread.is_alive():
                job.worker_thread.join(timeout=3.0)
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(job.job_dir, ignore_errors=True)
        return True

    def stop_job(self, job_id: str) -> JobState:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Không tìm thấy job {job_id}")
        job.stop_event.set()
        return job

    def start_job(self, job_id: str, raw_config: dict[str, Any]) -> JobState:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Không tìm thấy job {job_id}")

        # If this job was already run or is running, clone a fresh job directory for clean comparison
        if job.status in {"completed", "stopped", "error"}:
            job = self.clone_job_for_rerun(job_id)
        elif job.status in {"queued", "running"}:
            raise RuntimeError("Job này đang chạy. Hãy bấm Dừng hoặc chọn Chạy mới.")

        models = {m["key"]: m for m in get_available_models()}
        trackers = {t["key"]: t for t in get_available_trackers()}
        if not models:
            raise RuntimeError("Không tìm thấy file model trọng số nào.")

        default_model_key = next((k for k, v in models.items() if v.get("default")), next(iter(models)))
        default_tracker_key = next((k for k, v in trackers.items() if v.get("default")), next(iter(trackers)))

        model_key = raw_config.get("model_key", default_model_key)
        if model_key not in models:
            model_key = default_model_key
        model_spec = models[model_key]

        mode = raw_config.get("mode", "track")
        if mode not in {"track", "detect"}:
            mode = "track"

        tracker_key = raw_config.get("tracker_key", default_tracker_key)
        if tracker_key not in trackers:
            tracker_key = default_tracker_key

        imgsz = int(raw_config.get("imgsz", 640))
        if model_spec.get("imgsz_fixed"):
            imgsz = int(model_spec["imgsz_fixed"])

        config = {
            "model_key": model_key,
            "model_label": model_spec["label"],
            "model_short": model_spec["short_label"],
            "mode": mode,
            "tracker_key": tracker_key,
            "tracker_label": trackers[tracker_key]["label"],
            "tracker_path": trackers[tracker_key]["path"],
            "conf": max(0.05, min(0.95, float(raw_config.get("conf", 0.25)))),
            "iou": max(0.10, min(0.95, float(raw_config.get("iou", 0.45)))),
            "imgsz": imgsz,
            "frame_stride": max(1, min(10, int(raw_config.get("frame_stride", 1)))),
            "trail_len": max(0, min(150, int(raw_config.get("trail_len", 40)))),
            "show_boxes": bool(raw_config.get("show_boxes", True)),
            "show_ids": bool(raw_config.get("show_ids", True)),
            "show_conf": bool(raw_config.get("show_conf", True)),
            "show_trails": bool(raw_config.get("show_trails", True)),
            "show_pending": bool(raw_config.get("show_pending", True)),
        }

        job.config = config
        job.status = "queued"
        job.error = None
        job.stop_event.clear()
        job.current_frame = 0
        job.processed_frames = 0
        job.summary = {}
        self._persist_meta(job)

        t = threading.Thread(
            target=self._run_worker,
            args=(job, model_spec),
            name=f"infer-{job.job_id}",
            daemon=True,
        )
        job.worker_thread = t
        t.start()
        return job

    def _run_worker(self, job: JobState, model_spec: dict[str, Any]) -> None:
        cap = None
        writer = None
        frames_log: list[dict[str, Any]] = []
        trails: dict[int, deque[tuple[int, int]]] = {}
        track_stats: dict[int, dict[str, Any]] = {}

        with _INFERENCE_LOCK:
            try:
                job.status = "running"
                cfg = job.config
                mode = cfg["mode"]
                device = model_spec["device"]
                model = _get_or_load_model(cfg["model_key"], mode, model_spec)

                cap = cv2.VideoCapture(str(job.input_path))
                if not cap.isOpened():
                    raise RuntimeError("Không thể mở video đầu vào để xử lý.")

                fps = float(job.video_meta["fps"])
                width = int(job.video_meta["width"])
                height = int(job.video_meta["height"])
                total_frames = int(job.video_meta["total_frames"])

                fourcc_str, ext, mime, playable = detect_best_video_codec()
                out_path = job.job_dir / f"annotated{ext}"
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    # Hard fallback to mp4v if probed codec failed at this resolution
                    fourcc_str, ext, mime, playable = ("mp4v", ".mp4", "video/mp4", False)
                    out_path = job.job_dir / f"annotated{ext}"
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                job.output_path = out_path
                job.output_mime = mime
                job.browser_playable = playable
                job.codec_fourcc = fourcc_str

                stride = int(cfg["frame_stride"])
                trail_len = int(cfg["trail_len"])
                infer_times_ms: list[float] = []
                all_confs: list[float] = []
                peak_simultaneous = 0
                last_annotated: np.ndarray | None = None

                frame_idx = 0
                wall_start = time.perf_counter()

                while True:
                    if job.stop_event.is_set():
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if stride > 1 and (frame_idx % stride != 0) and last_annotated is not None:
                        writer.write(last_annotated)
                        frame_idx += 1
                        job.current_frame = frame_idx
                        continue

                    t0 = time.perf_counter()
                    if mode == "track":
                        results = model.track(
                            frame,
                            tracker=cfg["tracker_path"],
                            persist=True,
                            imgsz=cfg["imgsz"],
                            conf=cfg["conf"],
                            iou=cfg["iou"],
                            device=device,
                            verbose=False,
                        )
                    else:
                        results = model.predict(
                            frame,
                            imgsz=cfg["imgsz"],
                            conf=cfg["conf"],
                            iou=cfg["iou"],
                            device=device,
                            verbose=False,
                        )
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    infer_times_ms.append(infer_ms)

                    res = results[0]
                    names = res.names or {0: "drone"}
                    boxes = res.boxes

                    frame_dets: list[dict[str, Any]] = []
                    active_confirmed = 0
                    t_sec = frame_idx / fps if fps > 0 else 0.0

                    if boxes is not None and len(boxes) > 0:
                        xyxy_arr = boxes.xyxy.cpu().numpy()
                        conf_arr = boxes.conf.cpu().numpy()
                        cls_arr = boxes.cls.cpu().numpy()
                        ids_arr = boxes.id.cpu().numpy() if boxes.id is not None else None

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = [float(v) for v in xyxy_arr[i]]
                            c_val = float(conf_arr[i])
                            cls_id = int(cls_arr[i])
                            cls_name = str(names.get(cls_id, "drone"))
                            tid = int(ids_arr[i]) if ids_arr is not None else None
                            confirmed = (tid is not None) if mode == "track" else True

                            frame_dets.append({
                                "id": tid,
                                "x1": round(x1, 1),
                                "y1": round(y1, 1),
                                "x2": round(x2, 1),
                                "y2": round(y2, 1),
                                "conf": round(c_val, 4),
                                "cls": cls_name,
                                "confirmed": confirmed,
                            })

                            all_confs.append(c_val)
                            if confirmed and tid is not None:
                                active_confirmed += 1
                                cx = int(round((x1 + x2) * 0.5))
                                cy = int(round((y1 + y2) * 0.5))
                                if tid not in trails:
                                    trails[tid] = deque(maxlen=max(2, trail_len))
                                if trail_len > 0:
                                    trails[tid].append((cx, cy))

                                st = track_stats.get(tid)
                                if st is None:
                                    track_stats[tid] = {
                                        "track_id": tid,
                                        "color_hex": _bgr_to_hex(_track_color_bgr(tid)),
                                        "first_frame": frame_idx,
                                        "last_frame": frame_idx,
                                        "first_time": round(t_sec, 2),
                                        "last_time": round(t_sec, 2),
                                        "frame_count": 1,
                                        "conf_sum": c_val,
                                        "max_conf": c_val,
                                    }
                                else:
                                    st["last_frame"] = frame_idx
                                    st["last_time"] = round(t_sec, 2)
                                    st["frame_count"] += 1
                                    st["conf_sum"] += c_val
                                    if c_val > st["max_conf"]:
                                        st["max_conf"] = c_val

                    if len(frame_dets) > 0:
                        job.frames_with_drone += 1
                    peak_simultaneous = max(peak_simultaneous, len(frame_dets))

                    avg_ms = sum(infer_times_ms) / len(infer_times_ms)
                    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                    hud_mode = f"{cfg['tracker_key']}" if mode == "track" else "detect"
                    hud_text = (
                        f"{cfg['model_short']} | {hud_mode} | "
                        f"F {frame_idx + 1}/{max(total_frames, frame_idx + 1)} | "
                        f"{infer_ms:.1f} ms ({avg_fps:.1f} FPS) | "
                        f"Active: {active_confirmed if mode == 'track' else len(frame_dets)} | "
                        f"IDs: {len(track_stats)}"
                    )

                    annotated = draw_annotations(frame, frame_dets, trails, cfg, hud_text)
                    last_annotated = annotated
                    writer.write(annotated)

                    frames_log.append({
                        "f": frame_idx,
                        "t": round(t_sec, 3),
                        "ms": round(infer_ms, 2),
                        "dets": frame_dets,
                    })

                    frame_idx += 1
                    job.current_frame = frame_idx
                    job.processed_frames += 1
                    job.last_infer_ms = infer_ms
                    job.avg_infer_ms = avg_ms
                    job.avg_fps = avg_fps
                    job.active_tracks = active_confirmed if mode == "track" else len(frame_dets)
                    job.unique_track_ids = len(track_stats)

                    elapsed = time.perf_counter() - wall_start
                    if total_frames > 0 and frame_idx > 0:
                        job.eta_seconds = max(0.0, (elapsed / frame_idx) * (total_frames - frame_idx))

                    jpg = encode_preview_jpeg(annotated)
                    if jpg is not None:
                        with job.frame_cond:
                            job.latest_jpeg = jpg
                            job.frame_seq += 1
                            job.frame_cond.notify_all()

                # Finalize video writer BEFORE writing summary
                writer.release()
                writer = None
                cap.release()
                cap = None

                total_elapsed = time.perf_counter() - wall_start
                proc_count = len(infer_times_ms)
                mean_ms = (sum(infer_times_ms) / proc_count) if proc_count > 0 else 0.0
                mean_fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
                det_rate = (job.frames_with_drone / proc_count * 100.0) if proc_count > 0 else 0.0
                avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0

                tracks_summary = []
                for st in sorted(track_stats.values(), key=lambda x: (x["first_frame"], x["track_id"])):
                    fc = st["frame_count"]
                    tracks_summary.append({
                        "track_id": st["track_id"],
                        "color_hex": st["color_hex"],
                        "first_frame": st["first_frame"],
                        "last_frame": st["last_frame"],
                        "first_time": st["first_time"],
                        "last_time": st["last_time"],
                        "duration_s": round(max(0.0, st["last_time"] - st["first_time"]), 2),
                        "frame_count": fc,
                        "avg_conf": round(st["conf_sum"] / fc, 3) if fc > 0 else 0.0,
                        "max_conf": round(st["max_conf"], 3),
                    })

                job.summary = {
                    "total_video_frames": frame_idx,
                    "processed_frames": proc_count,
                    "wall_time_s": round(total_elapsed, 2),
                    "avg_infer_ms": round(mean_ms, 2),
                    "avg_fps": round(mean_fps, 2),
                    "frames_with_drone": job.frames_with_drone,
                    "detection_rate_pct": round(det_rate, 1),
                    "unique_track_ids": len(tracks_summary),
                    "peak_simultaneous": peak_simultaneous,
                    "avg_conf": round(avg_conf, 3),
                    "total_detections": len(all_confs),
                    "tracks": tracks_summary,
                }

                # Write tracks.json
                tracks_payload = {
                    "job_id": job.job_id,
                    "video_meta": job.video_meta,
                    "config": job.config,
                    "summary": job.summary,
                    "frames": frames_log,
                }
                (job.job_dir / "tracks.json").write_text(
                    json.dumps(tracks_payload, ensure_ascii=False), encoding="utf-8"
                )

                # Write tracks.csv
                with open(job.job_dir / "tracks.csv", "w", newline="", encoding="utf-8") as cf:
                    cw = csv.writer(cf)
                    cw.writerow(["frame", "time_sec", "track_id", "confirmed", "class", "conf", "x1", "y1", "x2", "y2"])
                    for f_rec in frames_log:
                        for d in f_rec["dets"]:
                            cw.writerow([
                                f_rec["f"],
                                f_rec["t"],
                                "" if d["id"] is None else d["id"],
                                int(bool(d["confirmed"])),
                                d["cls"],
                                d["conf"],
                                d["x1"],
                                d["y1"],
                                d["x2"],
                                d["y2"],
                            ])

                job.total_frames = max(job.total_frames, frame_idx)
                job.eta_seconds = 0.0
                job.status = "stopped" if job.stop_event.is_set() else "completed"
                _reset_model_tracker(model)
                self._persist_meta(job)

            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                self._persist_meta(job)
            finally:
                if writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                with job.frame_cond:
                    job.frame_cond.notify_all()
