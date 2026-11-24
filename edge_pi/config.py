"""
==============================================================================
Module: config.py
Mô tả : Quản lý cấu hình hệ thống, hằng số, trạng thái chia sẻ (SharedState),
        cầu nối dẫn hướng ROI (TargetRoiGuide) và ghi log telemetry có buffer.
==============================================================================
"""

import json
import os
import threading
import time
from queue import Empty, Queue
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEF_MODEL_PATH = os.path.join(BASE_DIR, "best.hef")
PT_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = int(os.getenv("DRONE_TRACKER_CAMERA_FPS", "120"))
CAMERA_WIDE_FOV = os.getenv("DRONE_TRACKER_WIDE_FOV", "0").strip().lower() in {"1", "true", "yes", "on"}
CAMERA_AF_MODE = os.getenv("DRONE_TRACKER_AF_MODE", "fast").strip().lower()
CAMERA_LENS_POS = float(os.getenv("DRONE_TRACKER_LENS_POS", "0.05"))
CAMERA_SHARPNESS = float(os.getenv("DRONE_TRACKER_SHARPNESS", "1.0"))
CAMERA_EV = float(os.getenv("DRONE_TRACKER_EV", "0.0"))
CAMERA_FAST_AE = os.getenv("DRONE_TRACKER_FAST_AE", "1").strip().lower() not in {"0", "false", "no"}
CAMERA_USE_LORES = os.getenv("DRONE_TRACKER_PISP_LORES", "1").strip().lower() not in {"0", "false", "no"}
CAMERA_SPOT_PREF_SHUTTER_US = int(os.getenv("DRONE_TRACKER_SPOT_PREF_SHUTTER_US", "6000"))

INFER_SIZE = int(os.getenv("DRONE_TRACKER_INFER_SIZE", "640"))
LORES_WIDTH = INFER_SIZE
LORES_HEIGHT = int(round(CAMERA_HEIGHT * (INFER_SIZE / float(CAMERA_WIDTH))))  # 360 cho 1280x720
LORES_PAD_Y = (INFER_SIZE - LORES_HEIGHT) // 2  # 140
LORES_GAIN = INFER_SIZE / float(CAMERA_WIDTH)   # 0.5

CONF_THRESHOLD = 0.20
DETECTION_FLOOR = 0.10
IOU_THRESHOLD = 0.45
NUM_CLASSES = 1
REG_MAX = 16

REACQUIRE_ID_GRACE_SEC = 15.0
MAX_FRAME_AGE_SEC = 0.20
STARTUP_HOME_SETTLE_SEC = 1.0

CAMERA_HFOV_DEG = float(os.getenv("DRONE_TRACKER_HFOV_DEG", "102.0" if CAMERA_WIDE_FOV else "47.0"))
CAMERA_VFOV_DEG = float(os.getenv("DRONE_TRACKER_VFOV_DEG", "67.0" if CAMERA_WIDE_FOV else "28.0"))
GIMBAL_PAN_SCALE = float(os.getenv("DRONE_TRACKER_PAN_SCALE", "0.85"))
GIMBAL_TILT_SCALE = float(os.getenv("DRONE_TRACKER_TILT_SCALE", "1.45"))
GIMBAL_EGO_SCALE = GIMBAL_PAN_SCALE
SERVO_LAG_TAU = float(os.getenv("DRONE_TRACKER_SERVO_TAU", "0.045"))
SERVO_LATENCY_SEC = float(os.getenv("DRONE_TRACKER_SERVO_LATENCY", "0.095"))

ROI_ENABLED = os.getenv("DRONE_TRACKER_ROI", "1").strip().lower() not in {"0", "false", "no"}
MOTION_CUE_ENABLED = os.getenv("DRONE_TRACKER_MOTION_CUE", "1").strip().lower() not in {"0", "false", "no"}
ROI_SIZE = INFER_SIZE
ROI_HINT_TTL = 0.7
ROI_FULL_EVERY = 10
ROI_SMALL_MAX = 72.0
ROI_HINT_DIST = 240.0

DEFAULT_PID = {
    "pan_kp": 0.006, "pan_ki": 0.0000, "pan_kd": 0.000,
    "tilt_kp": 0.0045, "tilt_ki": 0.0000, "tilt_kd": 0.000,
    "deadband": 20.0,
}

PID_FILE = os.path.join(BASE_DIR, "pid_config_tracking_v4.json")
PID_PARAM_LIMITS = {
    "pan_kp": (0.0, 0.2), "pan_ki": (0.0, 0.01), "pan_kd": (0.0, 0.1),
    "tilt_kp": (0.0, 0.2), "tilt_ki": (0.0, 0.01), "tilt_kd": (0.0, 0.1),
    "deadband": (0.0, 50.0),
}


def validate_pid(params):
    if not isinstance(params, dict):
        raise ValueError("PID config must be an object")
    checked = {}
    for key, value in params.items():
        if key not in PID_PARAM_LIMITS:
            continue
        value = float(value)
        lo, hi = PID_PARAM_LIMITS[key]
        if not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        checked[key] = value
    return checked


if os.path.exists(PID_FILE):
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            _saved = json.load(f)
            DEFAULT_PID.update(validate_pid(_saved))
        print(f"[CONFIG] Da tai cau hinh PID tu {PID_FILE}")
    except Exception as e:
        print(f"[CONFIG] Loi doc {PID_FILE}: {e}")

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_ENABLED = os.getenv("DRONE_TRACKER_WEB", "0").strip().lower() in {"1", "true", "yes", "on"}
LOCAL_DISPLAY_MODE = os.getenv("DRONE_TRACKER_HDMI", "auto").strip().lower()
LOCAL_DISPLAY_ENABLED = (
    LOCAL_DISPLAY_MODE in {"1", "true", "yes", "on"}
    or (LOCAL_DISPLAY_MODE == "auto" and bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")))
)
LOCAL_FULLSCREEN = os.getenv("DRONE_TRACKER_HDMI_FULLSCREEN", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
UI_FPS = max(1, int(os.getenv("DRONE_TRACKER_UI_FPS", "30")))


def put_latest(queue, value):
    if queue.full():
        try:
            queue.get_nowait()
        except Empty:
            pass
    queue.put_nowait(value)


class RateMeter:
    """Bộ đo tốc độ khung hình làm mượt bằng EMA."""

    def __init__(self, alpha=0.15):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.last_timestamp = None
        self.value = 0.0

    def update(self, timestamp):
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return self.value
        if not np.isfinite(timestamp):
            return self.value
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 1e-6:
                instant = 1.0 / dt
                self.value = instant if self.value <= 0 else (
                    (1.0 - self.alpha) * self.value + self.alpha * instant
                )
        self.last_timestamp = timestamp
        return self.value


class TargetRoiGuide:
    """Cầu nối thread-safe đưa vị trí & độ sáng mục tiêu từ T-PID về T-AI / T-Camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self.locked = False
        self.bbox = None       # (cx, cy, w, h)
        self.roi_bbox = None   # (cx, cy, w, h)
        self.lost_bbox = None  # (cx, cy, w, h)
        self.lost_time = None
        self.updated_at = None
        self.target_luma = None

    def set_locked(self, cx, cy, w, h, stamp, target_luma=None, roi_center=None):
        with self._lock:
            self.locked = True
            bw, bh = float(w), float(h)
            self.bbox = (float(cx), float(cy), bw, bh)
            if roi_center is not None and len(roi_center) >= 2:
                self.roi_bbox = (float(roi_center[0]), float(roi_center[1]), bw, bh)
            else:
                self.roi_bbox = self.bbox
            self.lost_bbox = None
            self.lost_time = None
            self.updated_at = float(stamp)
            if target_luma is not None and np.isfinite(target_luma):
                self.target_luma = float(target_luma)

    def set_lost(self, lost_bbox, lost_time):
        with self._lock:
            self.locked = False
            self.bbox = None
            self.roi_bbox = None
            self.target_luma = None
            if lost_bbox is not None and lost_time is not None:
                lx1, ly1, lx2, ly2 = lost_bbox
                self.lost_bbox = (
                    (lx1 + lx2) / 2.0,
                    (ly1 + ly2) / 2.0,
                    max(1.0, lx2 - lx1),
                    max(1.0, ly2 - ly1),
                )
                self.lost_time = float(lost_time)
            else:
                self.lost_bbox = None
                self.lost_time = None

    def snapshot(self, now, lost_ttl=1.8):
        with self._lock:
            if self.locked and self.bbox is not None:
                return {
                    "locked": True,
                    "bbox": self.bbox,
                    "roi_bbox": self.roi_bbox or self.bbox,
                    "lost_bbox": None,
                    "target_luma": self.target_luma,
                }
            if (
                self.lost_bbox is not None
                and self.lost_time is not None
                and 0.0 <= now - self.lost_time <= lost_ttl
            ):
                return {
                    "locked": False,
                    "bbox": None,
                    "roi_bbox": None,
                    "lost_bbox": self.lost_bbox,
                    "target_luma": None,
                }
            return {
                "locked": False,
                "bbox": None,
                "roi_bbox": None,
                "lost_bbox": None,
                "target_luma": None,
            }


class SharedState:
    """Quản lý trạng thái vận hành thread-safe giữa AI, PID và Web Dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self.ai_fps = 0.0
        self.drone_count = 0
        self.max_conf = 0.0
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.error_x = 0.0
        self.error_y = 0.0
        self.is_tracking = False
        self.tracker_state = "IDLE"
        self.control_state = "HOLD"
        self.track_id = 0
        self.total_detect = 0
        self.start_time = time.time()
        self.pid_params = dict(DEFAULT_PID)
        self.pid_dirty = False
        self.backend = "INIT"
        self.roi_mode = "FULL"
        self.stream_viewers = 0

    def update_stats(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def add_detections(self, count):
        with self._lock:
            self.total_detect += int(count)

    def get_stats(self):
        with self._lock:
            return {
                "ai_fps": round(self.ai_fps, 1),
                "drone_count": self.drone_count,
                "max_conf": round(self.max_conf * 100, 1),
                "pan_angle": round(self.pan_angle, 1),
                "tilt_angle": round(self.tilt_angle, 1),
                "error_x": round(self.error_x, 1),
                "error_y": round(self.error_y, 1),
                "is_tracking": self.is_tracking,
                "tracker_state": self.tracker_state,
                "control_state": self.control_state,
                "track_id": self.track_id,
                "total_detect": self.total_detect,
                "uptime": round(time.time() - self.start_time, 1),
                "backend": self.backend,
                "roi_mode": self.roi_mode,
                "pid": dict(self.pid_params),
            }

    def set_pid(self, params):
        params = validate_pid(params)
        if not params:
            raise ValueError("no valid PID parameters")
        with self._lock:
            self.pid_params.update(params)
            self.pid_dirty = True

    def consume_pid_dirty(self):
        with self._lock:
            d = self.pid_dirty
            self.pid_dirty = False
            return d, dict(self.pid_params)


STATE = SharedState()
TARGET_GUIDE = TargetRoiGuide()
STOP_EVENT = threading.Event()

q_raw = Queue(maxsize=1)
q_data = Queue(maxsize=1)
q_stream = Queue(maxsize=1)
q_local = Queue(maxsize=1)

CAM_DIAG_LOCK = threading.Lock()
CAM_DIAG = {
    "exp_us": 0.0,
    "gain": 0.0,
    "spot_y": 0.0,
    "frame_y": 0.0,
    "af_state": -1,
}

DIAG_LOG_PATH = os.path.join(BASE_DIR, "tracker_diag.log")
ID_LOG_PATH = os.path.join(BASE_DIR, "id_debug.log")
_DIAG_FH = None
_DIAG_WRITES = 0


def _diag_log_line(line):
    """Ghi log chẩn đoán với file descriptor được giữ mở (line-buffered) chống nghẽn I/O ở 120 FPS."""
    global _DIAG_FH, _DIAG_WRITES
    try:
        if _DIAG_FH is None or (_DIAG_WRITES % 200 == 0):
            if os.path.exists(DIAG_LOG_PATH) and os.path.getsize(DIAG_LOG_PATH) > 5 * 1024 * 1024:
                if _DIAG_FH is not None:
                    try:
                        _DIAG_FH.close()
                    except OSError:
                        pass
                    _DIAG_FH = None
                try:
                    os.replace(DIAG_LOG_PATH, DIAG_LOG_PATH + ".prev")
                except OSError:
                    pass
            if _DIAG_FH is None:
                _DIAG_FH = open(DIAG_LOG_PATH, "a", encoding="utf-8", buffering=1)
        _DIAG_FH.write(line + "\n")
        _DIAG_WRITES += 1
    except OSError:
        if _DIAG_FH is not None:
            try:
                _DIAG_FH.close()
            except OSError:
                pass
            _DIAG_FH = None


def _id_log(msg):
    try:
        with open(ID_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass
