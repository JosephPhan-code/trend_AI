"""
==============================================================================
  THIẾT KẾ HỆ THỐNG CAMERA TỰ ĐỘNG BÁM BẮT DRONE ỨNG DỤNG TRÍ TUỆ NHÂN TẠO
==============================================================================
  File    : main.py  —  Điểm vào trung tâm (OOP Entry Point & Backward-Compatible Facade)
  Kiến trúc 모듈 OOP:
      • config.py           : Cấu hình, hằng số, SharedState, TargetRoiGuide, Buffered Logger
      • servo_controller.py : Điều khiển 2 trục Pan/Tilt Hardware PWM Sysfs / Gpiozero (50Hz)
      • camera_engine.py    : CameraEngine (Pi Camera V3 + PiSP Dual-Stream lores + Dynamic AF/Spot AE)
      • detector_engine.py  : HailoDetector (UINT8/UINT16 Threshold-First Zero-Dequant + Async NPU)
                              & CpuFallbackDetector (Ultralytics YOLO)
      • tracker_ekf.py      : ExtendedKalmanTracker (8-state EKF + Smith-Predictor), DroneLockon,
                              MotionDetector (phaseCorrelate) & RoiPlanner (SAHI Overlap)
      • controller_pid.py   : StableTrackingController (Dual PID + Feed-Forward + Coasting)
                              & TrackingWorker
      • web_dashboard.py    : WebDashboardServer (Flask) & LocalDisplayWorker (HDMI)
==============================================================================
"""

import threading
import time

from config import (
    BASE_DIR,
    CAM_DIAG,
    CAM_DIAG_LOCK,
    CAMERA_AF_MODE,
    CAMERA_EV,
    CAMERA_FAST_AE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_HFOV_DEG,
    CAMERA_LENS_POS,
    CAMERA_SHARPNESS,
    CAMERA_SPOT_PREF_SHUTTER_US,
    CAMERA_USE_LORES,
    CAMERA_VFOV_DEG,
    CAMERA_WIDE_FOV,
    CAMERA_WIDTH,
    CONF_THRESHOLD,
    DEFAULT_PID,
    DETECTION_FLOOR,
    DIAG_LOG_PATH,
    GIMBAL_EGO_SCALE,
    GIMBAL_PAN_SCALE,
    GIMBAL_TILT_SCALE,
    HEF_MODEL_PATH,
    ID_LOG_PATH,
    INFER_SIZE,
    IOU_THRESHOLD,
    LOCAL_DISPLAY_ENABLED,
    LOCAL_DISPLAY_MODE,
    LOCAL_FULLSCREEN,
    LORES_GAIN,
    LORES_HEIGHT,
    LORES_PAD_Y,
    LORES_WIDTH,
    MAX_FRAME_AGE_SEC,
    MOTION_CUE_ENABLED,
    NUM_CLASSES,
    PID_FILE,
    PID_PARAM_LIMITS,
    PT_MODEL_PATH,
    REACQUIRE_ID_GRACE_SEC,
    REG_MAX,
    ROI_ENABLED,
    ROI_FULL_EVERY,
    ROI_HINT_DIST,
    ROI_HINT_TTL,
    ROI_SIZE,
    ROI_SMALL_MAX,
    SERVO_LAG_TAU,
    SERVO_LATENCY_SEC,
    STARTUP_HOME_SETTLE_SEC,
    STATE,
    STOP_EVENT,
    TARGET_GUIDE,
    UI_FPS,
    WEB_HOST,
    WEB_PORT,
    RateMeter,
    SharedState,
    TargetRoiGuide,
    _diag_log_line,
    _id_log,
    put_latest,
    q_data,
    q_local,
    q_raw,
    q_stream,
    validate_pid,
)
from servo_controller import ServoController
from tracker_ekf import (
    GIMBAL_POSE,
    DroneLockon,
    ExtendedKalmanTracker,
    GimbalPoseTracker,
    KalmanTracker,
    MotionDetector,
    RoiPlanner,
    _offset_detections,
    detect_with_roi,
)
from camera_engine import (
    CameraEngine,
    TargetSpotAeController,
    _build_fast_camera_tuning,
    _compute_af_window,
    _compute_target_ev,
    _measure_target_luma,
    _split_exposure_product,
    thread_camera,
)
from detector_engine import (
    _ANCHORS,
    _ANCHORS_BY_GRID,
    _LOGIT_FLOOR,
    _STRIDES,
    BaseDetector,
    CpuFallbackDetector,
    HailoDetector,
    _decode_end_to_end,
    _decode_result_dict,
    _dfl,
    _make_anchors,
    _nms,
    _push_results,
    _restore_boxes,
    _run_cpu_loop,
    _run_hailo_loop,
    letterbox_image,
    postprocess_hailo,
    postprocess_hailo_quant,
    thread_ai,
)
from controller_pid import (
    StableTrackingController,
    TrackingWorker,
    _annotate,
    _draw_crosshair,
    thread_pid_servo,
)
from web_dashboard import (
    HTML,
    LocalDisplayWorker,
    WebDashboardServer,
    _gen_stream,
    api_set_pid,
    api_shutdown,
    api_stats,
    app,
    index,
    thread_local_display,
    thread_web,
    video_feed,
)


_INSTANCE_LOCK = None


def _acquire_instance_lock():
    global _INSTANCE_LOCK
    try:
        import fcntl
        lock_path = "/tmp/drone_tracker_main.lock"
        lock = open(lock_path, "a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            print("[MAIN] Da co mot main.py dang chay; khong khoi dong ban thu hai.")
            return False
        _INSTANCE_LOCK = (lock, fcntl)
        return True
    except ImportError:
        return True
    except OSError as exc:
        print(f"[MAIN] Khong tao duoc lock tien trinh ({exc}); tiep tuc chay.")
        return True


def _release_instance_lock():
    global _INSTANCE_LOCK
    if _INSTANCE_LOCK is None:
        return
    lock, fcntl = _INSTANCE_LOCK
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        lock.close()
    except OSError:
        pass
    _INSTANCE_LOCK = None


class AegisTrackerApplication:
    """Lớp điều phối trung tâm (Orchestrator) quản lý vòng đời của toàn bộ hệ thống."""

    def __init__(self):
        self.servo = None
        self.threads = []

    def start(self):
        if not _acquire_instance_lock():
            return
        STOP_EVENT.clear()
        print("=" * 65)
        print("  THIET KE HE THONG CAMERA TU DONG BAM BAT DRONE UNG DUNG AI")
        print("  Pi 5 + Hailo-8 26 TOPS NPU + PID Servo + Web Dashboard (OOP)")
        print(
            f"[PID] Profile: pan Kp={DEFAULT_PID['pan_kp']:.4f}, "
            f"tilt Kp={DEFAULT_PID['tilt_kp']:.4f}, deadband={DEFAULT_PID['deadband']:.1f}px"
        )
        print("=" * 65)

        try:
            self.servo = ServoController()
            GIMBAL_POSE.bind_servo(self.servo)
            print("[MAIN] Khoi dong servo: dua ve Home (0°, 0°) trong 3 giay...")
            self.servo.go_home()
            time.sleep(3.0)

            print("[MAIN] Tam ngat xung PWM 1 giay de xa tai co khi va triet tieu rung giat...")
            self.servo.disable_pwm()
            time.sleep(1.0)

            print("[MAIN] Mo han PWM va bat dau he thong tracking...")
            self.servo.enable_pwm()
            time.sleep(0.5)

            self.threads = [
                threading.Thread(target=thread_camera, name="T-Camera", daemon=True),
                threading.Thread(target=thread_ai, name="T-AI", daemon=True),
                threading.Thread(target=thread_pid_servo, args=(self.servo,), name="T-PID", daemon=True),
                threading.Thread(target=thread_web, name="T-Web", daemon=True),
            ]
            if LOCAL_DISPLAY_ENABLED:
                self.threads.append(
                    threading.Thread(target=thread_local_display, name="T-HDMI", daemon=True)
                )
            for t in self.threads:
                t.start()
                print(f"[MAIN] Luong '{t.name}' da khoi dong.")

            print("[MAIN] He thong hoat dong! Nhan Ctrl+C hoac Click Chuot Phai de dung.\n")
            while not STOP_EVENT.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[MAIN] Dang tat an toan...")
        finally:
            STOP_EVENT.set()
            if self.servo is not None:
                self.servo.shutdown()
            _release_instance_lock()


def main():
    AegisTrackerApplication().start()


if __name__ == "__main__":
    main()
