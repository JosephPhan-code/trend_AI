"""
==============================================================================
Module: camera_engine.py
Mô tả : Lớp OOP CameraEngine quản lý Pi Camera V3 (picamera2),
        tích hợp luồng kép phần cứng PiSP (main 1280x720 + lores 640x360),
        lấy nét động theo mục tiêu (AfWindows) và đo sáng điểm bám chống nhòe
        (TargetSpotAeController).
==============================================================================
"""

import time
import cv2
import numpy as np

from config import (
    CAM_DIAG,
    CAM_DIAG_LOCK,
    CAMERA_AF_MODE,
    CAMERA_EV,
    CAMERA_FAST_AE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_LENS_POS,
    CAMERA_SHARPNESS,
    CAMERA_SPOT_PREF_SHUTTER_US,
    CAMERA_USE_LORES,
    CAMERA_WIDE_FOV,
    CAMERA_WIDTH,
    LORES_HEIGHT,
    LORES_WIDTH,
    STOP_EVENT,
    TARGET_GUIDE,
    put_latest,
    q_raw,
)
from tracker_ekf import GIMBAL_POSE


def _build_fast_camera_tuning(picam2_cls, wide_fov=False):
    """Tăng tốc độ hội tụ AE/AGC và ưu tiên màn trập nhanh chống nhòe drone."""
    if not CAMERA_FAST_AE:
        return None
    tuning_name = "imx708_wide.json" if wide_fov else "imx708.json"
    try:
        tuning = picam2_cls.load_tuning_file(tuning_name)
    except Exception:
        try:
            tuning = picam2_cls.load_tuning_file("imx708.json")
        except Exception:
            return None
    try:
        for alg in tuning.get("algorithms", []):
            if "rpi.agc" in alg:
                agc = alg["rpi.agc"]
                targets = agc["channels"][:1] if agc.get("channels") else [agc]
                for ch in targets:
                    ch["speed"] = 0.65
                    ch["fast_reduce_threshold"] = 0.35
                    ch["convergence_frames"] = 4
                    modes = ch.get("exposure_modes")
                    if isinstance(modes, dict):
                        modes["short"] = {
                            "shutter": [100, 4000, 8000, 14000, 32000],
                            "gain": [1.0, 2.5, 5.0, 9.0, 16.0],
                        }
        return tuning
    except Exception:
        return None


def _compute_af_window(frame_w, frame_h, scaler_crop, bbox=None):
    """Ánh xạ vùng lấy nét (drone bbox hoặc tâm 50%) sang tọa độ cảm biến ScalerCrop."""
    sx, sy, sw, sh = [int(v) for v in scaler_crop]
    if sw < 64 or sh < 64 or frame_w <= 0 or frame_h <= 0:
        return [(sx, sy, max(sw, 64), max(sh, 64))]
    if bbox is None:
        nx, ny, nw, nh = 0.25, 0.25, 0.50, 0.50
    else:
        cx, cy, bw, bh = bbox
        win_w = max(bw * 2.5, 180.0)
        win_h = max(bh * 2.5, 180.0)
        nw = min(0.65, win_w / frame_w)
        nh = min(0.65, win_h / frame_h)
        nx = min(max(cx / frame_w - nw * 0.5, 0.0), 1.0 - nw)
        ny = min(max(cy / frame_h - nh * 0.5, 0.0), 1.0 - nh)
    rx = int(round(sx + nx * sw))
    ry = int(round(sy + ny * sh))
    rw = max(64, int(round(nw * sw)))
    rh = max(64, int(round(nh * sh)))
    rx = min(max(rx, sx), sx + sw - rw)
    ry = min(max(ry, sy), sy + sh - rh)
    return [(rx, ry, rw, rh)]


def _measure_target_luma(frame, bbox_xyxy):
    """Đo độ sáng (0..255) tại lõi 60% bbox drone, chống nhiễu nền trời/phản quang."""
    if frame is None or frame.ndim != 3 or not frame.size or bbox_xyxy is None:
        return None
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if bw >= 10.0 and bh >= 10.0:
        mx, my = bw * 0.2, bh * 0.2
        x1, x2 = x1 + mx, x2 - mx
        y1, y2 = y1 + my, y2 - my
    ix1 = max(0, min(fw - 1, int(round(x1))))
    iy1 = max(0, min(fh - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(fw, int(round(x2))))
    iy2 = max(iy1 + 1, min(fh, int(round(y2))))
    sh = max(1, (iy2 - iy1) // 48)
    sw = max(1, (ix2 - ix1) // 48)
    crop = frame[iy1:iy2:sh, ix1:ix2:sw]
    if not crop.size:
        return None
    if crop.shape[0] * crop.shape[1] >= 16:
        y_map = (
            0.114 * crop[..., 0].astype(np.float32)
            + 0.587 * crop[..., 1].astype(np.float32)
            + 0.299 * crop[..., 2].astype(np.float32)
        )
        return float(0.65 * y_map.mean() + 0.35 * np.median(y_map))
    bgr = crop.mean(axis=(0, 1))
    return float(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])


def _compute_target_ev(base_ev, target_luma, prev_ev):
    if target_luma is None or not np.isfinite(target_luma):
        desired = float(base_ev)
    elif target_luma < 78.0:
        desired = float(base_ev) + min(1.4, (78.0 - target_luma) / 52.0)
    elif target_luma > 172.0:
        desired = float(base_ev) - min(1.0, (target_luma - 172.0) / 60.0)
    else:
        desired = float(base_ev)
    smoothed = 0.72 * float(prev_ev) + 0.28 * desired
    return float(np.clip(smoothed, base_ev - 1.0, base_ev + 1.4))


def _split_exposure_product(total_eg, pref_exp_us=CAMERA_SPOT_PREF_SHUTTER_US,
                            max_exp_us=32700, min_gain=1.13, max_gain=15.8):
    """Tách tích phơi sáng (us * gain) thành (ExposureTime, AnalogueGain) chống nhòe."""
    lo_p = 100.0 * float(min_gain)
    hi_p = float(max_exp_us) * float(max_gain)
    p = float(np.clip(total_eg, lo_p, hi_p))
    pref = float(min(max_exp_us, max(500, pref_exp_us)))
    mid_gain = 8.0
    if p <= pref * min_gain:
        exp_us = int(max(100, round(p / min_gain)))
        return exp_us, round(float(min_gain), 3)
    if p <= pref * mid_gain:
        return int(round(pref)), round(float(p / pref), 3)
    if p <= float(max_exp_us) * mid_gain:
        return int(round(p / mid_gain)), round(float(mid_gain), 3)
    gain = min(float(max_gain), p / float(max_exp_us))
    return int(max_exp_us), round(float(gain), 3)


class TargetSpotAeController:
    """Đo sáng điểm bám theo bbox drone bằng ExposureTime + AnalogueGain (AeEnable=False)."""

    def __init__(self, base_ev=0.0, max_exp_us=32700,
                 pref_exp_us=CAMERA_SPOT_PREF_SHUTTER_US,
                 min_gain=1.13, max_gain=15.8):
        self.base_ev = float(base_ev)
        self.max_exp_us = int(max(4000, max_exp_us))
        self.pref_exp_us = int(min(self.max_exp_us, max(500, pref_exp_us)))
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.target_y = float(np.clip(104.0 * (2.0 ** (self.base_ev * 0.45)), 60.0, 170.0))
        self.low_y = max(45.0, self.target_y - 24.0)
        self.high_y = min(205.0, self.target_y + 36.0)
        self.reset()

    def reset(self):
        self._active = False
        self._cmd_eg = None
        self._cmd_time = 0.0
        self._last_guide_luma = None
        self._last_det_time = 0.0

    def update(self, now, seen_exp, seen_gain, target_luma, frame_luma=None, guide_luma=None):
        now = float(now)
        if guide_luma is not None and np.isfinite(guide_luma):
            gl = float(guide_luma)
            if (
                self._last_guide_luma is None
                or abs(gl - self._last_guide_luma) > 1e-6
                or gl >= 245.0
            ):
                self._last_guide_luma = gl
                self._last_det_time = now
            fresh_det = (now - self._last_det_time) <= 0.35
        else:
            fresh_det = True

        if seen_exp is None or seen_exp <= 0 or seen_gain is None or seen_gain <= 0:
            return None
        seen_eg = float(seen_exp) * float(seen_gain)

        if not self._active:
            self._active = True
            self._last_det_time = now
            exp0, gain0 = _split_exposure_product(
                seen_eg,
                pref_exp_us=self.pref_exp_us,
                max_exp_us=self.max_exp_us,
                min_gain=self.min_gain,
                max_gain=self.max_gain,
            )
            self._cmd_eg = float(exp0) * float(gain0)
            self._cmd_time = now
            return {
                "AeEnable": False,
                "ExposureTime": exp0,
                "AnalogueGain": gain0,
            }

        settled = (
            self._cmd_eg is None
            or abs(seen_eg - self._cmd_eg) / max(1.0, self._cmd_eg) <= 0.08
        )
        if not settled and (now - self._cmd_time) < 0.38:
            return None

        if not fresh_det or target_luma is None or not np.isfinite(target_luma):
            return None

        y_spot = float(target_luma)
        if self.low_y <= y_spot <= self.high_y:
            return None

        if y_spot < self.low_y and frame_luma is not None and float(frame_luma) >= 210.0:
            return None

        ratio = self.target_y / max(10.0, y_spot)
        mult = float(np.clip(ratio ** 1.35, 0.45, 2.20))
        if mult > 1.0 and frame_luma is not None and float(frame_luma) > 130.0:
            max_safe = max(1.0, (220.0 / max(10.0, float(frame_luma))) ** 1.2)
            mult = min(mult, max_safe)

        new_eg = float(np.clip(
            seen_eg * mult,
            100.0 * self.min_gain,
            float(self.max_exp_us) * self.max_gain,
        ))
        exp_cmd, gain_cmd = _split_exposure_product(
            new_eg,
            pref_exp_us=self.pref_exp_us,
            max_exp_us=self.max_exp_us,
            min_gain=self.min_gain,
            max_gain=self.max_gain,
        )
        cmd_eg = float(exp_cmd) * float(gain_cmd)
        if abs(cmd_eg - seen_eg) / max(1.0, seen_eg) < 0.06:
            return None

        self._cmd_eg = cmd_eg
        self._cmd_time = now
        return {
            "ExposureTime": int(exp_cmd),
            "AnalogueGain": float(gain_cmd),
        }


class CameraEngine:
    """Đóng gói OOP luồng thu nhận hình ảnh từ Pi Camera V3 (hoặc OpenCV fallback)."""

    def __init__(self, raw_queue=None, target_guide=None, gimbal_pose=None, stop_event=None):
        self.raw_queue = raw_queue or q_raw
        self.target_guide = target_guide or TARGET_GUIDE
        self.gimbal_pose = gimbal_pose or GIMBAL_POSE
        self.stop_event = stop_event or STOP_EVENT

    def run(self):
        print("[CAM] Khoi dong Pi Camera V3 (picamera2)...")
        try:
            from picamera2 import Picamera2
            tuning = _build_fast_camera_tuning(Picamera2, wide_fov=CAMERA_WIDE_FOV)
            try:
                picam2 = Picamera2(tuning=tuning) if tuning is not None else Picamera2()
            except Exception:
                picam2 = Picamera2()

            cfg_kwargs = {
                "main": {"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            }
            if CAMERA_WIDE_FOV:
                cfg_kwargs["sensor"] = {"output_size": (2304, 1296)}

            has_lores = False
            if CAMERA_USE_LORES:
                try:
                    cfg = picam2.create_preview_configuration(
                        **cfg_kwargs,
                        lores={"size": (LORES_WIDTH, LORES_HEIGHT), "format": "RGB888"},
                        buffer_count=2,
                        queue=False,
                    )
                    has_lores = True
                except Exception:
                    has_lores = False

            if not has_lores:
                try:
                    cfg = picam2.create_preview_configuration(**cfg_kwargs, buffer_count=2, queue=False)
                except Exception:
                    try:
                        cfg = picam2.create_preview_configuration(**cfg_kwargs, buffer_count=3, queue=False)
                    except TypeError:
                        try:
                            cfg = picam2.create_preview_configuration(**cfg_kwargs)
                        except TypeError:
                            cfg = picam2.create_preview_configuration(
                                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                            )

            picam2.configure(cfg)
            picam2.start()

            if has_lores:
                print(f"[CAM] PiSP Dual-Stream Hardware Resize ON: main={CAMERA_WIDTH}x{CAMERA_HEIGHT}, lores={LORES_WIDTH}x{LORES_HEIGHT}")

            scaler_crop = (768, 432, 3072, 1728)
            try:
                md0 = picam2.capture_metadata()
                if md0 and "ScalerCrop" in md0:
                    scaler_crop = tuple(md0["ScalerCrop"])
            except Exception:
                pass

            try:
                pas = picam2.camera_properties.get("PixelArraySize", (4608, 2592))
                full_w, full_h = float(pas[0]), float(pas[1])
                base_hfov = 102.0 if CAMERA_WIDE_FOV else 66.0
                base_vfov = 67.0 if CAMERA_WIDE_FOV else 41.0
                crop_rw = float(scaler_crop[2]) / max(full_w, 1.0)
                crop_rh = float(scaler_crop[3]) / max(full_h, 1.0)
                eff_hfov = float(np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(base_hfov * 0.5)) * crop_rw)))
                eff_vfov = float(np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(base_vfov * 0.5)) * crop_rh)))
                self.gimbal_pose.set_optical_fov(eff_hfov, eff_vfov)
                print(f"[CAM] Active FOV calibrated from ScalerCrop={scaler_crop}: {eff_hfov:.1f}x{eff_vfov:.1f} deg")
            except Exception:
                pass

            use_dynamic_target_cam = False
            try:
                from libcamera import controls as lc
                cam_controls = {
                    "AeEnable": True,
                    "AwbEnable": True,
                    "AeExposureMode": lc.AeExposureModeEnum.Short,
                    "AeConstraintMode": lc.AeConstraintModeEnum.Normal,
                    "AeMeteringMode": lc.AeMeteringModeEnum.Matrix,
                    "ExposureValue": CAMERA_EV,
                    "Sharpness": CAMERA_SHARPNESS,
                }
                if hasattr(lc, "AwbModeEnum"):
                    cam_controls["AwbMode"] = lc.AwbModeEnum.Auto
                if CAMERA_AF_MODE == "manual":
                    cam_controls["AfMode"] = lc.AfModeEnum.Manual
                    cam_controls["LensPosition"] = CAMERA_LENS_POS
                else:
                    cam_controls["AfMode"] = lc.AfModeEnum.Continuous
                    if hasattr(lc, "AfSpeedEnum"):
                        cam_controls["AfSpeed"] = lc.AfSpeedEnum.Fast
                    if hasattr(lc, "AfRangeEnum"):
                        cam_controls["AfRange"] = lc.AfRangeEnum.Normal
                    if hasattr(lc, "AfMeteringEnum"):
                        cam_controls["AfMetering"] = lc.AfMeteringEnum.Auto
                        use_dynamic_target_cam = True
                if hasattr(lc, "draft") and hasattr(lc.draft, "NoiseReductionModeEnum"):
                    cam_controls["NoiseReductionMode"] = lc.draft.NoiseReductionModeEnum.Fast
                picam2.set_controls(cam_controls)
                print(
                    f"[CAM] Controls: AF={CAMERA_AF_MODE} (IDLE=FullFrame -> LOCKED=TargetAF+TargetAE={use_dynamic_target_cam}), "
                    f"FastAE={tuning is not None}, Sharp={CAMERA_SHARPNESS}, EV={CAMERA_EV}, WideFOV={CAMERA_WIDE_FOV}"
                )
            except Exception as ce:
                print(f"[CAM] Khong dat duoc full controls ({ce}); dung auto settings")
            try:
                frame_us = max(1000, int(round(1_000_000 / max(CAMERA_FPS, 1))))
                picam2.set_controls({"FrameDurationLimits": (frame_us, frame_us)})
                print(f"[CAM] OK Pi Camera V3: {CAMERA_WIDTH}x{CAMERA_HEIGHT} - Target FPS: {CAMERA_FPS}")
            except Exception:
                print(f"[CAM] OK Pi Camera V3: {CAMERA_WIDTH}x{CAMERA_HEIGHT} (Frame rate auto)")

            last_af_update = 0.0
            last_ae_luma_t = 0.0
            cam_tracking_mode = False
            last_af_center = None
            frame_period_us = max(1000, int(round(1_000_000 / max(CAMERA_FPS, 1))))
            ae_ctrl = TargetSpotAeController(
                base_ev=CAMERA_EV,
                max_exp_us=max(4000, min(40000, frame_period_us - 600)),
            )

            while not self.stop_event.is_set():
                cam_md = None
                lores_frame = None
                if hasattr(picam2, "captured_request"):
                    with picam2.captured_request() as req:
                        frame = req.make_array("main")
                        if has_lores:
                            try:
                                lores_frame = req.make_array("lores")
                            except Exception:
                                lores_frame = None
                        if use_dynamic_target_cam:
                            cam_md = req.get_metadata()
                else:
                    frame = picam2.capture_array()

                stamp = time.monotonic()
                gimbal_cap = self.gimbal_pose.sample_now(stamp)
                put_latest(
                    self.raw_queue,
                    {"frame": frame, "lores": lores_frame, "timestamp": stamp, "gimbal": gimbal_cap},
                )

                if cam_md:
                    with CAM_DIAG_LOCK:
                        CAM_DIAG["exp_us"] = float(cam_md.get("ExposureTime", 0.0) or 0.0)
                        CAM_DIAG["gain"] = float(cam_md.get("AnalogueGain", 0.0) or 0.0)
                        CAM_DIAG["af_state"] = int(cam_md.get("AfState", -1) if cam_md.get("AfState") is not None else -1)

                if use_dynamic_target_cam:
                    guide = self.target_guide.snapshot(stamp)
                    if guide["locked"] and guide["bbox"] is not None:
                        cur_bbox = guide["bbox"]
                        cur_center = (cur_bbox[0], cur_bbox[1])
                        upd = {}
                        moved = (
                            not cam_tracking_mode
                            or last_af_center is None
                            or np.hypot(cur_center[0] - last_af_center[0], cur_center[1] - last_af_center[1]) >= 28.0
                        )
                        if moved and (not cam_tracking_mode or stamp - last_af_update >= 0.12):
                            upd["AfMetering"] = lc.AfMeteringEnum.Windows
                            upd["AfWindows"] = _compute_af_window(
                                CAMERA_WIDTH, CAMERA_HEIGHT, scaler_crop, cur_bbox
                            )
                            last_af_center = cur_center
                            last_af_update = stamp

                        if (not cam_tracking_mode) or (stamp - last_ae_luma_t >= 0.05):
                            last_ae_luma_t = stamp
                            spot_xyxy = (
                                cur_bbox[0] - 0.5 * cur_bbox[2],
                                cur_bbox[1] - 0.5 * cur_bbox[3],
                                cur_bbox[0] + 0.5 * cur_bbox[2],
                                cur_bbox[1] + 0.5 * cur_bbox[3],
                            )
                            spot_luma = _measure_target_luma(frame, spot_xyxy)
                            if spot_luma is None:
                                spot_luma = guide["target_luma"]
                            sub_frame = lores_frame[::8, ::8] if lores_frame is not None else frame[::16, ::16]
                            bm, gm, rm = sub_frame.mean(axis=(0, 1))
                            frame_luma = float(0.114 * bm + 0.587 * gm + 0.299 * rm)
                            with CAM_DIAG_LOCK:
                                CAM_DIAG["spot_y"] = float(spot_luma) if spot_luma is not None else 0.0
                                CAM_DIAG["frame_y"] = frame_luma
                            ae_upd = ae_ctrl.update(
                                stamp,
                                cam_md.get("ExposureTime") if cam_md else None,
                                cam_md.get("AnalogueGain") if cam_md else None,
                                spot_luma,
                                frame_luma=frame_luma,
                                guide_luma=guide["target_luma"],
                            )
                            if ae_upd:
                                upd.update(ae_upd)

                        cam_tracking_mode = True
                        if upd:
                            try:
                                picam2.set_controls(upd)
                            except Exception:
                                use_dynamic_target_cam = False
                    elif cam_tracking_mode:
                        cam_tracking_mode = False
                        last_af_center = None
                        ae_ctrl.reset()
                        try:
                            picam2.set_controls({
                                "AfMetering": lc.AfMeteringEnum.Auto,
                                "AeEnable": True,
                                "AeMeteringMode": lc.AeMeteringModeEnum.Matrix,
                                "ExposureValue": CAMERA_EV,
                            })
                        except Exception:
                            use_dynamic_target_cam = False
            picam2.stop()
        except Exception as e:
            if self.stop_event.is_set():
                return
            print(f"[CAM] picamera2 loi ({e}) -> fallback VideoCapture(0)")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            print(f"[CAM] OK VideoCapture(0): {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                stamp = time.monotonic()
                gimbal_cap = self.gimbal_pose.sample_now(stamp)
                put_latest(self.raw_queue, {"frame": frame, "lores": None, "timestamp": stamp, "gimbal": gimbal_cap})
            cap.release()


def thread_camera():
    CameraEngine().run()
