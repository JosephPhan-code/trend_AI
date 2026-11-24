"""
==============================================================================
Module: controller_pid.py
Mô tả : Lớp OOP StableTrackingController (Dual PID hai trục theo thời gian,
        hysteresis, phanh tuyến tính, giảm chấn tâm, bù vận tốc đón đầu, coasting)
        và TrackingWorker điều phối luồng T-PID + vẽ OSD (_annotate).
==============================================================================
"""

import time
from queue import Empty
import cv2
import numpy as np

from camera_engine import _measure_target_luma
from config import (
    CAM_DIAG,
    CAM_DIAG_LOCK,
    CONF_THRESHOLD,
    LOCAL_DISPLAY_ENABLED,
    MAX_FRAME_AGE_SEC,
    STATE,
    STOP_EVENT,
    TARGET_GUIDE,
    UI_FPS,
    _diag_log_line,
    put_latest,
    q_data,
    q_local,
    q_stream,
)
from servo_controller import ServoController
from tracker_ekf import GIMBAL_POSE, DroneLockon


class StableTrackingController:
    """PID hai trục theo thời gian, hysteresis và giới hạn tốc độ lệnh."""

    REFERENCE_HZ = 30.0
    FILTER_TAU_NEAR = 0.038
    FILTER_TAU_FAR = 0.016
    VELOCITY_FILTER_TAU = 0.10
    MEASUREMENT_MEDIAN_WINDOW = 1
    CENTER_DAMPING_GAIN = np.array([0.0011, 0.00095])
    CENTER_DAMPING_LIMIT = np.array([14.0, 10.0])
    DERIVATIVE_TAU = 0.08
    MAX_SPEED = np.array([85.0, 50.0])
    MAX_ACCEL = np.array([420.0, 260.0])
    BRAKE_ZONE = np.array([250.0, 180.0])
    MAX_DELTA = 2.5
    TILT_MAX_DELTA = 2.0
    DEADBAND_SCALE = np.array([1.0, 1.35])
    VELOCITY_FF_GAIN = np.array([0.010, 0.007])
    VELOCITY_FF_LIMIT = np.array([12.0, 8.0])

    def __init__(self, params):
        self.params = dict(params)
        self.reset()

    def configure(self, params):
        self.params = dict(params)
        self.reset()

    def reset(self):
        self.last_time = None
        self.errors = np.zeros(2, dtype=float)
        self.raw_history = []
        self.integral = np.zeros(2, dtype=float)
        self.derivative = np.zeros(2, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.rates = np.zeros(2, dtype=float)
        self.moving = [False, False]

    def step(self, cx, cy, width, height, timestamp, prediction_scale=1.0,
             velocity=(0.0, 0.0)):
        values = np.asarray([cx, cy, width, height, timestamp, prediction_scale], dtype=float)
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        if velocity.size != 2 or not np.all(np.isfinite(velocity)):
            velocity = np.zeros(2, dtype=float)
        if not np.all(np.isfinite(values)) or width <= 0 or height <= 0:
            self.reset()
            return 0.0, 0.0
        if self.last_time is not None and timestamp <= self.last_time:
            return 0.0, 0.0
        raw = np.array([
            (cx - width / 2) * 1280.0 / width,
            (cy - height / 2) * 720.0 / height,
        ])
        first = self.last_time is None
        elapsed = 1.0 / self.REFERENCE_HZ if first else timestamp - self.last_time
        if elapsed > MAX_FRAME_AGE_SEC:
            self.reset()
            first = True
            elapsed = 1.0 / self.REFERENCE_HZ
        dt = min(elapsed, 0.1)
        previous = self.errors.copy()
        self.raw_history.append(raw.copy())
        if len(self.raw_history) > self.MEASUREMENT_MEDIAN_WINDOW:
            self.raw_history.pop(0)
        measured = np.median(np.vstack(self.raw_history), axis=0)
        distance = np.abs(measured)
        tau = np.interp(
            distance,
            [0.0, 80.0, 220.0, 600.0],
            [self.FILTER_TAU_NEAR, 0.080, self.FILTER_TAU_FAR, self.FILTER_TAU_FAR],
        )
        alpha = 1.0 - np.exp(-dt / tau)
        self.errors = measured if first else previous + alpha * (measured - previous)
        for i in range(2):
            if not first and measured[i] * previous[i] < 0.0 and abs(measured[i]) >= 12.0:
                self.errors[i] = float(measured[i])
                self.rates[i] = 0.0
                self.integral[i] = 0.0
        self.last_time = float(timestamp)
        d_raw = np.zeros(2) if first else (self.errors - previous) / dt
        d_alpha = 1.0 - np.exp(-dt / self.DERIVATIVE_TAU)
        self.derivative += d_alpha * (d_raw - self.derivative)
        v_alpha = 1.0 - np.exp(-dt / self.VELOCITY_FILTER_TAU)
        self.velocity += v_alpha * (velocity - self.velocity)
        scale = float(np.clip(prediction_scale, 0.0, 1.0))
        deltas = np.zeros(2, dtype=float)
        for i, axis in enumerate(("pan", "tilt")):
            error = float(self.errors[i])
            band = self.params["deadband"] * self.DEADBAND_SCALE[i]
            resume_band = max(band * 1.4, band + 3.0)
            if abs(error) <= band or scale == 0:
                self.moving[i] = False
            elif abs(error) >= resume_band:
                self.moving[i] = True
            if not self.moving[i]:
                self.integral[i] = self.derivative[i] = self.rates[i] = 0.0
                continue
            candidate_i = float(np.clip(self.integral[i] + error * dt, -500, 500))
            value = (
                self.params[f"{axis}_kp"] * error
                + self.params[f"{axis}_ki"] * candidate_i
                + self.params[f"{axis}_kd"] * self.derivative[i]
            )
            moving_away = (
                abs(error) > band * 1.5
                and abs(self.velocity[i]) >= 45.0
                and self.velocity[i] * error > 0.0
            )
            ff_rate = 0.0
            if moving_away:
                ff_rate = float(np.clip(
                    self.VELOCITY_FF_GAIN[i] * self.velocity[i],
                    -self.VELOCITY_FF_LIMIT[i],
                    self.VELOCITY_FF_LIMIT[i],
                ))
            damping_rate = 0.0
            if abs(error) < self.BRAKE_ZONE[i] * 1.25:
                damping_rate = float(np.clip(
                    self.CENTER_DAMPING_GAIN[i] * self.derivative[i],
                    -self.CENTER_DAMPING_LIMIT[i],
                    self.CENTER_DAMPING_LIMIT[i],
                ))
            raw_rate = (value * self.REFERENCE_HZ + damping_rate) * scale + ff_rate * scale
            if abs(raw_rate) <= self.MAX_SPEED[i] or raw_rate * error <= 0:
                self.integral[i] = candidate_i
            distance = max(abs(error) - band, 0.0)
            brake = float(np.clip(distance / self.BRAKE_ZONE[i], 0.0, 1.0))
            speed_limit = self.MAX_SPEED[i] * brake
            desired = float(np.clip(raw_rate, -speed_limit, speed_limit))
            if desired * self.rates[i] <= 0:
                self.rates[i] = 0.0
            if abs(desired) > abs(self.rates[i]):
                limit = self.MAX_ACCEL[i] * dt
                self.rates[i] += float(np.clip(desired - self.rates[i], -limit, limit))
            else:
                self.rates[i] = desired
            sign = -1.0 if i == 0 else 1.0
            max_delta = self.TILT_MAX_DELTA if i == 1 else self.MAX_DELTA
            deltas[i] = np.clip(sign * self.rates[i] * dt, -max_delta, max_delta)
        return float(deltas[0]), float(deltas[1])

    def coast(self, timestamp, decay_tau=0.14, max_coast_sec=0.18):
        try:
            stamp = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return 0.0, 0.0
        if not np.isfinite(stamp) or self.last_time is None or stamp <= self.last_time:
            return 0.0, 0.0
        dt = min(stamp - self.last_time, 0.1)
        self.last_time = stamp
        if dt <= 0.0 or not any(self.moving):
            self.rates[:] = 0.0
            return 0.0, 0.0
        if not hasattr(self, "_coast_elapsed"):
            self._coast_elapsed = 0.0
        self._coast_elapsed += dt
        if self._coast_elapsed > max_coast_sec:
            self.rates[:] = 0.0
            return 0.0, 0.0
        decay = float(np.exp(-dt / max(decay_tau, 1e-3)))
        self.rates *= decay
        deltas = np.zeros(2, dtype=float)
        for i in range(2):
            sign = -1.0 if i == 0 else 1.0
            deltas[i] = np.clip(sign * self.rates[i] * dt, -1.0, 1.0)
        return float(deltas[0]), float(deltas[1])

    def reset_coast(self):
        self._coast_elapsed = 0.0


def _draw_crosshair(f, cx, cy, color=(0, 255, 255), sz=40, th=2):
    cv2.line(f, (cx - sz, cy), (cx - 12, cy), color, th)
    cv2.line(f, (cx + 12, cy), (cx + sz, cy), color, th)
    cv2.line(f, (cx, cy - sz), (cx, cy - 12), color, th)
    cv2.line(f, (cx, cy + 12), (cx, cy + sz), color, th)
    cv2.circle(f, (cx, cy), 8, color, th)
    cv2.circle(f, (cx, cy), 22, color, 1)


def _annotate(frame, detections, ai_fps, backend, lockon=None, tracked_pos=None):
    h, w = frame.shape[:2]
    cx_s, cy_s = w // 2, h // 2

    for d in detections:
        if d["conf"] < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        bx, by = int(d["cx"]), int(d["cy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{d['conf']*100:.0f}%"
        selected = lockon.matched_detection if lockon is not None else None
        same_target = (
            selected is not None
            and abs(float(d.get("cx", (d["x1"] + d["x2"]) / 2)) - float(selected["cx"])) < 1.0
            and abs(float(d.get("cy", (d["y1"] + d["y2"]) / 2)) - float(selected["cy"])) < 1.0
        )
        if same_target and lockon.track_id > 0:
            label = f"DRONE #{lockon.track_id}  {label}"
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.circle(frame, (bx, by), 5, (0, 60, 255), -1)

    if tracked_pos is not None and lockon is not None:
        bx, by = int(np.clip(tracked_pos[0], 0, w - 1)), int(np.clip(tracked_pos[1], 0, h - 1))
        color = (0, 255, 80) if lockon.state == "LOCKED" else (0, 180, 255)
        cv2.circle(frame, (bx, by), 26, color, 2)
        cv2.line(frame, (cx_s, cy_s), (bx, by), color, 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"{lockon.state} #{lockon.track_id}",
            (bx + 28, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    _draw_crosshair(frame, cx_s, cy_s)

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (360, 86), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)

    fps_c = (0, 255, 80) if ai_fps > 20 else (0, 165, 255) if ai_fps > 10 else (0, 0, 255)
    cv2.putText(frame, f"AI STAGE FPS : {ai_fps:5.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, fps_c, 2)
    visible_count = sum(d["conf"] >= CONF_THRESHOLD for d in detections)
    cv2.putText(frame, f"DRONES  : {visible_count}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)

    status = lockon.state if lockon is not None else "IDLE"
    sc = (0, 200, 100)
    cv2.putText(frame, status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)
    tag = f"[{backend} | {STATE.roi_mode}]"
    cv2.putText(
        frame, tag,
        (max(370, w - len(tag) * 9 - 14), 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
    )


class TrackingWorker:
    """Đóng gói OOP luồng T-PID: kết nối DroneLockon (EKF) + StableTrackingController (PID) + ServoController."""

    def __init__(self, servo: ServoController, data_queue=None, stream_queue=None,
                 local_queue=None, target_guide=None, gimbal_pose=None, stop_event=None):
        self.servo = servo
        self.data_queue = data_queue or q_data
        self.stream_queue = stream_queue or q_stream
        self.local_queue = local_queue or q_local
        self.target_guide = target_guide or TARGET_GUIDE
        self.gimbal_pose = gimbal_pose or GIMBAL_POSE
        self.stop_event = stop_event or STOP_EVENT

    def run(self):
        print("[TRACKER] V6.3 OOP: Smith-Predictor EKF + Per-Axis Calibration + Low-Lag Camera + Smart ROI")
        self.gimbal_pose.bind_servo(self.servo)
        _, params = STATE.consume_pid_dirty()
        control = StableTrackingController(params)
        lockon = DroneLockon()
        last_packet_time = None
        active_track_id = None
        home_requested = False
        last_diag_t = 0.0
        last_diag_state = None
        last_ui_t = 0.0
        ui_period = 1.0 / max(UI_FPS, 1)
        _diag_log_line(
            "ts,state,id,roi,fps,det_n,det_conf,det_cx,det_cy,det_w,det_h,"
            "ekf_cx,ekf_cy,ekf_vx,ekf_vy,ekf_psi,ekf_nis,ctrl_cx,ctrl_cy,"
            "pan,tilt,dp,dt,ctrl_mode,err_x,err_y,exp_us,gain,spot_y,af"
        )

        def _emit_diag(now_ts, det_list, ctrl_pt, final_pt, dp_val, dt_val, c_mode):
            nonlocal last_diag_t, last_diag_state
            st = lockon.state
            is_active = (st != "IDLE") or bool(det_list) or (last_diag_state != st)
            if not is_active and (now_ts - last_diag_t) < 1.0:
                return
            last_diag_t = now_ts
            last_diag_state = st
            md = lockon.matched_detection
            top_d = md if md is not None else (max(det_list, key=lambda x: x["conf"]) if det_list else None)
            d_conf = float(top_d["conf"]) if top_d else 0.0
            d_cx = float(top_d["cx"]) if top_d else 0.0
            d_cy = float(top_d["cy"]) if top_d else 0.0
            d_w = float(top_d["w"]) if top_d else 0.0
            d_h = float(top_d["h"]) if top_d else 0.0
            if lockon.kalman.initialized:
                ex, ey = final_pt if final_pt is not None else lockon.kalman.get_pos()
                evx, evy = lockon.kalman.get_world_velocity_px()
                epsi = float(lockon.kalman.x[4, 0])
                enis = float(lockon.kalman.last_nis)
            else:
                ex = ey = evx = evy = epsi = enis = 0.0
            ccx, ccy = ctrl_pt if ctrl_pt is not None else (0.0, 0.0)
            pa, ta = self.servo.get_angles()
            with CAM_DIAG_LOCK:
                exp_u = CAM_DIAG["exp_us"]
                gain_v = CAM_DIAG["gain"]
                spot_y = CAM_DIAG["spot_y"]
                af_s = CAM_DIAG["af_state"]
            _diag_log_line(
                f"{time.strftime('%H:%M:%S')}.{int((time.time()%1)*1000):03d},"
                f"{st},{lockon.track_id},{STATE.roi_mode},{STATE.ai_fps:.1f},"
                f"{len(det_list)},{d_conf:.2f},{d_cx:.1f},{d_cy:.1f},{d_w:.1f},{d_h:.1f},"
                f"{ex:.1f},{ey:.1f},{evx:.1f},{evy:.1f},{epsi:.2f},{enis:.2f},"
                f"{ccx:.1f},{ccy:.1f},{pa:.2f},{ta:.2f},{dp_val:.3f},{dt_val:.3f},{c_mode},"
                f"{control.errors[0]:.1f},{control.errors[1]:.1f},"
                f"{exp_u:.0f},{gain_v:.2f},{spot_y:.1f},{af_s}"
            )

        def idle_or_hold(now):
            nonlocal home_requested
            control.reset()
            control.reset_coast()
            if lockon.state == "IDLE" and lockon.track_id == 0:
                if not home_requested:
                    self.servo.go_home()
                    home_requested = True
            elif lockon.state == "IDLE":
                home_requested = False
                self.servo.move(0.0, 0.0)
            else:
                self.servo.move(0.0, 0.0)
            pa, ta = self.servo.get_angles()
            STATE.update_stats(
                is_tracking=lockon.state != "IDLE",
                tracker_state=lockon.state,
                control_state="HOME" if home_requested else "HOLD",
                pan_angle=pa, tilt_angle=ta,
                error_x=0.0, error_y=0.0,
                track_id=lockon.track_id,
            )

        while not self.stop_event.is_set():
            dirty, params = STATE.consume_pid_dirty()
            if dirty:
                control.configure(params)
                print(f"[PID] Applied stable profile: Kp={params['pan_kp']:.4f}/{params['tilt_kp']:.4f}")
            try:
                data = self.data_queue.get(timeout=0.05)
            except Empty:
                now = time.monotonic()
                if last_packet_time is None or now - last_packet_time > MAX_FRAME_AGE_SEC:
                    if lockon.state != "IDLE":
                        lockon.reset(release_time=now)
                    self.target_guide.set_lost(lockon._lost_bbox, lockon._lost_time)
                    idle_or_hold(now)
                continue
            if self.stop_event.is_set():
                break
            now = time.monotonic()
            stamp = data["timestamp"]
            if (
                not np.isfinite(stamp)
                or stamp > now
                or (last_packet_time is not None and stamp <= last_packet_time)
            ):
                continue
            last_packet_time = stamp
            frame = data["frame"]
            fw, fh = data["frame_w"], data["frame_h"]
            detections = data["detections"]
            final_pos = None
            if now - stamp > MAX_FRAME_AGE_SEC:
                if lockon.state != "IDLE":
                    lockon.reset(release_time=stamp)
                self.target_guide.set_lost(lockon._lost_bbox, lockon._lost_time)
                idle_or_hold(now)
            else:
                gimbal_cap = data.get("gimbal")
                if gimbal_cap is None:
                    gimbal_cap = self.gimbal_pose.sample_now(stamp)
                result = lockon.update(
                    detections, timestamp=stamp,
                    frame_shape=(fh, fw), frame=frame,
                    gimbal_angles=gimbal_cap,
                )
                if result is None:
                    if lockon.state == "LOCKED" and lockon.locked_bbox is not None:
                        gimbal_now = self.gimbal_pose.sample_now(now)
                        kx_raw, ky_raw = lockon.kalman.get_pos(gimbal_angles=gimbal_now)
                        rkx_raw, rky_raw = lockon.kalman.predict_pos_ahead(
                            0.040, gimbal_angles=gimbal_now, max_lead_px=36.0
                        )
                        kx = float(np.clip(kx_raw, 0.0, float(fw)))
                        ky = float(np.clip(ky_raw, 0.0, float(fh)))
                        rkx = float(np.clip(rkx_raw, 0.0, float(fw)))
                        rky = float(np.clip(rky_raw, 0.0, float(fh)))
                        kw, kh = lockon.kalman.get_size()
                        self.target_guide.set_locked(
                            kx, ky, kw, kh, stamp, roi_center=(rkx, rky)
                        )
                        final_pos = (kx, ky)
                        dp, dt = control.coast(stamp, decay_tau=0.18, max_coast_sec=0.25)
                        if not self.stop_event.is_set():
                            self.servo.move(dp, dt)
                        pa, ta = self.servo.get_angles()
                        cmode = "COAST" if (abs(dp) > 1e-4 or abs(dt) > 1e-4) else "HOLD"
                        STATE.update_stats(
                            is_tracking=True, tracker_state=lockon.state,
                            control_state=cmode, track_id=lockon.track_id,
                            error_x=float(control.errors[0]), error_y=float(control.errors[1]),
                            pan_angle=pa, tilt_angle=ta,
                        )
                        _emit_diag(now, detections, (kx, ky), final_pos, dp, dt, cmode)
                    else:
                        self.target_guide.set_lost(lockon._lost_bbox, lockon._lost_time)
                        idle_or_hold(now)
                        _emit_diag(now, detections, None, None, 0.0, 0.0, STATE.control_state)
                else:
                    if active_track_id != lockon.track_id:
                        control.reset()
                        active_track_id = lockon.track_id
                    control.reset_coast()
                    home_requested = False
                    cx, cy, _ = result
                    live_gimbal = self.gimbal_pose.get_live_angles()
                    ctrl_cx, ctrl_cy = lockon.kalman.project_with_inflight_gimbal(
                        gimbal_cap, live_gimbal,
                        lead_sec=0.020, compensation=0.72, max_lead_px=18.0,
                    )
                    lb = lockon.locked_bbox
                    luma_box = lockon.raw_bbox if lockon.raw_bbox is not None else lb
                    if lb is not None:
                        tluma = _measure_target_luma(frame, luma_box)
                        rkx, rky = lockon.kalman.predict_pos_ahead(
                            0.040, gimbal_angles=live_gimbal, max_lead_px=36.0
                        )
                        self.target_guide.set_locked(
                            cx, cy, lb[2] - lb[0], lb[3] - lb[1], stamp,
                            target_luma=tluma,
                            roi_center=(
                                float(np.clip(rkx, 0.0, float(fw))),
                                float(np.clip(rky, 0.0, float(fh))),
                            ),
                        )
                    final_pos = (cx, cy)
                    velocity = lockon.kalman.get_world_velocity_px(gimbal_cap)
                    dp, dt = control.step(
                        ctrl_cx, ctrl_cy, fw, fh, stamp,
                        prediction_scale=1.0, velocity=velocity,
                    )
                    if not self.stop_event.is_set():
                        self.servo.move(dp, dt)
                    mode = "MOVING" if any(control.moving) else "HOLD"
                    pa, ta = self.servo.get_angles()
                    STATE.update_stats(
                        is_tracking=True, tracker_state=lockon.state,
                        control_state=mode, track_id=lockon.track_id,
                        error_x=float(control.errors[0]), error_y=float(control.errors[1]),
                        pan_angle=pa, tilt_angle=ta,
                    )
                    _emit_diag(now, detections, (ctrl_cx, ctrl_cy), final_pos, dp, dt, mode)

            need_ui = LOCAL_DISPLAY_ENABLED or (STATE.stream_viewers > 0)
            if need_ui and (now - last_ui_t) >= ui_period:
                last_ui_t = now
                annotated = frame.copy()
                _annotate(annotated, detections, data["ai_fps"], STATE.backend, lockon, final_pos)
                if STATE.stream_viewers > 0:
                    put_latest(self.stream_queue, {"annotated": annotated})
                if LOCAL_DISPLAY_ENABLED:
                    cv2.putText(
                        annotated, "[CHUOT PHAI / ESC: THOAT]",
                        (15, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
                    )
                    put_latest(self.local_queue, {"annotated": annotated})


def thread_pid_servo(servo: ServoController):
    TrackingWorker(servo).run()
