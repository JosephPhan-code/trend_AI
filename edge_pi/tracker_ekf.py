"""
==============================================================================
Module: tracker_ekf.py
Mô tả : Các lớp OOP cho theo dõi góc Gimbal (GimbalPoseTracker),
        Bộ lọc Extended Kalman 8 trạng thái (ExtendedKalmanTracker),
        Quản lý khóa mục tiêu & tái bắt dấu (DroneLockon),
        Phát hiện chuyển động bù dịch chuyển camera (MotionDetector)
        và Lập lịch cửa sổ cắt SAHI/ROI (RoiPlanner).
==============================================================================
"""

import threading
import time
import cv2
import numpy as np

from config import (
    CAMERA_HEIGHT,
    CAMERA_HFOV_DEG,
    CAMERA_VFOV_DEG,
    CAMERA_WIDTH,
    CONF_THRESHOLD,
    DETECTION_FLOOR,
    GIMBAL_PAN_SCALE,
    GIMBAL_TILT_SCALE,
    REACQUIRE_ID_GRACE_SEC,
    ROI_ENABLED,
    ROI_FULL_EVERY,
    ROI_HINT_DIST,
    ROI_HINT_TTL,
    ROI_SIZE,
    ROI_SMALL_MAX,
    SERVO_LAG_TAU,
    SERVO_LATENCY_SEC,
    STATE,
    _id_log,
)


class GimbalPoseTracker:
    """Theo dõi góc thực tế Pan/Tilt có bù trễ thời gian (lag + buffer) và cập nhật FOV cảm biến."""

    def __init__(self, tau=SERVO_LAG_TAU, latency=SERVO_LATENCY_SEC):
        self._lock = threading.Lock()
        self._servo = None
        self._tau = max(0.005, float(tau))
        self._latency = float(np.clip(latency, 0.0, 0.25))
        self._pan = 0.0
        self._tilt = 0.0
        self._last_t = None
        self._hist = []  # [(timestamp, pan_deg, tilt_deg)]
        self._hfov_deg = CAMERA_HFOV_DEG
        self._vfov_deg = CAMERA_VFOV_DEG

    def set_optical_fov(self, hfov_deg, vfov_deg):
        with self._lock:
            self._hfov_deg = float(np.clip(hfov_deg, 15.0, 150.0))
            self._vfov_deg = float(np.clip(vfov_deg, 10.0, 130.0))

    def get_optical_fov(self):
        with self._lock:
            return self._hfov_deg, self._vfov_deg

    def get_live_angles(self):
        with self._lock:
            if self._servo is None:
                return 0.0, 0.0
            try:
                p, t = self._servo.get_angles()
                return float(p), float(t)
            except Exception:
                return float(self._pan), float(self._tilt)

    def bind_servo(self, servo):
        with self._lock:
            self._servo = servo
            self._last_t = None
            self._hist.clear()
            if servo is not None:
                try:
                    p, t = servo.get_angles()
                    self._pan, self._tilt = float(p), float(t)
                except Exception:
                    self._pan, self._tilt = 0.0, 0.0

    def sample_now(self, timestamp=None):
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            if self._servo is None:
                return 0.0, 0.0
            try:
                cmd_p, cmd_t = self._servo.get_angles()
            except Exception:
                return self._pan, self._tilt
            if self._last_t is None or not np.isfinite(stamp) or stamp <= self._last_t:
                self._pan, self._tilt = float(cmd_p), float(cmd_t)
            else:
                dt = min(0.25, max(0.0, stamp - self._last_t))
                alpha = 1.0 - float(np.exp(-dt / self._tau))
                self._pan += alpha * (float(cmd_p) - self._pan)
                self._tilt += alpha * (float(cmd_t) - self._tilt)
            self._last_t = stamp
            self._hist.append((stamp, self._pan, self._tilt))
            if len(self._hist) > 48:
                self._hist.pop(0)
            target_t = stamp - self._latency
            if self._latency <= 1e-4 or len(self._hist) < 2 or target_t >= self._hist[-1][0]:
                return float(self._pan), float(self._tilt)
            if target_t <= self._hist[0][0]:
                return float(self._hist[0][1]), float(self._hist[0][2])
            for idx in range(len(self._hist) - 1, 0, -1):
                t0, p0, tl0 = self._hist[idx - 1]
                t1, p1, tl1 = self._hist[idx]
                if t0 <= target_t <= t1 and t1 > t0:
                    w = (target_t - t0) / (t1 - t0)
                    return float(p0 + w * (p1 - p0)), float(tl0 + w * (tl1 - tl0))
            return float(self._pan), float(self._tilt)


GIMBAL_POSE = GimbalPoseTracker()


class ExtendedKalmanTracker:
    """8-state Extended Kalman Filter (EKF) ước lượng quỹ đạo phi tuyến của drone:

        x = [theta_x, theta_y, omega_x, omega_y, turn_rate, ln_w, ln_h, d_ln_s]^T
    """

    VEL_DAMP_TAU = 2.60
    TURN_DAMP_TAU = 1.40
    SCALE_DAMP_TAU = 0.60
    ANG_ACCEL_NOISE = 0.014
    TURN_ACCEL_NOISE = 6.50
    SCALE_ACCEL_NOISE = 0.30
    ASPECT_DIFFUSION = 0.0036
    MAX_ANGLE_CAM = 1.22

    def __init__(self, frame_w=CAMERA_WIDTH, frame_h=CAMERA_HEIGHT,
                 hfov_deg=None, vfov_deg=None,
                 ego_scale=None, pan_scale=None, tilt_scale=None,
                 gimbal_pose=None):
        self._gimbal_pose = gimbal_pose or GIMBAL_POSE
        if ego_scale is not None:
            sc = float(np.clip(ego_scale, 0.0, 2.0))
            self.pan_scale = sc
            self.tilt_scale = sc
        else:
            self.pan_scale = float(np.clip(GIMBAL_PAN_SCALE if pan_scale is None else pan_scale, 0.0, 2.0))
            self.tilt_scale = float(np.clip(GIMBAL_TILT_SCALE if tilt_scale is None else tilt_scale, 0.0, 2.5))
        self.ego_scale = self.pan_scale
        self._explicit_fov = hfov_deg is not None or vfov_deg is not None
        g_hfov, g_vfov = self._gimbal_pose.get_optical_fov()
        self._hfov_deg = float(np.clip(g_hfov if hfov_deg is None else hfov_deg, 15.0, 150.0))
        self._vfov_deg = float(np.clip(g_vfov if vfov_deg is None else vfov_deg, 10.0, 130.0))
        self._set_fov(frame_w, frame_h)
        self.reset()

    def _sync_optical_fov(self):
        if not self._explicit_fov:
            g_hfov, g_vfov = self._gimbal_pose.get_optical_fov()
            if abs(g_hfov - self._hfov_deg) > 0.2 or abs(g_vfov - self._vfov_deg) > 0.2:
                self._hfov_deg = g_hfov
                self._vfov_deg = g_vfov
                self._set_fov(self.frame_w, self.frame_h)

    def _set_fov(self, frame_w, frame_h):
        fw = float(frame_w) if frame_w and frame_w > 0 else float(CAMERA_WIDTH)
        fh = float(frame_h) if frame_h and frame_h > 0 else float(CAMERA_HEIGHT)
        self.frame_w = fw
        self.frame_h = fh
        self.cx0 = fw * 0.5
        self.cy0 = fh * 0.5
        self.fx = (fw * 0.5) / float(np.tan(np.deg2rad(self._hfov_deg * 0.5)))
        self.fy = (fh * 0.5) / float(np.tan(np.deg2rad(self._vfov_deg * 0.5)))

    def set_frame_shape(self, frame_shape):
        if frame_shape is None or len(frame_shape) < 2:
            return
        fh, fw = float(frame_shape[0]), float(frame_shape[1])
        if np.isfinite([fh, fw]).all() and fh > 0 and fw > 0:
            if abs(fw - self.frame_w) > 0.5 or abs(fh - self.frame_h) > 0.5:
                self._set_fov(fw, fh)

    def _gimbal_rad(self, gimbal_angles):
        if gimbal_angles is None:
            return self._last_gimbal_rad
        try:
            pan_deg, tilt_deg = float(gimbal_angles[0]), float(gimbal_angles[1])
            if not np.isfinite([pan_deg, tilt_deg]).all():
                return self._last_gimbal_rad
            return (
                self.pan_scale * float(np.deg2rad(pan_deg)),
                self.tilt_scale * float(np.deg2rad(tilt_deg)),
            )
        except (TypeError, ValueError, IndexError):
            return self._last_gimbal_rad

    def _pixel_to_world(self, cx, cy, gimbal_rad):
        pan_r, tilt_r = gimbal_rad
        ax = float(np.arctan2(float(cx) - self.cx0, self.fx))
        ay = float(np.arctan2(float(cy) - self.cy0, self.fy))
        return ax - pan_r, ay + tilt_r

    def _world_to_pixel(self, theta_x, theta_y, gimbal_rad):
        pan_r, tilt_r = gimbal_rad
        ax = float(np.clip(theta_x + pan_r, -self.MAX_ANGLE_CAM, self.MAX_ANGLE_CAM))
        ay = float(np.clip(theta_y - tilt_r, -self.MAX_ANGLE_CAM, self.MAX_ANGLE_CAM))
        return (
            float(self.cx0 + self.fx * np.tan(ax)),
            float(self.cy0 + self.fy * np.tan(ay)),
        )

    def reset(self):
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[5, 0] = np.log(32.0)
        self.x[6, 0] = np.log(32.0)
        self.P = np.diag([
            1.6e-5, 1.6e-5,
            0.012, 0.012,
            0.36,
            0.05, 0.05,
            0.16,
        ]).astype(np.float64)
        self.initialized = False
        self.last_timestamp = None
        self._last_gimbal_rad = (0.0, 0.0)
        self.last_nis = 0.0
        self._maneuver_streak = 0

    def init(self, cx, cy, timestamp=None, w=None, h=None,
             gimbal_angles=None, frame_shape=None):
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite([cx, cy, stamp]).all():
            return False
        self._sync_optical_fov()
        self.set_frame_shape(frame_shape)
        self.reset()
        g_rad = (0.0, 0.0) if gimbal_angles is None else self._gimbal_rad(gimbal_angles)
        self._last_gimbal_rad = g_rad
        tx, ty = self._pixel_to_world(cx, cy, g_rad)
        self.x[0, 0] = tx
        self.x[1, 0] = ty
        bw = float(w) if (w is not None and np.isfinite(w) and w >= 2.0) else 32.0
        bh = float(h) if (h is not None and np.isfinite(h) and h >= 2.0) else 32.0
        self.x[5, 0] = float(np.log(np.clip(bw, 2.0, 1200.0)))
        self.x[6, 0] = float(np.log(np.clip(bh, 2.0, 1200.0)))
        self.last_timestamp = stamp
        self.initialized = True
        return True

    @classmethod
    def _transition_and_jacobian(cls, x_vec, dt):
        tx, ty, wx, wy, psi_d, lw, lh, sd = [float(v) for v in x_vec[:, 0]]
        gv = float(np.exp(-dt / cls.VEL_DAMP_TAU))
        gp = float(np.exp(-dt / cls.TURN_DAMP_TAU))
        gs = float(np.exp(-dt / cls.SCALE_DAMP_TAU))

        u = psi_d * dt
        c = float(np.cos(u))
        s = float(np.sin(u))
        if abs(u) < 1e-3:
            u2 = u * u
            u4 = u2 * u2
            A = dt * (1.0 - u2 / 6.0 + u4 / 120.0)
            B = 0.5 * dt * u * (1.0 - u2 / 12.0 + u4 / 360.0)
            dA_dpsi = (dt * dt) * (-u / 3.0 + (u * u2) / 30.0)
            dB_dpsi = 0.5 * (dt * dt) * (1.0 - u2 / 4.0 + u4 / 72.0)
        else:
            inv_p = 1.0 / psi_d
            inv_p2 = inv_p * inv_p
            A = s * inv_p
            B = (1.0 - c) * inv_p
            dA_dpsi = (dt * c * psi_d - s) * inv_p2
            dB_dpsi = (dt * s * psi_d - (1.0 - c)) * inv_p2

        nx = np.empty((8, 1), dtype=np.float64)
        nx[0, 0] = tx + A * wx - B * wy
        nx[1, 0] = ty + B * wx + A * wy
        nx[2, 0] = gv * (c * wx - s * wy)
        nx[3, 0] = gv * (s * wx + c * wy)
        nx[4, 0] = gp * psi_d
        nx[5, 0] = lw + dt * sd
        nx[6, 0] = lh + dt * sd
        nx[7, 0] = gs * sd

        F = np.zeros((8, 8), dtype=np.float64)
        F[0, 0] = 1.0
        F[0, 2] = A
        F[0, 3] = -B
        F[0, 4] = dA_dpsi * wx - dB_dpsi * wy

        F[1, 1] = 1.0
        F[1, 2] = B
        F[1, 3] = A
        F[1, 4] = dB_dpsi * wx + dA_dpsi * wy

        F[2, 2] = gv * c
        F[2, 3] = -gv * s
        F[2, 4] = gv * dt * (-s * wx - c * wy)

        F[3, 2] = gv * s
        F[3, 3] = gv * c
        F[3, 4] = gv * dt * (c * wx - s * wy)

        F[4, 4] = gp
        F[5, 5] = 1.0
        F[5, 7] = dt
        F[6, 6] = 1.0
        F[6, 7] = dt
        F[7, 7] = gs
        return nx, F

    @classmethod
    def _process_noise(cls, dt):
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = np.zeros((8, 8), dtype=np.float64)
        qa = cls.ANG_ACCEL_NOISE
        for idx in (0, 1):
            v_idx = idx + 2
            Q[idx, idx] = qa * dt3 / 3.0
            Q[idx, v_idx] = qa * dt2 * 0.5
            Q[v_idx, idx] = qa * dt2 * 0.5
            Q[v_idx, v_idx] = qa * dt
        Q[4, 4] = cls.TURN_ACCEL_NOISE * dt

        qs = cls.SCALE_ACCEL_NOISE
        s_pos = qs * dt3 / 3.0
        s_cross = qs * dt2 * 0.5
        s_vel = qs * dt
        ar = cls.ASPECT_DIFFUSION * dt
        Q[5, 5] = s_pos + ar
        Q[6, 6] = s_pos + ar
        Q[5, 6] = Q[6, 5] = s_pos * 0.85
        Q[5, 7] = Q[7, 5] = s_cross
        Q[6, 7] = Q[7, 6] = s_cross
        Q[7, 7] = s_vel
        return Q

    def _clamp_state(self):
        self.x[2, 0] = float(np.clip(self.x[2, 0], -1.8, 1.8))
        self.x[3, 0] = float(np.clip(self.x[3, 0], -1.8, 1.8))
        self.x[4, 0] = float(np.clip(self.x[4, 0], -4.5, 4.5))
        self.x[5, 0] = float(np.clip(self.x[5, 0], np.log(2.0), np.log(1200.0)))
        self.x[6, 0] = float(np.clip(self.x[6, 0], np.log(2.0), np.log(1200.0)))
        self.x[7, 0] = float(np.clip(self.x[7, 0], -1.5, 1.5))
        self.P = (self.P + self.P.T) * 0.5
        diag = np.clip(np.diag(self.P), 1e-8, 8.0)
        np.fill_diagonal(self.P, diag)

    def predict(self, timestamp=None, gimbal_angles=None):
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        self._sync_optical_fov()
        if gimbal_angles is not None:
            self._last_gimbal_rad = self._gimbal_rad(gimbal_angles)
        if not self.initialized or not np.isfinite(stamp):
            return self.get_pos()
        dt = stamp - self.last_timestamp
        if dt <= 0.0:
            return self.get_pos()
        dt_eff = min(dt, 0.50)
        omega_speed = float(np.hypot(self.x[2, 0], self.x[3, 0]))
        self.x, F = self._transition_and_jacobian(self.x, dt_eff)
        if omega_speed < 0.055:
            self.x[4, 0] *= float(np.exp(-dt_eff / 0.22))
        Q = self._process_noise(dt_eff)
        if omega_speed < 0.055:
            Q[4, 4] *= 0.08
        self.P = F @ self.P @ F.T + Q
        self._clamp_state()
        self.last_timestamp = stamp
        return self.get_pos()

    def _measurement_and_jacobian(self, x_vec, gimbal_rad, with_size=False):
        pan_r, tilt_r = gimbal_rad
        raw_ax = float(x_vec[0, 0]) + pan_r
        raw_ay = float(x_vec[1, 0]) - tilt_r
        ax = float(np.clip(raw_ax, -self.MAX_ANGLE_CAM, self.MAX_ANGLE_CAM))
        ay = float(np.clip(raw_ay, -self.MAX_ANGLE_CAM, self.MAX_ANGLE_CAM))
        tx = float(np.tan(ax))
        ty = float(np.tan(ay))
        cx_hat = self.cx0 + self.fx * tx
        cy_hat = self.cy0 + self.fy * ty
        dx = self.fx * (1.0 + tx * tx) if abs(raw_ax) < self.MAX_ANGLE_CAM else 0.0
        dy = self.fy * (1.0 + ty * ty) if abs(raw_ay) < self.MAX_ANGLE_CAM else 0.0

        if not with_size:
            z_hat = np.array([[cx_hat], [cy_hat]], dtype=np.float64)
            H = np.zeros((2, 8), dtype=np.float64)
            H[0, 0] = dx
            H[1, 1] = dy
            return z_hat, H

        z_hat = np.array([[cx_hat], [cy_hat], [float(x_vec[5, 0])], [float(x_vec[6, 0])]], dtype=np.float64)
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = dx
        H[1, 1] = dy
        H[2, 5] = 1.0
        H[3, 6] = 1.0
        return z_hat, H

    def _measurement_cov(self, conf=None, w=None, h=None, with_size=False):
        c = float(np.clip(0.55 if conf is None else conf, 0.08, 1.0))
        box_span = float(max(w or 32.0, h or 32.0))
        sigma_xy = float(np.clip(
            3.2 + 2.2 * ((0.50 / c) ** 0.75) + 0.032 * min(box_span, 500.0),
            4.0, 26.0,
        ))
        var_xy = sigma_xy * sigma_xy
        if not with_size:
            return np.diag([var_xy, var_xy]).astype(np.float64)
        sigma_ln = float(np.clip(0.09 + 0.06 / max(c, 0.12), 0.12, 0.45))
        var_ln = sigma_ln * sigma_ln
        return np.diag([var_xy, var_xy, var_ln, var_ln]).astype(np.float64)

    def mahalanobis_distance(self, cx, cy, conf=None, gimbal_angles=None):
        if not self.initialized or not np.isfinite([cx, cy]).all():
            return float("inf")
        g_rad = self._gimbal_rad(gimbal_angles)
        z_hat, H = self._measurement_and_jacobian(self.x, g_rad, with_size=False)
        bw, bh = self.get_size()
        R = self._measurement_cov(conf=conf, w=bw, h=bh, with_size=False)
        S = H @ self.P @ H.T + R
        innov = np.array([[float(cx) - z_hat[0, 0]], [float(cy) - z_hat[1, 0]]], dtype=np.float64)
        try:
            nis = float((innov.T @ np.linalg.solve(S, innov))[0, 0])
            return float(np.sqrt(max(0.0, nis)))
        except np.linalg.LinAlgError:
            return float("inf")

    def update(self, cx, cy, w=None, h=None, conf=None, gimbal_angles=None):
        if not self.initialized or not np.isfinite([cx, cy]).all():
            return self.get_pos()
        if gimbal_angles is not None:
            self._last_gimbal_rad = self._gimbal_rad(gimbal_angles)
        g_rad = self._last_gimbal_rad

        has_size = (
            w is not None and h is not None
            and np.isfinite([w, h]).all()
            and float(w) >= 2.0 and float(h) >= 2.0
        )
        z_hat, H = self._measurement_and_jacobian(self.x, g_rad, with_size=has_size)
        R = self._measurement_cov(conf=conf, w=w, h=h, with_size=has_size)

        if has_size:
            z = np.array([
                [float(cx)],
                [float(cy)],
                [float(np.log(np.clip(float(w), 2.0, 1200.0)))],
                [float(np.log(np.clip(float(h), 2.0, 1200.0)))],
            ], dtype=np.float64)
        else:
            z = np.array([[float(cx)], [float(cy)]], dtype=np.float64)

        innov = z - z_hat
        S_xy = H[:2, :] @ self.P @ H[:2, :].T + R[:2, :2]
        try:
            nis_xy = float((innov[:2].T @ np.linalg.solve(S_xy, innov[:2]))[0, 0])
        except np.linalg.LinAlgError:
            nis_xy = 0.0
        self.last_nis = max(0.0, nis_xy)

        if 10.6 < self.last_nis <= 48.0:
            self._maneuver_streak += 1
            if self._maneuver_streak >= 2:
                boost = float(np.clip(1.0 + 0.12 * (self.last_nis - 10.6), 1.0, 2.8))
                scale_vec = np.ones(8, dtype=np.float64)
                scale_vec[2:5] = np.sqrt(boost)
                self.P = (self.P * scale_vec[:, None]) * scale_vec[None, :]
        else:
            self._maneuver_streak = 0
            if self.last_nis > 48.0:
                huber = float(np.sqrt(self.last_nis) / 6.9)
                R[:2, :2] *= huber

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innov
        I_KH = np.eye(8, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._clamp_state()
        return self.get_pos()

    def get_pos(self, gimbal_angles=None):
        if not self.initialized:
            return 0.0, 0.0
        g_rad = self._last_gimbal_rad if gimbal_angles is None else self._gimbal_rad(gimbal_angles)
        return self._world_to_pixel(float(self.x[0, 0]), float(self.x[1, 0]), g_rad)

    def get_size(self):
        return float(np.exp(self.x[5, 0])), float(np.exp(self.x[6, 0]))

    def get_world_velocity_px(self, gimbal_angles=None):
        if not self.initialized:
            return 0.0, 0.0
        pan_r, tilt_r = self._last_gimbal_rad if gimbal_angles is None else self._gimbal_rad(gimbal_angles)
        ax = float(np.clip(float(self.x[0, 0]) + pan_r, -self.MAX_ANGLE_CAM, self.MAX_ANGLE_CAM))
        ay = float(np.clip(float(self.x[1, 0]) - tilt_r, -self.MAX_ANGLE_CAM, self.MAX_ANGLE_CAM))
        sec2_x = 1.0 + float(np.tan(ax)) ** 2
        sec2_y = 1.0 + float(np.tan(ay)) ** 2
        return (
            float(self.fx * sec2_x * float(self.x[2, 0])),
            float(self.fy * sec2_y * float(self.x[3, 0])),
        )

    def get_velocity(self):
        return self.get_world_velocity_px()

    def predict_pos_ahead(self, lead_sec, gimbal_angles=None, max_lead_px=36.0,
                          min_speed_px=45.0):
        if not self.initialized:
            return 0.0, 0.0
        g_rad = self._last_gimbal_rad if gimbal_angles is None else self._gimbal_rad(gimbal_angles)
        cx_now, cy_now = self._world_to_pixel(float(self.x[0, 0]), float(self.x[1, 0]), g_rad)
        dt_req = float(np.clip(lead_sec, 0.0, 0.15))
        if dt_req <= 1e-4:
            return cx_now, cy_now
        if min_speed_px > 0.0:
            vx_w, vy_w = self.get_world_velocity_px(gimbal_angles=gimbal_angles)
            spd = float(np.hypot(vx_w, vy_w))
            if spd <= min_speed_px:
                return cx_now, cy_now
            ramp = float(np.clip((spd - min_speed_px) / max(min_speed_px, 40.0), 0.0, 1.0))
            dt = dt_req * ramp
        else:
            dt = dt_req
        nx, _ = self._transition_and_jacobian(self.x, dt)
        cx_lead, cy_lead = self._world_to_pixel(float(nx[0, 0]), float(nx[1, 0]), g_rad)
        dx, dy = cx_lead - cx_now, cy_lead - cy_now
        step = float(np.hypot(dx, dy))
        if step > max_lead_px and step > 1e-6:
            scale = max_lead_px / step
            dx *= scale
            dy *= scale
        return float(cx_now + dx), float(cy_now + dy)

    def project_with_inflight_gimbal(self, capture_gimbal, live_gimbal,
                                     lead_sec=0.020, compensation=0.72,
                                     max_lead_px=18.0, max_shift_px=220.0,
                                     min_speed_px=45.0):
        if not self.initialized:
            return 0.0, 0.0
        cx_base, cy_base = self.predict_pos_ahead(
            lead_sec, gimbal_angles=capture_gimbal,
            max_lead_px=max_lead_px, min_speed_px=min_speed_px,
        )
        if capture_gimbal is None or live_gimbal is None:
            return cx_base, cy_base
        try:
            cp, ct = float(capture_gimbal[0]), float(capture_gimbal[1])
            lp, lt = float(live_gimbal[0]), float(live_gimbal[1])
            if not np.isfinite([cp, ct, lp, lt]).all():
                return cx_base, cy_base
        except (TypeError, ValueError, IndexError):
            return cx_base, cy_base
        comp = float(np.clip(compensation, 0.0, 1.0))
        eff_gimbal = (cp + comp * (lp - cp), ct + comp * (lt - ct))
        cx_live, cy_live = self.predict_pos_ahead(
            lead_sec, gimbal_angles=eff_gimbal,
            max_lead_px=max_lead_px, min_speed_px=min_speed_px,
        )
        dx, dy = cx_live - cx_base, cy_live - cy_base
        shift = float(np.hypot(dx, dy))
        if shift > max_shift_px and shift > 1e-6:
            scale = max_shift_px / shift
            dx *= scale
            dy *= scale
        return float(cx_base + dx), float(cy_base + dy)


KalmanTracker = ExtendedKalmanTracker


class DroneLockon:
    """Confirm one target, associate high then low confidence, expire by seconds."""

    LOW_CONFIDENCE = 0.10
    ACQUIRE_MAX_GAP_SEC = 0.55
    IOU_MIN = 0.05
    DIST_MAX_PX = 380.0
    REID_SIMILARITY_MIN = 0.45
    REID_SIMILARITY_SMALL_MIN = 0.32
    LOCK_HOLD_SEC = 0.60
    REACQUIRE_LOW_CONF = 0.12
    REACQUIRE_LOW_SEC = 2.5
    REACQUIRE_RADIUS_PX = 160.0
    APPEARANCE_WEIGHT = 0.70
    MAHALANOBIS_WEIGHT = 0.65
    WEAK_CONFIRM_CONF = 0.13
    WEAK_CONFIRM_HITS = 3
    WEAK_CONFIRM_WINDOW_SEC = 0.35
    WEAK_CONFIRM_GATE_PX = 75.0
    STATIC_CLUTTER_RADIUS_PX = 3.5
    STATIC_CLUTTER_MIN_HITS = 10

    def __init__(self, kalman=None):
        self.kalman = kalman or ExtendedKalmanTracker()
        self.track_id = 0
        self._last_release_time = None
        self._last_timestamp = None
        self._weak_tracks = []
        self._static_bins = []
        self.reset()

    def reset(self, release_time=None):
        if release_time is not None and np.isfinite(release_time):
            self._last_release_time = float(release_time)
        lb = getattr(self, "locked_bbox", None)
        if release_time is not None and lb is not None:
            self._lost_bbox = list(lb)
            self._lost_time = float(release_time)
        elif not hasattr(self, "_lost_bbox"):
            self._lost_bbox = None
            self._lost_time = None
        self.kalman.reset()
        self.state = "IDLE"
        self.locked_bbox = None
        self.raw_bbox = None
        self.last_match_time = None
        self.matched_detection = None
        self._last_match_center = None
        self.control_center = None
        self._candidate = None
        self._candidate_time = None
        self._candidate_gimbal = None
        self._rescue_cand = None
        self._rescue_hits = 0
        if not hasattr(self, "appearance_signature"):
            self.appearance_signature = None
            self.appearance_time = None

    @staticmethod
    def _bbox(d):
        return [d["x1"], d["y1"], d["x2"], d["y2"]]

    @staticmethod
    def _appearance_signature(frame, bbox):
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or not frame.size:
            return None
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                return None
            if bw >= 12 and bh >= 12:
                mx = int(round(bw * 0.175))
                my = int(round(bh * 0.175))
                if (x2 - mx) - (x1 + mx) >= 6 and (y2 - my) - (y1 + my) >= 6:
                    x1, x2 = x1 + mx, x2 - mx
                    y1, y2 = y1 + my, y2 - my
            crop = frame[y1:y2, x1:x2]
            if crop.size < 36:
                return None
            small = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
            if small.ndim == 2:
                gray = small
                hsv = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
            else:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).reshape(-1).astype(np.float32)
            gray = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
            gray = (gray - gray.mean()) / (gray.std() + 1e-6)
            sig = np.concatenate([hist, gray.reshape(-1)]).astype(np.float32)
            norm = float(np.linalg.norm(sig))
            return sig / norm if norm > 1e-6 else None
        except (ValueError, TypeError, cv2.error):
            return None

    @staticmethod
    def _appearance_similarity(a, b):
        if a is None or b is None:
            return None
        try:
            a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                return None
            return float(np.clip(np.dot(a, b), -1.0, 1.0))
        except (TypeError, ValueError):
            return None

    def _valid_detections(self, detections, frame_shape):
        result = []
        for d in detections:
            try:
                x1, y1, x2, y2, conf = [float(d[k]) for k in ("x1", "y1", "x2", "y2", "conf")]
                if not np.isfinite([x1, y1, x2, y2, conf]).all():
                    continue
                if not self.LOW_CONFIDENCE <= conf <= 1.0 or x2 <= x1 or y2 <= y1:
                    continue
                if frame_shape is not None:
                    h, w = float(frame_shape[0]), float(frame_shape[1])
                    if not np.isfinite([h, w]).all() or min(h, w) <= 0:
                        continue
                    x1, x2 = max(0.0, x1), min(w, x2)
                    y1, y2 = max(0.0, y1), min(h, y2)
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    continue
                clean = dict(d)
                clean.update(
                    x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                    cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1,
                )
                result.append(clean)
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                continue
        return result

    def _iou(self, a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    def _is_static_weak_clutter(self, d, stamp):
        self._static_bins = [b for b in self._static_bins if stamp - b["last"] <= 3.0]
        cx, cy = d["cx"], d["cy"]
        for b in self._static_bins:
            if np.hypot(cx - b["cx"], cy - b["cy"]) <= self.STATIC_CLUTTER_RADIUS_PX:
                b["hits"] += 1
                b["last"] = stamp
                return b["hits"] >= self.STATIC_CLUTTER_MIN_HITS
        self._static_bins.append({"cx": cx, "cy": cy, "hits": 1, "last": stamp})
        if len(self._static_bins) > 24:
            self._static_bins.pop(0)
        return False

    def _update_weak_confirmation(self, low_dets, stamp):
        self._weak_tracks = [
            t for t in self._weak_tracks
            if stamp - t["last"] <= self.WEAK_CONFIRM_WINDOW_SEC
        ]
        cands = [
            d for d in low_dets
            if d["conf"] >= self.WEAK_CONFIRM_CONF
            and not self._is_static_weak_clutter(d, stamp)
        ]
        if not cands:
            return None
        best_confirmed = None
        for d in cands:
            matched_t = None
            best_dist = self.WEAK_CONFIRM_GATE_PX
            for t in self._weak_tracks:
                if t["last"] == stamp:
                    continue
                dist = float(np.hypot(d["cx"] - t["cx"], d["cy"] - t["cy"]))
                if dist <= best_dist:
                    best_dist = dist
                    matched_t = t
            if matched_t is not None:
                matched_t["cx"] = d["cx"]
                matched_t["cy"] = d["cy"]
                matched_t["last"] = stamp
                matched_t["hits"] += 1
                matched_t["det"] = d
                travel = float(np.hypot(d["cx"] - matched_t["ox"], d["cy"] - matched_t["oy"]))
                if matched_t["hits"] >= self.WEAK_CONFIRM_HITS and travel >= 4.0:
                    if best_confirmed is None or d["conf"] > best_confirmed["conf"]:
                        best_confirmed = d
            else:
                self._weak_tracks.append({
                    "ox": d["cx"], "oy": d["cy"],
                    "cx": d["cx"], "cy": d["cy"],
                    "last": stamp, "hits": 1, "det": d,
                })
        return best_confirmed

    def _best_match(self, detections, bbox, center, age, low=False, acquire=False,
                    anchor_center=None, gimbal_angles=None):
        pred_cx, pred_cy = center
        if not acquire and self.kalman.initialized:
            w, h = self.kalman.get_size()
        else:
            w, h = max(2.0, bbox[2] - bbox[0]), max(2.0, bbox[3] - bbox[1])
        pred_bbox = [pred_cx - w / 2, pred_cy - h / 2, pred_cx + w / 2, pred_cy + h / 2]
        diagonal = float(np.hypot(w, h))
        small_box = min(w, h) < 40.0
        motion_speed = 0.0
        if self.kalman.initialized:
            motion_speed = float(np.hypot(*self.kalman.get_world_velocity_px(gimbal_angles)))
        motion_allowance = min(180.0, motion_speed * max(age, 1.0 / 30.0) * 0.90)
        gate = min(
            self.DIST_MAX_PX,
            max(50.0, diagonal * 1.05) + min(280.0, age * 600.0) + motion_allowance,
        )
        size_limit = 4.2 if (small_box or age > 0.12) else 3.4
        if low:
            gate *= 0.78
            size_limit = 2.8 if small_box else 2.4
        if acquire:
            gate = min(190.0, max(65.0, diagonal * 1.6) + min(120.0, age * 350.0))
            size_limit = 3.2 if small_box else 2.5
        best_d, best_score = None, -1.0
        for d in detections:
            size_ratio = max(d["w"] / w, w / d["w"], d["h"] / h, h / d["h"])
            if size_ratio > size_limit:
                continue
            iou = self._iou(pred_bbox, self._bbox(d))
            dist = float(np.hypot(pred_cx - d["cx"], pred_cy - d["cy"]))
            if anchor_center is not None:
                dist_anchor = float(np.hypot(anchor_center[0] - d["cx"], anchor_center[1] - d["cy"]))
                dist = min(dist, dist_anchor)
            mahal = None
            if not acquire and self.kalman.initialized:
                mahal = self.kalman.mahalanobis_distance(
                    d["cx"], d["cy"], conf=d["conf"], gimbal_angles=gimbal_angles
                )
            in_elliptical_gate = mahal is not None and mahal <= (5.0 if low else 6.8)
            if dist > gate and not in_elliptical_gate:
                continue
            if not small_box and not acquire and iou < self.IOU_MIN and dist > gate * 0.78 and not in_elliptical_gate:
                continue
            score = 2.0 * iou + max(0.0, 1.0 - dist / max(gate, 1.0)) + 0.15 * d["conf"]
            if mahal is not None and np.isfinite(mahal):
                score += self.MAHALANOBIS_WEIGHT * max(0.0, 1.0 - mahal / 6.5)
            similarity = self._appearance_similarity(
                self.appearance_signature, d.get("_appearance")
            )
            if similarity is not None:
                score += self.APPEARANCE_WEIGHT * max(similarity, 0.0)
            if score > best_score:
                best_score, best_d = score, d
        if best_d is None and not acquire and not low and len(detections) == 1 and age > 0.10:
            cand = detections[0]
            if cand["conf"] >= 0.35:
                size_ratio = max(cand["w"] / w, w / cand["w"], cand["h"] / h, h / cand["h"])
                wide_gate = min(640.0, 240.0 + age * 800.0)
                dist_p = float(np.hypot(pred_cx - cand["cx"], pred_cy - cand["cy"]))
                dist_c = (
                    float(np.hypot(anchor_center[0] - cand["cx"], anchor_center[1] - cand["cy"]))
                    if anchor_center is not None else dist_p
                )
                if size_ratio <= 5.0 and min(dist_p, dist_c) <= wide_gate:
                    sim = self._appearance_similarity(self.appearance_signature, cand.get("_appearance"))
                    if sim is None or cand["w"] * cand["h"] < 900.0 or sim >= 0.25:
                        return cand
        return best_d

    def update(self, detections, timestamp=None, frame_shape=None, frame=None, gimbal_angles=None):
        self.matched_detection = None
        try:
            stamp = time.monotonic() if timestamp is None else float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(stamp) or (self._last_timestamp is not None and stamp <= self._last_timestamp):
            return None
        self._last_timestamp = stamp
        self.kalman.set_frame_shape(frame_shape)
        valid = self._valid_detections(detections if detections is not None else [], frame_shape)
        if frame is not None:
            for d in valid:
                if d["conf"] >= CONF_THRESHOLD:
                    d["_appearance"] = self._appearance_signature(frame, self._bbox(d))
        high = [d for d in valid if d["conf"] >= CONF_THRESHOLD]
        low = [d for d in valid if d["conf"] < CONF_THRESHOLD]

        if self.state == "IDLE":
            match = None
            seed_cand = None
            seed_time = None
            seed_gimbal = None
            lost_center = None
            lost_age = (
                stamp - self._lost_time
                if self._lost_bbox is not None and self._lost_time is not None
                else None
            )
            if lost_age is not None and 0 <= lost_age <= self.REACQUIRE_LOW_SEC:
                lx1, ly1, lx2, ly2 = self._lost_bbox
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                lost_center = (lcx, lcy)
                radius = self.REACQUIRE_RADIUS_PX + max(lx2 - lx1, ly2 - ly1) + 220.0 * lost_age
                near_low = [
                    d for d in low
                    if d["conf"] >= self.REACQUIRE_LOW_CONF
                    and np.hypot(d["cx"] - lcx, d["cy"] - lcy) <= radius
                ]
                if near_low:
                    high = high + near_low
            elif lost_age is not None:
                self._lost_bbox = None
                self._lost_time = None
            if self._candidate is not None and stamp - self._candidate_time <= self.ACQUIRE_MAX_GAP_SEC:
                candidate = self._candidate
                match = self._best_match(
                    high, self._bbox(candidate),
                    (candidate["cx"], candidate["cy"]),
                    stamp - self._candidate_time, acquire=True,
                )
                if match is not None:
                    seed_cand = candidate
                    seed_time = self._candidate_time
                    seed_gimbal = self._candidate_gimbal
            if match is None:
                if high:
                    if lost_center is not None:
                        self._candidate = min(
                            high,
                            key=lambda d: np.hypot(d["cx"] - lost_center[0], d["cy"] - lost_center[1]) - 120.0 * d["conf"],
                        )
                    else:
                        self._candidate = max(high, key=lambda d: d["conf"])
                    self._candidate_time = stamp
                    self._candidate_gimbal = gimbal_angles
                    return None
                weak_match = self._update_weak_confirmation(low, stamp)
                if weak_match is None:
                    if self._candidate_time is not None and stamp - self._candidate_time > self.ACQUIRE_MAX_GAP_SEC:
                        self._candidate = None
                        self._candidate_time = None
                        self._candidate_gimbal = None
                    return None
                match = weak_match

            seed_dist = (
                float(np.hypot(match["cx"] - seed_cand["cx"], match["cy"] - seed_cand["cy"]))
                if seed_cand is not None else 0.0
            )
            min_seed_jump = max(12.0, 0.30 * float(np.hypot(match["w"], match["h"])))
            if (
                seed_cand is not None and seed_time is not None
                and 0.01 <= (stamp - seed_time) <= 0.40
                and seed_dist >= min_seed_jump
            ):
                self.kalman.init(
                    seed_cand["cx"], seed_cand["cy"], seed_time,
                    w=seed_cand["w"], h=seed_cand["h"],
                    gimbal_angles=seed_gimbal, frame_shape=frame_shape,
                )
                self.kalman.predict(stamp, gimbal_angles=gimbal_angles)
                self.kalman.update(
                    match["cx"], match["cy"],
                    w=match["w"], h=match["h"], conf=match["conf"],
                    gimbal_angles=gimbal_angles,
                )
            else:
                self.kalman.init(
                    match["cx"], match["cy"], stamp,
                    w=match["w"], h=match["h"],
                    gimbal_angles=gimbal_angles, frame_shape=frame_shape,
                )
            gap = stamp - self._last_release_time if self._last_release_time is not None else None
            same_id = self.track_id > 0 and gap is not None and gap <= REACQUIRE_ID_GRACE_SEC
            similarity = None
            dist_lost = None
            if self._lost_bbox is not None:
                lx1, ly1, lx2, ly2 = self._lost_bbox
                dist_lost = float(np.hypot(match["cx"] - (lx1 + lx2) / 2, match["cy"] - (ly1 + ly2) / 2))
            kinematic_reach = min(900.0, 260.0 + 320.0 * gap) if gap is not None else 0.0
            near_quick = (
                same_id and gap <= 4.0
                and dist_lost is not None
                and dist_lost <= kinematic_reach
                and (len(high) <= 1 or dist_lost <= self.REACQUIRE_RADIUS_PX * 2 + 150.0 * gap)
            )
            if same_id and not near_quick and self.appearance_signature is not None:
                similarity = self._appearance_similarity(
                    self.appearance_signature, match.get("_appearance")
                )
                box_area = match["w"] * match["h"]
                sim_thr = self.REID_SIMILARITY_SMALL_MIN if box_area < 900.0 else self.REID_SIMILARITY_MIN
                same_id = similarity is None or similarity >= sim_thr
            if not same_id:
                self.track_id += 1
            _id_log(
                f"reacquire id={self.track_id} new={not same_id} "
                f"gap={'-' if gap is None else f'{gap:.2f}s'} "
                f"dist={'-' if dist_lost is None else f'{dist_lost:.0f}px'} "
                f"sim={'-' if similarity is None else f'{similarity:.2f}'} "
                f"near_quick={near_quick} conf={match['conf']:.2f}"
            )
            self._last_release_time = None
            self._candidate = None
            self._candidate_time = None
            self._candidate_gimbal = None
            self._weak_tracks.clear()
        else:
            prev_kalman_stamp = self.kalman.last_timestamp
            self.kalman.predict(stamp, gimbal_angles=gimbal_angles)
            age = stamp - self.last_match_time
            if age > 0.04 and prev_kalman_stamp is not None:
                dt_step = max(0.0, min(stamp - prev_kalman_stamp, 0.25))
                self.kalman.x[2:5, 0] *= np.exp(-dt_step / 0.24)
                self.kalman.x[7, 0] *= np.exp(-dt_step / 0.20)
            center_raw = self.kalman.get_pos(gimbal_angles=gimbal_angles)
            fw_b, fh_b = (
                (float(frame_shape[1]), float(frame_shape[0]))
                if frame_shape is not None and len(frame_shape) >= 2
                else (self.kalman.frame_w, self.kalman.frame_h)
            )
            center = (
                float(np.clip(center_raw[0], -80.0, fw_b + 80.0)),
                float(np.clip(center_raw[1], -80.0, fh_b + 80.0)),
            )
            anchor = self._last_match_center if age > 0.05 else None
            match = self._best_match(
                high, self.locked_bbox, center, age,
                anchor_center=anchor, gimbal_angles=gimbal_angles,
            )
            if match is None:
                match = self._best_match(
                    low, self.locked_bbox, center, age, low=True,
                    anchor_center=anchor, gimbal_angles=gimbal_angles,
                )
            rescue_reinit = False
            if match is None and len(high) == 1 and age > 0.14:
                cand = high[0]
                kw_cur, kh_cur = self.kalman.get_size()
                size_ratio = max(
                    cand["w"] / max(kw_cur, 2.0), kw_cur / max(cand["w"], 2.0),
                    cand["h"] / max(kh_cur, 2.0), kh_cur / max(cand["h"], 2.0),
                )
                d_pred = float(np.hypot(center[0] - cand["cx"], center[1] - cand["cy"]))
                d_anch = (
                    float(np.hypot(anchor[0] - cand["cx"], anchor[1] - cand["cy"]))
                    if anchor is not None else d_pred
                )
                d_min = min(d_pred, d_anch)
                wide_gate = min(760.0, 260.0 + age * 900.0)
                if d_min <= wide_gate and size_ratio <= 8.0:
                    rescue_match = None
                    if cand["conf"] >= 0.35:
                        rescue_match = cand
                    else:
                        if (
                            self._rescue_cand is not None
                            and np.hypot(cand["cx"] - self._rescue_cand["cx"], cand["cy"] - self._rescue_cand["cy"]) <= 80.0
                        ):
                            self._rescue_hits += 1
                        else:
                            self._rescue_hits = 1
                        self._rescue_cand = cand
                        if self._rescue_hits >= 2:
                            rescue_match = cand
                    if rescue_match is not None:
                        match = rescue_match
                        self._rescue_cand = None
                        self._rescue_hits = 0
                        if d_min > 160.0 or size_ratio > 3.5:
                            rescue_reinit = True
            else:
                self._rescue_cand = None
                self._rescue_hits = 0

            if match is None and age <= self.LOCK_HOLD_SEC:
                self.raw_bbox = None
                self.control_center = None
                return None
            if match is None:
                _id_log(f"lost id={self.track_id} age={age:.2f}s high={len(high)} low={len(low)}")
                self.reset(release_time=stamp)
                return None
            if rescue_reinit:
                self.kalman.init(
                    match["cx"], match["cy"], stamp,
                    w=match["w"], h=match["h"],
                    gimbal_angles=gimbal_angles, frame_shape=frame_shape,
                )
            else:
                upd_w = match["w"] if match["conf"] >= CONF_THRESHOLD else None
                upd_h = match["h"] if match["conf"] >= CONF_THRESHOLD else None
                self.kalman.update(
                    match["cx"], match["cy"],
                    w=upd_w, h=upd_h, conf=match["conf"],
                    gimbal_angles=gimbal_angles,
                )

        self.raw_bbox = self._bbox(match)
        cx, cy = self.kalman.get_pos(gimbal_angles=gimbal_angles)
        kw, kh = self.kalman.get_size()
        self.locked_bbox = [cx - kw * 0.5, cy - kh * 0.5, cx + kw * 0.5, cy + kh * 0.5]
        self.last_match_time = stamp
        self.matched_detection = match
        self._last_match_center = (float(match["cx"]), float(match["cy"]))
        self.state = "LOCKED"
        if match.get("_appearance") is not None:
            new_sig = np.asarray(match["_appearance"], dtype=np.float32)
            old_sig = self.appearance_signature
            if old_sig is not None and np.shape(old_sig) == new_sig.shape:
                mix = 0.8 * np.asarray(old_sig, dtype=np.float32) + 0.2 * new_sig
                n = float(np.linalg.norm(mix))
                new_sig = mix / n if n > 1e-6 else new_sig
            self.appearance_signature = new_sig
            self.appearance_time = stamp
        self.control_center = self.kalman.predict_pos_ahead(
            0.022, gimbal_angles=gimbal_angles, max_lead_px=18.0
        )
        return cx, cy, False


class MotionDetector:
    """Phát hiện chuyển động nhỏ có bù dịch chuyển camera (phaseCorrelate) làm ROI hint."""

    PROC_W = 320
    PROC_H = 180
    BORDER_MARGIN = 14
    MAX_DT_SEC = 0.25
    MAX_GLOBAL_SHIFT = 35.0
    MIN_CORR_RESP = 0.15
    CONSISTENCY_DIST = 28.0
    COOLDOWN_SEC = 2.0
    COOLDOWN_RADIUS = 110.0

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.prev_gray = None
        self.prev_stamp = None
        self.prev_cand = None
        self.cooldowns = []
        self._hann = cv2.createHanningWindow((self.PROC_W, self.PROC_H), cv2.CV_32F)

    def reset(self):
        self.prev_gray = None
        self.prev_stamp = None
        self.prev_cand = None

    def cool_down(self, cx, cy, now):
        self.cooldowns = [c for c in self.cooldowns if c[2] > now]
        self.cooldowns.append((float(cx), float(cy), float(now) + self.COOLDOWN_SEC))
        if len(self.cooldowns) > 16:
            self.cooldowns.pop(0)

    def _in_cooldown(self, cx_full, cy_full, now):
        self.cooldowns = [c for c in self.cooldowns if c[2] > now]
        for x, y, _ in self.cooldowns:
            if np.hypot(cx_full - x, cy_full - y) <= self.COOLDOWN_RADIUS:
                return True
        return False

    def detect_hint(self, frame, now, lores_frame=None):
        """Trả về (cx_full, cy_full, area_small) của đốm chuyển động nhất quán, hoặc None."""
        if not self.enabled or frame is None or frame.ndim != 3:
            return None
        fh, fw = frame.shape[:2]
        if fw < 32 or fh < 32:
            return None
        src_img = lores_frame if (lores_frame is not None and lores_frame.ndim == 3) else frame
        gray = cv2.resize(
            cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY),
            (self.PROC_W, self.PROC_H),
            interpolation=cv2.INTER_AREA,
        )
        prev = self.prev_gray
        prev_t = self.prev_stamp
        self.prev_gray = gray
        self.prev_stamp = now
        if prev is None or prev_t is None:
            return None
        dt = now - prev_t
        if dt <= 0.0 or dt > self.MAX_DT_SEC:
            self.prev_cand = None
            return None

        if float(prev.std()) < 2.0 and float(gray.std()) < 2.0:
            aligned = prev
        else:
            shift, resp = cv2.phaseCorrelate(np.float32(prev), np.float32(gray), window=self._hann)
            if (
                not np.isfinite(resp) or not np.all(np.isfinite(shift))
                or resp < self.MIN_CORR_RESP
                or np.hypot(shift[0], shift[1]) > self.MAX_GLOBAL_SHIFT
            ):
                self.prev_cand = None
                return None
            M = np.float32([[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]])
            aligned = cv2.warpAffine(
                prev, M, (self.PROC_W, self.PROC_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
        diff = cv2.absdiff(gray, aligned)
        m = self.BORDER_MARGIN
        valid = diff[m:-m, m:-m]
        if not valid.size:
            return None
        thr = float(np.clip(valid.mean() + 3.5 * valid.std(), 16.0, 55.0))
        mask = np.zeros_like(diff)
        mask[m:-m, m:-m] = np.where(valid >= thr, np.uint8(255), np.uint8(0))
        mask = cv2.dilate(mask, None, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or len(cnts) > 8:
            self.prev_cand = None
            return None

        sx = fw / float(self.PROC_W)
        sy = fh / float(self.PROC_H)
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not 2.0 <= area <= 350.0:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            cx_s, cy_s = bx + bw * 0.5, by + bh * 0.5
            cx_f, cy_f = cx_s * sx, cy_s * sy
            if self._in_cooldown(cx_f, cy_f, now):
                continue
            candidates.append((cx_s, cy_s, cx_f, cy_f, area))

        if not candidates:
            self.prev_cand = None
            return None

        matched = None
        if self.prev_cand is not None and now - self.prev_cand[2] <= self.MAX_DT_SEC:
            px, py, _ = self.prev_cand
            near = [c for c in candidates if np.hypot(c[0] - px, c[1] - py) <= self.CONSISTENCY_DIST]
            if near:
                matched = min(near, key=lambda c: np.hypot(c[0] - px, c[1] - py))

        lead = matched if matched is not None else max(candidates, key=lambda c: c[4])
        self.prev_cand = (lead[0], lead[1], now)
        if matched is None:
            return None
        return (float(matched[2]), float(matched[3]), float(matched[4]))


class RoiPlanner:
    """Lập lịch cửa sổ ROI 640x640 theo chế độ LOCK / REACQ / HINT / MOT / SAHI-SCAN / FULL."""

    def __init__(self):
        self.hint = None
        self.hint_time = None
        self.count = 0
        self.scan_idx = 0
        self.last_mode = "FULL"
        self.active_motion_hint = None
        self.motion_misses = 0
        self.roi_miss_streak = 0
        self.roi_backoff = 0
        self._last_was_crop = False

    def _clamp_box(self, cx, cy, fw, fh, size):
        cw, ch = min(size, fw), min(size, fh)
        x0 = int(round(min(max(cx - cw / 2, 0), fw - cw)))
        y0 = int(round(min(max(cy - ch / 2, 0), fh - ch)))
        return (x0, y0, int(cw), int(ch))

    def next_region(self, frame_shape, now, target_guide=None, motion_hint=None):
        self.count += 1
        self.active_motion_hint = None
        self._last_was_crop = False
        if not ROI_ENABLED:
            self.last_mode = "FULL"
            return None
        fh, fw = frame_shape[:2]
        if fw <= ROI_SIZE and fh <= ROI_SIZE:
            self.last_mode = "FULL"
            return None

        guide = target_guide.snapshot(now) if target_guide is not None else None
        if guide is not None and guide["locked"] and guide["bbox"] is not None:
            cx, cy, w, h = guide.get("roi_bbox") or guide["bbox"]
            if self.roi_miss_streak >= 2 or self.roi_backoff > 0:
                if self.roi_backoff > 0:
                    self.roi_backoff -= 1
                self.last_mode = "FULL"
                return None
            if max(w, h) < ROI_SMALL_MAX and self.count % ROI_FULL_EVERY != 0:
                cx = float(np.clip(cx, 0.0, float(fw)))
                cy = float(np.clip(cy, 0.0, float(fh)))
                self.last_mode = "ROI-LOCK"
                self._last_was_crop = True
                return self._clamp_box(cx, cy, fw, fh, ROI_SIZE)
            self.last_mode = "FULL"
            return None

        if guide is not None and guide["lost_bbox"] is not None:
            cx, cy, w, h = guide["lost_bbox"]
            if self.roi_miss_streak < 2 and max(w, h) < ROI_SMALL_MAX and self.count % 5 != 0:
                self.last_mode = "ROI-REACQ"
                self._last_was_crop = True
                return self._clamp_box(cx, cy, fw, fh, ROI_SIZE)
            self.last_mode = "FULL"
            return None

        fresh = self.hint is not None and now - self.hint_time <= ROI_HINT_TTL
        if fresh:
            cx, cy, w, h = self.hint
            if self.roi_miss_streak < 2 and max(w, h) < ROI_SMALL_MAX and self.count % 5 != 0:
                self.last_mode = "ROI-HINT"
                self._last_was_crop = True
                return self._clamp_box(cx, cy, fw, fh, ROI_SIZE)
            self.last_mode = "FULL"
            return None

        if motion_hint is not None:
            mx, my = motion_hint[0], motion_hint[1]
            self.active_motion_hint = (mx, my)
            self.last_mode = "ROI-MOT"
            self._last_was_crop = True
            return self._clamp_box(mx, my, fw, fh, ROI_SIZE)

        step = self.scan_idx % 4
        self.scan_idx += 1
        if step == 1:
            self.last_mode = "FULL"
            return None
        if step == 0:
            cx = fw / 2.0
        elif step == 2:
            cx = ROI_SIZE / 2.0
        else:
            cx = fw - ROI_SIZE / 2.0
        self.last_mode = "ROI-SCAN"
        self._last_was_crop = True
        return self._clamp_box(cx, fh / 2.0, fw, fh, ROI_SIZE)

    def update(self, detections, now, target_guide=None, motion_detector=None):
        has_valid_det = any(d["conf"] >= DETECTION_FLOOR for d in detections)
        if has_valid_det:
            if not self._last_was_crop and self.roi_miss_streak >= 2:
                self.roi_backoff = 10
            self.roi_miss_streak = 0
        elif self._last_was_crop:
            self.roi_miss_streak += 1
        else:
            self.roi_miss_streak = min(self.roi_miss_streak, 1)

        guide = target_guide.snapshot(now) if target_guide is not None else None
        if guide is not None and guide["locked"] and guide["bbox"] is not None:
            self.hint = guide["bbox"]
            self.hint_time = now
            self.motion_misses = 0
            return

        if self.active_motion_hint is not None:
            has_det = any(d["conf"] >= 0.13 for d in detections)
            if has_det:
                self.motion_misses = 0
            else:
                self.motion_misses += 1
                if self.motion_misses >= 3 and motion_detector is not None:
                    motion_detector.cool_down(self.active_motion_hint[0], self.active_motion_hint[1], now)
                    self.motion_misses = 0

        best = None
        if self.hint is not None and now - self.hint_time <= ROI_HINT_TTL:
            hx, hy = self.hint[0], self.hint[1]
            near = [
                d for d in detections
                if np.hypot(d["cx"] - hx, d["cy"] - hy) <= ROI_HINT_DIST
            ]
            if near:
                best = min(near, key=lambda d: np.hypot(d["cx"] - hx, d["cy"] - hy) - 100.0 * d["conf"])
        if best is None:
            strong = [d for d in detections if d["conf"] >= CONF_THRESHOLD]
            if strong:
                best = max(strong, key=lambda d: d["conf"])
        if best is not None:
            self.hint = (best["cx"], best["cy"], best["w"], best["h"])
            self.hint_time = now


def _offset_detections(dets, x0, y0):
    for d in dets:
        for k in ("cx", "x1", "x2"):
            d[k] = float(d[k] + x0)
        for k in ("cy", "y1", "y2"):
            d[k] = float(d[k] + y0)
    return dets


def detect_with_roi(frame, planner, detect_fn, now, target_guide=None, motion_detector=None):
    """detect_fn(img) -> detections theo tọa độ của img."""
    motion_hint = None
    if motion_detector is not None:
        guide = target_guide.snapshot(now) if target_guide is not None else None
        is_locked = guide is not None and guide["locked"]
        has_hint = planner.hint is not None and now - planner.hint_time <= ROI_HINT_TTL
        if is_locked or has_hint:
            motion_detector.reset()
        else:
            motion_hint = motion_detector.detect_hint(frame, now)

    region = planner.next_region(frame.shape, now, target_guide=target_guide, motion_hint=motion_hint)
    if region is None:
        dets = detect_fn(frame)
    else:
        x0, y0, cw, ch = region
        crop = np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])
        dets = _offset_detections(detect_fn(crop), x0, y0)
    planner.update(dets, now, target_guide=target_guide, motion_detector=motion_detector)
    STATE.update_stats(roi_mode=planner.last_mode)
    return dets
