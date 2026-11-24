"""
==============================================================================
Module: servo_controller.py
Mô tả : Điều khiển 2 động cơ Servo qua Sysfs/Gpiozero.
        Tích hợp vòng lặp làm mượt (Smooth Loop) chạy ngầm ở 50Hz.
==============================================================================
"""

import time
import threading
from pathlib import Path
import subprocess
import shutil

GPIO_PAN       = 12
GPIO_TILT      = 13

PAN_MIN_ANGLE  = -90.0
PAN_MAX_ANGLE  = 90.0
TILT_MIN_ANGLE = -45.0
TILT_MAX_ANGLE = 60.0

MIN_PULSE_US = 500
MAX_PULSE_US = 2500
FREQUENCY_HZ = 50
PERIOD_NS = int(1_000_000_000 / FREQUENCY_HZ)

PAN_HOME  = 0.0
TILT_HOME = 0.0

# Bước tối đa của servo trong mỗi nhịp 20 ms.
# Tilt thường nhạy với rung và độ rơ cơ khí hơn pan, nên giảm riêng bước
# tilt. StableTrackingController vẫn giới hạn tilt ở 50 độ/giây; mức 1.8
# độ/nhịp tương đương 90 độ/giây vẫn đủ dư để không làm nghẽn lệnh bám.
MAX_STEP_PAN  = 2.5
MAX_STEP_TILT = 1.8

def _angle_to_duty_ns(angle: float, min_a: float, max_a: float) -> int:
    angle = max(min_a, min(max_a, angle))
    span = max(max_a - min_a, 1e-6)
    pulse_us = MIN_PULSE_US + ((angle - min_a) / span) * (MAX_PULSE_US - MIN_PULSE_US)
    return int(round(pulse_us * 1000))

def _ensure_pwm_pinmux(pan_gpio=12, tilt_gpio=13):
    cmds = []
    if shutil.which("pinctrl"):
        cmds = [
            ["pinctrl", "set", str(pan_gpio), "a0"],
            ["pinctrl", "set", str(tilt_gpio), "a0"],
        ]
    for c in cmds:
        try: subprocess.run(c, timeout=2, capture_output=True, check=False)
        except Exception: pass

class _HwPwmChannel:
    def __init__(self, chip: str, channel: int):
        _ensure_pwm_pinmux()
        self.base = Path(f"/sys/class/pwm/{chip}/pwm{channel}")
        export = Path(f"/sys/class/pwm/{chip}/export")
        if not self.base.exists():
            try:
                export.write_text(str(channel))
                for _ in range(20):
                    if self.base.exists(): break
                    time.sleep(0.05)
            except Exception as e:
                raise RuntimeError(f"cannot export {chip}/pwm{channel}: {e}") from e
        try: (self.base / "enable").write_text("0")
        except Exception: pass
        try: (self.base / "period").write_text(str(PERIOD_NS))
        except Exception: pass
        try: (self.base / "enable").write_text("1")
        except Exception: pass
        
        self.duty_path = self.base / "duty_cycle"

    def set_duty_ns(self, duty_ns: int) -> None:
        self.duty_path.write_text(str(int(duty_ns)))

    def disable(self) -> None:
        try: (self.base / "enable").write_text("0")
        except Exception: pass

    def enable(self) -> None:
        try: (self.base / "enable").write_text("1")
        except Exception: pass

    def close(self):
        try: (self.base / "enable").write_text("0")
        except Exception: pass

class _GpioZeroServo:
    def __init__(self, gpio: int):
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory
        factory = LGPIOFactory()
        self._s = Servo(
            gpio,
            min_pulse_width=MIN_PULSE_US / 1_000_000,
            max_pulse_width=MAX_PULSE_US / 1_000_000,
            frame_width=1 / FREQUENCY_HZ,
            pin_factory=factory
        )

    def set_angle(self, angle: float, min_a: float, max_a: float):
        angle = max(min_a, min(max_a, angle))
        v = ((angle - min_a) / (max_a - min_a)) * 2.0 - 1.0
        self._s.value = v

    def disable(self):
        try: self._s.detach()
        except Exception: pass

    def enable(self):
        pass

    def close(self):
        try: self._s.detach()
        except Exception: pass

class ServoController:
    def __init__(self):
        self._lock = threading.Lock()
        self._target_pan  = PAN_HOME
        self._target_tilt = TILT_HOME
        self._current_pan = PAN_HOME
        self._current_tilt = TILT_HOME
        self._stop_event = threading.Event()

        def _devpath(chip_name):
            try: return str(Path(f"/sys/class/pwm/{chip_name}").resolve())
            except: return ""

        chips = ["pwmchip0", "pwmchip1", "pwmchip2", "pwmchip3"]
        chips = sorted(chips, key=lambda c: ("/98000" not in _devpath(c), c))

        self._use_hw = False
        self._pan_hw = None
        self._tilt_hw = None
        self._pan_sw = None
        self._tilt_sw = None

        last_exc = None
        for candidate in chips:
            if not Path(f"/sys/class/pwm/{candidate}").exists():
                continue
            try:
                pan_hw = _HwPwmChannel(candidate, 0)
                tilt_hw = _HwPwmChannel(candidate, 1)
                
                pan_hw.set_duty_ns(_angle_to_duty_ns(PAN_HOME, PAN_MIN_ANGLE, PAN_MAX_ANGLE))
                tilt_hw.set_duty_ns(_angle_to_duty_ns(TILT_HOME, TILT_MIN_ANGLE, TILT_MAX_ANGLE))
                
                self._pan_hw = pan_hw
                self._tilt_hw = tilt_hw
                self._use_hw = True
                print(f"[SERVO] Hardware PWM Sysfs OK: {candidate}")
                break
            except Exception as exc:
                last_exc = exc
                continue
                
        if not self._use_hw:
            print(f"[SERVO] Khong the dung Sysfs (Thieu sudo?): {last_exc}")
            print(f"[SERVO] Da fallback sang Software PWM (gpiozero)...")
            self._pan_sw = _GpioZeroServo(GPIO_PAN)
            self._tilt_sw = _GpioZeroServo(GPIO_TILT)

        self._pwm_enabled = True
        self._apply(self._current_pan, self._current_tilt)
        self._worker = threading.Thread(target=self._smooth_loop, daemon=True)
        self._worker.start()

    def disable_pwm(self):
        """Tạm ngắt xung PWM để servo xả tải cơ khí, chống rung giật."""
        with self._lock:
            self._pwm_enabled = False
            if self._use_hw:
                if self._pan_hw: self._pan_hw.disable()
                if self._tilt_hw: self._tilt_hw.disable()
            else:
                if self._pan_sw: self._pan_sw.disable()
                if self._tilt_sw: self._tilt_sw.disable()
        print("[SERVO] Tam ngat xung PWM (Giam tai & Chong rung giat).")

    def enable_pwm(self):
        """Mở lại xung PWM chính thức để điều khiển servo."""
        with self._lock:
            self._pwm_enabled = True
            if self._use_hw:
                if self._pan_hw: self._pan_hw.enable()
                if self._tilt_hw: self._tilt_hw.enable()
            else:
                if self._pan_sw: self._pan_sw.enable()
                if self._tilt_sw: self._tilt_sw.enable()
            self._apply(self._current_pan, self._current_tilt)
        print("[SERVO] Da mo lai xung PWM (San sang bám mục tiêu).")

    def _apply(self, pan, tilt):
        if not getattr(self, "_pwm_enabled", True):
            return
        if self._use_hw:
            self._pan_hw.set_duty_ns(_angle_to_duty_ns(pan, PAN_MIN_ANGLE, PAN_MAX_ANGLE))
            self._tilt_hw.set_duty_ns(_angle_to_duty_ns(tilt, TILT_MIN_ANGLE, TILT_MAX_ANGLE))
        else:
            self._pan_sw.set_angle(pan, PAN_MIN_ANGLE, PAN_MAX_ANGLE)
            self._tilt_sw.set_angle(tilt, TILT_MIN_ANGLE, TILT_MAX_ANGLE)

    def _approach(self, cur: float, tgt: float, step: float) -> float:
        d = tgt - cur
        if abs(d) <= step: return tgt
        return cur + (step if d > 0 else -step)

    def move(self, delta_pan: float, delta_tilt: float):
        with self._lock:
            delta_pan  = max(-2.5, min(2.5, delta_pan))
            delta_tilt = max(-2.5, min(2.5, delta_tilt))

            new_pan  = self._current_pan + delta_pan
            new_tilt = self._current_tilt + delta_tilt

            self._target_pan  = max(PAN_MIN_ANGLE,  min(PAN_MAX_ANGLE,  new_pan))
            self._target_tilt = max(TILT_MIN_ANGLE, min(TILT_MAX_ANGLE, new_tilt))

    def _smooth_loop(self):
        last_pan, last_tilt = None, None
        while not self._stop_event.wait(0.02):
            with self._lock:
                self._current_pan = self._approach(self._current_pan, self._target_pan, MAX_STEP_PAN)
                self._current_tilt = self._approach(self._current_tilt, self._target_tilt, MAX_STEP_TILT)
                pan, tilt = self._current_pan, self._current_tilt

            if (last_pan is None or abs(pan - last_pan) > 0.02 or 
                last_tilt is None or abs(tilt - last_tilt) > 0.02):
                self._apply(pan, tilt)
                last_pan, last_tilt = pan, tilt

    def go_home(self):
        with self._lock:
            self._target_pan = PAN_HOME
            self._target_tilt = TILT_HOME
        print("[SERVO] Về Home (0°, 0°)")

    def get_angles(self) -> tuple:
        with self._lock:
            return (self._current_pan, self._current_tilt)

    def shutdown(self):
        self.go_home()
        time.sleep(0.5)
        self._stop_event.set()
        self._worker.join(timeout=1.0)
        if self._use_hw:
            self._pan_hw.close()
            self._tilt_hw.close()
        else:
            self._pan_sw.close()
            self._tilt_sw.close()
        print("[SERVO] Đã tắt an toàn.")
