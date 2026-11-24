#!/usr/bin/env bash
# Khởi động hệ thống pan-tilt tracker trên Raspberry Pi 5.
# Đặt file này cùng thư mục với main.py, best.hef và servo_controller.py.

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo "Không tìm thấy main.py trong: $SCRIPT_DIR"
    [ -t 0 ] && read -r -p "Nhấn Enter để đóng..."
    exit 1
fi

# Đặt CPU scaling governor sang performance (nếu có quyền sudo không mật khẩu) để chống trễ nhảy xung
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] || sudo -n true 2>/dev/null; then
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | sudo -n tee "$gov" >/dev/null 2>&1 || true
    done
fi

# Ưu tiên virtual environment đi kèm dự án.
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "Không tìm thấy Python 3. Hãy cài Python hoặc tạo thư mục venv."
    [ -t 0 ] && read -r -p "Nhấn Enter để đóng..."
    exit 1
fi

# Mặc định mở full tốc camera (120 FPS ở mode mặc định 1536x864; 56 FPS nếu bật DRONE_TRACKER_WIDE_FOV=1).
# Vẫn cho phép ghi đè khi chạy, ví dụ: DRONE_TRACKER_CAMERA_FPS=30 ./run_drone_tracker.sh
if [ -z "${DRONE_TRACKER_CAMERA_FPS:-}" ]; then
    case "${DRONE_TRACKER_WIDE_FOV:-0}" in
        1|true|TRUE|yes|YES|on|ON) export DRONE_TRACKER_CAMERA_FPS=56 ;;
        *) export DRONE_TRACKER_CAMERA_FPS=120 ;;
    esac
else
    export DRONE_TRACKER_CAMERA_FPS
fi

export PYTHONUNBUFFERED=1
export DRONE_TRACKER_WEB="${DRONE_TRACKER_WEB:-0}"
echo "Drone tracker đang chạy bằng: $PYTHON"
echo "Thư mục: $SCRIPT_DIR"
echo "Camera Target FPS: $DRONE_TRACKER_CAMERA_FPS | Web Dashboard: $DRONE_TRACKER_WEB (set DRONE_TRACKER_WEB=1 để bật)"
echo "Nhấn Ctrl+C để dừng an toàn."

"$PYTHON" "$SCRIPT_DIR/main.py"
status=$?

if [ "$status" -ne 0 ]; then
    echo "Drone tracker đã dừng với mã lỗi: $status"
fi

# Giữ cửa sổ khi chạy từ terminal tương tác; khi chạy nền thì thoát luôn.
if [ -t 0 ]; then
    read -r -p "Nhấn Enter để đóng..."
fi
exit "$status"
