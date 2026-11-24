#!/bin/bash
# ==============================================================================
# Script tự động chạy Hailo Dataflow Compiler (DFC) trong Docker container
# Dành cho việc compile ONNX -> HEF (INT8) trên Linux x86_64 / WSL2
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo " HAILO DOCKER COMPILER RUNNER (ONNX -> HEF)"
echo "=========================================================="

DOCKER_IMAGE=${1:-"hailo_dfc:latest"}
TARGET_CHIP=${2:-"hailo8"}  # 'hailo8' hoặc 'hailo8l'

echo "[*] Thư mục làm việc: $DIR"
echo "[*] Docker Image: $DOCKER_IMAGE"
echo "[*] NPU Target: $TARGET_CHIP"

# Kiểm tra Docker có sẵn không
if ! command -v docker &> /dev/null; then
    echo "[!] Lỗi: Docker chưa được cài đặt hoặc daemon chưa bật."
    echo "    Vui lòng cài đặt Docker trên Linux hoặc WSL2."
    exit 1
fi

echo "[*] Đang khởi chạy Docker container để biên dịch HEF..."

docker run --rm \
    -v "$DIR":/workspace \
    -w /workspace \
    "$DOCKER_IMAGE" \
    python3 step2_onnx_to_hef.py \
        --onnx models/yolo26s_p2_epoch14_640.onnx \
        --calib_dir calib_images \
        --output_hef "models/yolo26s_p2_epoch14_640_${TARGET_CHIP}.hef" \
        --target "$TARGET_CHIP" \
        --imgsz 640

echo ""
echo "[+] Hoàn tất! File .hef đã được tạo bên trong thư mục models/."
