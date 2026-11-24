#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "  🍎 THIET LAP MOI TRUONG & TRAIN DRONE TREN MAC M4"
echo "============================================================"

# 1. Tao virtual environment neu chua co
if [ ! -d ".venv_m4" ]; then
    echo "[1/3] Dang tao moi truong Python (.venv_m4)..."
    python3 -m venv .venv_m4
fi

source .venv_m4/bin/activate

# 2. Cai dat Ultralytics + PyTorch MPS + CoreMLTools
echo "[2/3] Kiem tra & cai dat thu vien Ultralytics + CoreML..."
pip install --upgrade pip -q
pip install -U ultralytics coremltools opencv-python -q

# 3. Bat MPS fallback & dung caffeinate giu Mac M4 khong bi Sleep khi tat man hinh qua dem
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "[3/3] Bat dau Train (da bat caffeinate chong ngu dong Mac qua dem)..."
echo "      Neu ban nhan Ctrl+C tam dung, lan sau chay lai file nay se TU DONG RESUME!"
echo "============================================================"

caffeinate -dims python3 -u train_mac_m4.py "$@"
