#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=================================================================="
echo "  🎯 FINETUNE 15 EPOCHS CHONG BAO GIA BAU TROI (MAC M4 MPS)"
echo "  +10,000 anh nen bau troi rong (chim, may bay, may, dinh ang-ten)"
echo "=================================================================="

if [ ! -d ".venv_m4" ]; then
    echo "[1/3] Dang tao moi truong Python (.venv_m4)..."
    python3 -m venv .venv_m4
fi

source .venv_m4/bin/activate

echo "[2/3] Kiem tra Ultralytics + CoreML..."
pip install --upgrade pip -q
pip install -U ultralytics coremltools opencv-python -q

export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "[3/3] Bat dau Fine-tune 15 epochs (co caffeinate chong Sleep)..."
echo "      Mac dinh --mode focused (nhanh 5x, tap trung chong FP)."
echo "      Vi du: ./run_finetune_15ep_mac.sh --weights runs_mac_m4/.../best.pt --mode full"
echo "=================================================================="

caffeinate -dims python3 -u finetune_sky_hn_15ep.py --epochs 15 "$@"
