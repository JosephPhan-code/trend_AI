#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 train_unified.py --model yolo11s.pt --epochs 80 --imgsz 640 --batch 32 --workers 8
