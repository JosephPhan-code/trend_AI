"""
Anti-Overfitting / Anti-Shortcut Drone Detector Training Script (Unified Dataset)
Compatible with Apple Silicon (MPS), NVIDIA CUDA, or CPU.
Anti-memorization design:
  - Mosaic (1.0, closed last 15 epochs) + MixUp (0.15) + Copy-Paste (0.15)
  - Random Erasing / Cutout (0.35) -> prevents memorizing single rotor/LED spots
  - Strong HSV + Scale + Degrees + Perspective augmentations -> destroys background/sensor shortcuts
  - Cosine LR schedule + Weight Decay (0.001) + Dropout (0.10) + Early Stopping (patience=20)
  - Automatic locked Test-Set evaluation at the end of training!
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolo26n.pt", help="Base checkpoint (e.g. yolo26n.pt, yolov8s.pt, yolo11s.pt)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--name", type=str, default="drone_unified_anti_overfit")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_yaml = root / "data.yaml"
    dev = args.device if args.device else pick_device()
    print(f"Training on device={dev} | data={data_yaml} | model={args.model}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=dev,
        project=str(root / "runs"),
        name=args.name,
        exist_ok=True,
        # Optimization & Regularization (anti-overfitting)
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.001,
        warmup_epochs=3.0,
        dropout=0.10,
        patience=20,
        # Anti-Shortcut Augmentations (destroys background/HUD/position memorization)
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.15,
        copy_paste=0.15,
        erasing=0.35,
        hsv_h=0.02,
        hsv_s=0.65,
        hsv_v=0.45,
        degrees=12.0,
        translate=0.15,
        scale=0.60,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        cache=False,
        amp=True,
        save_period=10,
    )

    # Automatic final unbiased evaluation on locked test set
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print("\n=== EVALUATING BEST WEIGHTS ON LOCKED TEST SET ===")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=dev)
        print(f"LOCKED TEST -> mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f} | P={metrics.box.mp:.4f} | R={metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
