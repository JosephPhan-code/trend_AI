"""
Anti-Overfitting Drone Detection Training Script for Apple Silicon Mac M4 (MPS) / CUDA
Built for Block-Stratified 80/10/10 Dataset + Hard Negatives!

Anti-Overfitting ("Chống học vẹt") Engineering Included:
1. Breaks static HUD/background/center-reticle shortcuts:
   - mosaic=1.0, mixup=0.15, copy_paste=0.15, erasing=0.35 (Random Erasing)
   - translate=0.20, degrees=10.0, perspective=0.0005, fliplr=0.5
   - Strong photometric jitter: hsv_h=0.02, hsv_s=0.75, hsv_v=0.45
2. Regularization & Generalization:
   - weight_decay=0.001, dropout=0.10, cos_lr=True, patience=10
   - close_mosaic=10 (sharpens tight bounding boxes in final 10 epochs)
   - Trained with ~3.5%-5% Hard Negative background images (birds, airplanes, empty HUD, buildings)
3. Objective Evaluation:
   - Automatically evaluates best.pt on the LOCKED unseen test/ split after training!
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import platform
import subprocess
from pathlib import Path


def get_system_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(out) / (1024 ** 3)
    except Exception:
        pass
    return 16.0


def recommend_preset(ram_gb: float):
    if ram_gb >= 30:
        return "s", 960, 16
    elif ram_gb >= 22:
        return "s", 960, 12
    else:
        return "n", 800, 12


def main():
    parser = argparse.ArgumentParser(description="Anti-Overfitting YOLO26-P2 Drone Training")
    parser.add_argument("--scale", choices=["n", "s", "m"], default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weights", type=str, default=None, help="Optional pretrained .pt for fine-tuning")
    parser.add_argument("--export-coreml", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    dataset_dir = Path(__file__).resolve().parent
    runs_dir = dataset_dir / "runs_mac_m4"

    portable_yaml = dataset_dir / "data_local.yaml"
    portable_yaml.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: drone\n",
        encoding="utf-8",
    )

    ram_gb = get_system_ram_gb()
    rec_scale, rec_imgsz, rec_batch = recommend_preset(ram_gb)
    scale = args.scale or rec_scale
    imgsz = args.imgsz or rec_imgsz
    batch = args.batch or rec_batch

    device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    run_name = f"yolo26{scale}_p2_img{imgsz}_anti_overfit"
    last_ckpt = runs_dir / run_name / "weights" / "last.pt"

    print("=" * 68)
    print(f"🚀 ANTI-OVERFITTING DRONE TRAINING (YOLO26-{scale.upper()}-P2)")
    print(f"  Device / RAM    : {device} ({platform.system()}, {ram_gb:.1f} GB)")
    print(f"  Dataset         : {dataset_dir} (80% Train / 10% Val / 10% Test + Hard Negatives)")
    print(f"  Settings        : imgsz={imgsz}, batch={batch}, epochs={args.epochs}, workers={args.workers}")
    print("  Anti-Shortcut   : MixUp=0.15 | CopyPaste=0.15 | RandomErasing=0.35 | Rot=10° | CosLR | WeightDecay")
    print("=" * 68)

    if last_ckpt.exists():
        print(f"🔄 Resuming from {last_ckpt}...")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        if args.weights:
            model = YOLO(args.weights)
        else:
            model = YOLO(f"yolo26{scale}-p2.yaml").load(f"yolo26{scale}.pt")

        model.train(
            data=str(portable_yaml),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            amp=(device != "mps"),
            workers=args.workers,
            optimizer="auto",
            cos_lr=True,             # Smooth cosine LR schedule prevents late-epoch memorization
            weight_decay=0.001,      # Stronger L2 weight decay anti-overfitting
            dropout=0.10,            # Dropout regularization
            patience=10,
            save=True,
            save_period=1,
            # --- Strong Anti-Shortcut / Anti-Pattern-Memorization Augmentations ---
            mosaic=1.0,              # Breaks fixed center-reticle / HUD layout patterns
            close_mosaic=10,         # Tight box refinement on clean images in last 10 epochs
            mixup=0.15,              # Blends scenes so model cannot memorize backgrounds
            copy_paste=0.15,         # Pastes drones across totally different skies/cities/mountains
            erasing=0.35,            # Randomly masks HUD text/watermarks/partial drone wings
            hsv_h=0.02,              # Color/hue shift (breaks camera sensor tint shortcuts)
            hsv_s=0.75,              # Saturation diversity (daylight/fog/dusk/desaturated)
            hsv_v=0.45,              # Brightness/exposure diversity (low-light & glare)
            degrees=10.0,            # Roll/tilt invariance (breaks horizon-level bias)
            translate=0.20,          # Position invariance (breaks crosshair-center bias)
            scale=0.35,              # Preserves tiny distant drones without erasing 4x4px targets
            perspective=0.0005,      # Mild camera tilt/perspective shift
            fliplr=0.5,
            flipud=0.0,
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
        )

    best_pt = runs_dir / run_name / "weights" / "best.pt"
    if best_pt.exists():
        print("\n📊 Running FINAL UNSEEN TEST-SET EVALUATION (split='test')...")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(data=str(portable_yaml), split="test", imgsz=imgsz, device=device)
        print(f"🏆 LOCKED TEST SET RESULTS: mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f} | P={metrics.box.mp:.4f} | R={metrics.box.mr:.4f}")

        if args.export_coreml and platform.system() == "Darwin":
            print("\n🍎 Exporting best.pt -> CoreML (.mlpackage) for M4 Neural Engine...")
            try:
                coreml_p = eval_model.export(format="coreml", imgsz=imgsz, half=True)
                print(f"✅ CoreML ready: {coreml_p}")
            except Exception as e:
                print(f"⚠️ CoreML export skipped: {e}")


if __name__ == "__main__":
    main()
