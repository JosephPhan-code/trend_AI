"""
15-Epoch Empty-Sky Hard-Negative Fine-Tuning Script (Mac M4 MPS / CUDA)
=======================================================================
Purpose:
  Fine-tune an existing trained Drone Detection checkpoint (`best.pt`) for 15 epochs
  using the 10,000 newly added Empty-Sky Hard Negatives:
    - Distant flying birds (chim bay xa)
    - Civilian commercial airplanes (may bay dan dung)
    - Backlit clouds & empty sky glare (may nguoc sang)
    - Antenna tips, telecom/cell/transmission towers against sky (dinh ang-ten)

Two training modes:
  --mode focused (default): ~50,266 images (36,000 stratified RGB+IR drones + ALL ~14,266 train Hard Negatives
                            including all 8,000 new sky hard negatives -> ~28% hard negative ratio).
                            Fast (~5x faster per epoch), laser-focused on crushing sky false positives
                            while keeping all 7 drone sources represented to prevent catastrophic forgetting.
  --mode full             : Full 237,636-image data2 dataset (~7.7% total hard negatives).

Fine-Tuning Hyperparameters (15 epochs):
  - Low learning rate: AdamW, lr0=0.0008, lrf=0.05, cos_lr=True, warmup_epochs=1.5
  - Boosted classification loss: cls=0.70 (sharpens drone vs bird/airplane/cloud/antenna boundary)
  - Clean final 5 epochs: mosaic=0.80, close_mosaic=5 (pure empty-sky frames punish FP heavily)
  - Anti-shortcut regularization: mixup=0.10, copy_paste=0.15, erasing=0.25, weight_decay=0.001, dropout=0.10

Post-Training Automatic Verification:
  1. Full locked test/ set mAP50, mAP50-95, Precision, Recall
  2. Dedicated 1,000-image Sky Hard-Negative False Positive Rate (FPR%) breakdown across all 4 categories!
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import platform
import subprocess
from pathlib import Path
from collections import Counter


SKY_PREFIX_TO_CAT = {
    "d2_hn_skybird_": "sky_bird (chim bay xa)",
    "d2_hn_skyair_": "sky_airplane (may bay dan dung)",
    "d2_hn_skycloud_": "sky_cloud (may nguoc sang)",
    "d2_hn_skyant_": "sky_antenna (dinh ang-ten)",
}


def get_system_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(out) / (1024 ** 3)
    except Exception:
        pass
    return 16.0


def find_default_weights(dataset_dir: Path) -> Path | None:
    """Auto-discovers the most recently modified best.pt inside runs_mac_m4 or workspace."""
    candidates = []
    for search_root in [dataset_dir / "runs_mac_m4", dataset_dir / "runs", dataset_dir.parent / "clean_relabel_train"]:
        if search_root.exists():
            candidates.extend(search_root.rglob("best.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def evaluate_sky_hard_negatives(model, dataset_dir: Path, imgsz: int, device, save_json: Path):
    """Runs inference on all locked test/ Sky Hard-Negative images and computes FPR% per category."""
    test_dir = dataset_dir / "images" / "test"
    by_cat = {cat: [] for cat in SKY_PREFIX_TO_CAT.values()}
    for e in sorted(os.scandir(test_dir), key=lambda x: x.name):
        if not e.is_file():
            continue
        for pref, cat in SKY_PREFIX_TO_CAT.items():
            if e.name.startswith(pref):
                by_cat[cat].append(e.path)
                break

    total_imgs = sum(len(v) for v in by_cat.values())
    if total_imgs == 0:
        print("⚠️ No sky hard-negative test images found in images/test.")
        return

    print("\n" + "=" * 70)
    print(f"🧪 SKY HARD-NEGATIVE FALSE POSITIVE STRESS TEST ({total_imgs} locked test images)")
    print("=" * 70)

    report = {"total_images": total_imgs, "per_category": {}}
    all_fp25 = all_fp50 = 0
    for cat, paths in by_cat.items():
        if not paths:
            continue
        fp25 = fp50 = total_boxes25 = 0
        # Batch predict for high throughput
        bs = 32
        for i in range(0, len(paths), bs):
            batch_p = paths[i : i + bs]
            results = model.predict(
                source=batch_p,
                imgsz=imgsz,
                conf=0.25,
                device=device,
                verbose=False,
            )
            for res in results:
                n_b = len(res.boxes)
                if n_b > 0:
                    fp25 += 1
                    total_boxes25 += n_b
                    max_c = float(res.boxes.conf.max().item())
                    if max_c >= 0.50:
                        fp50 += 1

        fpr25 = fp25 / len(paths) * 100.0
        fpr50 = fp50 / len(paths) * 100.0
        all_fp25 += fp25
        all_fp50 += fp50
        report["per_category"][cat] = {
            "tested": len(paths),
            "fp_images_conf25": fp25,
            "fpr_pct_conf25": round(fpr25, 2),
            "fp_images_conf50": fp50,
            "fpr_pct_conf50": round(fpr50, 2),
            "total_fp_boxes_conf25": total_boxes25,
        }
        print(f"  - {cat:36s}: n={len(paths):3d} | FPR@0.25={fpr25:5.2f}% ({fp25:3d}) | FPR@0.50={fpr50:5.2f}% ({fp50:3d})")

    overall25 = all_fp25 / total_imgs * 100.0
    overall50 = all_fp50 / total_imgs * 100.0
    report["overall_fpr_pct_conf25"] = round(overall25, 2)
    report["overall_fpr_pct_conf50"] = round(overall50, 2)
    print("-" * 70)
    print(f"  🎯 OVERALL SKY FALSE POSITIVE RATE : FPR@0.25 = {overall25:.2f}% ({all_fp25}/{total_imgs}) | FPR@0.50 = {overall50:.2f}% ({all_fp50}/{total_imgs})")
    print("=" * 70)

    save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="15-Epoch Empty-Sky Hard-Negative Fine-Tuning")
    parser.add_argument("--weights", type=str, default=None, help="Path to base trained best.pt to fine-tune")
    parser.add_argument("--mode", choices=["focused", "full"], default="focused",
                        help="'focused' (~50K imgs, 28%% HN, 5x faster) or 'full' (all 237K data2 imgs)")
    parser.add_argument("--epochs", type=int, default=15, help="Fine-tuning epochs (default: 15)")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (auto: 960 on Mac >=22GB, 800 on 16GB, 640 on 4GB CUDA)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (auto-selected by hardware)")
    parser.add_argument("--lr0", type=float, default=0.0008, help="Initial fine-tune LR (default: 0.0008)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate Sky Hard-Negative FPR without training")
    parser.add_argument("--export-coreml", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    dataset_dir = Path(__file__).resolve().parent
    runs_dir = dataset_dir / "runs_finetune_15ep"

    # Determine device and safe defaults
    if torch.backends.mps.is_available():
        device = "mps"
        ram_gb = get_system_ram_gb()
        rec_imgsz = 960 if ram_gb >= 22 else 800
        rec_batch = 16 if ram_gb >= 30 else 12
    elif torch.cuda.is_available():
        device = 0
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        rec_imgsz = 960 if vram_gb >= 12 else (800 if vram_gb >= 7 else 640)
        rec_batch = 16 if vram_gb >= 12 else (8 if vram_gb >= 7 else 8)
    else:
        device = "cpu"
        rec_imgsz, rec_batch = 640, 8

    imgsz = args.imgsz or rec_imgsz
    batch = args.batch or rec_batch

    # Resolve base weights
    if args.weights:
        base_weights = Path(args.weights)
    else:
        base_weights = find_default_weights(dataset_dir)
    if not base_weights or not base_weights.exists():
        raise FileNotFoundError(
            "Khong tim thay file weights best.pt de finetune! "
            "Hay truyen: python finetune_sky_hn_15ep.py --weights /duong/dan/toi/best.pt"
        )

    # Create portable data YAML for selected mode
    focused_txt = dataset_dir / "finetune_15ep_train.txt"
    train_entry = "finetune_15ep_train.txt" if (args.mode == "focused" and focused_txt.exists()) else "images/train"
    yaml_path = dataset_dir / f"data_finetune_15ep_{args.mode}.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        f"train: {train_entry}\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: drone\n",
        encoding="utf-8",
    )

    run_name = f"finetune_15ep_{args.mode}_img{imgsz}"
    run_out = runs_dir / run_name
    last_ckpt = run_out / "weights" / "last.pt"

    print("=" * 70)
    print(f"🎯 15-EPOCH SKY HARD-NEGATIVE FINE-TUNING ({args.mode.upper()} MODE)")
    print(f"  Base Weights : {base_weights}")
    print(f"  Device       : {device} ({platform.system()})")
    print(f"  Train Source : {train_entry}")
    print(f"  Settings     : epochs={args.epochs} | imgsz={imgsz} | batch={batch} | lr0={args.lr0} | cls=0.70")
    print("=" * 70)

    if args.eval_only:
        model = YOLO(str(base_weights))
        evaluate_sky_hard_negatives(model, dataset_dir, imgsz, device, dataset_dir / "sky_hn_baseline_fp_eval.json")
        return

    if last_ckpt.exists():
        print(f"🔄 Resuming interrupted 15-epoch fine-tune from {last_ckpt}...")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        model = YOLO(str(base_weights))
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            amp=(device != "mps"),
            workers=args.workers,
            # --- Fine-Tuning Optimizer & LR Schedule ---
            optimizer="AdamW",
            lr0=args.lr0,            # Gentle fine-tuning LR preserves drone recall
            lrf=0.05,                # Smooth decay to 5% of lr0
            warmup_epochs=1.5,
            cos_lr=True,
            weight_decay=0.001,
            dropout=0.10,
            cls=0.70,                # Stronger classification penalty against sky false positives
            patience=10,
            save=True,
            save_period=1,
            # --- Anti-FP & Anti-Shortcut Augmentations ---
            mosaic=0.80,
            close_mosaic=5,          # Last 5 epochs on clean native images (full empty-sky penalty!)
            mixup=0.10,
            copy_paste=0.15,
            erasing=0.25,
            hsv_h=0.02,
            hsv_s=0.75,
            hsv_v=0.45,
            degrees=10.0,
            translate=0.20,
            scale=0.35,
            perspective=0.0005,
            fliplr=0.5,
            flipud=0.0,
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
        )

    best_pt = run_out / "weights" / "best.pt"
    if best_pt.exists():
        print("\n📊 Running LOCKED TEST-SET EVALUATION (split='test')...")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(data=str(yaml_path), split="test", imgsz=imgsz, device=device)
        print(
            f"🏆 POST-FINETUNE TEST SET: mAP50={metrics.box.map50:.4f} | "
            f"mAP50-95={metrics.box.map:.4f} | P={metrics.box.mp:.4f} | R={metrics.box.mr:.4f}"
        )

        # Evaluate specifically on the 1,000 locked Sky Hard-Negative test images!
        evaluate_sky_hard_negatives(eval_model, dataset_dir, imgsz, device, run_out / "sky_hn_post_finetune_fp_eval.json")

        if args.export_coreml and platform.system() == "Darwin":
            print("\n🍎 Exporting fine-tuned best.pt -> CoreML (.mlpackage)...")
            try:
                coreml_p = eval_model.export(format="coreml", imgsz=imgsz, half=True)
                print(f"✅ CoreML ready: {coreml_p}")
            except Exception as e:
                print(f"⚠️ CoreML export skipped: {e}")


if __name__ == "__main__":
    main()
