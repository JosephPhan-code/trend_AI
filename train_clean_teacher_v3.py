"""
Train Clean Teacher v3 (High-Purity Anti-Pole / Anti-Shadow / Anti-Bird Detector) on RTX 2050
Trained EXCLUSIVELY on verified 100%-clean sources (d2_* + 6 Hard Negative domains + c1_orig) —
ZERO c1_datatrain auto-labeled noise!

Purpose:
  1. Learn exact visual distinction between real drones vs stadium light poles, power pylons,
     drone ground shadows, birds, airplanes, helicopters, and grass debris.
  2. Immediately validate on both clean val set AND Hard Negative false-positive rate (hn_skyant, hn_bird)
     AND the 4 residual c1_datatrain error cases (drone_039127, 036071, 063668, 086194).
  3. Power Step 2 (purify_with_teacher_v3.py) to push unified dataset label cleanliness to >98-99%.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from ultralytics import YOLO

WORKSPACE = Path(r"e:\trend_AI")
UNIFIED = WORKSPACE / "drone_dataset_unified"
TRAIN_TXT = UNIFIED / "teacher_v3_train.txt"
VAL_TXT = UNIFIED / "teacher_v3_val.txt"
DATA_YAML = UNIFIED / "data_teacher_v3.yaml"
PROJECT_DIR = WORKSPACE / "clean_relabel_train"
RUN_NAME = "clean_teacher_v3"

# Curated 20,000 train + 2,500 val high-purity subset (0% c1_datatrain noise!)
TRAIN_QUOTAS = {
    "d2_rgb_pathikg": 7500,     # Exact outdoor park/desert/poles/shadows domain with 100% human GT!
    "d2_rgb_antiuav300": 2500,  # RGB tracking + HUD + night LEDs
    "d2_rgb_mavvid": 2000,      # Multi-drone & outdoor landscapes
    "d2_rgb_dvb10k": 1500,      # Clean drones alongside birds/helicopters/airliners
    "d2_rgb_detfly": 800,       # Top-down small drones
    "d2_ir_410": 800,           # Thermal IR single/building
    "d2_ir_2025": 400,          # Thermal IR swarms
    "c1_orig": 1000,            # Diverse close-up & handheld drones
    # 6 Verified Pure Hard Negative Domains (3,500 = 17.5% HNs to crush pole/pylon/bird FPs!)
    "hn_skyant": 1200,          # Utility poles, power pylons, antennas, towers
    "hn_bird": 800,             # Birds, helicopters, airliners
    "hn_skyair": 400,           # Commercial airplanes
    "hn_skybird": 400,          # Birds & thermal sky/branches
    "hn_skycloud": 400,         # Clouds & rooftops
    "hn_antiuav": 300,          # Purified empty HUD crosshairs/buildings/cranes
}

VAL_QUOTAS = {
    "d2_rgb_pathikg": 900,
    "d2_rgb_antiuav300": 300,
    "d2_rgb_mavvid": 250,
    "d2_rgb_dvb10k": 200,
    "d2_rgb_detfly": 60,
    "d2_ir_410": 100,
    "d2_ir_2025": 50,
    "c1_orig": 140,
    "hn_skyant": 150,
    "hn_bird": 120,
    "hn_skyair": 60,
    "hn_skybird": 60,
    "hn_skycloud": 60,
    "hn_antiuav": 50,
}


def src_of_stem(stem: str) -> str:
    if stem.startswith("d2_antiuav_2025_"):
        return "d2_ir_2025"
    if stem.startswith("d2_anti_uav410_"):
        return "d2_ir_410"
    if stem.startswith("d2_rgb_antiuav300_"):
        return "d2_rgb_antiuav300"
    if stem.startswith("d2_rgb_pathikg_"):
        return "d2_rgb_pathikg"
    if stem.startswith("d2_rgb_mavvid_"):
        return "d2_rgb_mavvid"
    if stem.startswith("d2_rgb_dvb10k_"):
        return "d2_rgb_dvb10k"
    if stem.startswith("d2_rgb_detfly_"):
        return "d2_rgb_detfly"
    if stem.startswith("d2_hn_") or stem.startswith("b1_hn_"):
        return "hn_" + stem.split("_")[2]
    if stem.startswith("drone_"):
        idx = int(stem.split("_")[1])
        if idx <= 9312:
            return "c1_orig"
        if idx <= 32222:
            return "c1_tonghop"
        return "c1_datatrain"
    return "other"


def build_clean_lists():
    rng = random.Random(2026)
    summary = {}
    for sp, quotas, out_txt in (("train", TRAIN_QUOTAS, TRAIN_TXT), ("val", VAL_QUOTAS, VAL_TXT)):
        idir = UNIFIED / "images" / sp
        by_src = defaultdict(list)
        for fn in sorted(os.listdir(idir)):
            stem = os.path.splitext(fn)[0]
            src = src_of_stem(stem)
            if src in quotas:
                by_src[src].append(str(idir / fn))
        chosen = []
        counts = {}
        for src, q in quotas.items():
            pool = by_src[src]
            # Evenly strided sampling across sorted frame IDs maximizes scene diversity!
            if len(pool) <= q:
                picked = list(pool)
            else:
                step = len(pool) / q
                picked = [pool[int(i * step)] for i in range(q)]
            chosen.extend(picked)
            counts[src] = len(picked)
        rng.shuffle(chosen)
        out_txt.write_text("\n".join(chosen) + "\n", encoding="utf-8")
        summary[sp] = {"total": len(chosen), "per_source": counts}
        print(f"Built {out_txt.name}: {len(chosen)} clean images | {counts}", flush=True)

    DATA_YAML.write_text(
        f"# Clean Teacher v3 Gold Subset (100% Clean d2_* + 6 HN domains + c1_orig, ZERO c1_datatrain)\n"
        f"path: {UNIFIED}\n"
        f"train: {TRAIN_TXT.name}\n"
        f"val: {VAL_TXT.name}\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: drone\n",
        encoding="utf-8",
    )
    return summary


def main():
    t0 = time.time()
    print("=== STEP 1: BUILDING GOLD CLEAN TEACHER V3 LISTS ===", flush=True)
    summary = build_clean_lists()

    # Clear any old cache for teacher txt lists
    for cp in UNIFIED.glob("teacher_v3_*.cache"):
        try:
            cp.unlink()
        except OSError:
            pass

    print(f"\n=== STEP 2: TRAINING CLEAN TEACHER V3 ON RTX 2050 (CUDA={torch.cuda.is_available()}) ===", flush=True)
    # Start fresh from pretrained COCO yolov8n.pt so ZERO old pole/shadow bias is inherited!
    model = YOLO(str(WORKSPACE / "yolov8n.pt"))
    results = model.train(
        data=str(DATA_YAML),
        epochs=12,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        amp=True,
        cos_lr=True,
        lr0=0.0025,
        lrf=0.01,
        weight_decay=0.0008,
        warmup_epochs=2.0,
        patience=6,
        mosaic=1.0,
        close_mosaic=3,
        mixup=0.10,
        erasing=0.25,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.12,
        scale=0.55,
        fliplr=0.5,
        cache=False,
        plots=False,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n=== TRAINING COMPLETE in {(time.time()-t0)/60:.1f} min | best={best_pt} ===", flush=True)

    # Quick validation on Hard Negative test poles/birds + 4 blind-audit c1_datatrain error images
    if best_pt.exists():
        bm = YOLO(str(best_pt))
        # Check false positive rate on 500 unseen test Hard Negatives (hn_skyant + hn_bird)
        test_hn_ant = sorted((UNIFIED / "images" / "test").glob("d2_hn_skyant_*.jpg"))[:250]
        test_hn_bird = sorted((UNIFIED / "images" / "test").glob("d2_hn_bird_*.jpg"))[:250]
        for tag, paths in (("test_hn_skyant_poles", test_hn_ant), ("test_hn_bird_planes", test_hn_bird)):
            fp_imgs = 0
            for i in range(0, len(paths), 32):
                rs = bm.predict([str(p) for p in paths[i : i + 32]], imgsz=640, conf=0.35, device=0, verbose=False)
                for r in rs:
                    if r.boxes is not None and len(r.boxes) > 0:
                        fp_imgs += 1
            print(f"Unseen Test HN Check [{tag}] (n={len(paths)}, conf>=0.35): FP={fp_imgs} ({100*fp_imgs/max(1,len(paths)):.2f}%)", flush=True)

        # Check predictions on the 4 specific c1_datatrain residual error cases we spotted earlier
        print("\nSpot-check on 4 blind-audit c1_datatrain residual errors:")
        for stem in ("drone_039127", "drone_036071", "drone_063668", "drone_086194"):
            for sp in ("train", "val", "test"):
                ip = UNIFIED / "images" / sp / f"{stem}.jpg"
                lp = UNIFIED / "labels" / sp / f"{stem}.txt"
                if ip.exists():
                    r = bm.predict(str(ip), imgsz=640, conf=0.20, device=0, verbose=False)[0]
                    gt = [l.strip() for l in open(lp) if l.strip()]
                    pr = [
                        ([round(float(v), 3) for v in b], round(float(cf), 3))
                        for b, cf in zip(r.boxes.xywhn.cpu().numpy(), r.boxes.conf.cpu().numpy())
                    ] if r.boxes is not None and len(r.boxes) else []
                    print(f"  {stem} ({sp}) | GT={len(gt)}: {gt} | Teacher_v3={len(pr)}: {pr}", flush=True)
                    break

    (PROJECT_DIR / RUN_NAME / "teacher_v3_meta.json").write_text(
        json.dumps({"summary": summary, "elapsed_sec": round(time.time() - t0, 1), "best_pt": str(best_pt)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
