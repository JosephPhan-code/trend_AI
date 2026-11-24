"""
Stratified Anti-Leakage 80/10/10 (Train / Val / Test) Split + Anti-Overfitting Engineering
Applies to BOTH e:\trend_AI\data2 (219K+) and e:\trend_AI\clean_dataset (100K+):

1. Anti-Shortcut Hard Negative Injection (~3.5% - 5% empty-label 0-box background frames):
   - Injects real Anti-UAV Hard Negatives from e:\trend_AI\quarantine_non_drone
     (empty HUD crosshairs, buildings, dark clouds, civil airplanes '民航飞机')
   - Downloads & injects ~4,600 pure Bird / Airplane / Helicopter RGB Hard Negatives
     from athkot/drone-vs-bird-synthetic-10k (frames with ONLY birds/airplanes/helicopters, 0 drones!)
   -> Destroys 'always-predict-drone' & 'predict-crosshair/bird/airplane' shortcuts!

2. Sequence-Block Stratified Split (80% Train / 10% Val / 10% Test):
   - Groups consecutive frame IDs within each source into contiguous blocks (BLOCK_SIZE=150)
     so consecutive video frames NEVER leak across train / val / test!
   - Stratifies blocks independently per source (ir_2025, ir_410, rgb_antiuav300, rgb_pathikg,
     rgb_mavvid, rgb_dvb10k, rgb_detfly, hard_neg_bird, hard_neg_antiuav) so train, val, and test
     have IDENTICAL source, RGB/IR, and hard-negative distributions!
   - Uses atomic os.replace() on NTFS (instantaneous, 0 extra disk space).

3. Upgrades data.yaml (adds test: images/test) and train_mac_m4.py (adds anti-overfitting
   regularization: mixup, copy_paste, random erasing, rotation/perspective, cosine LR,
   weight_decay, dropout, and automatic locked test-set evaluation!).
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import io
import json
import time
import shutil
import random
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import requests
import pyarrow.parquet as pq
from PIL import Image

WORKSPACE = Path(r"e:\trend_AI")
DATA2_DIR = WORKSPACE / "data2"
BATCH1_DIR = WORKSPACE / "clean_dataset"
QUARANTINE_DIR = WORKSPACE / "quarantine_non_drone"
TMP_HN = WORKSPACE / "_tmp_hard_neg"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
BLOCK_SIZE = 150
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_bird_airplane_hard_negatives(out_dir: Path, max_count: int = 5000):
    """Downloads pure bird/airplane/helicopter (0 drone) RGB images from athkot/drone-vs-bird-synthetic-10k."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n🐦 Downloading pure Bird / Airplane / Helicopter Hard Negatives (0 drones)...", flush=True)
    r = requests.get("https://huggingface.co/api/datasets/athkot/drone-vs-bird-synthetic-10k/tree/main/data", timeout=15)
    if not r.ok:
        print(f"  ⚠️ Could not fetch dvb10k shards: HTTP {r.status_code}", flush=True)
        return []
    shards = sorted([x["path"] for x in r.json() if x["path"].endswith(".parquet")])
    saved = []
    seen_h = set()

    for shard in shards:
        if len(saved) >= max_count:
            break
        local_p = out_dir / os.path.basename(shard)
        url = f"https://huggingface.co/datasets/athkot/drone-vs-bird-synthetic-10k/resolve/main/{shard}"
        with requests.get(url, stream=True, allow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(local_p, "wb") as f:
                for ch in resp.iter_content(1024 * 1024):
                    if ch:
                        f.write(ch)
        try:
            tbl = pq.read_table(local_p)
            img_col = tbl["image"].to_pylist()
            obj_col = tbl["objects"].to_pylist()
            for i in range(tbl.num_rows):
                if len(saved) >= max_count:
                    break
                objs = obj_col[i] or {}
                cats = objs.get("category", []) or []
                # Pure Hard Negative = ZERO drones (cat == 0), and HAS bird(1)/airplane(2)/helicopter(3) or pure sky!
                if 0 in cats:
                    continue
                img_d = img_col[i]
                raw = img_d.get("bytes") if isinstance(img_d, dict) else None
                if not raw:
                    continue
                h = hashlib.md5(raw).hexdigest()
                if h in seen_h:
                    continue
                seen_h.add(h)
                dst = out_dir / f"hn_bird_air_{len(saved):05d}.jpg"
                if raw[:2] == b"\xff\xd8":
                    dst.write_bytes(raw)
                else:
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    im.save(dst, format="JPEG", quality=90)
                saved.append(dst)
        finally:
            if local_p.exists():
                local_p.unlink()
    print(f"  ✅ Extracted {len(saved)} pure Bird / Airplane / Helicopter RGB Hard Negatives!", flush=True)
    return saved


def inject_hard_negatives(dataset_dir: Path, bird_paths: list, antiuav_hn_paths: list, tag_prefix: str):
    """Copies hard negatives (with empty 0-byte .txt labels) into dataset_dir/images/train temporarily before split."""
    img_dst = dataset_dir / "images" / "train"
    lbl_dst = dataset_dir / "labels" / "train"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Check if already injected
    already_b = sum(1 for e in os.scandir(img_dst) if e.name.startswith(f"{tag_prefix}_hn_bird_"))
    already_a = sum(1 for e in os.scandir(img_dst) if e.name.startswith(f"{tag_prefix}_hn_antiuav_"))
    if already_b > 0 or already_a > 0:
        print(f"  Hard negatives already present in {dataset_dir.name} (bird={already_b}, antiuav={already_a}), skipping re-copy.", flush=True)
        return

    added_b = 0
    for i, src_p in enumerate(bird_paths):
        stem = f"{tag_prefix}_hn_bird_{i:05d}"
        shutil.copy2(src_p, img_dst / f"{stem}.jpg")
        (lbl_dst / f"{stem}.txt").write_text("", encoding="utf-8")  # Empty file = Hard Negative (0 drones)
        added_b += 1

    added_a = 0
    for i, src_p in enumerate(antiuav_hn_paths):
        ext = src_p.suffix.lower()
        stem = f"{tag_prefix}_hn_antiuav_{i:05d}"
        shutil.copy2(src_p, img_dst / f"{stem}{ext}")
        (lbl_dst / f"{stem}.txt").write_text("", encoding="utf-8")
        added_a += 1

    print(f"  ✅ Injected Hard Negatives into {dataset_dir.name}: +{added_b} bird/airplane RGB + {added_a} empty-HUD/airplane AntiUAV", flush=True)


def get_source_key(filename: str, is_data2: bool) -> str:
    if "_hn_bird_" in filename:
        return "hard_neg_bird_air"
    if "_hn_antiuav_" in filename:
        return "hard_neg_antiuav"
    if is_data2:
        if filename.startswith("d2_rgb_"):
            return "rgb_" + filename.split("_")[2]
        if filename.startswith("d2_antiuav_2025"):
            return "ir_2025"
        if filename.startswith("d2_anti_uav410"):
            return "ir_410"
        return "other"
    else:
        # Batch 1 clean_dataset: group by ID range (0..9312 original, 9313..32222 autolabel1, 32223+ data_train)
        if filename.startswith("drone_"):
            try:
                num = int(os.path.splitext(filename)[0].split("_")[1])
                if num < 9313:
                    return "b1_orig_mixed"
                elif num < 32223:
                    return "b1_tonghop_rgbt"
                else:
                    return "b1_datatrain_mixed"
            except ValueError:
                pass
        return "b1_other"


def block_stratified_split_80_10_10(dataset_dir: Path, is_data2: bool):
    """
    Re-splits ALL images in dataset_dir across train (80%) / val (10%) / test (10%) using
    contiguous temporal blocks (BLOCK_SIZE=150) stratified by source -> SAME distribution + NO frame leakage!
    """
    print(f"\n{'=' * 68}")
    print(f"📊 BLOCK-STRATIFIED 80/10/10 SPLIT (ANTI-LEAKAGE): {dataset_dir}")
    print(f"{'=' * 68}", flush=True)

    for sp in ["train", "val", "test"]:
        (dataset_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # 1. Gather all (current_split, filename) grouped by source
    by_source = defaultdict(list)
    for sp in ["train", "val", "test"]:
        img_d = dataset_dir / "images" / sp
        if not img_d.exists():
            continue
        for e in os.scandir(img_d):
            if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                skey = get_source_key(e.name, is_data2)
                by_source[skey].append((sp, e.name))

    rng = random.Random(20261122)
    assignments = {"train": [], "val": [], "test": []}
    dist_table = {}

    for skey in sorted(by_source.keys()):
        items = sorted(by_source[skey], key=lambda x: x[1])  # Sort by frame/file ID to keep sequences contiguous!
        n = len(items)
        # Adaptive block size so small sources (like detfly 998) still get clean ~10 blocks!
        blk_sz = min(BLOCK_SIZE, max(10, n // 20))
        blocks = [items[i : i + blk_sz] for i in range(0, n, blk_sz)]
        rng.shuffle(blocks)

        n_test_target = int(round(n * TEST_RATIO))
        n_val_target = int(round(n * VAL_RATIO))

        c_test = c_val = c_train = 0
        for blk in blocks:
            if c_test < n_test_target:
                for it in blk:
                    assignments["test"].append((skey, it[0], it[1]))
                c_test += len(blk)
            elif c_val < n_val_target:
                for it in blk:
                    assignments["val"].append((skey, it[0], it[1]))
                c_val += len(blk)
            else:
                for it in blk:
                    assignments["train"].append((skey, it[0], it[1]))
                c_train += len(blk)

        dist_table[skey] = {
            "total": n,
            "train": c_train,
            "val": c_val,
            "test": c_test,
            "train_pct": round(c_train / max(n, 1) * 100, 1),
            "val_pct": round(c_val / max(n, 1) * 100, 1),
            "test_pct": round(c_test / max(n, 1) * 100, 1),
        }

    # 2. Move files atomically on NTFS (only when target_split != current_split!)
    moved = 0
    for target_sp, lst in assignments.items():
        for _, cur_sp, fn in lst:
            if cur_sp == target_sp:
                continue
            stem = os.path.splitext(fn)[0]
            src_img = dataset_dir / "images" / cur_sp / fn
            dst_img = dataset_dir / "images" / target_sp / fn
            src_lbl = dataset_dir / "labels" / cur_sp / f"{stem}.txt"
            dst_lbl = dataset_dir / "labels" / target_sp / f"{stem}.txt"
            if src_img.exists():
                os.replace(src_img, dst_img)
            if src_lbl.exists():
                os.replace(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
            moved += 1
            if moved % 25000 == 0:
                print(f"  Moved {moved} image+label pairs...", flush=True)

    # 3. Verify final distribution & exact counts
    tot_tr = len(assignments["train"])
    tot_va = len(assignments["val"])
    tot_te = len(assignments["test"])
    grand = tot_tr + tot_va + tot_te

    print(f"  ✅ Atomic moves done ({moved} pairs relocated). Grand total: {grand}", flush=True)
    print(f"     TRAIN: {tot_tr} ({tot_tr/grand*100:.1f}%) | VAL: {tot_va} ({tot_va/grand*100:.1f}%) | TEST: {tot_te} ({tot_te/grand*100:.1f}%)", flush=True)
    print(f"\n  Per-source distribution check (TRAIN% | VAL% | TEST% of each split):", flush=True)
    for skey, d in dist_table.items():
        tr_share = d["train"] / max(tot_tr, 1) * 100
        va_share = d["val"] / max(tot_va, 1) * 100
        te_share = d["test"] / max(tot_te, 1) * 100
        print(f"    - {skey:20s}: total={d['total']:6d} | Train={tr_share:5.2f}% | Val={va_share:5.2f}% | Test={te_share:5.2f}%", flush=True)

    # Update data.yaml with train, val, AND test!
    (dataset_dir / "data.yaml").write_text(
        "# Cross-platform Anti-Leakage Stratified Drone Dataset (80% Train / 10% Val / 10% Test)\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: drone\n",
        encoding="utf-8",
    )

    # Save split report JSON
    split_rep = {
        "updated_at": datetime.now().isoformat(),
        "split_method": f"Sequence-Block Stratified Split (block_size<={BLOCK_SIZE}) — Zero Frame Leakage",
        "anti_overfitting_features": [
            "Contiguous video-block isolation across train/val/test",
            "Hard-Negative injection (birds, airplanes, helicopters, empty HUD crosshairs, buildings)",
            "Stratified identical distribution across train/val/test",
        ],
        "totals": {
            "grand_total": grand,
            "train": tot_tr,
            "val": tot_va,
            "test": tot_te,
        },
        "per_source_distribution": dist_table,
    }
    (dataset_dir / "split_distribution_report.json").write_text(
        json.dumps(split_rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return split_rep


def write_anti_overfit_train_script(target_dir: Path):
    """Upgrades train_mac_m4.py with strong Anti-Overfitting / Anti-Shortcut regularization + locked Test eval."""
    code = '''"""
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
        f"path: {dataset_dir.as_posix()}\\n"
        "train: images/train\\n"
        "val: images/val\\n"
        "test: images/test\\n"
        "nc: 1\\n"
        "names:\\n"
        "  0: drone\\n",
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
        print("\\n📊 Running FINAL UNSEEN TEST-SET EVALUATION (split='test')...")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(data=str(portable_yaml), split="test", imgsz=imgsz, device=device)
        print(f"🏆 LOCKED TEST SET RESULTS: mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f} | P={metrics.box.mp:.4f} | R={metrics.box.mr:.4f}")

        if args.export_coreml and platform.system() == "Darwin":
            print("\\n🍎 Exporting best.pt -> CoreML (.mlpackage) for M4 Neural Engine...")
            try:
                coreml_p = eval_model.export(format="coreml", imgsz=imgsz, half=True)
                print(f"✅ CoreML ready: {coreml_p}")
            except Exception as e:
                print(f"⚠️ CoreML export skipped: {e}")


if __name__ == "__main__":
    main()
'''
    (target_dir / "train_mac_m4.py").write_text(code, encoding="utf-8")


def main():
    t0 = time.time()
    print("=" * 68)
    print("🛡️ STRATIFIED 80/10/10 SPLIT + ANTI-OVERFITTING PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 68, flush=True)

    # Step 1: Collect Hard Negatives (pure Birds/Airplanes/Helicopters + Anti-UAV empty HUD/airplanes)
    bird_hn = extract_bird_airplane_hard_negatives(TMP_HN / "bird_air", max_count=5000)

    antiuav_hn = []
    if QUARANTINE_DIR.exists():
        for sp in ["train", "val"]:
            d = QUARANTINE_DIR / "images" / sp
            if d.exists():
                for e in sorted(os.scandir(d), key=lambda x: x.name):
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                        antiuav_hn.append(Path(e.path))
    print(f"🏢 Found {len(antiuav_hn)} real Anti-UAV Hard Negatives in quarantine_non_drone (empty HUD / civil airplanes).", flush=True)

    # Step 2: Inject Hard Negatives into data2 (~8,400 HN = ~3.7%) and clean_dataset (~4,500 HN = ~4.3%)
    inject_hard_negatives(DATA2_DIR, bird_hn, antiuav_hn, tag_prefix="d2")
    inject_hard_negatives(BATCH1_DIR, bird_hn[:2500], antiuav_hn[:2000], tag_prefix="b1")

    # Clean temp bird download cache
    shutil.rmtree(TMP_HN, ignore_errors=True)

    # Step 3: Perform Block-Stratified 80/10/10 Split on BOTH datasets (anti-frame-leakage + identical distributions!)
    rep2 = block_stratified_split_80_10_10(DATA2_DIR, is_data2=True)
    rep1 = block_stratified_split_80_10_10(BATCH1_DIR, is_data2=False)

    # Step 4: Upgrade train_mac_m4.py in BOTH folders with anti-overfitting augmentations + locked test eval
    write_anti_overfit_train_script(DATA2_DIR)
    write_anti_overfit_train_script(BATCH1_DIR)

    print("\n" + "=" * 68)
    print(f"✅ ALL DONE IN {time.time()-t0:.1f}s!")
    print(f"  data2         : {rep2['totals']['grand_total']} ({rep2['totals']['train']} train / {rep2['totals']['val']} val / {rep2['totals']['test']} test)")
    print(f"  clean_dataset : {rep1['totals']['grand_total']} ({rep1['totals']['train']} train / {rep1['totals']['val']} val / {rep1['totals']['test']} test)")
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
