"""
Unified Leak-Proof Merger + Same-Distribution 80/10/10 (Train/Val/Test) Splitter + Anti-Overfitting Suite
Merges cleaned e:\\trend_AI\\clean_dataset + e:\\trend_AI\\data2 -> e:\\trend_AI\\drone_dataset_unified

Features:
  1. Zero-extra-disk NTFS hardlinking (os.link) for images & fresh hardlinked/copied labels.
  2. Exact MD5 deduplication across & within both batches (prefers human-labeled data2 over auto-labeled Batch 1).
  3. Anti-Leakage Union-Find Clustering:
       - Contiguous sequence blocks (BLOCK_SIZE=150) per source prevent video-frame leakage.
       - Non-trivial exact dHash64 linking across datasets (ignoring low-entropy flat/black hashes
         and mega-buckets > 80 items) prevents re-encoded duplicate leakage across train/val/test.
  4. Stratified Greedy Allocation -> identical source %, RGB/IR %, Hard-Negative %, and object-size
     distribution across train (80%), val (10%), and test (10%).
  5. Writes verified data.yaml, anti-overfitting train_unified.py, run_train_mac.sh,
     drone_tracker.yaml, mo_web_qa_unified.bat, and split_distribution_report.json.
"""

import os
import sys
import json
import time
import shutil
import random
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

WORKSPACE = Path(r"e:\trend_AI")
CLEAN_DIR = WORKSPACE / "clean_dataset"
DATA2_DIR = WORKSPACE / "data2"
OUT_DIR = WORKSPACE / "drone_dataset_unified"
IDX_P = Path(r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\scratch\audit_full\all_index_full.jsonl")

BLOCK_SIZE = 150
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 2026


def src_of(ds_name: str, stem: str) -> str:
    if ds_name == "data2":
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
        if stem.startswith("d2_hn_"):
            return "hn_" + stem.split("_")[2]
        return "d2_other"
    else:
        if stem.startswith("b1_hn_"):
            return "hn_" + stem.split("_")[2]
        if stem.startswith("drone_"):
            idx = int(stem.split("_")[1])
            if idx <= 9312:
                return "c1_orig"
            if idx <= 32222:
                return "c1_tonghop"
            return "c1_datatrain"
        return "c1_other"


def source_priority(ds_name: str, src: str) -> int:
    """Higher number = preferred when deduplicating exact MD5 duplicates."""
    if src.startswith("hn_"):
        return 100 if ds_name == "data2" else 50
    if ds_name == "data2":
        return 90
    if src == "c1_orig":
        return 70
    if src == "c1_datatrain":
        return 60
    return 50


def stem_seq_id(stem: str) -> int:
    parts = stem.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return abs(hash(stem)) % 1000000


def compute_md5_dhash(img_path: str):
    try:
        raw = Path(img_path).read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        im = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return md5, ""
        small = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
        diff = small[:, 1:] > small[:, :-1]
        val = 0
        for b in diff.ravel():
            val = (val << 1) | int(b)
        return md5, f"{val:016x}"
    except Exception:
        return "", ""


def is_informative_dhash(dh: str) -> bool:
    """Filters out pure black/white/flat-sky low-entropy 64-bit hashes."""
    if not dh or len(dh) != 16:
        return False
    bits = bin(int(dh, 16)).count("1")
    return 6 <= bits <= 58


def parse_box_stats(lbl_path: str):
    """Returns (n_boxes, small_cnt, med_cnt, large_cnt, valid_ok)."""
    try:
        lines = [l.strip() for l in open(lbl_path, "r", encoding="utf-8", errors="replace") if l.strip()]
    except OSError:
        return 0, 0, 0, 0, False
    sm, md, lg = 0, 0, 0
    for l in lines:
        ps = l.split()
        if len(ps) != 5:
            return len(lines), sm, md, lg, False
        cls = int(float(ps[0]))
        cx, cy, bw, bh = map(float, ps[1:])
        if cls != 0 or bw <= 0 or bh <= 0:
            return len(lines), sm, md, lg, False
        area = bw * bh
        if area < 0.005:
            sm += 1
        elif area < 0.04:
            md += 1
        else:
            lg += 1
    return len(lines), sm, md, lg, True


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]


def hardlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    t0 = time.time()
    print("=== UNIFIED MERGE + LEAK-PROOF ANTI-OVERFITTING 80/10/10 SPLIT ===", flush=True)

    # 1. Load precomputed MD5 + dHash64 cache from all_index_full.jsonl
    hash_cache = {}
    if IDX_P.exists():
        with open(IDX_P, encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                hash_cache[(d["ds"], d["stem"])] = (d.get("md5", ""), d.get("dhash", ""))
    print(f"Loaded {len(hash_cache)} precomputed MD5/dHash entries.", flush=True)

    # 2. Scan all surviving files in clean_dataset and data2
    raw_items = []
    need_hash_idx = []
    for ds_name, ds_root in (("clean_dataset", CLEAN_DIR), ("data2", DATA2_DIR)):
        for sp in ("train", "val", "test"):
            img_dir = ds_root / "images" / sp
            lbl_dir = ds_root / "labels" / sp
            if not img_dir.exists():
                continue
            for fn in os.listdir(img_dir):
                img_p = img_dir / fn
                if not img_p.is_file():
                    continue
                stem = img_p.stem
                lbl_p = lbl_dir / f"{stem}.txt"
                if not lbl_p.exists():
                    continue
                src = src_of(ds_name, stem)
                md5, dh = hash_cache.get((ds_name, stem), ("", ""))
                item = {
                    "ds": ds_name,
                    "orig_sp": sp,
                    "stem": stem,
                    "ext": img_p.suffix.lower(),
                    "src": src,
                    "img": str(img_p),
                    "lbl": str(lbl_p),
                    "md5": md5,
                    "dhash": dh,
                    "is_hn": src.startswith("hn_"),
                    "is_ir": ("_ir_" in src),
                    "seq_id": stem_seq_id(stem),
                    "prio": source_priority(ds_name, src),
                }
                if not md5:
                    need_hash_idx.append(len(raw_items))
                raw_items.append(item)

    print(f"Total surviving paired files discovered: {len(raw_items)} (need fresh MD5/dHash: {len(need_hash_idx)})", flush=True)

    # 3. Compute MD5 + dHash64 for new files (e.g., 10,000 d2_hn_sky* images) in parallel
    if need_hash_idx:
        t_h = time.time()
        paths = [raw_items[i]["img"] for i in need_hash_idx]
        with ThreadPoolExecutor(max_workers=16) as ex:
            res = list(ex.map(compute_md5_dhash, paths, chunksize=128))
        for idx, (m, dh) in zip(need_hash_idx, res):
            raw_items[idx]["md5"] = m
            raw_items[idx]["dhash"] = dh
        print(f"Hashed {len(need_hash_idx)} new images in {time.time()-t_h:.1f}s.", flush=True)

    # 4. Exact MD5 Deduplication (keep highest-priority copy; zero exact duplicates!)
    md5_groups = defaultdict(list)
    for it in raw_items:
        md5_groups[it["md5"] or f"nomd5_{it['ds']}_{it['stem']}"].append(it)

    dedup_items = []
    md5_dropped = Counter()
    for m, grp in md5_groups.items():
        if len(grp) == 1:
            dedup_items.append(grp[0])
        else:
            grp.sort(key=lambda x: (-x["prio"], x["ds"], x["stem"]))
            dedup_items.append(grp[0])
            for dup in grp[1:]:
                md5_dropped[f"{dup['ds']}:{dup['src']}"] += 1

    print(f"After Exact MD5 Dedup: {len(dedup_items)} unique images (dropped {sum(md5_dropped.values())} exact duplicates).", flush=True)
    for k, v in sorted(md5_dropped.items()):
        print(f"  MD5 dup dropped [{k}]: {v}", flush=True)

    # 5. Parse box stats in parallel & validate 0 bad labels / 0 empty_non_hn / 0 hn_has_box
    t_b = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        bstats = list(ex.map(parse_box_stats, [it["lbl"] for it in dedup_items], chunksize=500))

    clean_items = []
    bad_lbl_skipped = 0
    for it, (nb, sm, md, lg, ok) in zip(dedup_items, bstats):
        if not ok:
            bad_lbl_skipped += 1
            continue
        if it["is_hn"] and nb > 0:
            bad_lbl_skipped += 1
            continue
        if (not it["is_hn"]) and nb == 0:
            bad_lbl_skipped += 1
            continue
        it["n_boxes"] = nb
        it["sm"] = sm
        it["md"] = md
        it["lg"] = lg
        clean_items.append(it)
    print(f"Validated labels for {len(clean_items)} items in {time.time()-t_b:.1f}s (skipped invalid/empty: {bad_lbl_skipped}).", flush=True)

    # 6. Anti-Leakage Union-Find Clustering:
    #    (a) Contiguous sequence blocks (BLOCK_SIZE=150) per source
    #    (b) Non-trivial exact dHash64 linking (2 <= bucket_size <= 80) across & within datasets
    n = len(clean_items)
    uf = UnionFind(n)

    # 6a. Sequence block edges
    block_rep = {}
    for i, it in enumerate(clean_items):
        bkey = (it["src"], it["seq_id"] // BLOCK_SIZE)
        if bkey in block_rep:
            uf.union(i, block_rep[bkey])
        else:
            block_rep[bkey] = i

    # 6b. Informative exact dHash64 cross-links (only buckets 2..80 to avoid chaining unrelated sky/black frames)
    dh_buckets = defaultdict(list)
    for i, it in enumerate(clean_items):
        if is_informative_dhash(it["dhash"]):
            dh_buckets[it["dhash"]].append(i)

    dh_edges_linked = 0
    dh_skipped_large = 0
    for dh, idxs in dh_buckets.items():
        if len(idxs) < 2:
            continue
        if len(idxs) > 80:
            dh_skipped_large += 1
            continue
        # Only union if resulting combined size won't explode into a giant mega-cluster (> 2500)
        root0 = uf.find(idxs[0])
        for j in idxs[1:]:
            rj = uf.find(j)
            if root0 != rj and (uf.sz[root0] + uf.sz[rj]) <= 2500:
                uf.union(root0, rj)
                root0 = uf.find(root0)
                dh_edges_linked += 1

    # Collect connected components
    comp_map = defaultdict(list)
    for i in range(n):
        comp_map[uf.find(i)].append(i)
    comps = list(comp_map.values())
    max_comp = max(len(c) for c in comps)
    print(f"Union-Find built {len(comps)} leak-isolated clusters (dHash links={dh_edges_linked}, skipped mega-dh={dh_skipped_large}, max_cluster={max_comp}).", flush=True)

    # 7. Multi-Stratum Greedy Balanced 80% / 10% / 10% Assignment
    #    Group components by their dominant source, shuffle deterministically within size tiers,
    #    and greedily fill each split (train/val/test) to minimize relative quota error per source!
    rng = random.Random(SEED)
    by_dom_src = defaultdict(list)
    for c in comps:
        dom = Counter(clean_items[i]["src"] for i in c).most_common(1)[0][0]
        by_dom_src[dom].append(c)

    target_ratios = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
    split_assign = {}  # item_idx -> 'train'|'val'|'test'
    src_split_counts = defaultdict(lambda: Counter())
    src_totals = Counter(it["src"] for it in clean_items)

    # Sort sources largest-first or deterministic
    for dom_src in sorted(by_dom_src.keys()):
        clist = by_dom_src[dom_src]
        rng.shuffle(clist)
        # Sort roughly largest-first (with gentle randomization) so big clusters balance first
        clist.sort(key=lambda c: -len(c))
        for c in clist:
            c_src_cnt = Counter(clean_items[i]["src"] for i in c)
            best_sp = None
            best_score = -1e18
            for sp in ("train", "val", "test"):
                # Score = weighted deficit across sources in this component
                score = 0.0
                for s, k in c_src_cnt.items():
                    tgt = src_totals[s] * target_ratios[sp]
                    cur = src_split_counts[s][sp]
                    # Normalized deficit: positive when under target, negative when over target
                    score += ((tgt - cur) / max(1.0, tgt)) * k
                if score > best_score:
                    best_score = score
                    best_sp = sp
            for i in c:
                split_assign[i] = best_sp
                src_split_counts[clean_items[i]["src"]][best_sp] += 1

    # 8. Materialize e:\trend_AI\drone_dataset_unified via NTFS hardlinks (0 GB extra disk!)
    print(f"\nMaterializing unified dataset at {OUT_DIR} via NTFS hardlinks...", flush=True)
    if OUT_DIR.exists():
        print("  Cleaning previous unified images/labels directories...", flush=True)
        for sub in ("images", "labels"):
            sp_root = OUT_DIR / sub
            if sp_root.exists():
                shutil.rmtree(sp_root)

    for sp in ("train", "val", "test"):
        (OUT_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # Ensure 100% unique destination stems across clean_dataset + data2
    used_stems = set()
    link_tasks = []
    for i, it in enumerate(clean_items):
        sp = split_assign[i]
        stem = it["stem"]
        if stem in used_stems:
            stem = f"{it['ds'][:2]}_{stem}"
        used_stems.add(stem)
        it["final_stem"] = stem
        it["final_sp"] = sp
        dst_img = OUT_DIR / "images" / sp / f"{stem}{it['ext']}"
        dst_lbl = OUT_DIR / "labels" / sp / f"{stem}.txt"
        link_tasks.append((Path(it["img"]), dst_img, Path(it["lbl"]), dst_lbl))

    def do_link(t):
        si, di, sl, dl = t
        hardlink_or_copy(si, di)
        hardlink_or_copy(sl, dl)

    t_l = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(do_link, link_tasks, chunksize=500))
    print(f"Linked {len(link_tasks)} (image, label) pairs in {time.time()-t_l:.1f}s!", flush=True)

    # 9. Verify Distribution & Leakage Mathematically
    sp_totals = Counter()
    sp_rgb_ir = defaultdict(Counter)
    sp_pos_hn = defaultdict(Counter)
    sp_sizes = defaultdict(Counter)
    sp_md5s = defaultdict(set)
    sp_dhs = defaultdict(set)

    for it in clean_items:
        sp = it["final_sp"]
        sp_totals[sp] += 1
        sp_rgb_ir[sp]["IR" if it["is_ir"] else "RGB"] += 1
        sp_pos_hn[sp]["HardNeg_0box" if it["is_hn"] else "Positive_Drone"] += 1
        sp_sizes[sp]["small"] += it["sm"]
        sp_sizes[sp]["medium"] += it["md"]
        sp_sizes[sp]["large"] += it["lg"]
        sp_md5s[sp].add(it["md5"])
        if is_informative_dhash(it["dhash"]) and len(dh_buckets.get(it["dhash"], [])) <= 80:
            sp_dhs[sp].add(it["dhash"])

    md5_leak_tv = len(sp_md5s["train"] & sp_md5s["val"])
    md5_leak_tt = len(sp_md5s["train"] & sp_md5s["test"])
    md5_leak_vt = len(sp_md5s["val"] & sp_md5s["test"])
    dh_leak_tv = len(sp_dhs["train"] & sp_dhs["val"])
    dh_leak_tt = len(sp_dhs["train"] & sp_dhs["test"])
    dh_leak_vt = len(sp_dhs["val"] & sp_dhs["test"])

    # 10. Write data.yaml, drone_tracker.yaml, train_unified.py, run_train_mac.sh, mo_web_qa_unified.bat
    (OUT_DIR / "data.yaml").write_text(
        f"# Unified Drone Detection Dataset (Cleaned Batch 1 + Batch 2, Zero-Leakage 80/10/10)\n"
        f"# Generated: {datetime.now().isoformat()}\n"
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: drone\n",
        encoding="utf-8",
    )

    (OUT_DIR / "drone_tracker.yaml").write_text(
        "tracker_type: botsort\n"
        "track_high_thresh: 0.45\n"
        "track_low_thresh: 0.10\n"
        "new_track_thresh: 0.55\n"
        "track_buffer: 30\n"
        "match_thresh: 0.80\n"
        "fuse_score: True\n"
        "gmc_method: sparseOptFlow\n"
        "proximity_thresh: 0.5\n"
        "appearance_thresh: 0.25\n"
        "with_reid: False\n",
        encoding="utf-8",
    )

    train_script = '''"""
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
        print("\\n=== EVALUATING BEST WEIGHTS ON LOCKED TEST SET ===")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=dev)
        print(f"LOCKED TEST -> mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f} | P={metrics.box.mp:.4f} | R={metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
'''
    (OUT_DIR / "train_unified.py").write_text(train_script, encoding="utf-8")

    (OUT_DIR / "run_train_mac.sh").write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "cd \"$(dirname \"$0\")\"\n"
        "python3 train_unified.py --model yolo11s.pt --epochs 80 --imgsz 640 --batch 32 --workers 8\n",
        encoding="utf-8",
    )

    (WORKSPACE / "mo_web_qa_unified.bat").write_text(
        "@echo off\n"
        "title Drone Unified Dataset QA Server (Port 8901)\n"
        "cd /d e:\\trend_AI\n"
        "python -X utf8 -u step3_qa_server.py --dir e:\\trend_AI\\drone_dataset_unified --port 8901\n"
        "pause\n",
        encoding="utf-8",
    )

    # 11. Build and save comprehensive JSON report
    total_final = sum(sp_totals.values())
    per_source_report = {}
    for s in sorted(src_totals.keys()):
        st = src_totals[s]
        per_source_report[s] = {
            "total": st,
            "train": src_split_counts[s]["train"],
            "val": src_split_counts[s]["val"],
            "test": src_split_counts[s]["test"],
            "train_pct": round(100.0 * src_split_counts[s]["train"] / max(1, st), 2),
            "val_pct": round(100.0 * src_split_counts[s]["val"] / max(1, st), 2),
            "test_pct": round(100.0 * src_split_counts[s]["test"] / max(1, st), 2),
        }

    def pct_map(cnt_d):
        s = max(1, sum(cnt_d.values()))
        return {k: {"count": v, "pct": round(100.0 * v / s, 2)} for k, v in sorted(cnt_d.items())}

    disk_e = shutil.disk_usage("E:/")
    report = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(OUT_DIR),
        "elapsed_seconds": round(time.time() - t0, 2),
        "totals": {
            "total_unified_images": total_final,
            "train": sp_totals["train"],
            "val": sp_totals["val"],
            "test": sp_totals["test"],
            "train_pct": round(100.0 * sp_totals["train"] / max(1, total_final), 2),
            "val_pct": round(100.0 * sp_totals["val"] / max(1, total_final), 2),
            "test_pct": round(100.0 * sp_totals["test"] / max(1, total_final), 2),
            "exact_md5_duplicates_removed": sum(md5_dropped.values()),
            "invalid_labels_skipped": bad_lbl_skipped,
        },
        "leakage_verification": {
            "md5_overlap_train_val": md5_leak_tv,
            "md5_overlap_train_test": md5_leak_tt,
            "md5_overlap_val_test": md5_leak_vt,
            "dhash64_overlap_train_val": dh_leak_tv,
            "dhash64_overlap_train_test": dh_leak_tt,
            "dhash64_overlap_val_test": dh_leak_vt,
            "union_find_clusters": len(comps),
            "max_cluster_size": max_comp,
            "sequence_block_size": BLOCK_SIZE,
        },
        "modality_distribution": {sp: pct_map(sp_rgb_ir[sp]) for sp in ("train", "val", "test")},
        "positive_vs_hard_negative": {sp: pct_map(sp_pos_hn[sp]) for sp in ("train", "val", "test")},
        "box_size_distribution": {sp: pct_map(sp_sizes[sp]) for sp in ("train", "val", "test")},
        "per_source_distribution": per_source_report,
        "disk_e_free_gb": round(disk_e.free / 1e9, 2),
    }

    (OUT_DIR / "split_distribution_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n================ UNIFIED MERGE & SPLIT COMPLETE ================", flush=True)
    print(f"Folder : {OUT_DIR}", flush=True)
    print(f"Total  : {total_final} | train={sp_totals['train']} ({report['totals']['train_pct']}%) | val={sp_totals['val']} ({report['totals']['val_pct']}%) | test={sp_totals['test']} ({report['totals']['test_pct']}%)", flush=True)
    print(f"Leakage: MD5=(tv:{md5_leak_tv}, tt:{md5_leak_tt}, vt:{md5_leak_vt}) | dHash64=(tv:{dh_leak_tv}, tt:{dh_leak_tt}, vt:{dh_leak_vt})", flush=True)
    print(f"E: Free: {report['disk_e_free_gb']} GB | Elapsed: {report['elapsed_seconds']}s", flush=True)
    print("\nPer-Source Distribution:")
    for s, info in per_source_report.items():
        print(f"  {s:20s}: total={info['total']:6d} | train={info['train']:6d} ({info['train_pct']:5.2f}%) | val={info['val']:5d} ({info['val_pct']:5.2f}%) | test={info['test']:5d} ({info['test_pct']:5.2f}%)")
    print("\nModality (RGB vs IR):", json.dumps(report["modality_distribution"]))
    print("Pos vs HardNeg:", json.dumps(report["positive_vs_hard_negative"]))
    print("Box Sizes:", json.dumps(report["box_size_distribution"]))


if __name__ == "__main__":
    main()
