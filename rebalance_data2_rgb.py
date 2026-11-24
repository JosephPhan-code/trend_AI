"""
Rebalance & Enrich e:\trend_AI\data2 -> Balanced Diverse Dataset (> 210,000 images)
Target Composition:
  - ~50,000 Best Thermal IR images (25,000 Anti-UAV 2025 MultiUAV + 25,000 Anti-UAV410)
  - ~165,000+ COLORFUL, DIVERSE RGB Drone Images from 5 distinct public HuggingFace sources:
      1. pathikg/drone-detection-dataset       (~54,071 real-world colorful RGB outdoor photos)
      2. guaguarobotics/mav-vid                (~40,232 Full-HD colorful RGB multi-view frames)
      3. DestinyLove/Det-Fly                   (~13,000 colorful Air-to-Air RGB drone images)
      4. athkot/drone-vs-bird-synthetic-10k    (~9,950 colorful RGB Drone-vs-Bird, drone class only!)
      5. huseyincavus/anti-uav-rgbt-300        (visible.mp4 COLOR RGB streams ONLY, daytime/dusk)

Guarantees:
- 0% overlap with Batch 1 (e:\trend_AI\clean_dataset) via MD5 deduplication
- Every single image passes 5-layer geometry + pixel contrast audit + Saliency Tight-Fit alignment
- Re-encodes bloated PNGs to crisp JPEG Q=92 to keep disk usage sleek (~18-24 GB total)
- Hard disk safety floor: >= 3.5 GB free on E:\ at all times
- Deletes each downloaded Parquet shard / ZIP immediately after streaming
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import io
import csv
import json
import time
import shutil
import random
import hashlib
import zipfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import pyarrow.parquet as pq
import cv2
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
Image.MAX_IMAGE_PIXELS = None

WORKSPACE = Path(r"e:\trend_AI")
BATCH1_DIR = WORKSPACE / "clean_dataset"
DATA2_DIR = WORKSPACE / "data2"
TMP_DL = DATA2_DIR / "_rgb_tmp"
AUDIT_LOG = DATA2_DIR / "relabel_audit_log.csv"
REPORT_JSON = DATA2_DIR / "data2_report.json"

KEEP_IR_2025 = 25000
KEEP_IR_410 = 25000
TARGET_RGB_TOTAL = 165000
FREE_DISK_FLOOR_GB = 3.5
TRAIN_RATIO = 0.85
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def free_gb() -> float:
    return shutil.disk_usage(str(WORKSPACE)).free / (1024 ** 3)


def file_md5_path(filepath: str) -> str:
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(131072), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def bytes_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def box_iou(b1, b2) -> float:
    x1_min, y1_min = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x1_max, y1_max = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x2_min, y2_min = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2_max, y2_max = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    ix1, iy1 = max(x1_min, x2_min), max(y1_min, y2_min)
    ix2, iy2 = min(x1_max, x2_max), min(y1_max, y2_max)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def has_hud_distractor_watermark(gray: np.ndarray) -> bool:
    h, w = gray.shape
    if (w, h) != (1920, 1080):
        return False
    roi = gray[1008:1066, 12:175]
    bg_med = float(np.median(gray[960:1005, 12:200]))
    if bg_med > 35.0:
        return False
    return 0.04 <= float((roi > 220).mean()) <= 0.35


def is_geometry_clean(w: int, h: int, cx: float, cy: float, bw: float, bh: float) -> bool:
    area = bw * bh
    if area < 0.00010 or area > 0.72 or bw < 0.0025 or bh < 0.0025:
        return False
    if max(bw / max(bh, 1e-6), bh / max(bw, 1e-6)) > 4.5:
        return False
    if (w, h) == (1920, 1080) and area > 0.08:
        return False
    if (w, h) == (640, 512):
        if area > 0.06:
            return False
        if (cy + bh / 2 > 0.93 or cx - bw / 2 < 0.02 or cx + bw / 2 > 0.98) and area > 0.015:
            return False
    return True


def tighten_box_to_drone(gray: np.ndarray, cx: float, cy: float, bw: float, bh: float):
    h, w = gray.shape
    if bw * bh > 0.025:
        return cx, cy, bw, bh, False
    x1 = max(0, int(round((cx - bw / 2) * w)))
    y1 = max(0, int(round((cy - bh / 2) * h)))
    x2 = min(w, int(round((cx + bw / 2) * w)))
    y2 = min(h, int(round((cy + bh / 2) * h)))
    bw_px, bh_px = x2 - x1, y2 - y1
    if bw_px < 5 or bh_px < 5:
        return cx, cy, bw, bh, False

    pad_x = max(6, int(round(0.28 * bw_px)))
    pad_y = max(6, int(round(0.28 * bh_px)))
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = gray[ry1:ry2, rx1:rx2]

    bmask = np.zeros(roi.shape, dtype=bool)
    bmask[:3, :] = True
    bmask[-3:, :] = True
    bmask[:, :3] = True
    bmask[:, -3:] = True
    bg_med = float(np.median(roi[bmask])) if bmask.any() else float(np.median(roi))

    diff = np.abs(roi - bg_med)
    if diff.shape[0] >= 5 and diff.shape[1] >= 5:
        pdiff = np.pad(diff, 1, mode="edge")
        diff = (
            pdiff[:-2, :-2] + pdiff[:-2, 1:-1] + pdiff[:-2, 2:] +
            pdiff[1:-1, :-2] + pdiff[1:-1, 1:-1] + pdiff[1:-1, 2:] +
            pdiff[2:, :-2] + pdiff[2:, 1:-1] + pdiff[2:, 2:]
        ) / 9.0

    sx1 = max(0, (x1 - rx1) - max(2, int(0.15 * bw_px)))
    sy1 = max(0, (y1 - ry1) - max(2, int(0.15 * bh_px)))
    sx2 = min(roi.shape[1], (x2 - rx1) + max(2, int(0.15 * bw_px)))
    sy2 = min(roi.shape[0], (y2 - ry1) + max(2, int(0.15 * bh_px)))
    sub = diff[sy1:sy2, sx1:sx2]
    if sub.size < 9:
        return cx, cy, bw, bh, False
    peak = float(np.percentile(sub, 97))
    if peak < 6.0:
        return cx, cy, bw, bh, False

    ys, xs = np.nonzero(sub >= max(6.0, 0.36 * peak))
    if xs.size < 4:
        return cx, cy, bw, bh, False

    tx1_s, tx2_s = int(xs.min()), int(xs.max()) + 1
    ty1_s, ty2_s = int(ys.min()), int(ys.max()) + 1
    mx = max(3, int(round(0.14 * (tx2_s - tx1_s))))
    my = max(3, int(round(0.14 * (ty2_s - ty1_s))))

    nx1 = max(0, rx1 + sx1 + tx1_s - mx)
    ny1 = max(0, ry1 + sy1 + ty1_s - my)
    nx2 = min(w, rx1 + sx1 + tx2_s + mx)
    ny2 = min(h, ry1 + sy1 + ty2_s + my)

    ncx = ((nx1 + nx2) / 2.0) / w
    ncy = ((ny1 + ny2) / 2.0) / h
    nbw = min(max(0.003, (nx2 - nx1) / w), ncx * 2.0, (1.0 - ncx) * 2.0)
    nbh = min(max(0.003, (ny2 - ny1) / h), ncy * 2.0, (1.0 - ncy) * 2.0)
    ar = max(nbw / max(nbh, 1e-6), nbh / max(nbw, 1e-6))
    if ar <= 4.2 and box_iou((cx, cy, bw, bh), (ncx, ncy, nbw, nbh)) >= 0.22:
        return ncx, ncy, nbw, nbh, True
    return cx, cy, bw, bh, False


def validate_rgb_pil_and_boxes(im_rgb: Image.Image, raw_boxes: list, require_daylight_color: bool = False):
    """
    Validates RGB PIL Image + YOLO boxes [(cx, cy, bw, bh)].
    Runs 5-layer geometry + pixel contrast check + auto tight-fit alignment.
    Returns (ok, final_lines, was_fixed, reason).
    """
    if not raw_boxes:
        return False, [], False, "empty_boxes"
    w, h = im_rgb.size
    if w < 32 or h < 32:
        return False, [], False, "tiny_img"

    rgb_arr = np.asarray(im_rgb, dtype=np.float32)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
        return False, [], False, "not_rgb"

    gray = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]

    if has_hud_distractor_watermark(gray):
        return False, [], False, "hud_airplane_watermark"

    img_mean = float(gray.mean())
    if require_daylight_color:
        # Reject pitch-black night or pure grayscale frames when filtering RGB video streams
        color_diff = float(np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 2]).mean())
        if img_mean < 28.0 or color_diff < 2.2:
            return False, [], False, "too_dark_or_gray"

    is_dark = img_mean < 52.0
    cleaned = []
    was_fixed = False
    for cx, cy, bw, bh in raw_boxes:
        orig = (cx, cy, bw, bh)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        if cx - bw / 2 < 0:
            bw = cx * 2
        if cx + bw / 2 > 1:
            bw = (1.0 - cx) * 2
        if cy - bh / 2 < 0:
            bh = cy * 2
        if cy + bh / 2 > 1:
            bh = (1.0 - cy) * 2
        if abs(cx - orig[0]) > 1e-4 or abs(cy - orig[1]) > 1e-4 or abs(bw - orig[2]) > 1e-4 or abs(bh - orig[3]) > 1e-4:
            was_fixed = True
        if is_geometry_clean(w, h, cx, cy, bw, bh):
            cleaned.append((cx, cy, bw, bh))
        else:
            was_fixed = True

    if not cleaned:
        return False, [], False, "geometry_rejected"

    # IoU dedup
    if len(cleaned) > 1:
        keep = [True] * len(cleaned)
        for i in range(len(cleaned)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(cleaned)):
                if not keep[j]:
                    continue
                if box_iou(cleaned[i], cleaned[j]) > 0.65:
                    if cleaned[j][2] * cleaned[j][3] > cleaned[i][2] * cleaned[i][3]:
                        keep[i] = False
                    else:
                        keep[j] = False
                    was_fixed = True
        cleaned = [cleaned[i] for i in range(len(cleaned)) if keep[i]]

    verified = []
    for cx, cy, bw, bh in cleaned:
        def check_px(bcx, bcy, bbw, bbh):
            x1 = max(0, int(round((bcx - bbw / 2) * w)))
            y1 = max(0, int(round((bcy - bbh / 2) * h)))
            x2 = min(w, int(round((bcx + bbw / 2) * w)))
            y2 = min(h, int(round((bcy + bbh / 2) * h)))
            if x2 - x1 < 3 or y2 - y1 < 3:
                return "sub3px"
            crop = gray[y1:y2, x1:x2]
            c_std = float(crop.std())
            pad = max(6, int(0.20 * max(x2 - x1, y2 - y1)))
            rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
            rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
            roi = gray[ry1:ry2, rx1:rx2]
            rmask = np.ones(roi.shape, dtype=bool)
            rmask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
            rvals = roi[rmask]
            bg = float(np.median(rvals)) if rvals.size > 0 else float(np.median(crop))
            diff_c = np.abs(crop - bg)
            fg_peak = float(np.percentile(diff_c, 96))
            if c_std < 3.8 and fg_peak < 6.5:
                return "ghost_blank"
            is_small = (bbw * bbh) < 0.005
            gx = np.abs(np.diff(crop, axis=1)).mean() if crop.shape[1] > 1 else 0.0
            gy = np.abs(np.diff(crop, axis=0)).mean() if crop.shape[0] > 1 else 0.0
            if is_small or is_dark or (gx + gy) < 5.5:
                if fg_peak < 5.5 and c_std < 4.5:
                    return "dark_blur_ghost"
                sal = diff_c >= max(5.5, 0.35 * fg_peak)
                if int(sal.sum()) >= 4:
                    ys, xs = np.nonzero(sal)
                    ch_p, cw_p = crop.shape
                    if abs(float(xs.mean()) / max(cw_p, 1) - 0.5) > 0.28 or abs(float(ys.mean()) / max(ch_p, 1) - 0.5) > 0.28:
                        return "off_center"
                    if cw_p > 18 and ch_p > 18 and (xs.max() - xs.min() + 1) / cw_p < 0.18 and (ys.max() - ys.min() + 1) / ch_p < 0.18:
                        return "bloated"
            return "ok"

        st = check_px(cx, cy, bw, bh)
        if st == "ok":
            verified.append((cx, cy, bw, bh))
        elif st in ("off_center", "bloated"):
            tcx, tcy, tbw, tbh, snapped = tighten_box_to_drone(gray, cx, cy, bw, bh)
            if snapped and is_geometry_clean(w, h, tcx, tcy, tbw, tbh) and check_px(tcx, tcy, tbw, tbh) == "ok":
                verified.append((tcx, tcy, tbw, tbh))
                was_fixed = True
            else:
                was_fixed = True
        else:
            was_fixed = True

    if not verified:
        return False, [], False, "pixel_rejected"
    return True, [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in verified], was_fixed, "ok"


def pil_to_jpeg_bytes(im_rgb: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    im_rgb.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def download_file(url: str, dest: Path, label: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    existing = dest.stat().st_size if dest.exists() else 0
    if total > 0 and existing == total:
        return
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
    mode = "ab" if existing > 0 else "wb"
    t0 = time.time()
    last_p = t0
    got = existing
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=90) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                if total > 500_000_000 and now - last_p >= 6.0:
                    spd = (got - existing) / max(now - t0, 0.1) / 1e6
                    pct = got / max(total, 1) * 100
                    print(f"    DL {label or dest.name}: {pct:5.1f}% ({got/1e9:.2f}/{total/1e9:.2f}GB) | {spd:.1f} MB/s", flush=True)
                    last_p = now


def phase0_rebalance_ir():
    """Prunes existing ~236K Thermal IR in data2 down to best 50,000 (25K 2025 + 25K 410), uniform stride."""
    print("=" * 68)
    print("PHASE 0: Rebalancing existing Thermal IR in data2 -> Best 50,000 IR images")
    print("=" * 68, flush=True)

    by_src = {"antiuav_2025": [], "anti_uav410": []}
    for split in ["train", "val"]:
        img_dir = DATA2_DIR / "images" / split
        if not img_dir.exists():
            continue
        for e in sorted(os.scandir(img_dir), key=lambda x: x.name):
            if not e.is_file():
                continue
            if e.name.startswith("d2_antiuav_2025_"):
                by_src["antiuav_2025"].append((split, e.name))
            elif e.name.startswith("d2_anti_uav410_"):
                by_src["anti_uav410"].append((split, e.name))

    kept_total = 0
    removed_total = 0
    for key, target_n in [("antiuav_2025", KEEP_IR_2025), ("anti_uav410", KEEP_IR_410)]:
        items = by_src[key]
        if len(items) <= target_n:
            kept_total += len(items)
            print(f"  {key}: already {len(items)} <= {target_n}, keeping all.", flush=True)
            continue
        # Uniform stride sampling across sorted list to maximize sequence diversity!
        keep_indices = set(int(round(i * (len(items) - 1) / (target_n - 1))) for i in range(target_n))
        to_remove = [items[i] for i in range(len(items)) if i not in keep_indices]
        print(f"  {key}: keeping {len(keep_indices)} / {len(items)} diverse IR frames (pruning {len(to_remove)})...", flush=True)

        def rm_pair(item):
            sp, fn = item
            stem = os.path.splitext(fn)[0]
            try:
                os.remove(DATA2_DIR / "images" / sp / fn)
            except OSError:
                pass
            try:
                os.remove(DATA2_DIR / "labels" / sp / f"{stem}.txt")
            except OSError:
                pass

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(rm_pair, to_remove))

        kept_total += len(keep_indices)
        removed_total += len(to_remove)

    print(f"  ✅ Thermal IR rebalanced: {kept_total} kept, {removed_total} pruned | Free E:: {free_gb():.2f} GB\n", flush=True)
    return kept_total


def index_existing_hashes():
    print("PHASE 1: Indexing MD5 hashes of Batch 1 (clean_dataset) + kept IR in data2...", flush=True)
    paths = []
    for base in [BATCH1_DIR, DATA2_DIR]:
        for split in ["train", "val"]:
            d = base / "images" / split
            if d.exists():
                for e in os.scandir(d):
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                        paths.append(e.path)
    hashes = set()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, h in enumerate(pool.map(file_md5_path, paths), 1):
            if h:
                hashes.add(h)
            if i % 30000 == 0 or i == len(paths):
                print(f"  Indexed {i}/{len(paths)} existing hashes ({time.time()-t0:.1f}s)", flush=True)
    return hashes


def ingest_hf_parquet_rgb(repo: str, tag: str, drone_cat_id: int, existing_hashes: set,
                          next_id: int, max_keep: int, stats: Counter, audit_writer):
    print(f"\n{'=' * 68}")
    print(f"🎨 RGB PARQUET SOURCE: {repo} (max_keep={max_keep}, free={free_gb():.2f} GB)")
    print(f"{'=' * 68}", flush=True)

    r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main/data", timeout=20)
    if not r.ok:
        print(f"  ⚠️ Failed to list shards for {repo}: HTTP {r.status_code}", flush=True)
        return next_id, 0
    shards = sorted([x["path"] for x in r.json() if x["path"].endswith(".parquet")])
    print(f"  Found {len(shards)} parquet shards.", flush=True)

    rng = random.Random(hash(repo) & 0xFFFFFFFF)
    kept = 0
    for s_idx, shard in enumerate(shards, 1):
        if kept >= max_keep or free_gb() < FREE_DISK_FLOOR_GB:
            break
        local_p = TMP_DL / f"{tag}_{os.path.basename(shard)}"
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard}"
        try:
            download_file(url, local_p, label=f"{tag} shard {s_idx}/{len(shards)}")
            table = pq.read_table(local_p)
            cols = set(table.column_names)
            n_rows = table.num_rows
            img_col = table["image"].to_pylist()
            obj_col = table["objects"].to_pylist()
            w_col = table["width"].to_pylist() if "width" in cols else [None] * n_rows
            h_col = table["height"].to_pylist() if "height" in cols else [None] * n_rows

            for i in range(n_rows):
                if kept >= max_keep:
                    break
                img_obj = img_col[i]
                raw_bytes = img_obj.get("bytes") if isinstance(img_obj, dict) else None
                if not raw_bytes:
                    continue

                raw_h = bytes_md5(raw_bytes)
                if raw_h in existing_hashes:
                    stats["dup_skipped"] += 1
                    continue

                objs = obj_col[i]
                if not isinstance(objs, dict):
                    continue
                bboxes = objs.get("bbox", []) or []
                cats = objs.get("category", []) or []
                if not bboxes:
                    stats["rejected_empty"] += 1
                    continue

                try:
                    im_rgb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                except Exception:
                    stats["corrupt"] += 1
                    continue

                iw, ih = im_rgb.size
                # Convert COCO [x, y, w, h] pixel bbox -> normalized YOLO [cx, cy, bw, bh]
                # Filter strictly to drone_cat_id (crucial for drone-vs-bird where bird=1, airplane=2, heli=3!)
                yolo_boxes = []
                for idx_b, bb in enumerate(bboxes):
                    if cats and idx_b < len(cats) and cats[idx_b] != drone_cat_id:
                        continue
                    if not bb or len(bb) < 4:
                        continue
                    bx, by, bw_p, bh_p = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                    if max(bx, by, bw_p, bh_p) <= 1.05:
                        # Already normalized
                        yolo_boxes.append((bx, by, bw_p, bh_p))
                    else:
                        # COCO pixel [xmin, ymin, width, height]
                        yolo_boxes.append(((bx + bw_p / 2.0) / iw, (by + bh_p / 2.0) / ih, bw_p / iw, bh_p / ih))

                ok, lines, fixed, reason = validate_rgb_pil_and_boxes(im_rgb, yolo_boxes, require_daylight_color=False)
                if not ok:
                    stats[f"rejected_{reason}"] += 1
                    continue

                # Store crisp JPEG (if already JPEG and small keep bytes; otherwise encode Q=92)
                if raw_bytes[:2] == b"\xff\xd8" and len(raw_bytes) <= 350_000:
                    out_bytes = raw_bytes
                else:
                    out_bytes = pil_to_jpeg_bytes(im_rgb, quality=92)
                out_h = bytes_md5(out_bytes)
                if out_h in existing_hashes:
                    stats["dup_skipped"] += 1
                    continue
                existing_hashes.add(raw_h)
                existing_hashes.add(out_h)

                split = "train" if rng.random() < TRAIN_RATIO else "val"
                stem = f"d2_rgb_{tag}_{next_id:06d}"
                next_id += 1
                (DATA2_DIR / "images" / split / f"{stem}.jpg").write_bytes(out_bytes)
                (DATA2_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

                kept += 1
                stats["rgb_kept_total"] += 1
                stats[f"rgb_{tag}"] += 1
                stats["rgb_boxes"] += len(lines)
                if fixed:
                    stats["rgb_fixed"] += 1
                    audit_writer.writerow({
                        "image": f"{stem}.jpg", "split": split, "source": f"rgb_{tag}",
                        "action": "rgb_tight_fixed", "final_boxes": len(lines), "timestamp": datetime.now().isoformat()
                    })

            print(
                f"  [{tag}] shard {s_idx}/{len(shards)} done | kept={kept} | "
                f"total_rgb={stats['rgb_kept_total']} | free={free_gb():.1f}GB",
                flush=True,
            )
        except Exception as e:
            print(f"  ⚠️ Error on shard {shard}: {e}", flush=True)
        finally:
            if local_p.exists():
                local_p.unlink()
    return next_id, kept


def ingest_detfly_zip(existing_hashes: set, next_id: int, max_keep: int, stats: Counter, audit_writer):
    tag = "detfly"
    url = "https://huggingface.co/datasets/DestinyLove/Det-Fly/resolve/main/Det-Fly/Det-Fly.zip"
    zip_p = TMP_DL / "Det-Fly.zip"
    print(f"\n{'=' * 68}")
    print(f"🎨 RGB AIR-TO-AIR SOURCE: DestinyLove/Det-Fly (2.14 GB, free={free_gb():.2f} GB)")
    print(f"{'=' * 68}", flush=True)
    kept = 0
    rng = random.Random(777)
    try:
        download_file(url, zip_p, label="Det-Fly.zip")
        with zipfile.ZipFile(zip_p, "r") as zf:
            names = zf.namelist()
            xml_by_stem = {}
            img_entries = []
            for n in names:
                low = n.lower()
                if low.endswith(".xml"):
                    xml_by_stem[Path(n).stem] = n
                elif os.path.splitext(low)[1] in IMAGE_EXTS:
                    img_entries.append(n)
            print(f"  Det-Fly ZIP: {len(img_entries)} images, {len(xml_by_stem)} XML labels.", flush=True)
            rng.shuffle(img_entries)

            for idx, img_e in enumerate(img_entries, 1):
                if kept >= max_keep or free_gb() < FREE_DISK_FLOOR_GB:
                    break
                xml_e = xml_by_stem.get(Path(img_e).stem)
                if not xml_e:
                    continue
                try:
                    raw_bytes = zf.read(img_e)
                    raw_h = bytes_md5(raw_bytes)
                    if raw_h in existing_hashes:
                        stats["dup_skipped"] += 1
                        continue
                    im_rgb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    iw, ih = im_rgb.size
                    root = ET.fromstring(zf.read(xml_e))
                    yolo_boxes = []
                    for obj in root.findall("object"):
                        bb = obj.find("bndbox")
                        if bb is None:
                            continue
                        x1 = float(bb.find("xmin").text)
                        y1 = float(bb.find("ymin").text)
                        x2 = float(bb.find("xmax").text)
                        y2 = float(bb.find("ymax").text)
                        yolo_boxes.append((((x1 + x2) / 2.0) / iw, ((y1 + y2) / 2.0) / ih, (x2 - x1) / iw, (y2 - y1) / ih))
                except Exception:
                    stats["corrupt"] += 1
                    continue

                ok, lines, fixed, reason = validate_rgb_pil_and_boxes(im_rgb, yolo_boxes)
                if not ok:
                    stats[f"rejected_{reason}"] += 1
                    continue

                out_bytes = raw_bytes if (raw_bytes[:2] == b"\xff\xd8" and len(raw_bytes) <= 400_000) else pil_to_jpeg_bytes(im_rgb, 92)
                out_h = bytes_md5(out_bytes)
                if out_h in existing_hashes:
                    continue
                existing_hashes.add(raw_h)
                existing_hashes.add(out_h)

                split = "train" if rng.random() < TRAIN_RATIO else "val"
                stem = f"d2_rgb_{tag}_{next_id:06d}"
                next_id += 1
                (DATA2_DIR / "images" / split / f"{stem}.jpg").write_bytes(out_bytes)
                (DATA2_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

                kept += 1
                stats["rgb_kept_total"] += 1
                stats[f"rgb_{tag}"] += 1
                stats["rgb_boxes"] += len(lines)
                if fixed:
                    stats["rgb_fixed"] += 1
                    audit_writer.writerow({
                        "image": f"{stem}.jpg", "split": split, "source": f"rgb_{tag}",
                        "action": "rgb_tight_fixed", "final_boxes": len(lines), "timestamp": datetime.now().isoformat()
                    })
                if idx % 2000 == 0:
                    print(f"  [detfly] scanned={idx}/{len(img_entries)} | kept={kept} | free={free_gb():.1f}GB", flush=True)
    except Exception as e:
        print(f"  ⚠️ Det-Fly error: {e}", flush=True)
    finally:
        if zip_p.exists():
            zip_p.unlink()
    print(f"  [detfly] finished: kept={kept}", flush=True)
    return next_id, kept


def ingest_antiuav300_visible_rgb(existing_hashes: set, next_id: int, max_keep: int, stats: Counter, audit_writer):
    """Extracts ONLY colorful daytime/dusk visible.mp4 RGB streams from anti-uav-rgbt-300.zip!"""
    tag = "antiuav300_rgb"
    url = "https://huggingface.co/datasets/huseyincavus/anti-uav-rgbt-300/resolve/main/anti-uav-rgbt-300.zip"
    zip_p = TMP_DL / "anti-uav-rgbt-300.zip"
    print(f"\n{'=' * 68}")
    print(f"🎨 RGB VISIBLE VIDEO SOURCE: anti-uav-rgbt-300.zip (ONLY visible.mp4 color, max_keep={max_keep})")
    print(f"{'=' * 68}", flush=True)
    kept = 0
    rng = random.Random(300)
    try:
        download_file(url, zip_p, label="anti-uav-rgbt-300.zip")
        with zipfile.ZipFile(zip_p, "r") as zf:
            names = set(zf.namelist())
            vis_videos = sorted([n for n in names if n.endswith("visible.mp4")])
            print(f"  Found {len(vis_videos)} visible RGB videos (skipping all infrared.mp4).", flush=True)
            target_per_vid = max(120, int(max_keep / max(len(vis_videos), 1) * 1.4))

            for v_idx, v_entry in enumerate(vis_videos, 1):
                if kept >= max_keep or free_gb() < FREE_DISK_FLOOR_GB:
                    break
                j_entry = v_entry[:-4] + ".json"
                if j_entry not in names:
                    continue
                try:
                    meta = json.loads(zf.read(j_entry).decode("utf-8", errors="ignore"))
                    exists = meta.get("exist", [])
                    rects = meta.get("gt_rect", [])
                except Exception:
                    continue
                n_ann = min(len(exists), len(rects))
                if n_ann < 10:
                    continue

                valid_frames = [i for i in range(n_ann) if exists[i] == 1 and isinstance(rects[i], list) and len(rects[i]) >= 4 and rects[i][2] > 2 and rects[i][3] > 2]
                if not valid_frames:
                    continue
                stride = max(1, len(valid_frames) // target_per_vid)
                wanted = set(valid_frames[::stride])

                tmp_mp4 = TMP_DL / "cur_vis.mp4"
                tmp_mp4.write_bytes(zf.read(v_entry))
                cap = cv2.VideoCapture(str(tmp_mp4))
                f_i = 0
                vid_kept = 0
                try:
                    while cap.isOpened() and kept < max_keep:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        if f_i in wanted:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            im_rgb = Image.fromarray(frame_rgb)
                            iw, ih = im_rgb.size
                            rx, ry, rw, rh = [float(x) for x in rects[f_i][:4]]
                            cx, cy, bw, bh = (rx + rw / 2.0) / iw, (ry + rh / 2.0) / ih, rw / iw, rh / ih

                            # Strictly require colorful daylight/dusk RGB (rejects pitch-black night & airplane watermark)
                            ok, lines, fixed, reason = validate_rgb_pil_and_boxes(
                                im_rgb, [(cx, cy, bw, bh)], require_daylight_color=True
                            )
                            if ok:
                                out_bytes = pil_to_jpeg_bytes(im_rgb, quality=90)
                                out_h = bytes_md5(out_bytes)
                                if out_h not in existing_hashes:
                                    existing_hashes.add(out_h)
                                    split = "train" if rng.random() < TRAIN_RATIO else "val"
                                    stem = f"d2_rgb_{tag}_{next_id:06d}"
                                    next_id += 1
                                    (DATA2_DIR / "images" / split / f"{stem}.jpg").write_bytes(out_bytes)
                                    (DATA2_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                                    kept += 1
                                    vid_kept += 1
                                    stats["rgb_kept_total"] += 1
                                    stats[f"rgb_{tag}"] += 1
                                    stats["rgb_boxes"] += len(lines)
                                    if fixed:
                                        stats["rgb_fixed"] += 1
                                        audit_writer.writerow({
                                            "image": f"{stem}.jpg", "split": split, "source": f"rgb_{tag}",
                                            "action": "rgb_tight_fixed", "final_boxes": len(lines), "timestamp": datetime.now().isoformat()
                                        })
                                else:
                                    stats["dup_skipped"] += 1
                            else:
                                stats[f"rejected_{reason}"] += 1
                        f_i += 1
                finally:
                    cap.release()
                    if tmp_mp4.exists():
                        tmp_mp4.unlink()

                if v_idx % 20 == 0 or v_idx == len(vis_videos):
                    print(
                        f"  [{tag}] video {v_idx}/{len(vis_videos)} | kept={kept} | "
                        f"total_rgb={stats['rgb_kept_total']} | free={free_gb():.1f}GB",
                        flush=True,
                    )
    except Exception as e:
        print(f"  ⚠️ AntiUAV300 RGB error: {e}", flush=True)
    finally:
        if zip_p.exists():
            zip_p.unlink()
    return next_id, kept


def main():
    t0 = time.time()
    print("=" * 68)
    print("🌈 REBALANCE & ENRICH DATA2: 50,000 IR + 165,000+ COLORFUL RGB (>215K)")
    print(f"Started: {datetime.now().isoformat()} | Free E:: {free_gb():.2f} GB")
    print("=" * 68, flush=True)

    TMP_DL.mkdir(parents=True, exist_ok=True)

    # Step 0: Prune excess IR frames down to best 50,000 diverse IR
    ir_kept = phase0_rebalance_ir()

    # Step 1: Index Batch 1 + kept IR hashes for 0% duplication
    existing_hashes = index_existing_hashes()

    stats = Counter()
    stats["ir_kept_total"] = ir_kept
    next_id = 0

    audit_exists = AUDIT_LOG.exists()
    audit_f = open(AUDIT_LOG, "a", newline="", encoding="utf-8")
    audit_writer = csv.DictWriter(
        audit_f, fieldnames=["image", "split", "source", "action", "final_boxes", "timestamp"]
    )
    if not audit_exists:
        audit_writer.writeheader()

    try:
        # RGB 1: pathikg/drone-detection-dataset (~54K real-world outdoor color RGB)
        next_id, _ = ingest_hf_parquet_rgb(
            "pathikg/drone-detection-dataset", "pathikg", 0, existing_hashes, next_id, 55000, stats, audit_writer
        )

        # RGB 2: athkot/drone-vs-bird-synthetic-10k (~10K color Drone-vs-Bird, drone=0 only!)
        next_id, _ = ingest_hf_parquet_rgb(
            "athkot/drone-vs-bird-synthetic-10k", "dvb10k", 0, existing_hashes, next_id, 12000, stats, audit_writer
        )

        # RGB 3: DestinyLove/Det-Fly (Air-to-Air color outdoor RGB)
        next_id, _ = ingest_detfly_zip(
            existing_hashes, next_id, 15000, stats, audit_writer
        )

        # RGB 4: guaguarobotics/mav-vid (~40K Full-HD multi-view color RGB)
        next_id, _ = ingest_hf_parquet_rgb(
            "guaguarobotics/mav-vid", "mavvid", 0, existing_hashes, next_id, 42000, stats, audit_writer
        )

        # RGB 5: anti-uav-rgbt-300 visible.mp4 ONLY (daytime/dusk color RGB to hit >=165K RGB!)
        rem_rgb = max(0, TARGET_RGB_TOTAL - stats["rgb_kept_total"])
        if rem_rgb > 0 and free_gb() > 10.0:
            next_id, _ = ingest_antiuav300_visible_rgb(
                existing_hashes, next_id, rem_rgb + 10000, stats, audit_writer
            )

    finally:
        audit_f.close()
        shutil.rmtree(TMP_DL, ignore_errors=True)

    # Final verification count
    final_train = sum(1 for e in os.scandir(DATA2_DIR / "images" / "train") if e.is_file())
    final_val = sum(1 for e in os.scandir(DATA2_DIR / "images" / "val") if e.is_file())
    lbl_train = sum(1 for e in os.scandir(DATA2_DIR / "labels" / "train") if e.is_file())
    lbl_val = sum(1 for e in os.scandir(DATA2_DIR / "labels" / "val") if e.is_file())
    total_final = final_train + final_val
    rgb_pct = (stats["rgb_kept_total"] / max(total_final, 1)) * 100.0
    ir_pct = (ir_kept / max(total_final, 1)) * 100.0

    report = {
        "completed_at": datetime.now().isoformat(),
        "output_dir": str(DATA2_DIR),
        "final_train": final_train,
        "final_val": final_val,
        "final_total": total_final,
        "labels_paired_100pct": (final_train == lbl_train and final_val == lbl_val),
        "composition": {
            "colorful_rgb_images": stats["rgb_kept_total"],
            "colorful_rgb_percent": round(rgb_pct, 2),
            "thermal_ir_images": ir_kept,
            "thermal_ir_percent": round(ir_pct, 2),
        },
        "rgb_breakdown": {k: v for k, v in stats.items() if k.startswith("rgb_")},
        "stats": dict(stats),
        "free_disk_e_gb_remaining": round(free_gb(), 2),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print("✅ DATA2 BALANCED DIVERSE DATASET COMPLETE!")
    print("=" * 68)
    print(f"Final data2 total       : {total_final} images ({final_train} train / {final_val} val)")
    print(f"  🎨 Colorful RGB       : {stats['rgb_kept_total']} ({rgb_pct:.1f}%)")
    print(f"  🌡️ Thermal IR         : {ir_kept} ({ir_pct:.1f}%)")
    print(f"Labels paired 100%      : {final_train == lbl_train and final_val == lbl_val}")
    print(f"RGB Tight-fit fixed     : {stats['rgb_fixed']}")
    print(f"Duplicates skipped      : {stats['dup_skipped']}")
    print(f"Remaining free on E:    : {free_gb():.2f} GB")
    print(f"Total time              : {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
