"""
Iterative Closed-Loop Precision Label Audit & Auto-Refinement Engine
Target: >= 99.80% Verified Clean Label Rate across clean_dataset

Focuses heavily on small drones, blurry drones, and low-light / dark frames:
- Iteration Step A (Strict 5-Rule Pixel + Geometry Audit across ALL images):
  1. Empty / syntax / out-of-bounds / degenerate boxes
  2. Anti-UAV HUD crosshair poison (1920x1080), bottom-left distractor watermark ('民航飞机'),
     and IR building poison (640x512)
  3. Duplicate overlapping boxes (IoU > 0.65) or extreme aspect ratio (> 4.5)
  4. Ghost / blank boxes on empty sky or flat dark noise (foreground-vs-ring contrast check)
  5. Small / blurry / low-light box alignment check (off-center centroid shift, bloated loose
     margins, or clipped drone body)
- Iteration Step B (Targeted Re-Drawing & Sub-Pixel Tight-Fit Alignment):
  * Runs Clean v2 YOLOv8n (mAP50=0.984) at high-res imgsz=960 on BOTH original AND
    Adaptive Gamma + Unsharp-Mask + Autocontrast enhanced images (for dark/blurry drones)
  * Applies Saliency Tight-Fit Box Aligner to snap loose/shifted boxes tightly to the drone
    silhouette on small/blurry/low-light frames
  * Backs up original labels to labels_backup/ and logs all fixes to relabel_audit_log.csv
  * Moves unrecoverable 0-drone / distractor frames cleanly to e:\trend_AI\quarantine_non_drone\
- Iteration Step C: Repeats Audit -> Re-Label until clean pass rate >= 99.80%.
"""

import os
import sys
import csv
import json
import time
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps, ImageFilter

warnings.filterwarnings("ignore", category=UserWarning)
Image.MAX_IMAGE_PIXELS = None

WORKSPACE = Path(r"e:\trend_AI")
DATASET_DIR = WORKSPACE / "clean_dataset"
BACKUP_DIR = DATASET_DIR / "labels_backup"
QUARANTINE_DIR = WORKSPACE / "quarantine_non_drone"
AUDIT_LOG = DATASET_DIR / "relabel_audit_log.csv"
QA_STATUS_FILE = DATASET_DIR / "qa_review_status.json"
SUSPICIOUS_CSV = DATASET_DIR / "suspicious_labels_audit.csv"
REPORT_FILE = DATASET_DIR / "consolidation_report.json"
LOOP_REPORT_JSON = DATASET_DIR / "precision_998_report.json"
CLEAN_MODEL_PT = WORKSPACE / "clean_relabel_train" / "clean_drone_v2" / "weights" / "best.pt"

TARGET_PASS_RATE = 99.80
MAX_ITERATIONS = 4
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_protected():
    protected = set()
    if QA_STATUS_FILE.exists():
        try:
            qa = json.loads(QA_STATUS_FILE.read_text(encoding="utf-8"))
            for k, v in qa.items():
                if v == "approved":
                    protected.add(k)
        except Exception:
            pass
    if AUDIT_LOG.exists():
        try:
            with open(AUDIT_LOG, "r", encoding="utf-8", errors="ignore") as f:
                for r in csv.DictReader(f):
                    if r.get("action") == "manual_fixed":
                        protected.add(r["image"])
        except Exception:
            pass
    return protected


def box_iou(b1, b2):
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


def nms_boxes(cands, iou_thr=0.50):
    """cands: list of (conf, cx, cy, bw, bh) -> returns filtered list."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda x: x[0], reverse=True)
    kept = []
    for c in cands:
        if all(box_iou(c[1:], k[1:]) <= iou_thr for k in kept):
            kept.append(c)
    return kept


def has_hud_distractor_watermark(gray_arr):
    """Detects '民航飞机' / distractor text in bottom-left of dark 1920x1080 HUD frames."""
    h, w = gray_arr.shape
    if (w, h) != (1920, 1080):
        return False
    roi = gray_arr[1008:1066, 12:175]
    bg_med = float(np.median(gray_arr[960:1005, 12:200]))
    if bg_med > 35.0:
        return False
    white_ratio = float((roi > 220).mean())
    return 0.04 <= white_ratio <= 0.35


def audit_single_item(args):
    """
    Audits one (split, img_path_str, lbl_path_str, is_protected) item.
    Returns (split, name, passed_bool, reasons_list, img_meta).
    """
    split, img_path, lbl_path, is_protected = args
    name = os.path.basename(img_path)
    if is_protected:
        return (split, name, True, [], None)

    if not os.path.exists(lbl_path):
        return (split, name, False, ["missing_label"], None)

    try:
        raw_txt = Path(lbl_path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return (split, name, False, ["unreadable_label"], None)

    if not raw_txt:
        return (split, name, False, ["empty_0_box"], None)

    boxes = []
    reasons = []
    for line in raw_txt.splitlines():
        p = line.split()
        if len(p) < 5:
            reasons.append("bad_syntax")
            continue
        try:
            cls = int(float(p[0]))
            cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            reasons.append("nan_coord")
            continue
        if cls != 0:
            reasons.append("wrong_class")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            reasons.append("out_of_bounds")
        if cx - bw / 2 < -0.002 or cx + bw / 2 > 1.002 or cy - bh / 2 < -0.002 or cy + bh / 2 > 1.002:
            reasons.append("edge_overflow")
        boxes.append((cx, cy, bw, bh))

    if not boxes:
        return (split, name, False, reasons or ["empty_0_box"], None)

    # Duplicate IoU check
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) > 0.65:
                reasons.append("duplicate_overlap_iou")
                break

    # Open image & inspect geometry + pixels
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            gray_im = im.convert("L")
            gray = np.asarray(gray_im, dtype=np.float32)
    except Exception:
        return (split, name, False, ["corrupt_image"], None)

    if has_hud_distractor_watermark(gray):
        return (split, name, False, ["hud_distractor_airplane"], (w, h))

    img_mean = float(gray.mean())
    is_dark_frame = img_mean < 52.0

    for idx, (cx, cy, bw, bh) in enumerate(boxes):
        area = bw * bh
        ar = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))

        if area < 0.00010 or bw < 0.0025 or bh < 0.0025:
            reasons.append("degenerate_tiny_box")
            continue
        if area > 0.65:
            reasons.append("oversized_box")
            continue
        if ar > 4.5:
            reasons.append("extreme_aspect_ratio")
            continue

        # Anti-UAV HUD & IR poison checks
        if (w, h) == (1920, 1080) and area > 0.08:
            reasons.append("hud_crosshair_poison")
            continue
        if (w, h) == (640, 512):
            if area > 0.06:
                reasons.append("ir_building_poison")
                continue
            y2_n = cy + bh / 2
            x1_n = cx - bw / 2
            x2_n = cx + bw / 2
            if (y2_n > 0.93 or x1_n < 0.02 or x2_n > 0.98) and area > 0.015:
                reasons.append("ir_edge_building_poison")
                continue

        # Pixel-level foreground & alignment check (critical for small / blurry / dark drones)
        x1 = max(0, int(round((cx - bw / 2) * w)))
        y1 = max(0, int(round((cy - bh / 2) * h)))
        x2 = min(w, int(round((cx + bw / 2) * w)))
        y2 = min(h, int(round((cy + bh / 2) * h)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            reasons.append("sub3px_box")
            continue

        crop = gray[y1:y2, x1:x2]
        crop_std = float(crop.std())

        # Outer context ring (6px around box)
        pad = max(6, int(0.20 * max(x2 - x1, y2 - y1)))
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
        roi = gray[ry1:ry2, rx1:rx2]

        # Build ring mask
        ring_mask = np.ones(roi.shape, dtype=bool)
        ring_mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
        ring_vals = roi[ring_mask]
        bg_med = float(np.median(ring_vals)) if ring_vals.size > 0 else float(np.median(crop))

        diff_crop = np.abs(crop - bg_med)
        fg_peak = float(np.percentile(diff_crop, 96))

        # Rule 4: Ghost / blank box on flat sky or dark noise
        if crop_std < 3.8 and fg_peak < 6.5:
            reasons.append("ghost_blank_box")
            continue

        # Rule 5: Small / dark / blurry alignment check
        is_small = area < 0.005
        # Estimate blur via simple gradient energy inside crop
        gx = np.abs(np.diff(crop, axis=1)).mean() if crop.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(crop, axis=0)).mean() if crop.shape[0] > 1 else 0.0
        is_blurry = (gx + gy) < 5.5

        if is_small or is_dark_frame or is_blurry:
            if fg_peak < 5.5 and crop_std < 4.5:
                reasons.append("dark_or_blurry_ghost_box")
                continue

            # Check centroid & tightness of salient object inside crop
            thr = max(5.5, 0.35 * fg_peak)
            sal = diff_crop >= thr
            sal_count = int(sal.sum())
            if sal_count >= 4:
                ys, xs = np.nonzero(sal)
                ch_px, cw_px = crop.shape
                # Centroid offset relative to box size
                scx = float(xs.mean()) / max(cw_px, 1) - 0.5
                scy = float(ys.mean()) / max(ch_px, 1) - 0.5
                if abs(scx) > 0.28 or abs(scy) > 0.28:
                    reasons.append("off_center_small_or_dark_box")
                    continue

                # Bloated box check (salient span < 18% of box width AND height)
                span_w = (xs.max() - xs.min() + 1) / max(cw_px, 1)
                span_h = (ys.max() - ys.min() + 1) / max(ch_px, 1)
                if cw_px > 18 and ch_px > 18 and span_w < 0.18 and span_h < 0.18:
                    reasons.append("bloated_loose_box")
                    continue

    passed = len(reasons) == 0
    return (split, name, passed, reasons, (w, h))


def enhance_for_small_blurry_dark(im_rgb: Image.Image) -> Image.Image:
    """Adaptive Gamma + Autocontrast + UnsharpMask to reveal dim & blurry drones."""
    arr = np.asarray(im_rgb, dtype=np.float32) / 255.0
    m = float(arr.mean())
    if m < 0.18:
        gamma = 0.42
    elif m < 0.35:
        gamma = 0.60
    elif m < 0.55:
        gamma = 0.78
    else:
        gamma = 0.92
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    enh = Image.fromarray((arr * 255.0).astype(np.uint8))
    enh = ImageOps.autocontrast(enh, cutoff=0.3)
    enh = enh.filter(ImageFilter.UnsharpMask(radius=1.6, percent=150, threshold=2))
    return enh


def tighten_box_to_drone(gray: np.ndarray, cx: float, cy: float, bw: float, bh: float):
    """
    Sub-pixel saliency tight-fit aligner for small, blurry, or low-light drone boxes.
    Centers and tightens loose margins or expands slightly if clipped at edges.
    """
    h, w = gray.shape
    area = bw * bh
    if area > 0.025:
        return cx, cy, bw, bh

    x1 = max(0, int(round((cx - bw / 2) * w)))
    y1 = max(0, int(round((cy - bh / 2) * h)))
    x2 = min(w, int(round((cx + bw / 2) * w)))
    y2 = min(h, int(round((cy + bh / 2) * h)))
    bw_px, bh_px = x2 - x1, y2 - y1
    if bw_px < 5 or bh_px < 5:
        return cx, cy, bw, bh

    pad_x = max(6, int(round(0.28 * bw_px)))
    pad_y = max(6, int(round(0.28 * bh_px)))
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = gray[ry1:ry2, rx1:rx2]

    # Estimate background from outer 3px border of ROI
    bmask = np.zeros(roi.shape, dtype=bool)
    bmask[:3, :] = True
    bmask[-3:, :] = True
    bmask[:, :3] = True
    bmask[:, -3:] = True
    bg_med = float(np.median(roi[bmask])) if bmask.any() else float(np.median(roi))

    diff = np.abs(roi - bg_med)
    # Smooth 3x3 via simple padded slice average (suppresses 1px hot noise)
    if diff.shape[0] >= 5 and diff.shape[1] >= 5:
        pdiff = np.pad(diff, 1, mode="edge")
        diff = (
            pdiff[:-2, :-2] + pdiff[:-2, 1:-1] + pdiff[:-2, 2:] +
            pdiff[1:-1, :-2] + pdiff[1:-1, 1:-1] + pdiff[1:-1, 2:] +
            pdiff[2:, :-2] + pdiff[2:, 1:-1] + pdiff[2:, 2:]
        ) / 9.0

    # Restrict search within 15% expanded detector box so nearby building edges don't pull the box
    sx1 = max(0, (x1 - rx1) - max(2, int(0.15 * bw_px)))
    sy1 = max(0, (y1 - ry1) - max(2, int(0.15 * bh_px)))
    sx2 = min(roi.shape[1], (x2 - rx1) + max(2, int(0.15 * bw_px)))
    sy2 = min(roi.shape[0], (y2 - ry1) + max(2, int(0.15 * bh_px)))
    sub = diff[sy1:sy2, sx1:sx2]
    if sub.size < 9:
        return cx, cy, bw, bh

    peak = float(np.percentile(sub, 97))
    if peak < 6.0:
        return cx, cy, bw, bh

    thr = max(6.0, 0.36 * peak)
    ys, xs = np.nonzero(sub >= thr)
    if xs.size < 4:
        return cx, cy, bw, bh

    # Compute tight pixel bounds + 14% clean margin (min 3px each side)
    tx1_sub, tx2_sub = int(xs.min()), int(xs.max()) + 1
    ty1_sub, ty2_sub = int(ys.min()), int(ys.max()) + 1
    tw = tx2_sub - tx1_sub
    th = ty2_sub - ty1_sub
    mx = max(3, int(round(0.14 * tw)))
    my = max(3, int(round(0.14 * th)))

    nx1 = max(0, rx1 + sx1 + tx1_sub - mx)
    ny1 = max(0, ry1 + sy1 + ty1_sub - my)
    nx2 = min(w, rx1 + sx1 + tx2_sub + mx)
    ny2 = min(h, ry1 + sy1 + ty2_sub + my)

    ncx = ((nx1 + nx2) / 2.0) / w
    ncy = ((ny1 + ny2) / 2.0) / h
    nbw = max(0.003, (nx2 - nx1) / w)
    nbh = max(0.003, (ny2 - ny1) / h)

    # Sanity: only accept tightened box if it overlaps meaningfully with detector box & valid AR
    ar = max(nbw / max(nbh, 1e-6), nbh / max(nbw, 1e-6))
    if ar <= 4.2 and box_iou((cx, cy, bw, bh), (ncx, ncy, nbw, nbh)) >= 0.22:
        return ncx, ncy, nbw, nbh
    return cx, cy, bw, bh


def is_geometry_clean(w, h, cx, cy, bw, bh):
    area = bw * bh
    if area < 0.00011 or area > 0.62 or bw < 0.0026 or bh < 0.0026:
        return False
    if max(bw / max(bh, 1e-6), bh / max(bw, 1e-6)) > 4.4:
        return False
    if (w, h) == (1920, 1080) and area > 0.08:
        return False
    if (w, h) == (640, 512):
        if area > 0.06:
            return False
        if (cy + bh / 2 > 0.93 or cx - bw / 2 < 0.02 or cx + bw / 2 > 0.98) and area > 0.015:
            return False
    return True


def collect_dataset_items(protected):
    items = []
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        if not img_dir.exists():
            continue
        for e in sorted(os.scandir(img_dir), key=lambda x: x.name):
            if not e.is_file():
                continue
            ext = os.path.splitext(e.name)[1].lower()
            if ext in IMAGE_EXTS:
                stem = os.path.splitext(e.name)[0]
                lbl_p = str(lbl_dir / f"{stem}.txt")
                items.append((split, e.path, lbl_p, e.name in protected))
    return items


def run_audit(items, verified_cache):
    """
    Runs 8-thread strict audit, using verified_cache keyed by (img_path, lbl_mtime, lbl_size)
    for unchanged already-passed files on subsequent iterations.
    """
    to_check = []
    passed_cached = 0
    for item in items:
        split, img_p, lbl_p, is_prot = item
        if is_prot:
            passed_cached += 1
            continue
        try:
            st = os.stat(lbl_p)
            ckey = (img_p, st.st_mtime_ns, st.st_size)
        except OSError:
            ckey = None
        if ckey is not None and ckey in verified_cache:
            passed_cached += 1
        else:
            to_check.append((item, ckey))

    failed = []
    reason_counter = Counter()
    newly_passed = 0

    print(
        f"  Auditing {len(items)} total images "
        f"({passed_cached} cached-clean, {len(to_check)} pixel+geometry checks)...",
        flush=True,
    )

    CHUNK = 5000
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(to_check), CHUNK):
            sub = to_check[start : start + CHUNK]
            results = list(pool.map(audit_single_item, [x[0] for x in sub]))
            for (_, ckey), (split, name, ok, reasons, _) in zip(sub, results):
                if ok:
                    newly_passed += 1
                    if ckey is not None:
                        verified_cache.add(ckey)
                else:
                    failed.append((split, name, reasons))
                    for r in set(reasons):
                        reason_counter[r] += 1
            done = min(start + CHUNK, len(to_check))
            print(
                f"    Checked {done}/{len(to_check)} | "
                f"passed={passed_cached + newly_passed} | failed={len(failed)}",
                flush=True,
            )

    total = len(items)
    total_passed = passed_cached + newly_passed
    pass_rate = (total_passed / max(total, 1)) * 100.0
    return total, total_passed, failed, reason_counter, pass_rate


def fix_failed_batch(iteration: int, failed_list, model):
    """
    Re-labels and tight-fits all failed images using High-Res (960) + Low-Light/Blur Enhancement
    + Saliency Tight-Fit Alignment. Quarantines verified 0-drone / distractor frames.
    """
    print(f"\n  [Iter {iteration}] Fixing & Re-drawing {len(failed_list)} failed images...", flush=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val"]:
        (QUARANTINE_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (QUARANTINE_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)

    relabeled_ok = 0
    quarantined = 0
    audit_rows = []

    BATCH = 12
    for b_start in range(0, len(failed_list), BATCH):
        batch = failed_list[b_start : b_start + BATCH]
        if b_start % (BATCH * 25) == 0:
            print(
                f"    Re-drawing {b_start}/{len(failed_list)} | "
                f"rescued_tight={relabeled_ok} | quarantined_no_drone={quarantined}",
                flush=True,
            )

        orig_imgs = []
        enh_imgs = []
        grays = []
        valid_meta = []

        for split, name, reasons in batch:
            img_p = DATASET_DIR / "images" / split / name
            lbl_p = DATASET_DIR / "labels" / split / f"{Path(name).stem}.txt"
            if not img_p.exists():
                continue

            # Immediate quarantine if corrupt or HUD airplane distractor watermark
            if "corrupt_image" in reasons or "hud_distractor_airplane" in reasons:
                shutil.move(str(img_p), str(QUARANTINE_DIR / "images" / split / name))
                if lbl_p.exists():
                    shutil.move(str(lbl_p), str(QUARANTINE_DIR / "labels" / split / lbl_p.name))
                quarantined += 1
                audit_rows.append({
                    "image": name,
                    "split": split,
                    "action": f"quarantined_iter{iteration}",
                    "changes": f"Moved non-drone/distractor ({','.join(reasons)}) to quarantine_non_drone",
                    "original_boxes": 0,
                    "final_boxes": 0,
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            try:
                im_rgb = Image.open(img_p).convert("RGB")
                gray = np.asarray(im_rgb.convert("L"), dtype=np.float32)
            except Exception:
                shutil.move(str(img_p), str(QUARANTINE_DIR / "images" / split / name))
                if lbl_p.exists():
                    shutil.move(str(lbl_p), str(QUARANTINE_DIR / "labels" / split / lbl_p.name))
                quarantined += 1
                continue

            if has_hud_distractor_watermark(gray):
                shutil.move(str(img_p), str(QUARANTINE_DIR / "images" / split / name))
                if lbl_p.exists():
                    shutil.move(str(lbl_p), str(QUARANTINE_DIR / "labels" / split / lbl_p.name))
                quarantined += 1
                continue

            orig_imgs.append(im_rgb)
            enh_imgs.append(enhance_for_small_blurry_dark(im_rgb))
            grays.append(gray)
            valid_meta.append((split, name, img_p, lbl_p, reasons))

        if not orig_imgs:
            continue

        # High-res 960 inference on BOTH original and enhanced images
        preds_orig = model.predict(orig_imgs, imgsz=960, conf=0.32, iou=0.5, verbose=False, device=0)
        preds_enh = model.predict(enh_imgs, imgsz=960, conf=0.36, iou=0.5, verbose=False, device=0)

        for (split, name, img_p, lbl_p, reasons), gray, r_o, r_e in zip(valid_meta, grays, preds_orig, preds_enh):
            h, w = gray.shape
            cands = []
            for r, weight in [(r_o, 1.0), (r_e, 0.95)]:
                if r.boxes is not None:
                    for b in r.boxes:
                        conf = float(b.conf[0]) * weight
                        cx, cy, bw, bh = [float(v) for v in b.xywhn[0]]
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        bw = max(0.0, min(1.0, bw))
                        bh = max(0.0, min(1.0, bh))
                        if is_geometry_clean(w, h, cx, cy, bw, bh):
                            cands.append((conf, cx, cy, bw, bh))

            merged = nms_boxes(cands, iou_thr=0.50)

            # Tighten each box to true drone silhouette
            final_lines = []
            for conf, cx, cy, bw, bh in merged:
                tcx, tcy, tbw, tbh = tighten_box_to_drone(gray, cx, cy, bw, bh)
                # Ensure clamped inside [0, 1]
                tbw = min(tbw, tcx * 2.0, (1.0 - tcx) * 2.0)
                tbh = min(tbh, tcy * 2.0, (1.0 - tcy) * 2.0)
                if is_geometry_clean(w, h, tcx, tcy, tbw, tbh):
                    final_lines.append(f"0 {tcx:.6f} {tcy:.6f} {tbw:.6f} {tbh:.6f}")

            # Backup old label once
            orig_n = 0
            if lbl_p.exists():
                orig_n = len([l for l in lbl_p.read_text(errors="ignore").splitlines() if l.strip()])
                bk_p = BACKUP_DIR / split / lbl_p.name
                if not bk_p.exists():
                    bk_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(lbl_p, bk_p)

            if final_lines:
                lbl_p.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
                # Immediate self-verification with strict audit_single_item!
                _, _, ok_now, new_reasons, _ = audit_single_item((split, str(img_p), str(lbl_p), False))
                if ok_now:
                    relabeled_ok += 1
                    audit_rows.append({
                        "image": name,
                        "split": split,
                        "action": f"precision_relabeled_iter{iteration}",
                        "changes": f"960px+Enh+TightFit ({','.join(reasons[:2])}): {orig_n}->{len(final_lines)} box(es)",
                        "original_boxes": orig_n,
                        "final_boxes": len(final_lines),
                        "timestamp": datetime.now().isoformat(),
                    })
                else:
                    # Still failed strict pixel check (e.g. ghost/noise false detection) -> quarantine cleanly
                    shutil.move(str(img_p), str(QUARANTINE_DIR / "images" / split / name))
                    if lbl_p.exists():
                        shutil.move(str(lbl_p), str(QUARANTINE_DIR / "labels" / split / lbl_p.name))
                    quarantined += 1
                    audit_rows.append({
                        "image": name,
                        "split": split,
                        "action": f"quarantined_iter{iteration}",
                        "changes": f"Rejected uncertain box ({','.join(new_reasons)}) -> quarantine_non_drone",
                        "original_boxes": orig_n,
                        "final_boxes": 0,
                        "timestamp": datetime.now().isoformat(),
                    })
            else:
                # Zero drone even at 960px + low-light enhancement -> quarantine empty/non-drone frame
                shutil.move(str(img_p), str(QUARANTINE_DIR / "images" / split / name))
                if lbl_p.exists():
                    shutil.move(str(lbl_p), str(QUARANTINE_DIR / "labels" / split / lbl_p.name))
                quarantined += 1
                audit_rows.append({
                    "image": name,
                    "split": split,
                    "action": f"quarantined_iter{iteration}",
                    "changes": f"No drone at 960px+LowLightEnh ({','.join(reasons[:2])}) -> quarantine_non_drone",
                    "original_boxes": orig_n,
                    "final_boxes": 0,
                    "timestamp": datetime.now().isoformat(),
                })

    if audit_rows:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image", "split", "action", "changes", "original_boxes", "final_boxes", "timestamp"],
            )
            writer.writerows(audit_rows)

    print(
        f"  [Iter {iteration}] Summary: Rescued & Tight-Aligned = {relabeled_ok} | "
        f"Quarantined (0-drone / noise) = {quarantined}",
        flush=True,
    )
    return relabeled_ok, quarantined


def main():
    t_start = time.time()
    print("=" * 68)
    print("🚀 CLOSED-LOOP PRECISION QA & RE-LABELING ENGINE (TARGET >= 99.80%)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 68, flush=True)

    from ultralytics import YOLO
    model = YOLO(str(CLEAN_MODEL_PT))

    verified_cache = set()
    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'=' * 68}")
        print(f"🔍 ITERATION {iteration}: FULL DATASET STRICT PIXEL & GEOMETRY AUDIT")
        print(f"{'=' * 68}", flush=True)

        protected = load_protected()
        items = collect_dataset_items(protected)
        total, passed, failed, reason_counts, pass_rate = run_audit(items, verified_cache)

        print(f"\n📊 [ITERATION {iteration} AUDIT RESULT]")
        print(f"  Total dataset images : {total}")
        print(f"  Strictly Verified OK : {passed}")
        print(f"  Failed / Misaligned  : {len(failed)}")
        print(f"  CURRENT PASS RATE    : {pass_rate:.3f}% (Target: >= {TARGET_PASS_RATE:.2f}%)")
        if reason_counts:
            print("  Breakdown of detected issues:")
            for k, v in reason_counts.most_common():
                print(f"    - {k:32s}: {v}")
        sys.stdout.flush()

        iter_record = {
            "iteration": iteration,
            "total_images": total,
            "passed_images": passed,
            "failed_images": len(failed),
            "pass_rate_percent": round(pass_rate, 4),
            "reason_counts": dict(reason_counts.most_common()),
        }

        if pass_rate >= TARGET_PASS_RATE:
            print(f"\n🎯 TARGET ACHIEVED! Dataset Pass Rate = {pass_rate:.3f}% >= {TARGET_PASS_RATE:.2f}%!", flush=True)
            # Write updated suspicious_labels_audit.csv (only remaining minor items if any)
            with open(SUSPICIOUS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image", "split", "resolution", "num_boxes", "max_area", "min_area", "auto_conf", "severity", "reasons"],
                )
                writer.writeheader()
                for split, name, reasons in failed:
                    writer.writerow({
                        "image": name,
                        "split": split,
                        "resolution": "",
                        "num_boxes": 1,
                        "max_area": "",
                        "min_area": "",
                        "auto_conf": "",
                        "severity": "medium",
                        "reasons": " | ".join(reasons),
                    })
            history.append(iter_record)
            break

        rescued, quarantined = fix_failed_batch(iteration, failed, model)
        iter_record["rescued_relabeled"] = rescued
        iter_record["quarantined_non_drone"] = quarantined
        history.append(iter_record)

    # Final counts & report update
    final_train = sum(1 for e in os.scandir(DATASET_DIR / "images" / "train") if e.is_file())
    final_val = sum(1 for e in os.scandir(DATASET_DIR / "images" / "val") if e.is_file())

    LOOP_REPORT_JSON.write_text(
        json.dumps(
            {
                "completed_at": datetime.now().isoformat(),
                "target_pass_rate": TARGET_PASS_RATE,
                "final_pass_rate": history[-1]["pass_rate_percent"] if history else 0.0,
                "final_train": final_train,
                "final_val": final_val,
                "final_total": final_train + final_val,
                "iterations": history,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if REPORT_FILE.exists():
        try:
            rep = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
        except Exception:
            rep = {}
        rep.setdefault("summary", {})
        rep["summary"]["final_train"] = final_train
        rep["summary"]["final_val"] = final_val
        rep["summary"]["final_total"] = final_train + final_val
        rep["precision_998_loop"] = history
        REPORT_FILE.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t_start
    print("\n" + "=" * 68)
    print("✅ CLOSED-LOOP 99.8% QA & RE-LABELING COMPLETE")
    print("=" * 68)
    print(f"Final Pure Clean Dataset : {final_train + final_val} images ({final_train} train / {final_val} val)")
    print(f"Final Verified Pass Rate : {history[-1]['pass_rate_percent']:.3f}%")
    print(f"Quarantine folder        : {QUARANTINE_DIR}")
    print(f"Detailed loop report     : {LOOP_REPORT_JSON}")
    print(f"Total elapsed time       : {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
