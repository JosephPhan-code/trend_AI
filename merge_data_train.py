"""
Smart Incremental Merge + Clean Auto-Label for e:\trend_AI\data_train -> clean_dataset
1. Automatically waits until e:\trend_AI\data_train finishes copying/extracting on disk
   (0 file count & byte size change for 20 consecutive seconds).
2. Indexes all existing images in clean_dataset by MD5 hash (100% duplicate elimination).
3. Scans all images in data_train (train/val/test or nested folders):
   - Validates image integrity & dimensions via PIL
   - Deduplicates against clean_dataset and within data_train
   - Parses & audits paired YOLO .txt (or VOC .xml) labels with strict anti-poison filters:
     * Rejects 1920x1080 HUD crosshair giant boxes (area > 0.08)
     * Rejects 640x512 IR building giant/edge boxes (area > 0.06 or edge > 0.015)
     * Clamps coordinates to [0, 1] & removes duplicate boxes (IoU > 0.75)
   - Any image with missing/empty label OR poisoned label is queued for GPU inference
     using the Clean v2 YOLOv8n model (mAP50=0.984, conf >= 0.40, batch=16).
4. Splits all valid new (image, clean_labels) pairs 85% train / 15% val, copies to
   clean_dataset, updates relabel_audit_log.csv & consolidation_report.json.
"""

import os
import sys
import time
import json
import csv
import shutil
import random
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WORKSPACE = Path(r"e:\trend_AI")
SOURCE_DIR = WORKSPACE / "data_train"
CLEAN_DATASET = WORKSPACE / "clean_dataset"
CLEAN_MODEL_PT = WORKSPACE / "clean_relabel_train" / "clean_drone_v2" / "weights" / "best.pt"
AUDIT_LOG = CLEAN_DATASET / "relabel_audit_log.csv"
REPORT_FILE = CLEAN_DATASET / "consolidation_report.json"

TRAIN_RATIO = 0.85
CONF_THR = 0.40
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def wait_until_copy_finished(folder: Path, stable_seconds: int = 20):
    print(f"Checking if {folder} is still being copied/extracted...", flush=True)
    def snapshot():
        count = 0
        total_bytes = 0
        for dp, _, fns in os.walk(folder):
            for fn in fns:
                count += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(dp, fn))
                except OSError:
                    pass
        return count, total_bytes

    prev_c, prev_b = snapshot()
    stable_elapsed = 0
    check_interval = 5
    while stable_elapsed < stable_seconds:
        time.sleep(check_interval)
        cur_c, cur_b = snapshot()
        if cur_c == prev_c and cur_b == prev_b and cur_c > 0:
            stable_elapsed += check_interval
            print(f"  Folder stable ({cur_c} files, {cur_b/1e9:.2f} GB) for {stable_elapsed}/{stable_seconds}s...", flush=True)
        else:
            print(
                f"  Still copying: {prev_c} -> {cur_c} files (+{cur_c - prev_c}), "
                f"{cur_b/1e9:.2f} GB... waiting",
                flush=True,
            )
            prev_c, prev_b = cur_c, cur_b
            stable_elapsed = 0
    print(f"Copy finished! Final snapshot: {prev_c} files ({prev_b/1e9:.2f} GB).", flush=True)


def file_md5(filepath: str, block_size: int = 131072):
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while True:
                buf = f.read(block_size)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return None


def check_image(filepath: str):
    try:
        with Image.open(filepath) as im:
            w, h = im.size
            im.verify()
        if w < 10 or h < 10:
            return False, 0, 0
        return True, w, h
    except Exception:
        return False, 0, 0


def iou(b1, b2):
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


def is_valid_clean_box(w_img, h_img, cx, cy, bw, bh):
    area = bw * bh
    if area < 0.00012 or area > 0.65:
        return False
    if bw < 0.0025 or bh < 0.0025:
        return False
    ar = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
    if ar > 5.0:
        return False
    if (w_img, h_img) == (1920, 1080) and area > 0.08:
        return False
    if (w_img, h_img) == (640, 512):
        if area > 0.06:
            return False
        y2 = cy + bh / 2
        x1 = cx - bw / 2
        x2 = cx + bw / 2
        if (y2 > 0.93 or x1 < 0.02 or x2 > 0.98) and area > 0.015:
            return False
    return True


def find_paired_label(img_path: Path):
    """Find matching .txt or .xml label for an image in standard YOLO or flat layouts."""
    parts = list(img_path.parts)
    # Check .../images/... -> .../labels/...
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            cand_parts = parts.copy()
            cand_parts[i] = "labels"
            cand_txt = Path(*cand_parts).with_suffix(".txt")
            if cand_txt.exists():
                return cand_txt, "txt"
            cand_xml = Path(*cand_parts).with_suffix(".xml")
            if cand_xml.exists():
                return cand_xml, "xml"
            break
    # Fallback: same directory
    same_txt = img_path.with_suffix(".txt")
    if same_txt.exists() and same_txt.name.lower() != "classes.txt":
        return same_txt, "txt"
    same_xml = img_path.with_suffix(".xml")
    if same_xml.exists():
        return same_xml, "xml"
    return None, None


def parse_and_clean_label(lbl_path: Path, fmt: str, w_img: int, h_img: int):
    """Returns (clean_lines, had_raw_boxes, changes_list)."""
    changes = []
    raw_boxes = []

    if fmt == "txt":
        try:
            raw_lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        except Exception:
            return [], False, []
        for idx, line in enumerate(raw_lines):
            p = line.split()
            if len(p) < 5:
                continue
            try:
                cls = int(float(p[0]))
                cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                raw_boxes.append((cls, cx, cy, bw, bh))
            except ValueError:
                continue

    elif fmt == "xml":
        try:
            root = ET.parse(lbl_path).getroot()
            size_el = root.find("size")
            if size_el is not None:
                sw = int(float(size_el.find("width").text))
                sh = int(float(size_el.find("height").text))
                if sw > 0 and sh > 0:
                    w_img, h_img = sw, sh
            for obj in root.findall("object"):
                bb = obj.find("bndbox")
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
                cx = ((xmin + xmax) / 2.0) / w_img
                cy = ((ymin + ymax) / 2.0) / h_img
                bw = (xmax - xmin) / w_img
                bh = (ymax - ymin) / h_img
                raw_boxes.append((0, cx, cy, bw, bh))
        except Exception:
            return [], False, []

    if not raw_boxes:
        return [], False, []

    cleaned = []
    for idx, (cls, cx, cy, bw, bh) in enumerate(raw_boxes):
        if cls != 0:
            changes.append(f"box{idx}:cls {cls}->0")
            cls = 0
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        if cx - bw / 2 < 0:
            bw = cx * 2
            changes.append(f"box{idx}:clamp_L")
        if cx + bw / 2 > 1:
            bw = (1.0 - cx) * 2
            changes.append(f"box{idx}:clamp_R")
        if cy - bh / 2 < 0:
            bh = cy * 2
            changes.append(f"box{idx}:clamp_T")
        if cy + bh / 2 > 1:
            bh = (1.0 - cy) * 2
            changes.append(f"box{idx}:clamp_B")

        if not is_valid_clean_box(w_img, h_img, cx, cy, bw, bh):
            changes.append(f"box{idx}:rejected_poison_or_invalid(area={bw*bh:.4f})")
            continue
        cleaned.append((0, cx, cy, bw, bh))

    # IoU dedup
    if len(cleaned) > 1:
        keep = [True] * len(cleaned)
        for i in range(len(cleaned)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(cleaned)):
                if not keep[j]:
                    continue
                if iou(cleaned[i][1:], cleaned[j][1:]) > 0.75:
                    if cleaned[j][3] * cleaned[j][4] > cleaned[i][3] * cleaned[i][4]:
                        keep[i] = False
                    else:
                        keep[j] = False
                    changes.append("dedup_iou")
        cleaned = [cleaned[i] for i in range(len(cleaned)) if keep[i]]

    lines = [f"0 {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in cleaned]
    return lines, True, changes


def main():
    start_time = datetime.now()
    print("=" * 60)
    print("MERGE + CLEAN AUTO-LABEL: data_train -> clean_dataset")
    print(f"Started: {start_time.isoformat()}")
    print("=" * 60, flush=True)

    if not SOURCE_DIR.exists():
        print(f"ERROR: {SOURCE_DIR} not found!")
        sys.exit(1)

    # Step 0: Wait until folder copy/extract finishes
    wait_until_copy_finished(SOURCE_DIR, stable_seconds=20)

    # Step 1: Index existing clean_dataset hashes & find next drone_XXXXXX index
    print("\nPhase 1: Indexing existing clean_dataset MD5 hashes...", flush=True)
    existing_hashes = set()
    max_idx = 0
    existing_total = 0
    for split in ["train", "val"]:
        img_dir = CLEAN_DATASET / "images" / split
        if not img_dir.exists():
            continue
        for entry in os.scandir(img_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in IMAGE_EXTS:
                existing_total += 1
                stem = os.path.splitext(entry.name)[0]
                if stem.startswith("drone_"):
                    try:
                        max_idx = max(max_idx, int(stem.split("_")[1]))
                    except ValueError:
                        pass
                h = file_md5(entry.path)
                if h:
                    existing_hashes.add(h)
                if existing_total % 5000 == 0:
                    print(f"  Indexed {existing_total} existing images...", flush=True)
    next_idx = max_idx + 1
    print(f"  Indexed {existing_total} existing images. Next ID: drone_{next_idx:06d}", flush=True)

    # Step 2: Scan all images in data_train
    print("\nPhase 2: Scanning & deduplicating data_train...", flush=True)
    stats = {
        "scanned": 0,
        "corrupt": 0,
        "dup_existing": 0,
        "dup_internal": 0,
        "clean_gt_kept": 0,
        "gt_fixed": 0,
        "queued_autolabel": 0,
        "autolabeled_kept": 0,
        "autolabel_no_drone": 0,
        "total_added": 0,
        "boxes_added": 0,
    }

    seen_new_hashes = set()
    ready_pairs = []      # (Path, lines, action_str, changes_str, orig_boxes)
    need_autolabel = []   # (Path, w, h, had_poisoned_gt)

    all_img_paths = []
    for dp, _, fns in os.walk(SOURCE_DIR):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                all_img_paths.append(Path(dp) / fn)
    all_img_paths.sort()
    print(f"  Found {len(all_img_paths)} candidate image files in data_train.", flush=True)

    for idx, img_p in enumerate(all_img_paths):
        stats["scanned"] += 1
        if stats["scanned"] % 5000 == 0:
            print(
                f"  Scanned {stats['scanned']}/{len(all_img_paths)} | "
                f"clean_gt={stats['clean_gt_kept']} | need_auto={len(need_autolabel)} | "
                f"dup={stats['dup_existing']+stats['dup_internal']}",
                flush=True,
            )

        h = file_md5(str(img_p))
        if not h:
            stats["corrupt"] += 1
            continue
        if h in existing_hashes:
            stats["dup_existing"] += 1
            continue
        if h in seen_new_hashes:
            stats["dup_internal"] += 1
            continue

        ok, w_img, h_img = check_image(str(img_p))
        if not ok:
            stats["corrupt"] += 1
            continue

        seen_new_hashes.add(h)

        lbl_p, fmt = find_paired_label(img_p)
        if lbl_p is not None:
            clean_lines, had_raw, changes = parse_and_clean_label(lbl_p, fmt, w_img, h_img)
            if clean_lines:
                stats["clean_gt_kept"] += 1
                if changes:
                    stats["gt_fixed"] += 1
                    ready_pairs.append((img_p, clean_lines, "merge_fixed", " | ".join(changes), len(clean_lines)))
                else:
                    ready_pairs.append((img_p, clean_lines, "original", "", len(clean_lines)))
                continue
            else:
                need_autolabel.append((img_p, w_img, h_img, had_raw))
        else:
            need_autolabel.append((img_p, w_img, h_img, False))

    stats["queued_autolabel"] = len(need_autolabel)
    print(
        f"\nScan summary:\n"
        f"  Total scanned:        {stats['scanned']}\n"
        f"  Corrupt skipped:      {stats['corrupt']}\n"
        f"  Duplicate (existing): {stats['dup_existing']}\n"
        f"  Duplicate (internal): {stats['dup_internal']}\n"
        f"  Valid clean GT pairs: {stats['clean_gt_kept']} ({stats['gt_fixed']} fixed)\n"
        f"  Queued for AutoLabel: {stats['queued_autolabel']}",
        flush=True,
    )

    # Step 3: Auto-label unlabeled / poisoned-GT images with Clean v2 model
    if need_autolabel:
        print(f"\nPhase 3: Auto-labeling {len(need_autolabel)} images with Clean v2 model (conf >= {CONF_THR})...", flush=True)
        from ultralytics import YOLO

        model = YOLO(str(CLEAN_MODEL_PT))
        BATCH = 16
        for i in range(0, len(need_autolabel), BATCH):
            batch = need_autolabel[i : i + BATCH]
            if i % (BATCH * 50) == 0:
                print(
                    f"  Auto-labeling {i}/{len(need_autolabel)}... "
                    f"(kept={stats['autolabeled_kept']}, no_drone={stats['autolabel_no_drone']})",
                    flush=True,
                )
            try:
                results = model.predict(
                    source=[str(item[0]) for item in batch],
                    conf=CONF_THR,
                    iou=0.5,
                    imgsz=640,
                    verbose=False,
                    device=0,
                )
                for (img_p, w_img, h_img, had_poison), res in zip(batch, results):
                    lines = []
                    confs = []
                    if res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            conf = float(box.conf[0])
                            cx, cy, bw, bh = [float(v) for v in box.xywhn[0]]
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            bw = max(0.0, min(1.0, bw))
                            bh = max(0.0, min(1.0, bh))
                            if is_valid_clean_box(w_img, h_img, cx, cy, bw, bh):
                                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                                confs.append(conf)
                    if lines:
                        stats["autolabeled_kept"] += 1
                        avg_c = sum(confs) / len(confs)
                        note = "Re-labeled poisoned GT" if had_poison else "Auto-labeled"
                        ready_pairs.append((
                            img_p,
                            lines,
                            "auto_labeled",
                            f"{note} via Clean v2: {len(lines)} box(es) (avg_conf={avg_c:.3f})",
                            0,
                        ))
                    else:
                        stats["autolabel_no_drone"] += 1
            except Exception as e:
                print(f"  Batch warning at {i}: {e}", flush=True)

    stats["total_added"] = len(ready_pairs)
    if not ready_pairs:
        print("\nNo new valid unique images to add from data_train!", flush=True)
        return

    # Step 4: Split 85/15 and copy into clean_dataset
    print(f"\nPhase 4: Merging {len(ready_pairs)} new images into clean_dataset (85/15 split)...", flush=True)
    random.seed(42)
    random.shuffle(ready_pairs)
    split_idx = int(len(ready_pairs) * TRAIN_RATIO)
    train_add = ready_pairs[:split_idx]
    val_add = ready_pairs[split_idx:]

    audit_rows = []
    cur_id = next_idx

    for split_name, subset in [("train", train_add), ("val", val_add)]:
        for i, (img_p, lines, action, changes, orig_n) in enumerate(subset):
            ext = img_p.suffix.lower()
            new_base = f"drone_{cur_id:06d}"
            cur_id += 1

            dst_img = CLEAN_DATASET / "images" / split_name / f"{new_base}{ext}"
            dst_lbl = CLEAN_DATASET / "labels" / split_name / f"{new_base}.txt"
            shutil.copy2(img_p, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["boxes_added"] += len(lines)

            if action != "original":
                audit_rows.append({
                    "image": f"{new_base}{ext}",
                    "split": split_name,
                    "action": action,
                    "changes": changes,
                    "original_boxes": orig_n,
                    "final_boxes": len(lines),
                    "timestamp": datetime.now().isoformat(),
                })
            if (i + 1) % 2500 == 0:
                print(f"  Copied {split_name}: {i+1}/{len(subset)}...", flush=True)

    if audit_rows:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image", "split", "action", "changes", "original_boxes", "final_boxes", "timestamp"],
            )
            writer.writerows(audit_rows)

    # Update consolidation_report.json
    final_train = sum(1 for e in os.scandir(CLEAN_DATASET / "images" / "train") if e.is_file())
    final_val = sum(1 for e in os.scandir(CLEAN_DATASET / "images" / "val") if e.is_file())
    report = {}
    if REPORT_FILE.exists():
        try:
            report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    report.setdefault("summary", {})
    report["summary"]["final_train"] = final_train
    report["summary"]["final_val"] = final_val
    report["summary"]["final_total"] = final_train + final_val
    report["data_train_merge"] = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("✅ MERGE data_train COMPLETE")
    print("=" * 60)
    print(f"Scanned in data_train:  {stats['scanned']}")
    print(f"Duplicates skipped:     {stats['dup_existing'] + stats['dup_internal']}")
    print(f"Corrupt skipped:        {stats['corrupt']}")
    print(f"Clean GT added:         {stats['clean_gt_kept']} ({stats['gt_fixed']} fixed)")
    print(f"Auto-labeled added:     {stats['autolabeled_kept']} (skipped {stats['autolabel_no_drone']} with no drone)")
    print(f"Total new images added: {stats['total_added']} (+{stats['boxes_added']} boxes)")
    print(f"FINAL clean_dataset:    {final_train + final_val} ({final_train} train / {final_val} val)")
    print(f"Elapsed:                {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
