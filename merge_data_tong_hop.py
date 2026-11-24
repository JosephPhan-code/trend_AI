"""
Incremental Merge: Add Data_tong_hop into existing clean_dataset
- Dedup against existing images by hash
- Skip empty labels
- Validate & fix labels (same rules as Step 2)
- Append new data to train/val maintaining 85/15 ratio
- Update relabel_audit_log.csv
- Update consolidation_report.json
"""

import os
import sys
import json
import hashlib
import shutil
import random
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    print("ERROR: Pillow not installed")
    sys.exit(1)

# ============ CONFIG ============
WORKSPACE = Path(r"e:\trend_AI")
NEW_SOURCE = WORKSPACE / "Data_tong_hop"
CLEAN_DATASET = WORKSPACE / "clean_dataset"
TRAIN_RATIO = 0.85
AUDIT_LOG = CLEAN_DATASET / "relabel_audit_log.csv"
REPORT_FILE = CLEAN_DATASET / "consolidation_report.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Label thresholds
MIN_BOX_AREA = 0.0001
MAX_BOX_AREA = 0.85
MIN_BOX_DIM = 0.003
IOU_THRESHOLD = 0.75

# ============ HELPERS ============

def file_hash(filepath, block_size=65536):
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()
    except Exception:
        return None

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            w, h = img.size
            if w < 10 or h < 10:
                return False
            return True
    except Exception:
        return False

def iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def parse_and_fix_label(label_path):
    """Parse, validate and fix a YOLO label file. Returns (lines, changes)."""
    changes = []
    lines = []
    try:
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [l.strip() for l in f.readlines() if l.strip()]
    except:
        return [], ["Could not read file"]
    
    if not raw_lines:
        return [], []
    
    boxes = []
    for idx, raw in enumerate(raw_lines):
        parts = raw.split()
        if len(parts) < 5:
            changes.append(f"Removed unparseable line {idx}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            changes.append(f"Removed invalid line {idx}")
            continue
        
        # Fix class
        if cls != 0:
            changes.append(f"Box {idx}: class {cls}->0")
            cls = 0
        
        # Clamp
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        # Ensure box within image
        half_w, half_h = w / 2, h / 2
        if cx - half_w < 0:
            w = cx * 2
            changes.append(f"Box {idx}: clamped left edge")
        if cx + half_w > 1:
            w = (1 - cx) * 2
            changes.append(f"Box {idx}: clamped right edge")
        if cy - half_h < 0:
            h = cy * 2
            changes.append(f"Box {idx}: clamped top edge")
        if cy + half_h > 1:
            h = (1 - cy) * 2
            changes.append(f"Box {idx}: clamped bottom edge")
        
        area = w * h
        if area < MIN_BOX_AREA:
            changes.append(f"Box {idx}: REMOVED (area={area:.6f} too small)")
            continue
        if area > MAX_BOX_AREA:
            changes.append(f"Box {idx}: REMOVED (area={area:.4f} too large)")
            continue
        if w < MIN_BOX_DIM or h < MIN_BOX_DIM:
            changes.append(f"Box {idx}: REMOVED (dim too small)")
            continue
        
        boxes.append((cls, cx, cy, w, h))
    
    # Remove duplicate overlapping boxes
    if len(boxes) > 1:
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            if not keep[i]: continue
            for j in range(i + 1, len(boxes)):
                if not keep[j]: continue
                if boxes[i][0] == boxes[j][0]:
                    if iou(boxes[i][1:], boxes[j][1:]) > IOU_THRESHOLD:
                        area_i = boxes[i][3] * boxes[i][4]
                        area_j = boxes[j][3] * boxes[j][4]
                        if area_j > area_i:
                            keep[i] = False
                            changes.append(f"Box dup: removed smaller box {i}")
                        else:
                            keep[j] = False
                            changes.append(f"Box dup: removed smaller box {j}")
        boxes = [boxes[i] for i in range(len(boxes)) if keep[i]]
    
    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    return lines, changes


def main():
    start_time = datetime.now()
    print("=" * 60)
    print("INCREMENTAL MERGE: Data_tong_hop → clean_dataset")
    print(f"Started: {start_time.isoformat()}")
    print("=" * 60)
    
    # Phase 0: Build hash index of existing clean_dataset
    print("\n🔑 Phase 0: Building hash index of existing clean_dataset...")
    existing_hashes = set()
    existing_count = 0
    for split in ["train", "val"]:
        img_dir = CLEAN_DATASET / "images" / split
        if not img_dir.exists():
            continue
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in IMAGE_EXTS:
                h = file_hash(img_file)
                if h:
                    existing_hashes.add(h)
                    existing_count += 1
                if existing_count % 2000 == 0:
                    print(f"   Indexed {existing_count} existing images...")
    print(f"   Total existing: {existing_count} images indexed")
    
    # Find next drone_XXXXXX index
    max_idx = 0
    for split in ["train", "val"]:
        for f in (CLEAN_DATASET / "images" / split).iterdir():
            if f.stem.startswith("drone_"):
                try:
                    idx = int(f.stem.split("_")[1])
                    max_idx = max(max_idx, idx)
                except:
                    pass
    next_idx = max_idx + 1
    print(f"   Next file index: drone_{next_idx:06d}")
    
    # Phase 1: Scan new source, dedup, validate
    print(f"\n📂 Phase 1: Scanning Data_tong_hop...")
    new_pairs = []  # (img_path, label_lines, changes)
    seen_hashes = set()
    
    stats = {
        "source_total": 0,
        "empty_labels": 0,
        "corrupt_images": 0,
        "duplicate_with_existing": 0,
        "duplicate_within_new": 0,
        "valid_new_pairs": 0,
        "labels_fixed": 0,
        "boxes_added": 0,
    }
    
    for split in ["train", "val"]:
        img_dir = NEW_SOURCE / "images" / split
        lbl_dir = NEW_SOURCE / "labels" / split
        if not img_dir.exists():
            continue
        
        img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        print(f"   {split}: {len(img_files)} image files")
        
        for idx, img_file in enumerate(img_files):
            stats["source_total"] += 1
            if idx % 3000 == 0 and idx > 0:
                print(f"   Processing {split} {idx}/{len(img_files)}...")
            
            # Check label
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                stats["empty_labels"] += 1
                continue
            
            label_lines, changes = parse_and_fix_label(lbl_file)
            if not label_lines:
                stats["empty_labels"] += 1
                continue
            
            # Validate image
            if not is_valid_image(img_file):
                stats["corrupt_images"] += 1
                continue
            
            # Hash dedup
            h = file_hash(img_file)
            if not h:
                stats["corrupt_images"] += 1
                continue
            
            if h in existing_hashes:
                stats["duplicate_with_existing"] += 1
                continue
            
            if h in seen_hashes:
                stats["duplicate_within_new"] += 1
                continue
            
            seen_hashes.add(h)
            
            if changes:
                stats["labels_fixed"] += 1
            
            stats["boxes_added"] += len(label_lines)
            new_pairs.append((img_file, label_lines, changes))
    
    stats["valid_new_pairs"] = len(new_pairs)
    print(f"\n📊 Scan results:")
    print(f"   Total in source:         {stats['source_total']}")
    print(f"   Empty/no labels:         {stats['empty_labels']}")
    print(f"   Corrupt images:          {stats['corrupt_images']}")
    print(f"   Duplicate (existing):    {stats['duplicate_with_existing']}")
    print(f"   Duplicate (within new):  {stats['duplicate_within_new']}")
    print(f"   Labels fixed:            {stats['labels_fixed']}")
    print(f"   ✅ Valid new pairs:       {stats['valid_new_pairs']}")
    print(f"   Total new boxes:         {stats['boxes_added']}")
    
    if not new_pairs:
        print("\n⚠️ No new unique images to add!")
        return
    
    # Phase 2: Shuffle and split new data
    print(f"\n📦 Phase 2: Splitting new data ({TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%})...")
    random.seed(42)
    random.shuffle(new_pairs)
    
    split_idx = int(len(new_pairs) * TRAIN_RATIO)
    train_new = new_pairs[:split_idx]
    val_new = new_pairs[split_idx:]
    print(f"   New train: {len(train_new)} | New val: {len(val_new)}")
    
    # Phase 3: Copy new files
    print(f"\n📁 Phase 3: Copying new files to clean_dataset...")
    audit_rows = []
    global_idx = next_idx
    
    def copy_pair(pair, split):
        nonlocal global_idx
        img_path, label_lines, changes = pair
        ext = img_path.suffix.lower()
        new_name = f"drone_{global_idx:06d}"
        
        dst_img = CLEAN_DATASET / "images" / split / f"{new_name}{ext}"
        dst_lbl = CLEAN_DATASET / "labels" / split / f"{new_name}.txt"
        
        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
        
        if changes:
            audit_rows.append({
                "image": f"{new_name}{ext}",
                "split": split,
                "action": "merge_fixed",
                "changes": " | ".join(changes),
                "original_boxes": "?",
                "final_boxes": len(label_lines),
                "timestamp": datetime.now().isoformat(),
            })
        
        global_idx += 1
        return new_name
    
    for i, pair in enumerate(train_new):
        copy_pair(pair, "train")
        if (i + 1) % 2000 == 0:
            print(f"   Copied train {i+1}/{len(train_new)}...")
    
    for i, pair in enumerate(val_new):
        copy_pair(pair, "val")
        if (i + 1) % 2000 == 0:
            print(f"   Copied val {i+1}/{len(val_new)}...")
    
    # Phase 4: Append to audit log
    print(f"\n📝 Appending {len(audit_rows)} entries to audit log...")
    if audit_rows:
        file_exists = AUDIT_LOG.exists()
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "split", "action", "changes", "original_boxes", "final_boxes", "timestamp"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(audit_rows)
    
    # Phase 5: Update report
    print("📊 Updating consolidation report...")
    report = {}
    if REPORT_FILE.exists():
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            report = json.load(f)
    
    # Count final totals
    final_train = len(list((CLEAN_DATASET / "images" / "train").glob("*")))
    final_val = len(list((CLEAN_DATASET / "images" / "val").glob("*")))
    
    report["last_merge"] = {
        "timestamp": datetime.now().isoformat(),
        "source": "Data_tong_hop",
        "source_total": stats["source_total"],
        "new_images_added": stats["valid_new_pairs"],
        "new_boxes_added": stats["boxes_added"],
        "duplicates_skipped": stats["duplicate_with_existing"] + stats["duplicate_within_new"],
        "empty_labels_skipped": stats["empty_labels"],
        "labels_fixed": stats["labels_fixed"],
    }
    report["summary"]["final_train"] = final_train
    report["summary"]["final_val"] = final_val
    report["summary"]["final_total"] = final_train + final_val
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Update data.yaml
    yaml_content = f"""path: {CLEAN_DATASET}
train: images/train
val: images/val
nc: 1
names:
  0: drone
"""
    with open(CLEAN_DATASET / "data.yaml", "w") as f:
        f.write(yaml_content)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'=' * 60}")
    print(f"✅ MERGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"New images added:     {stats['valid_new_pairs']}")
    print(f"New boxes added:      {stats['boxes_added']}")
    print(f"Duplicates skipped:   {stats['duplicate_with_existing'] + stats['duplicate_within_new']}")
    print(f"Empty labels skipped: {stats['empty_labels']}")
    print(f"Labels fixed:         {stats['labels_fixed']}")
    print(f"")
    print(f"📊 FINAL DATASET:")
    print(f"   Train: {final_train}")
    print(f"   Val:   {final_val}")
    print(f"   Total: {final_train + final_val}")
    print(f"Time elapsed: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
