"""
Step 1: Dataset Consolidation & Cleaning
- Scans all data sources
- Converts VOC XML to YOLO TXT
- Deduplicates by image hash
- Removes corrupt images, empty labels, invalid boxes
- Outputs clean unified dataset to e:\trend_AI\clean_dataset
"""

import os
import sys
import json
import hashlib
import shutil
import random
import csv
import xml.etree.ElementTree as ET
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
OUTPUT_DIR = WORKSPACE / "clean_dataset"
TRAIN_RATIO = 0.85
CLASS_NAMES = {0: "drone"}
REPORT_FILE = OUTPUT_DIR / "consolidation_report.json"

# Sources to scan (skip the duplicate data/images-20260818T... directory)
SOURCES = [
    {
        "name": "dataset_v2",
        "type": "yolo_split",  # has images/train, images/val, labels/train, labels/val
        "path": WORKSPACE / "dataset_v2",
    },
    {
        "name": "output_frames",
        "type": "yolo_flat",  # images and labels in same flat dir
        "path": WORKSPACE / "output_frames",
    },
    {
        "name": "archive_xml",
        "type": "voc_xml",
        "path": WORKSPACE / "trend_AI" / "data" / "archive" / "dataset_xml_format" / "dataset_xml_format",
    },
    {
        "name": "archive_yolo",
        "type": "yolo_flat",
        "path": WORKSPACE / "trend_AI" / "data" / "archive" / "drone_dataset_yolo" / "dataset_txt",
    },
    {
        "name": "large_dataset",
        "type": "yolo_split",
        "path": WORKSPACE / "trend_AI" / "images-20260818T094512Z-1-002",
    },
    {
        "name": "unified_prev",
        "type": "yolo_split",
        "path": WORKSPACE / "unified_dataset",
    },
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ============ HELPERS ============

def file_hash(filepath, block_size=65536):
    """Compute MD5 hash of file for dedup."""
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
    """Check if file is a valid, non-corrupt image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Re-open to get size (verify() may leave it in bad state)
        with Image.open(filepath) as img:
            w, h = img.size
            if w < 10 or h < 10:
                return False, 0, 0
            return True, w, h
    except Exception:
        return False, 0, 0


def parse_voc_xml(xml_path, img_w, img_h):
    """Convert Pascal VOC XML annotation to YOLO format lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = []
        
        # Try to get size from XML
        size_elem = root.find("size")
        if size_elem is not None:
            w = int(size_elem.find("width").text)
            h = int(size_elem.find("height").text)
            if w > 0 and h > 0:
                img_w, img_h = w, h
        
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            # Map all class names to 0 (drone)
            class_id = 0  # single class dataset
            
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            
            # Convert to YOLO format (cx, cy, w, h) normalized
            cx = ((xmin + xmax) / 2.0) / img_w
            cy = ((ymin + ymax) / 2.0) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))
            
            if bw > 0.001 and bh > 0.001:  # skip degenerate boxes
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        return lines
    except Exception as e:
        return []


def parse_yolo_label(label_path):
    """Parse and validate YOLO label file. Returns cleaned lines."""
    lines = []
    try:
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parts = raw_line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except (ValueError, IndexError):
                    continue
                
                # Clamp coordinates
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                # Skip degenerate boxes
                area = w * h
                if area < 0.0001:  # too small
                    continue
                if area > 0.85:  # too large (covers >85% of image)
                    continue
                if w < 0.002 or h < 0.002:  # essentially a line
                    continue
                
                # Map class to 0 (all are drone)
                class_id = 0
                
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    except Exception:
        pass
    
    # Remove duplicate overlapping boxes (IoU > 0.8)
    lines = remove_duplicate_boxes(lines)
    return lines


def iou(box1, box2):
    """Calculate IoU between two YOLO boxes (cx, cy, w, h)."""
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
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def remove_duplicate_boxes(lines):
    """Remove overlapping boxes with IoU > 0.8."""
    if len(lines) <= 1:
        return lines
    
    boxes = []
    for line in lines:
        parts = line.split()
        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            if boxes[i][0] == boxes[j][0]:  # same class
                box_i = boxes[i][1:]
                box_j = boxes[j][1:]
                if iou(box_i, box_j) > 0.8:
                    # Keep the larger box
                    area_i = box_i[2] * box_i[3]
                    area_j = box_j[2] * box_j[3]
                    if area_j > area_i:
                        keep[i] = False
                    else:
                        keep[j] = False
    
    return [lines[i] for i in range(len(lines)) if keep[i]]


def collect_image_label_pairs(source):
    """Collect all (image_path, label_lines) pairs from a source."""
    pairs = []
    src_path = source["path"]
    src_type = source["type"]
    src_name = source["name"]
    
    if not src_path.exists():
        print(f"  [SKIP] {src_name}: path not found {src_path}")
        return pairs
    
    if src_type == "yolo_split":
        img_dir = src_path / "images"
        lbl_dir = src_path / "labels"
        for split in ["train", "val", "test"]:
            split_img = img_dir / split
            split_lbl = lbl_dir / split
            if not split_img.exists():
                continue
            for img_file in split_img.iterdir():
                if img_file.suffix.lower() in IMAGE_EXTS:
                    lbl_file = split_lbl / (img_file.stem + ".txt")
                    label_lines = []
                    if lbl_file.exists():
                        label_lines = parse_yolo_label(lbl_file)
                    pairs.append((img_file, label_lines, src_name))
    
    elif src_type == "yolo_flat":
        for img_file in src_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTS:
                lbl_file = src_path / (img_file.stem + ".txt")
                label_lines = []
                if lbl_file.exists():
                    label_lines = parse_yolo_label(lbl_file)
                pairs.append((img_file, label_lines, src_name))
    
    elif src_type == "voc_xml":
        for img_file in src_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTS:
                xml_file = src_path / (img_file.stem + ".xml")
                label_lines = []
                if xml_file.exists():
                    # Need image dims for conversion
                    valid, w, h = is_valid_image(img_file)
                    if valid:
                        label_lines = parse_voc_xml(xml_file, w, h)
                pairs.append((img_file, label_lines, src_name))
    
    return pairs


# ============ MAIN ============

def main():
    start_time = datetime.now()
    print(f"=" * 60)
    print(f"STEP 1: Dataset Consolidation & Cleaning")
    print(f"Started: {start_time.isoformat()}")
    print(f"=" * 60)
    
    # Create output structure
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Stats
    stats = {
        "sources_scanned": 0,
        "total_images_found": 0,
        "corrupt_images": 0,
        "empty_labels": 0,
        "duplicate_images": 0,
        "images_without_labels": 0,
        "valid_pairs": 0,
        "boxes_removed_degenerate": 0,
        "boxes_removed_duplicate_iou": 0,
        "xml_converted": 0,
        "final_train": 0,
        "final_val": 0,
        "final_total": 0,
        "total_boxes": 0,
        "per_source": {},
    }
    
    corrupt_list = []
    duplicate_list = []
    
    # Collect all pairs from all sources
    all_pairs = []
    for source in SOURCES:
        print(f"\n📂 Scanning: {source['name']} ({source['path']})")
        stats["sources_scanned"] += 1
        pairs = collect_image_label_pairs(source)
        src_count = len(pairs)
        stats["per_source"][source["name"]] = {"found": src_count, "valid": 0}
        stats["total_images_found"] += src_count
        all_pairs.extend(pairs)
        print(f"   Found {src_count} image files")
    
    print(f"\n📊 Total images found across all sources: {stats['total_images_found']}")
    
    # Phase 1: Validate images and dedup by hash
    print(f"\n🔍 Phase 1: Validating images & deduplication...")
    hash_map = {}  # hash -> (img_path, label_lines, source_name)
    valid_pairs = []
    
    for idx, (img_path, label_lines, src_name) in enumerate(all_pairs):
        if idx % 2000 == 0:
            print(f"   Processing {idx}/{len(all_pairs)}...")
        
        # Validate image
        valid, w, h = is_valid_image(img_path)
        if not valid:
            stats["corrupt_images"] += 1
            corrupt_list.append(str(img_path))
            continue
        
        # Check hash for dedup
        h_val = file_hash(img_path)
        if h_val is None:
            stats["corrupt_images"] += 1
            corrupt_list.append(str(img_path))
            continue
        
        if h_val in hash_map:
            stats["duplicate_images"] += 1
            existing_path, existing_labels, existing_src = hash_map[h_val]
            duplicate_list.append({
                "duplicate": str(img_path),
                "original": str(existing_path),
            })
            # If existing has no labels but this one does, prefer this one
            if len(existing_labels) == 0 and len(label_lines) > 0:
                hash_map[h_val] = (img_path, label_lines, src_name)
            continue
        
        hash_map[h_val] = (img_path, label_lines, src_name)
    
    # Collect unique valid pairs
    for h_val, (img_path, label_lines, src_name) in hash_map.items():
        if len(label_lines) == 0:
            stats["images_without_labels"] += 1
            stats["empty_labels"] += 1
            continue  # Skip images without labels for clean dataset
        valid_pairs.append((img_path, label_lines, src_name))
        stats["per_source"][src_name]["valid"] = stats["per_source"].get(src_name, {}).get("valid", 0) + 1
    
    stats["valid_pairs"] = len(valid_pairs)
    print(f"   Unique valid image+label pairs: {stats['valid_pairs']}")
    print(f"   Corrupt images: {stats['corrupt_images']}")
    print(f"   Duplicate images removed: {stats['duplicate_images']}")
    print(f"   Images without labels (skipped): {stats['images_without_labels']}")
    
    # Phase 2: Shuffle and split
    print(f"\n📦 Phase 2: Splitting into train/val ({TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%})...")
    random.seed(42)
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * TRAIN_RATIO)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"   Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    
    # Phase 3: Copy files
    print(f"\n📁 Phase 3: Copying files to {OUTPUT_DIR}...")
    total_boxes = 0
    
    def copy_pair(pair, split, global_idx):
        nonlocal total_boxes
        img_path, label_lines, src_name = pair
        ext = img_path.suffix.lower()
        new_name = f"drone_{global_idx:06d}"
        
        dst_img = OUTPUT_DIR / "images" / split / f"{new_name}{ext}"
        dst_lbl = OUTPUT_DIR / "labels" / split / f"{new_name}.txt"
        
        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
        
        total_boxes += len(label_lines)
        return new_name
    
    global_idx = 0
    for pair in train_pairs:
        copy_pair(pair, "train", global_idx)
        global_idx += 1
        if global_idx % 2000 == 0:
            print(f"   Copied {global_idx}/{len(valid_pairs)}...")
    
    for pair in val_pairs:
        copy_pair(pair, "val", global_idx)
        global_idx += 1
        if global_idx % 2000 == 0:
            print(f"   Copied {global_idx}/{len(valid_pairs)}...")
    
    stats["final_train"] = len(train_pairs)
    stats["final_val"] = len(val_pairs)
    stats["final_total"] = len(valid_pairs)
    stats["total_boxes"] = total_boxes
    
    # Write data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: 1
names:
  0: drone
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    
    # Write report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_ratio": TRAIN_RATIO,
            "output_dir": str(OUTPUT_DIR),
            "class_names": CLASS_NAMES,
        },
        "summary": stats,
        "corrupt_images": corrupt_list[:100],  # limit report size
        "corrupt_images_total": len(corrupt_list),
        "duplicate_images_total": len(duplicate_list),
        "duplicate_images_sample": duplicate_list[:50],
    }
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'=' * 60}")
    print(f"✅ STEP 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Final dataset:    {stats['final_total']} images ({stats['final_train']} train / {stats['final_val']} val)")
    print(f"Total boxes:      {stats['total_boxes']}")
    print(f"Corrupted:        {stats['corrupt_images']}")
    print(f"Duplicates:       {stats['duplicate_images']}")
    print(f"No labels:        {stats['images_without_labels']}")
    print(f"Time elapsed:     {elapsed:.1f} seconds")
    print(f"Report:           {REPORT_FILE}")
    

if __name__ == "__main__":
    main()
