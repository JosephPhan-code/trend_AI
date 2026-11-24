"""
Step 2: Label Quality Audit & Fix
- Validates all YOLO labels in clean_dataset
- Fixes: clamp coordinates, remove degenerate boxes, remove high-IoU duplicates
- Logs all changes to relabel_audit_log.csv
- Backs up original labels before modifying
"""

import os
import sys
import csv
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATASET_DIR = Path(r"e:\trend_AI\clean_dataset")
LABELS_DIR = DATASET_DIR / "labels"
BACKUP_DIR = DATASET_DIR / "labels_backup"
AUDIT_LOG = DATASET_DIR / "relabel_audit_log.csv"

# Thresholds
MIN_BOX_AREA = 0.0001      # Minimum normalized area
MAX_BOX_AREA = 0.85         # Maximum normalized area
MIN_BOX_DIM = 0.003         # Minimum width or height
IOU_THRESHOLD = 0.75        # IoU threshold for duplicate removal
EDGE_TOLERANCE = 0.001      # Tolerance for edge clamping

def iou(box1, box2):
    """Calculate IoU between two boxes (cx, cy, w, h)."""
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


def parse_yolo_line(line):
    """Parse a YOLO label line into (class_id, cx, cy, w, h)."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        return (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except (ValueError, IndexError):
        return None


def format_yolo_line(box):
    """Format a box tuple back to YOLO line."""
    return f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}"


def audit_and_fix_label(label_path):
    """
    Audit and fix a single label file.
    Returns (changes_list, original_boxes, final_boxes, was_modified)
    """
    changes = []
    
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Parse original boxes
    original_boxes = []
    for line in raw_lines:
        box = parse_yolo_line(line)
        if box is not None:
            original_boxes.append(box)
        else:
            changes.append(f"Removed unparseable line: '{line[:50]}'")
    
    if not original_boxes:
        return changes, 0, 0, len(changes) > 0
    
    fixed_boxes = []
    
    for idx, box in enumerate(original_boxes):
        cls, cx, cy, w, h = box
        modified = False
        box_changes = []
        
        # 1. Fix class ID (all should be 0 = drone)
        if cls != 0:
            box_changes.append(f"Box {idx}: class {cls}->0")
            cls = 0
            modified = True
        
        # 2. Clamp coordinates to [0, 1]
        new_cx = max(0, min(1, cx))
        new_cy = max(0, min(1, cy))
        new_w = max(0, min(1, w))
        new_h = max(0, min(1, h))
        
        # Ensure box doesn't exceed image boundary
        half_w = new_w / 2
        half_h = new_h / 2
        
        if new_cx - half_w < 0:
            new_w = new_cx * 2
            box_changes.append(f"Box {idx}: clamped left edge")
            modified = True
        if new_cx + half_w > 1:
            new_w = (1 - new_cx) * 2
            box_changes.append(f"Box {idx}: clamped right edge")
            modified = True
        if new_cy - half_h < 0:
            new_h = new_cy * 2
            box_changes.append(f"Box {idx}: clamped top edge")
            modified = True
        if new_cy + half_h > 1:
            new_h = (1 - new_cy) * 2
            box_changes.append(f"Box {idx}: clamped bottom edge")
            modified = True
        
        if abs(new_cx - cx) > EDGE_TOLERANCE or abs(new_cy - cy) > EDGE_TOLERANCE:
            box_changes.append(f"Box {idx}: center shifted ({cx:.4f},{cy:.4f})->({new_cx:.4f},{new_cy:.4f})")
            modified = True
        
        # 3. Check area
        area = new_w * new_h
        if area < MIN_BOX_AREA:
            box_changes.append(f"Box {idx}: REMOVED (area={area:.6f} < {MIN_BOX_AREA})")
            changes.extend(box_changes)
            continue
        if area > MAX_BOX_AREA:
            box_changes.append(f"Box {idx}: REMOVED (area={area:.4f} > {MAX_BOX_AREA})")
            changes.extend(box_changes)
            continue
        
        # 4. Check minimum dimensions
        if new_w < MIN_BOX_DIM or new_h < MIN_BOX_DIM:
            box_changes.append(f"Box {idx}: REMOVED (dim too small w={new_w:.4f}, h={new_h:.4f})")
            changes.extend(box_changes)
            continue
        
        fixed_boxes.append((cls, new_cx, new_cy, new_w, new_h))
        changes.extend(box_changes)
    
    # 5. Remove duplicate overlapping boxes
    if len(fixed_boxes) > 1:
        keep = [True] * len(fixed_boxes)
        for i in range(len(fixed_boxes)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(fixed_boxes)):
                if not keep[j]:
                    continue
                if fixed_boxes[i][0] == fixed_boxes[j][0]:
                    box_i = fixed_boxes[i][1:]
                    box_j = fixed_boxes[j][1:]
                    iou_val = iou(box_i, box_j)
                    if iou_val > IOU_THRESHOLD:
                        area_i = box_i[2] * box_i[3]
                        area_j = box_j[2] * box_j[3]
                        if area_j > area_i:
                            keep[i] = False
                            changes.append(f"Box {i}: REMOVED (IoU={iou_val:.3f} with box {j}, keeping larger)")
                        else:
                            keep[j] = False
                            changes.append(f"Box {j}: REMOVED (IoU={iou_val:.3f} with box {i}, keeping larger)")
        
        fixed_boxes = [fixed_boxes[i] for i in range(len(fixed_boxes)) if keep[i]]
    
    was_modified = len(changes) > 0 or len(fixed_boxes) != len(original_boxes)
    
    return changes, len(original_boxes), len(fixed_boxes), was_modified, fixed_boxes


def main():
    start_time = datetime.now()
    print(f"=" * 60)
    print(f"STEP 2: Label Quality Audit & Fix")
    print(f"Started: {start_time.isoformat()}")
    print(f"=" * 60)
    
    if not DATASET_DIR.exists():
        print("ERROR: clean_dataset not found. Run Step 1 first.")
        sys.exit(1)
    
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Stats
    total_files = 0
    modified_files = 0
    total_boxes_before = 0
    total_boxes_after = 0
    total_changes = 0
    files_emptied = 0  # Files that ended up with 0 boxes
    
    # Audit log
    audit_rows = []
    
    for split in ["train", "val"]:
        split_dir = LABELS_DIR / split
        backup_split = BACKUP_DIR / split
        backup_split.mkdir(parents=True, exist_ok=True)
        
        if not split_dir.exists():
            continue
        
        label_files = sorted(split_dir.glob("*.txt"))
        print(f"\n📋 Processing {split}: {len(label_files)} label files")
        
        for idx, label_path in enumerate(label_files):
            if idx % 2000 == 0 and idx > 0:
                print(f"   Processed {idx}/{len(label_files)}...")
            
            total_files += 1
            
            result = audit_and_fix_label(label_path)
            changes, orig_count, final_count, was_modified, fixed_boxes = result
            
            total_boxes_before += orig_count
            total_boxes_after += final_count
            
            if was_modified:
                modified_files += 1
                total_changes += len(changes)
                
                # Backup original
                shutil.copy2(label_path, backup_split / label_path.name)
                
                # Write fixed label
                with open(label_path, "w", encoding="utf-8") as f:
                    for box in fixed_boxes:
                        f.write(format_yolo_line(box) + "\n")
                
                if final_count == 0:
                    files_emptied += 1
                
                # Log to audit
                action = "label_fixed"
                if final_count == 0 and orig_count > 0:
                    action = "all_boxes_removed"
                
                audit_rows.append({
                    "image": label_path.stem + ".jpg",
                    "split": split,
                    "action": action,
                    "changes": " | ".join(changes),
                    "original_boxes": orig_count,
                    "final_boxes": final_count,
                    "timestamp": datetime.now().isoformat(),
                })
    
    # Write audit log
    print(f"\n📝 Writing audit log to {AUDIT_LOG}")
    with open(AUDIT_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "action", "changes", "original_boxes", "final_boxes", "timestamp"])
        writer.writeheader()
        writer.writerows(audit_rows)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'=' * 60}")
    print(f"✅ STEP 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total label files:    {total_files}")
    print(f"Modified files:       {modified_files}")
    print(f"Total changes:        {total_changes}")
    print(f"Boxes before:         {total_boxes_before}")
    print(f"Boxes after:          {total_boxes_after}")
    print(f"Boxes removed:        {total_boxes_before - total_boxes_after}")
    print(f"Files emptied:        {files_emptied}")
    print(f"Backup dir:           {BACKUP_DIR}")
    print(f"Audit log:            {AUDIT_LOG}")
    print(f"Time elapsed:         {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
