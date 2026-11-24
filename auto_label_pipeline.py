"""
Auto-Label Pipeline:
1. Train YOLOv8n on existing clean_dataset (quick 30 epochs)
2. Run inference on all unlabeled images from Data_tong_hop
3. Keep images with confident detections (conf >= 0.25)
4. Dedup against existing clean_dataset
5. Merge new labeled images into clean_dataset
6. Update audit log
"""

import os
import sys
import json
import hashlib
import shutil
import csv
import random
from pathlib import Path
from datetime import datetime

# ============ CONFIG ============
WORKSPACE = Path(r"e:\trend_AI")
CLEAN_DATASET = WORKSPACE / "clean_dataset"
DATA_TONG_HOP = WORKSPACE / "Data_tong_hop"
TRAIN_DIR = WORKSPACE / "autolabel_train"
CONF_THRESHOLD = 0.25  # Minimum confidence for auto-labels
TRAIN_EPOCHS = 30
TRAIN_IMGSZ = 640
TRAIN_BATCH = 16
TRAIN_RATIO = 0.85
AUDIT_LOG = CLEAN_DATASET / "relabel_audit_log.csv"
REPORT_FILE = CLEAN_DATASET / "consolidation_report.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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

def build_existing_hash_index():
    """Build hash set of all images already in clean_dataset."""
    hashes = set()
    count = 0
    for split in ["train", "val"]:
        img_dir = CLEAN_DATASET / "images" / split
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                h = file_hash(f)
                if h:
                    hashes.add(h)
                    count += 1
                if count % 2000 == 0:
                    print(f"   Indexed {count}...")
    print(f"   Total existing: {count} images indexed")
    return hashes

def find_next_index():
    """Find the next drone_XXXXXX index."""
    max_idx = 0
    for split in ["train", "val"]:
        d = CLEAN_DATASET / "images" / split
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.stem.startswith("drone_"):
                try:
                    idx = int(f.stem.split("_")[1])
                    max_idx = max(max_idx, idx)
                except:
                    pass
    return max_idx + 1

def collect_unlabeled_images():
    """Collect images from Data_tong_hop that have empty or missing labels."""
    unlabeled = []
    for split in ["train", "val"]:
        img_dir = DATA_TONG_HOP / "images" / split
        lbl_dir = DATA_TONG_HOP / "labels" / split
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_file = lbl_dir / (f.stem + ".txt")
            is_empty = True
            if lbl_file.exists():
                content = lbl_file.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    # Has actual label content - not empty
                    is_empty = False
            if is_empty:
                unlabeled.append(f)
    return unlabeled

def main():
    start_time = datetime.now()
    print("=" * 60)
    print("AUTO-LABEL PIPELINE")
    print(f"Started: {start_time.isoformat()}")
    print("=" * 60)

    # ========================================
    # PHASE 1: Train YOLOv8n on existing data
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training YOLOv8n on existing clean_dataset")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    # Use the existing data.yaml
    data_yaml = CLEAN_DATASET / "data.yaml"
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {TRAIN_EPOCHS} | ImgSize: {TRAIN_IMGSZ} | Batch: {TRAIN_BATCH}")
    
    model = YOLO("yolov8n.pt")  # Start from pretrained COCO weights
    
    results = model.train(
        data=str(data_yaml),
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=TRAIN_BATCH,
        project=str(TRAIN_DIR),
        name="drone_detector",
        exist_ok=True,
        verbose=True,
        device=0,  # GPU
        workers=4,
        patience=10,
        save=True,
        plots=False,
    )
    
    # Load best model
    best_model_path = TRAIN_DIR / "drone_detector" / "weights" / "best.pt"
    if not best_model_path.exists():
        # Fallback to last
        best_model_path = TRAIN_DIR / "drone_detector" / "weights" / "last.pt"
    
    print(f"\n✅ Training complete! Best model: {best_model_path}")
    model = YOLO(str(best_model_path))
    
    # ========================================
    # PHASE 2: Collect unlabeled images & dedup
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Collecting unlabeled images & dedup")
    print("=" * 60)
    
    print("Building hash index of existing clean_dataset...")
    existing_hashes = build_existing_hash_index()
    
    print("Collecting unlabeled images from Data_tong_hop...")
    unlabeled_images = collect_unlabeled_images()
    print(f"Found {len(unlabeled_images)} unlabeled images")
    
    # Dedup against existing
    print("Deduplicating...")
    unique_unlabeled = []
    seen_hashes = set()
    dup_count = 0
    
    for idx, img_path in enumerate(unlabeled_images):
        if idx % 3000 == 0 and idx > 0:
            print(f"   Hashing {idx}/{len(unlabeled_images)}...")
        h = file_hash(img_path)
        if h is None:
            continue
        if h in existing_hashes or h in seen_hashes:
            dup_count += 1
            continue
        seen_hashes.add(h)
        unique_unlabeled.append(img_path)
    
    print(f"Unique unlabeled: {len(unique_unlabeled)} (removed {dup_count} duplicates)")
    
    if not unique_unlabeled:
        print("⚠️ No unique unlabeled images to process!")
        return
    
    # ========================================
    # PHASE 3: Run inference / auto-label
    # ========================================
    print("\n" + "=" * 60)
    print(f"PHASE 3: Auto-labeling {len(unique_unlabeled)} images (conf >= {CONF_THRESHOLD})")
    print("=" * 60)
    
    labeled_pairs = []  # (img_path, label_lines, num_boxes, avg_conf)
    no_detection = 0
    
    # Process in batches for efficiency
    BATCH_SIZE = 32
    for batch_start in range(0, len(unique_unlabeled), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(unique_unlabeled))
        batch_paths = unique_unlabeled[batch_start:batch_end]
        
        if batch_start % (BATCH_SIZE * 50) == 0:
            print(f"   Inferencing {batch_start}/{len(unique_unlabeled)}... "
                  f"(labeled so far: {len(labeled_pairs)}, no detection: {no_detection})")
        
        try:
            batch_results = model.predict(
                source=[str(p) for p in batch_paths],
                conf=CONF_THRESHOLD,
                iou=0.5,
                imgsz=TRAIN_IMGSZ,
                verbose=False,
                device=0,
            )
            
            for img_path, result in zip(batch_paths, batch_results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    no_detection += 1
                    continue
                
                # Convert to YOLO format
                label_lines = []
                confs = []
                img_h, img_w = result.orig_shape
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # xywhn = normalized center x, y, w, h
                    xywhn = box.xywhn[0]
                    cx = float(xywhn[0])
                    cy = float(xywhn[1])
                    bw = float(xywhn[2])
                    bh = float(xywhn[3])
                    
                    # Clamp
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    bw = max(0, min(1, bw))
                    bh = max(0, min(1, bh))
                    
                    # Skip degenerate
                    if bw * bh < 0.0001 or bw * bh > 0.85:
                        continue
                    
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    confs.append(conf)
                
                if label_lines:
                    avg_conf = sum(confs) / len(confs)
                    labeled_pairs.append((img_path, label_lines, len(label_lines), avg_conf))
                else:
                    no_detection += 1
        except Exception as e:
            print(f"   ⚠️ Batch error at {batch_start}: {e}")
            continue
    
    print(f"\n📊 Auto-label results:")
    print(f"   Images with detections: {len(labeled_pairs)}")
    print(f"   Images without detections: {no_detection}")
    
    if not labeled_pairs:
        print("⚠️ No detections found in unlabeled images!")
        return
    
    # ========================================
    # PHASE 4: Merge into clean_dataset
    # ========================================
    print("\n" + "=" * 60)
    print(f"PHASE 4: Merging {len(labeled_pairs)} auto-labeled images into clean_dataset")
    print("=" * 60)
    
    next_idx = find_next_index()
    print(f"Starting index: drone_{next_idx:06d}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled_pairs)
    split_idx = int(len(labeled_pairs) * TRAIN_RATIO)
    train_new = labeled_pairs[:split_idx]
    val_new = labeled_pairs[split_idx:]
    print(f"New train: {len(train_new)} | New val: {len(val_new)}")
    
    # Copy files
    audit_rows = []
    total_new_boxes = 0
    
    def copy_pair(img_path, label_lines, num_boxes, avg_conf, split, idx):
        nonlocal total_new_boxes
        ext = img_path.suffix.lower()
        new_name = f"drone_{idx:06d}"
        
        dst_img = CLEAN_DATASET / "images" / split / f"{new_name}{ext}"
        dst_lbl = CLEAN_DATASET / "labels" / split / f"{new_name}.txt"
        
        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
        
        total_new_boxes += num_boxes
        
        audit_rows.append({
            "image": f"{new_name}{ext}",
            "split": split,
            "action": "auto_labeled",
            "changes": f"Added {num_boxes} box(es) via YOLOv8n (avg_conf={avg_conf:.3f})",
            "original_boxes": 0,
            "final_boxes": num_boxes,
            "timestamp": datetime.now().isoformat(),
        })
        return new_name
    
    global_idx = next_idx
    for img_path, label_lines, num_boxes, avg_conf in train_new:
        copy_pair(img_path, label_lines, num_boxes, avg_conf, "train", global_idx)
        global_idx += 1
        if (global_idx - next_idx) % 2000 == 0:
            print(f"   Copied {global_idx - next_idx}/{len(labeled_pairs)}...")
    
    for img_path, label_lines, num_boxes, avg_conf in val_new:
        copy_pair(img_path, label_lines, num_boxes, avg_conf, "val", global_idx)
        global_idx += 1
        if (global_idx - next_idx) % 2000 == 0:
            print(f"   Copied {global_idx - next_idx}/{len(labeled_pairs)}...")
    
    # Append audit log
    print(f"\n📝 Appending {len(audit_rows)} entries to audit log...")
    with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "action", "changes", "original_boxes", "final_boxes", "timestamp"])
        writer.writerows(audit_rows)
    
    # Update report
    final_train = len(list((CLEAN_DATASET / "images" / "train").glob("*")))
    final_val = len(list((CLEAN_DATASET / "images" / "val").glob("*")))
    
    report = {}
    if REPORT_FILE.exists():
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            report = json.load(f)
    
    report["auto_label_merge"] = {
        "timestamp": datetime.now().isoformat(),
        "source": "Data_tong_hop (unlabeled)",
        "model_used": str(best_model_path),
        "conf_threshold": CONF_THRESHOLD,
        "train_epochs": TRAIN_EPOCHS,
        "unlabeled_scanned": len(unique_unlabeled),
        "images_with_detections": len(labeled_pairs),
        "images_no_detection": no_detection,
        "new_boxes_added": total_new_boxes,
        "duplicates_skipped": dup_count,
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
    print(f"✅ AUTO-LABEL PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model trained:        {best_model_path}")
    print(f"Unlabeled scanned:    {len(unique_unlabeled)}")
    print(f"New images labeled:   {len(labeled_pairs)}")
    print(f"New boxes added:      {total_new_boxes}")
    print(f"No detection (skip):  {no_detection}")
    print(f"")
    print(f"📊 FINAL DATASET:")
    print(f"   Train: {final_train}")
    print(f"   Val:   {final_val}")
    print(f"   Total: {final_train + final_val}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Best model saved: {best_model_path}")


if __name__ == "__main__":
    main()
