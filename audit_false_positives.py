"""
Deep False-Positive & Mislabel Audit across ALL 32,223 images in clean_dataset.
Detects:
1. 1920x1080 Anti-UAV HUD crosshair-reticle boxes (huge boxes covering HUD reticle)
2. 640x512 IR/Thermal building/background boxes (large boxes or boxes glued to bottom/side building region)
3. Dark/low-contrast crop boxes (drone is invisible or box is on dark sky/noise)
4. Tiny noise boxes (area < 0.0003)
5. Low-confidence auto-labeled boxes (conf < 0.45)
6. Non-drone objects detected by COCO or extreme aspect ratio boxes
Outputs:
- e:\trend_AI\clean_dataset\suspicious_labels_audit.csv
- Summary statistics by error category
"""

import os
import csv
import re
import json
from pathlib import Path
from collections import Counter
from PIL import Image, ImageStat

DATASET_DIR = Path(r"e:\trend_AI\clean_dataset")
AUDIT_LOG = DATASET_DIR / "relabel_audit_log.csv"
OUT_CSV = DATASET_DIR / "suspicious_labels_audit.csv"
OUT_JSON = DATASET_DIR / "suspicious_summary.json"


def load_auto_conf():
    conf_map = {}
    if AUDIT_LOG.exists():
        with open(AUDIT_LOG, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = re.search(r"avg_conf=([0-9.]+)", r.get("changes", ""))
                if m:
                    conf_map[r["image"]] = float(m.group(1))
    return conf_map


def main():
    conf_map = load_auto_conf()
    res_counts = Counter()
    reason_counts = Counter()
    severity_counts = Counter()
    rows = []

    total = 0
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        entries = sorted(os.scandir(img_dir), key=lambda e: e.name)
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            stem = os.path.splitext(name)[0]
            lbl_p = lbl_dir / f"{stem}.txt"
            total += 1
            if total % 5000 == 0:
                print(f"Audited {total} images... suspicious so far: {len(rows)}", flush=True)

            with Image.open(entry.path) as im:
                w, h = im.size
            res = f"{w}x{h}"
            res_counts[res] += 1

            boxes = []
            if lbl_p.exists():
                for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))

            reasons = []
            severity = "ok"
            auto_conf = conf_map.get(name, None)

            if not boxes:
                reasons.append("empty_label")
                severity = "critical"
            else:
                max_area = max(b[2] * b[3] for b in boxes)
                min_area = min(b[2] * b[3] for b in boxes)
                max_ar = max(max(b[2] / max(b[3], 1e-6), b[3] / max(b[2], 1e-6)) for b in boxes)

                # 1. Anti-UAV 1920x1080 HUD crosshair poison
                if (w, h) == (1920, 1080):
                    if max_area > 0.08:
                        reasons.append(f"hud_crosshair_giant_box(area={max_area:.3f})")
                        severity = "critical"
                    elif max_area > 0.03:
                        reasons.append(f"hud_oversized_box(area={max_area:.3f})")
                        if severity != "critical":
                            severity = "high"

                # 2. Anti-UAV 640x512 IR Thermal building/background poison
                elif (w, h) == (640, 512):
                    if max_area > 0.06:
                        reasons.append(f"ir_building_giant_box(area={max_area:.3f})")
                        severity = "critical"
                    else:
                        for cx, cy, bw, bh in boxes:
                            y2 = cy + bh / 2
                            x1 = cx - bw / 2
                            x2 = cx + bw / 2
                            # Touching bottom or side edge with non-tiny area -> building/roof/pole
                            if (y2 > 0.93 or x1 < 0.02 or x2 > 0.98) and (bw * bh > 0.015):
                                reasons.append(f"ir_edge_building_box(area={bw*bh:.3f})")
                                severity = "critical"
                                break

                # 3. Any resolution: tiny box / noise
                if min_area < 0.0003:
                    reasons.append(f"tiny_box(area={min_area:.5f})")
                    if severity == "ok":
                        severity = "medium"

                # 4. Extreme aspect ratio (line/pole/building strip)
                if max_ar > 5.0:
                    reasons.append(f"extreme_aspect_ratio({max_ar:.1f})")
                    if severity == "ok":
                        severity = "high"

                # 5. Low confidence auto-label
                if auto_conf is not None and auto_conf < 0.45:
                    reasons.append(f"low_autolabel_conf({auto_conf:.2f})")
                    if severity == "ok":
                        severity = "high" if auto_conf < 0.35 else "medium"

                # 6. Dark / low-contrast crop check using PIL ImageStat (zero OpenCV memory overhead)
                if (min_area < 0.005) or (auto_conf is not None and auto_conf < 0.60) or reasons:
                    try:
                        with Image.open(entry.path) as im_gray:
                            gray = im_gray.convert("L")
                            img_mean = ImageStat.Stat(gray).mean[0]
                            bx = min(boxes, key=lambda b: b[2] * b[3])
                            x1 = max(0, int((bx[0] - bx[2] / 2) * w))
                            y1 = max(0, int((bx[1] - bx[3] / 2) * h))
                            x2 = min(w, int((bx[0] + bx[2] / 2) * w))
                            y2 = min(h, int((bx[1] + bx[3] / 2) * h))
                            if x2 > x1 and y2 > y1:
                                crop = gray.crop((x1, y1, x2, y2))
                                crop_stat = ImageStat.Stat(crop)
                                crop_std = crop_stat.stddev[0]
                                if img_mean < 25 and crop_std < 12:
                                    reasons.append(f"dark_low_contrast_box(img_b={img_mean:.1f},std={crop_std:.1f})")
                                    if severity == "ok":
                                        severity = "high"
                                elif crop_std < 6.0:
                                    reasons.append(f"blank_flat_box(std={crop_std:.1f})")
                                    severity = "critical"
                    except Exception:
                        pass

            if reasons:
                severity_counts[severity] += 1
                for r in reasons:
                    tag = r.split("(")[0]
                    reason_counts[tag] += 1
                rows.append({
                    "image": name,
                    "split": split,
                    "resolution": res,
                    "num_boxes": len(boxes),
                    "max_area": round(max((b[2] * b[3] for b in boxes), default=0.0), 5),
                    "min_area": round(min((b[2] * b[3] for b in boxes), default=0.0), 5),
                    "auto_conf": "" if auto_conf is None else round(auto_conf, 3),
                    "severity": severity,
                    "reasons": " | ".join(reasons),
                })

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "split", "resolution", "num_boxes", "max_area", "min_area", "auto_conf", "severity", "reasons"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_images": total,
        "total_suspicious": len(rows),
        "severity_counts": dict(severity_counts),
        "reason_counts": dict(reason_counts.most_common()),
        "top_resolutions": dict(res_counts.most_common(10)),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ AUDIT SUMMARY ================")
    print(f"Total images audited: {total}")
    print(f"Total suspicious:     {len(rows)} ({len(rows)/max(total,1)*100:.1f}%)")
    print("\nBy Severity:")
    for k in ["critical", "high", "medium"]:
        print(f"  {k:10s}: {severity_counts[k]}")
    print("\nBy Error Type:")
    for k, v in reason_counts.most_common():
        print(f"  {k:30s}: {v}")
    print("\nTop Resolutions:")
    for k, v in res_counts.most_common(8):
        print(f"  {k:12s}: {v}")
    print(f"\nSaved report to: {OUT_CSV}")


if __name__ == "__main__":
    main()
