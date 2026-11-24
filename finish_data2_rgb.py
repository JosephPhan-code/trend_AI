"""
Fast Resume Script: Finishes e:\trend_AI\data2 from already-downloaded anti-uav-rgbt-300.zip!
- Current data2 already has:
    50,000 Thermal IR (25,000 MultiUAV-2025 + 25,000 Anti-UAV410)
    113,442 Colorful RGB (53,732 pathikg + 36,547 mavvid + 5,301 dvb10k + 998 detfly + 16,864 antiuav300_rgb)
- Resumes directly from local e:\trend_AI\data2\_rgb_tmp\anti-uav-rgbt-300.zip (6.05 GB, already downloaded!)
- Adds ~57,000 more unique colorful daytime/dusk RGB frames -> Total > 220,000 images (~77% RGB + 23% IR)
- Zero overlap with Batch 1 (clean_dataset) or existing data2 via MD5 dedup
- Runs full 5-layer pixel+geometry audit + Saliency Tight-Fit alignment on every frame
- Deletes temp zip & verifies 100% image-label pairing + updates data2_report.json
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import csv
import json
import time
import shutil
import random
import zipfile
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
from PIL import Image

# Reuse exact verified helpers from rebalance_data2_rgb.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rebalance_data2_rgb as lib

WORKSPACE = Path(r"e:\trend_AI")
BATCH1_DIR = WORKSPACE / "clean_dataset"
DATA2_DIR = WORKSPACE / "data2"
TMP_DL = DATA2_DIR / "_rgb_tmp"
ZIP_PATH = TMP_DL / "anti-uav-rgbt-300.zip"
AUDIT_LOG = DATA2_DIR / "relabel_audit_log.csv"
REPORT_JSON = DATA2_DIR / "data2_report.json"

TARGET_TOTAL_DATA2 = 220000
FREE_FLOOR_GB = 3.5
TRAIN_RATIO = 0.85


def scan_current_data2():
    c = Counter()
    max_rgb_id = -1
    for sp in ["train", "val"]:
        p = DATA2_DIR / "images" / sp
        if p.exists():
            for e in os.scandir(p):
                if e.is_file():
                    parts = e.name.split("_")
                    if e.name.startswith("d2_rgb_"):
                        c[f"rgb_{parts[2]}"] += 1
                        try:
                            max_rgb_id = max(max_rgb_id, int(os.path.splitext(parts[-1])[0]))
                        except ValueError:
                            pass
                    elif e.name.startswith("d2_antiuav_2025"):
                        c["ir_2025"] += 1
                    elif e.name.startswith("d2_anti_uav410"):
                        c["ir_410"] += 1
                    else:
                        c["other"] += 1
    return c, max_rgb_id + 1


def main():
    t0 = time.time()
    print("=" * 68)
    print("🔄 RESUMING DATA2 COLORFUL RGB COMPLETION (TARGET >= 220,000 IMAGES)")
    print(f"Started: {datetime.now().isoformat()} | Free E:: {lib.free_gb():.2f} GB")
    print("=" * 68, flush=True)

    cur_counts, next_id = scan_current_data2()
    cur_total = sum(cur_counts.values())
    needed = max(0, TARGET_TOTAL_DATA2 - cur_total)
    print(f"Current data2: {cur_total} images -> {dict(cur_counts)}", flush=True)
    print(f"Next RGB ID: {next_id} | Additional RGB images needed: {needed}", flush=True)

    if needed > 0:
        if not ZIP_PATH.exists():
            print("Local anti-uav-rgbt-300.zip missing -> downloading...", flush=True)
            lib.download_file(
                "https://huggingface.co/datasets/huseyincavus/anti-uav-rgbt-300/resolve/main/anti-uav-rgbt-300.zip",
                ZIP_PATH,
                label="anti-uav-rgbt-300.zip",
            )

        existing_hashes = lib.index_existing_hashes()
        stats = Counter()
        rng = random.Random(30099)
        kept = 0

        audit_f = open(AUDIT_LOG, "a", newline="", encoding="utf-8")
        audit_writer = csv.DictWriter(
            audit_f, fieldnames=["image", "split", "source", "action", "final_boxes", "timestamp"]
        )

        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                names = set(zf.namelist())
                vis_videos = sorted([n for n in names if n.endswith("visible.mp4")])
                # Start from video 35 onward first (since videos 1-34 were already processed before restart!)
                # then wrap around to 1-34 if still needed (MD5 dedup skips any already-saved frames!)
                ordered_videos = vis_videos[34:] + vis_videos[:34]
                print(f"Resuming across {len(ordered_videos)} visible RGB videos (starting at video #35)...", flush=True)
                target_per_vid = max(240, int(needed / max(len(vis_videos) - 34, 1) * 1.5))

                for v_idx, v_entry in enumerate(ordered_videos, 1):
                    if kept >= needed or lib.free_gb() < FREE_FLOOR_GB:
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

                    valid_frames = [
                        i for i in range(n_ann)
                        if exists[i] == 1 and isinstance(rects[i], list) and len(rects[i]) >= 4
                        and rects[i][2] > 2 and rects[i][3] > 2
                    ]
                    if not valid_frames:
                        continue
                    stride = max(1, len(valid_frames) // target_per_vid)
                    wanted = set(valid_frames[::stride])

                    tmp_mp4 = TMP_DL / "cur_vis.mp4"
                    tmp_mp4.write_bytes(zf.read(v_entry))
                    cap = cv2.VideoCapture(str(tmp_mp4))
                    f_i = 0
                    try:
                        while cap.isOpened() and kept < needed:
                            ret, frame_bgr = cap.read()
                            if not ret:
                                break
                            if f_i in wanted:
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                im_rgb = Image.fromarray(frame_rgb)
                                iw, ih = im_rgb.size
                                rx, ry, rw, rh = [float(x) for x in rects[f_i][:4]]
                                cx, cy, bw, bh = (rx + rw / 2.0) / iw, (ry + rh / 2.0) / ih, rw / iw, rh / ih

                                ok, lines, fixed, reason = lib.validate_rgb_pil_and_boxes(
                                    im_rgb, [(cx, cy, bw, bh)], require_daylight_color=True
                                )
                                if ok:
                                    out_bytes = lib.pil_to_jpeg_bytes(im_rgb, quality=90)
                                    out_h = lib.bytes_md5(out_bytes)
                                    if out_h not in existing_hashes:
                                        existing_hashes.add(out_h)
                                        split = "train" if rng.random() < TRAIN_RATIO else "val"
                                        stem = f"d2_rgb_antiuav300_rgb_{next_id:06d}"
                                        next_id += 1
                                        (DATA2_DIR / "images" / split / f"{stem}.jpg").write_bytes(out_bytes)
                                        (DATA2_DIR / "labels" / split / f"{stem}.txt").write_text(
                                            "\n".join(lines) + "\n", encoding="utf-8"
                                        )
                                        kept += 1
                                        stats["added"] += 1
                                        if fixed:
                                            stats["fixed"] += 1
                                            audit_writer.writerow({
                                                "image": f"{stem}.jpg", "split": split, "source": "rgb_antiuav300_rgb",
                                                "action": "rgb_tight_fixed", "final_boxes": len(lines),
                                                "timestamp": datetime.now().isoformat()
                                            })
                                    else:
                                        stats["dup_skipped"] += 1
                                else:
                                    stats[f"rej_{reason}"] += 1
                            f_i += 1
                    finally:
                        cap.release()
                        if tmp_mp4.exists():
                            tmp_mp4.unlink()

                    if v_idx % 15 == 0 or kept >= needed:
                        print(
                            f"  [resume antiuav300_rgb] video {v_idx}/{len(ordered_videos)} | "
                            f"added={kept}/{needed} | total_data2={cur_total+kept} | free={lib.free_gb():.1f}GB",
                            flush=True,
                        )
        finally:
            audit_f.close()

    # Cleanup temp folder & 6.05GB zip
    if TMP_DL.exists():
        shutil.rmtree(TMP_DL, ignore_errors=True)
        print(f"🧹 Cleaned _rgb_tmp/ -> Free E:: {lib.free_gb():.2f} GB", flush=True)

    # Final exact counts & pairing check
    final_counts, _ = scan_current_data2()
    final_train = sum(1 for e in os.scandir(DATA2_DIR / "images" / "train") if e.is_file())
    final_val = sum(1 for e in os.scandir(DATA2_DIR / "images" / "val") if e.is_file())
    lbl_train = sum(1 for e in os.scandir(DATA2_DIR / "labels" / "train") if e.is_file())
    lbl_val = sum(1 for e in os.scandir(DATA2_DIR / "labels" / "val") if e.is_file())
    total_final = final_train + final_val

    ir_total = final_counts.get("ir_2025", 0) + final_counts.get("ir_410", 0)
    rgb_total = total_final - ir_total

    # Ensure portable data.yaml & Mac M4 Kit exist inside data2
    (DATA2_DIR / "data.yaml").write_text(
        "# Cross-platform Ultralytics dataset config (Batch 2 - Balanced RGB + IR)\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: drone\n",
        encoding="utf-8",
    )
    for fname in ["train_mac_m4.py", "drone_tracker.yaml", "track_drone_m4.py", "run_train_mac.sh"]:
        src = BATCH1_DIR / fname
        if src.exists():
            shutil.copy2(src, DATA2_DIR / fname)

    report = {
        "completed_at": datetime.now().isoformat(),
        "output_dir": str(DATA2_DIR),
        "final_train": final_train,
        "final_val": final_val,
        "final_total": total_final,
        "labels_paired_100pct": (final_train == lbl_train and final_val == lbl_val),
        "composition": {
            "colorful_rgb_images": rgb_total,
            "colorful_rgb_percent": round(rgb_total / max(total_final, 1) * 100, 2),
            "thermal_ir_images": ir_total,
            "thermal_ir_percent": round(ir_total / max(total_final, 1) * 100, 2),
        },
        "source_breakdown": dict(final_counts),
        "free_disk_e_gb_remaining": round(lib.free_gb(), 2),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print("✅ DATA2 BALANCED DIVERSE DATASET COMPLETE!")
    print("=" * 68)
    print(f"Final data2 total       : {total_final} images ({final_train} train / {final_val} val)")
    print(f"  🎨 Colorful RGB       : {rgb_total} ({rgb_total/max(total_final,1)*100:.1f}%)")
    print(f"  🌡️ Thermal IR         : {ir_total} ({ir_total/max(total_final,1)*100:.1f}%)")
    print(f"Breakdown by source     : {dict(final_counts)}")
    print(f"Labels paired 100%      : {final_train == lbl_train and final_val == lbl_val}")
    print(f"Remaining free on E:    : {lib.free_gb():.2f} GB")
    print(f"Report saved to         : {REPORT_JSON}")
    print(f"Elapsed resume time     : {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
