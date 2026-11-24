"""
Comprehensive Multi-Signal Drone Label Error Scanner
Scans 100% of e:\\trend_AI\\clean_dataset and e:\\trend_AI\\data2:
  1. Per-file MD5 + 64-bit dHash (exact + near-duplicate cross-dataset leakage audit)
  2. Full label geometry checks (empty_non_hn, hn_has_box, bad_syntax, wrong_cls, oob,
     dup_iou, nested_dup, split_pair + night brightness, edge_sliver, huge, extreme_ar, tiny)
  3. OpenCV pixel-crop checks (ghost_blank, white_bg_truncated, flat_2d_graphic, pole_below)
  4. RTX 2050 FP16 dual-model GPU cross-check (clean_drone_v2 + yolov8n COCO) on all
     high-risk sources + all Hard Negatives + CPU-flagged anomalies.
"""

import os
import sys
import json
import time
import queue
import hashlib
import argparse
import threading
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

CLEAN_MODEL_PATH = r"e:\trend_AI\clean_relabel_train\clean_drone_v2\weights\best.pt"
COCO_MODEL_PATH = r"yolov8n.pt"
OUT_DIR = Path(
    r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\scratch\audit_full"
)

# COCO classes that contradict a 'drone' GT label when overlapping with high IoU
# Note: 'airplane' & 'kite' are tracked separately because real quadcopters often trigger them.
NON_DRONE_COCO = {
    "bird", "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "cat", "dog",
    "horse", "sheep", "cow", "chair", "couch", "potted plant", "bed", "dining table",
    "tv", "laptop", "cell phone", "book", "clock", "scissors", "tie", "backpack",
    "umbrella", "handbag", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
}


def src_of(dataset_name: str, stem: str) -> str:
    if dataset_name == "data2":
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
            return "d2_hn_" + stem.split("_")[2]
        return "d2_other"
    else:
        if stem.startswith("b1_hn_"):
            return "b1_hn_" + stem.split("_")[2]
        if stem.startswith("drone_"):
            idx = int(stem.split("_")[1])
            if idx <= 9312:
                return "c1_orig"
            if idx <= 32222:
                return "c1_tonghop"
            return "c1_datatrain"
        return "c1_other"


def needs_gpu(src: str, stem: str) -> str:
    """Returns 'dual', 'hn_only', or 'cpu_only'."""
    if "_hn_" in stem:
        return "hn_only"
    if src in ("d2_ir_2025", "d2_ir_410"):
        return "cpu_only"
    if src in (
        "c1_orig", "c1_tonghop", "c1_datatrain", "c1_other",
        "d2_rgb_dvb10k", "d2_rgb_detfly", "d2_rgb_pathikg", "d2_other",
    ):
        return "dual"
    # d2_rgb_antiuav300 tail added by turbo_finish_data2 ([210:] videos, >= 145000 approx)
    if src == "d2_rgb_antiuav300":
        try:
            num = int(stem.rsplit("_", 1)[-1])
            if num >= 140000:
                return "dual"
        except ValueError:
            pass
    return "cpu_only"


def compute_dhash64(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    val = 0
    for b in diff.ravel():
        val = (val << 1) | int(b)
    return f"{val:016x}"


def iou_xyxy(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def inter_over_min(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


def letterbox_640(img_bgr: np.ndarray, size: int = 640):
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas, (nw, nh, pad_x, pad_y)


def inspect_image_cpu(task):
    """Reads image + label once, computes MD5, dHash64, pixel/geometry flags, and optional 640 canvas."""
    ds_name, ds_root, sp, stem, img_path, lbl_path = task
    src = src_of(ds_name, stem)
    is_hn = "_hn_" in stem
    gpu_mode = needs_gpu(src, stem)

    flags = []
    boxes = []
    bad_syntax = []

    try:
        with open(lbl_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = [l.strip() for l in f if l.strip()]
    except OSError as e:
        return {"ds": ds_name, "sp": sp, "stem": stem, "src": src, "flags": [f"lbl_read_err:{e}"], "gpu_mode": "skip"}

    if not raw_lines:
        if not is_hn:
            flags.append("empty_non_hn")
    else:
        if is_hn:
            flags.append(f"hn_has_box:{len(raw_lines)}")
        for l in raw_lines:
            ps = l.split()
            if len(ps) != 5:
                bad_syntax.append(l)
                continue
            try:
                cls = int(float(ps[0]))
                cx, cy, bw, bh = map(float, ps[1:])
            except ValueError:
                bad_syntax.append(l)
                continue
            if cls != 0:
                flags.append(f"wrong_cls:{cls}")
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01 or bw <= 0 or bh <= 0:
                flags.append("oob_box")
            boxes.append((cx, cy, bw, bh, max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)))

    if bad_syntax:
        flags.append(f"bad_syntax:{len(bad_syntax)}")

    # Geometry checks
    for idx_b, (cx, cy, bw, bh, x1, y1, x2, y2) in enumerate(boxes):
        area = bw * bh
        ar = max(bw / (bh + 1e-9), bh / (bw + 1e-9))
        if area > 0.35:
            flags.append(f"huge_box:{idx_b}:{area:.3f}")
        if bw < 0.004 or bh < 0.004 or area < 0.000018:
            flags.append(f"tiny_box:{idx_b}:{bw:.4f}x{bh:.4f}")
        if ar > 4.5:
            flags.append(f"extreme_ar:{idx_b}:{ar:.2f}")
        if src != "d2_ir_2025":
            if ((y2 >= 0.992 or y1 <= 0.008) and bh < 0.032 and ar > 2.2) or (
                (x2 >= 0.992 or x1 <= 0.008) and bw < 0.032 and ar > 2.2
            ):
                flags.append(f"edge_sliver:{idx_b}:{bw:.3f}x{bh:.3f}")

    # Pair checks (duplicate / nested / split-LED)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i], boxes[j]
            iou_v = iou_xyxy(b1[4:8], b2[4:8])
            iom_v = inter_over_min(b1[4:8], b2[4:8])
            if iou_v >= 0.45 or iom_v >= 0.72:
                flags.append(f"dup_overlap:{i}-{j}:iou={iou_v:.2f}:iom={iom_v:.2f}")
            elif src != "d2_ir_2025" and len(boxes) <= 4:
                ux1, uy1 = min(b1[4], b2[4]), min(b1[5], b2[5])
                ux2, uy2 = max(b1[6], b2[6]), max(b1[7], b2[7])
                dist = ((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2) ** 0.5
                if dist < 0.065 and (ux2 - ux1) < 0.12 and (uy2 - uy1) < 0.12:
                    flags.append(f"split_pair:{i}-{j}:dist={dist:.3f}")

    # Read image bytes for MD5 + OpenCV decode
    try:
        raw_bytes = Path(img_path).read_bytes()
        md5 = hashlib.md5(raw_bytes).hexdigest()
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        im = None
        md5 = ""

    if im is None or im.size == 0:
        flags.append("corrupt_image")
        return {
            "ds": ds_name, "sp": sp, "stem": stem, "src": src,
            "img": img_path, "lbl": lbl_path, "w": 0, "h": 0,
            "md5": md5, "dhash": "", "n_gt": len(boxes), "boxes": boxes,
            "flags": flags, "gpu_mode": "skip", "canvas": None, "lb": None,
        }

    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_mean_gray = float(gray.mean())
    dhash = compute_dhash64(gray)

    # Whole-image studio white border & 2D flat graphic checks
    border = np.concatenate([gray[:4, :].ravel(), gray[-4:, :].ravel(), gray[:, :4].ravel(), gray[:, -4:].ravel()])
    border_white_pct = float((border > 230).mean())
    flat_pct = float((np.abs(np.diff(gray.astype(np.int16), axis=1)) <= 1).mean())
    if border_white_pct > 0.70 and flat_pct > 0.78:
        flags.append(f"flat_white_graphic:{flat_pct:.2f}")

    # Box-level OpenCV pixel checks
    for idx_b, (cx, cy, bw, bh, x1, y1, x2, y2) in enumerate(boxes):
        px1, py1 = max(0, int(round(x1 * w))), max(0, int(round(y1 * h)))
        px2, py2 = min(w, int(round(x2 * w))), min(h, int(round(y2 * h)))
        bw_px, bh_px = px2 - px1, py2 - py1
        if bw_px < 2 or bh_px < 2:
            flags.append(f"zero_px_box:{idx_b}")
            continue
        crop = gray[py1:py2, px1:px2]
        c_std = float(crop.std())
        p5, p95 = np.percentile(crop, [5, 95])
        ptp = float(int(crop.max()) - int(crop.min()))
        if bw_px >= 10 and bh_px >= 10 and (c_std < 3.5 or (ptp < 20.0 and (p95 - p5) < 10.0)):
            flags.append(f"ghost_blank:{idx_b}:std={c_std:.1f}:rng={p95-p5:.1f}")

        # White-bg truncated drone check (box slices off black arms/legs inside white image)
        if border_white_pct >= 0.72:
            m = 5
            strips = []
            if py1 >= 4:
                strips.append(gray[max(0, py1 - m) : py1, px1:px2])
            if py2 <= h - 4:
                strips.append(gray[py2 : min(h, py2 + m), px1:px2])
            if px1 >= 4:
                strips.append(gray[py1:py2, max(0, px1 - m) : px1])
            if px2 <= w - 4:
                strips.append(gray[py1:py2, px2 : min(w, px2 + m)])
            if strips:
                max_bleed = max(float((s < 205).mean()) for s in strips if s.size > 0)
                if max_bleed >= 0.16:
                    flags.append(f"white_bg_truncated:{idx_b}:bleed={max_bleed:.2f}")

        # Stadium light-pole / vertical stem directly under box bottom center (sky flanks on both sides)
        if src != "d2_ir_2025" and 0.012 <= bw <= 0.18 and 0.012 <= bh <= 0.18 and len(boxes) <= 6 and py1 >= 4:
            top_sky = gray[max(0, py1 - 5) : py1, px1:px2]
            top_m = float(top_sky.mean()) if top_sky.size else 0.0
            top_s = float(top_sky.std()) if top_sky.size else 999.0
            pole_len = min(h - py2, max(18, int(bh_px * 2.5)))
            if top_m > 75 and top_s < 14.0 and pole_len >= 18:
                cx_px = (px1 + px2) // 2
                col_w = max(2, int(bw_px * 0.16))
                mid = gray[py2 : py2 + pole_len, max(0, cx_px - col_w) : min(w, cx_px + col_w + 1)]
                side_l = gray[py2 : py2 + pole_len, max(0, px1 - max(4, int(bw_px * 0.4))) : max(1, px1)]
                side_r = gray[py2 : py2 + pole_len, min(w - 1, px2) : min(w, px2 + max(4, int(bw_px * 0.4)))]
                if mid.size and side_l.size and side_r.size:
                    sl_m = side_l.mean(axis=1)
                    sr_m = side_r.mean(axis=1)
                    side_s = float(np.concatenate([side_l, side_r], axis=1).std())
                    mid_min = mid.min(axis=1).astype(np.float32)
                    both_flanks = (((sl_m - mid_min) > 20.0) & ((sr_m - mid_min) > 20.0)).mean()
                    sky_cont = abs(0.5 * float(sl_m.mean() + sr_m.mean()) - top_m)
                    if side_s < 28.0 and sky_cont < 35.0 and float(both_flanks) >= 0.80:
                        flags.append(f"pole_below:{idx_b}:{float(both_flanks):.2f}")

    # Annotate night brightness if split_pair exists
    for idx_f, fl in enumerate(flags):
        if fl.startswith("split_pair:"):
            parts = fl.split(":")[1].split("-")
            b1, b2 = boxes[int(parts[0])], boxes[int(parts[1])]
            ux1, uy1 = max(0, int(min(b1[4], b2[4]) * w)), max(0, int(min(b1[5], b2[5]) * h))
            ux2, uy2 = min(w, int(max(b1[6], b2[6]) * w)), min(h, int(max(b1[7], b2[7]) * h))
            mg = float(gray[uy1:uy2, ux1:ux2].mean()) if ux2 > ux1 and uy2 > uy1 else 999.0
            flags[idx_f] = fl + f":gray={mg:.1f}"

    # Force GPU dual-check if CPU flagged RGB anomalies on a cpu_only source
    eff_gpu = gpu_mode
    if eff_gpu == "cpu_only" and flags and "ir_" not in src:
        eff_gpu = "dual"

    canvas, lb = (None, None)
    if eff_gpu != "skip":
        canvas, lb = letterbox_640(im, 640)

    return {
        "ds": ds_name, "sp": sp, "stem": stem, "src": src,
        "img": img_path, "lbl": lbl_path, "w": w, "h": h, "mgray": round(img_mean_gray, 1),
        "md5": md5, "dhash": dhash, "n_gt": len(boxes), "boxes": boxes,
        "flags": flags, "gpu_mode": eff_gpu, "canvas": canvas, "lb": lb,
    }


def unletterbox_xyxyn(xyxy_640, lb):
    nw, nh, px, py = lb
    return (
        max(0.0, min(1.0, (float(xyxy_640[0]) - px) / max(1, nw))),
        max(0.0, min(1.0, (float(xyxy_640[1]) - py) / max(1, nh))),
        max(0.0, min(1.0, (float(xyxy_640[2]) - px) / max(1, nw))),
        max(0.0, min(1.0, (float(xyxy_640[3]) - py) / max(1, nh))),
    )


def run_audit(limit: int = 0, target_stems=None, out_tag: str = "full"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tasks = []
    for ds_name, ds_root in [("clean_dataset", r"e:\trend_AI\clean_dataset"), ("data2", r"e:\trend_AI\data2")]:
        for sp in ("train", "val", "test"):
            img_dir = os.path.join(ds_root, "images", sp)
            lbl_dir = os.path.join(ds_root, "labels", sp)
            for fn in sorted(os.listdir(img_dir)):
                stem, ext = os.path.splitext(fn)
                if target_stems and stem not in target_stems:
                    continue
                img_p = os.path.join(img_dir, fn)
                lbl_p = os.path.join(lbl_dir, stem + ".txt")
                tasks.append((ds_name, ds_root, sp, stem, img_p, lbl_p))

    if limit and limit > 0 and not target_stems:
        # Stratified slice
        step = max(1, len(tasks) // limit)
        tasks = tasks[::step][:limit]

    print(f"Total tasks scheduled ({out_tag}): {len(tasks)}", flush=True)

    m_coco = YOLO(COCO_MODEL_PATH).model.cuda().half().eval()
    m_clean = YOLO(CLEAN_MODEL_PATH).model.cuda().half().eval()
    coco_names = YOLO(COCO_MODEL_PATH).names

    index_f = open(OUT_DIR / f"all_index_{out_tag}.jsonl", "w", encoding="utf-8")
    issues_f = open(OUT_DIR / f"flagged_{out_tag}.jsonl", "w", encoding="utf-8")

    stat_counts = Counter()
    dual_buf = []
    hn_buf = []
    processed = 0

    def flush_gpu(buf, mode):
        if not buf:
            return
        arr = np.stack([x["canvas"] for x in buf])
        with torch.inference_mode():
            ten = torch.from_numpy(arr).pin_memory().to("cuda", non_blocking=True).permute(0, 3, 1, 2).half().div_(255.0)
            p_clean = non_max_suppression(m_clean(ten)[0], conf_thres=0.20, iou_thres=0.45)
            p_coco = non_max_suppression(m_coco(ten)[0], conf_thres=0.22, iou_thres=0.45) if mode == "dual" else [None] * len(buf)

        for rec, dc, dcc in zip(buf, p_clean, p_coco):
            rec.pop("canvas", None)
            lb = rec.pop("lb", None)
            gt_boxes = [b[4:8] for b in rec["boxes"]]

            d_preds = []
            if dc is not None and len(dc):
                for row in dc.cpu().numpy():
                    d_preds.append((unletterbox_xyxyn(row[:4], lb), float(row[4])))

            c_preds = []
            if dcc is not None and len(dcc):
                for row in dcc.cpu().numpy():
                    cname = coco_names[int(row[5])]
                    c_preds.append((unletterbox_xyxyn(row[:4], lb), cname, float(row[4])))

            gpu_flags = []
            if mode == "hn_only":
                if d_preds:
                    top_cf = max(cf for _, cf in d_preds)
                    if top_cf >= 0.35:
                        gpu_flags.append(f"hn_drone_detected:cf={top_cf:.2f}")
            else:
                gt_matched = [False] * len(gt_boxes)
                for gi, gb in enumerate(gt_boxes):
                    best_d_iou = 0.0
                    best_d_cf = 0.0
                    for db, dcf in d_preds:
                        iv = iou_xyxy(gb, db)
                        if iv > best_d_iou:
                            best_d_iou = iv
                            best_d_cf = dcf
                    if best_d_iou >= 0.25 and best_d_cf >= 0.22:
                        gt_matched[gi] = True

                    best_coco = None
                    best_c_iou = 0.0
                    gb_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    for cb, cn, ccf in c_preds:
                        if cn not in NON_DRONE_COCO:
                            continue
                        if cn == "traffic light" and rec.get("mgray", 128) < 35 and ccf < 0.60:
                            continue
                        iv = iou_xyxy(gb, cb)
                        iom = inter_over_min(gb, cb)
                        is_person_swallow = (cn == "person" and iom >= 0.85 and gb_area < 0.025 and best_d_cf < 0.65)
                        is_person_overlap = (cn == "person" and iv >= 0.42 and best_d_cf < 0.25)
                        is_other = (cn != "person" and iv >= 0.35)
                        if is_person_swallow or is_person_overlap or is_other:
                            if iv > best_c_iou or best_coco is None:
                                best_c_iou = iv
                                best_coco = (cn, ccf, iv, iom)
                    if best_coco and (
                        best_coco[0] in {"clock", "tie", "tv", "book", "cell phone", "scissors"}
                        or (best_coco[0] == "traffic light" and best_coco[1] >= 0.50)
                        or best_coco[0] == "person"
                        or not (gt_matched[gi] and best_d_cf >= 0.65)
                    ):
                        gpu_flags.append(f"coco_conflict:{gi}:{best_coco[0]}:{best_coco[1]:.2f}:iou={best_coco[2]:.2f}:dcf={best_d_cf:.2f}")

                    if rec["src"] == "d2_rgb_dvb10k":
                        best_air = max((ccf for cb, cn, ccf in c_preds if cn == "airplane" and iou_xyxy(gb, cb) >= 0.35), default=0.0)
                        if best_air >= 0.40 and (gb_area > 0.12 or best_d_cf < 0.25):
                            gpu_flags.append(f"dvb_airplane_suspect:{gi}:cf={best_air:.2f}:a={gb_area:.2f}:dcf={best_d_cf:.2f}")

                if gt_boxes and not any(gt_matched):
                    gpu_flags.append(f"clean_v2_missed_all:n={len(gt_boxes)}")
                elif 0 < sum(gt_matched) < len(gt_boxes):
                    gpu_flags.append(f"clean_v2_partial:{sum(gt_matched)}/{len(gt_boxes)}")

                for db, dcf in d_preds:
                    if dcf >= 0.55 and all(iou_xyxy(db, gb) < 0.12 for gb in gt_boxes):
                        gpu_flags.append(f"unlabeled_clean_pred:cf={dcf:.2f}")
                        break

            rec["flags"].extend(gpu_flags)
            rec["d_preds"] = [[round(v, 5) for v in db] + [round(dcf, 3)] for db, dcf in d_preds]
            rec["c_preds"] = [[round(v, 5) for v in cb] + [cn, round(ccf, 3)] for cb, cn, ccf in c_preds]
            write_record(rec)

    def write_record(rec):
        rec.pop("canvas", None)
        rec.pop("lb", None)
        compact = {
            "ds": rec["ds"], "sp": rec["sp"], "stem": rec["stem"], "src": rec["src"],
            "w": rec["w"], "h": rec["h"], "n": rec["n_gt"], "md5": rec["md5"], "dhash": rec["dhash"],
            "flags": rec["flags"],
        }
        index_f.write(json.dumps(compact) + "\n")
        if rec["flags"]:
            compact["boxes"] = rec["boxes"]
            compact["img"] = rec["img"]
            compact["lbl"] = rec["lbl"]
            if "d_preds" in rec:
                compact["d_preds"] = rec["d_preds"]
            if "c_preds" in rec:
                compact["c_preds"] = rec["c_preds"]
            issues_f.write(json.dumps(compact) + "\n")
            seen_cats = set(f.split(":")[0] for f in rec["flags"])
            for c in seen_cats:
                stat_counts[(rec["ds"], rec["src"], c)] += 1

    CHUNK = 256
    with ThreadPoolExecutor(max_workers=12) as ex, ThreadPoolExecutor(max_workers=1) as pf_ex:
        chunk_slices = [tasks[i : i + CHUNK] for i in range(0, len(tasks), CHUNK)]
        fut = pf_ex.submit(lambda sl: list(ex.map(inspect_image_cpu, sl)), chunk_slices[0]) if chunk_slices else None
        for idx_c in range(len(chunk_slices)):
            results = fut.result()
            if idx_c + 1 < len(chunk_slices):
                fut = pf_ex.submit(lambda sl: list(ex.map(inspect_image_cpu, sl)), chunk_slices[idx_c + 1])
            for res in results:
                processed += 1
                mode = res.get("gpu_mode", "skip")
                if mode == "dual":
                    dual_buf.append(res)
                    if len(dual_buf) >= 64:
                        flush_gpu(dual_buf, "dual")
                        dual_buf.clear()
                elif mode == "hn_only":
                    hn_buf.append(res)
                    if len(hn_buf) >= 64:
                        flush_gpu(hn_buf, "hn_only")
                        hn_buf.clear()
                else:
                    write_record(res)

            if processed % 10240 == 0 or idx_c == len(chunk_slices) - 1:
                index_f.flush()
                issues_f.flush()
                elapsed = time.time() - t0
                print(f"[{processed}/{len(tasks)}] {processed/elapsed:.1f} img/s | elapsed={elapsed:.1f}s", flush=True)

    flush_gpu(dual_buf, "dual")
    flush_gpu(hn_buf, "hn_only")
    index_f.close()
    issues_f.close()

    elapsed = time.time() - t0
    print(f"\n=== AUDIT COMPLETE ({out_tag}): {processed} files in {elapsed:.1f}s ({processed/max(1e-3,elapsed):.1f} img/s) ===", flush=True)
    for (ds, src, cat), cnt in sorted(stat_counts.items()):
        print(f"  {ds:13s} | {src:18s} | {cat:22s} : {cnt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--stems", type=str, default="")
    args = ap.parse_args()
    stems = set(args.stems.split(",")) if args.stems else None
    run_audit(limit=args.limit, target_stems=stems, out_tag=args.tag)
