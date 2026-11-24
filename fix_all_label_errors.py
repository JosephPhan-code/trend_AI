"""
Surgical Label Error Fixer across e:\\trend_AI\\clean_dataset and e:\\trend_AI\\data2
Driven strictly by the verified 333,009-image multi-signal audit (flagged_full.jsonl).

Actions (all 100% reversible with backups in labels_backup_v3/ and quarantine_audit_v3/):
  1. Hard Negatives (_hn_antiuav_*): purge real drones that leaked from old quarantine
     (hn_has_box, hn_drone_detected >= 0.35, studio white-bg drones, or night LED drones).
  2. Stadium Light Poles (pole_below on c1_datatrain / c1_tonghop auto_labeled outdoor sky):
     drop ONLY the pole box(es); quarantine image if 0 real boxes remain.
  3. Duplicate / Nested / Split-LED Boxes:
     - c1_tonghop & c1_orig night split-LED pairs -> merge into 1 tight union box (and drop bright yellow streetlamps).
     - d2_rgb_pathikg & c1_datatrain nested inner stickers/logos/GPS pucks (iom >= 0.72, tiny) -> drop inner box.
     - c1_datatrain, c1_tonghop, c1_orig, d2_rgb_dvb10k high-overlap dups (IoU >= 0.50) -> keep best single box.
  4. d2_rgb_dvb10k Synthetic Non-Drones:
     - Drop unconfirmed bird/helicopter/airplane false boxes in multi-box images.
     - Quarantine chaotic AI swarms/collages (n >= 5), airliner extreme_ar (> 4.2),
       huge unconfirmed helicopters (area > 0.35, dcf < 0.25), and COCO-bird-only images.
  5. Shifted Wrong Boxes, Non-Drone 2D Graphics, Human-Body False Boxes, Ghosts & Edge Slivers:
     - c1_orig shifted pole/balcony-person boxes (missed_all + unlabeled_clean >= 0.55) -> replace with verified clean_v2 box.
     - c1_datatrain false boxes swallowed inside person (Nike logo, sunglasses, shorts, audience head) -> drop box.
     - 2D flowcharts / vest icons / shirt logos / wireframes (flat_white_graphic + missed_all or coco clock/tie/book) -> quarantine.
     - ghost_blank (non-ir2025, non-fog-antiuav300) & RGB edge_slivers -> drop bad box / quarantine if 0 left.
  6. White-Background Truncated Boxes (white_bg_truncated on c1_orig / c1_datatrain / pathikg):
     - Auto-expand via connected-component silhouette on white background (locked strictly to the
       drone component inside GT so nearby remote controllers/text are NOT swallowed).
"""

import os
import sys
import csv
import json
import time
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

WORKSPACE = Path(r"e:\trend_AI")
QUARANTINE_V3 = WORKSPACE / "quarantine_audit_v3"
FLG_P = Path(r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\scratch\audit_full\flagged_full.jsonl")
IDX_P = Path(r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\scratch\audit_full\all_index_full.jsonl")
VERIFY_DIR = Path(r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\scratch\fix_verify")
REPORT_P = WORKSPACE / "label_fix_v3_report.json"


def iou_xyxy(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def inter_over_min(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


def xyxy_to_yolo(x1, y1, x2, y2):
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    bw, bh = x2 - x1, y2 - y1
    return (x1 + bw / 2.0, y1 + bh / 2.0, bw, bh)


def backup_label(ds_root: Path, sp: str, stem: str, lbl_path: Path):
    bdir = ds_root / "labels_backup_v3" / sp
    bdir.mkdir(parents=True, exist_ok=True)
    bp = bdir / f"{stem}.txt"
    if not bp.exists() and lbl_path.exists():
        shutil.copy2(lbl_path, bp)


def quarantine_pair(ds_name: str, sp: str, stem: str, img_path: Path, lbl_path: Path, reason: str):
    q_img_dir = QUARANTINE_V3 / ds_name / reason / "images" / sp
    q_lbl_dir = QUARANTINE_V3 / ds_name / reason / "labels" / sp
    q_img_dir.mkdir(parents=True, exist_ok=True)
    q_lbl_dir.mkdir(parents=True, exist_ok=True)
    if img_path.exists():
        os.replace(img_path, q_img_dir / img_path.name)
    if lbl_path.exists():
        os.replace(lbl_path, q_lbl_dir / lbl_path.name)


def best_clean_conf_for(gb_xyxy, d_preds):
    best_cf = 0.0
    for dp in d_preds:
        if iou_xyxy(gb_xyxy, dp[:4]) >= 0.25 and dp[4] > best_cf:
            best_cf = dp[4]
    return best_cf


def is_outdoor_sky_above(im_bgr, x1, y1, x2, y2):
    """Distinguishes outdoor sky (B >= R - 8) from warm indoor white-room wall/ceiling."""
    h, w = im_bgr.shape[:2]
    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2 = min(w, int(x2 * w))
    if py1 < 4 or px2 <= px1:
        return False
    strip = im_bgr[max(0, py1 - 6) : py1, px1:px2]
    if strip.size == 0:
        return False
    b_m, g_m, r_m = float(strip[:, :, 0].mean()), float(strip[:, :, 1].mean()), float(strip[:, :, 2].mean())
    # Indoor synthetic white-room ceiling/wall is warm beige (R > B + 8); sky is bluish/gray (B >= R - 6)
    return b_m >= r_m - 6.0


def is_bright_yellow_lamp(im_bgr, x1, y1, x2, y2):
    h, w = im_bgr.shape[:2]
    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
    crop = im_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return False
    # Yellow streetlight glow: R > 180, G > 150, B < 100 on >= 8% of box pixels
    ylw = (crop[:, :, 2] > 180) & (crop[:, :, 1] > 150) & (crop[:, :, 0] < 100)
    return float(ylw.mean()) >= 0.08


def has_night_led_dot(im_bgr):
    """Detects real glowing red/green/orange drone LED inside center crosshair of dark Anti-UAV frame."""
    h, w = im_bgr.shape[:2]
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) > 22.0:
        return False
    # Ignore top HUD timestamp/text strip (y < 14%) and bottom watermark (y > 90%)
    roi = im_bgr[int(h * 0.14) : int(h * 0.90), int(w * 0.10) : int(w * 0.90)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Saturated bright colored LED pixels (not white HUD crosshair lines which have S ~ 0)
    sat_bright = (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 140)
    n_px = int(sat_bright.sum())
    return 6 <= n_px <= 15000


def expand_white_bg_box(im_bgr, x1, y1, x2, y2):
    """Expands a truncated drone box on white studio background via connected components."""
    h, w = im_bgr.shape[:2]
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    # Foreground mask: non-white pixels (< 232)
    fg = (gray < 232).astype(np.uint8)
    # Slight dilation bridges thin rotor blades to motor hubs while keeping distant remotes separate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_dil = cv2.dilate(fg, kernel, iterations=1)
    num_lbls, labels, stats, _ = cv2.connectedComponentsWithStats(fg_dil, connectivity=8)

    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
    gt_region = labels[py1:py2, px1:px2]
    if gt_region.size == 0:
        return None
    # Find dominant foreground component overlapping the GT box (>= 8% of GT area)
    vals, cnts = np.unique(gt_region[gt_region > 0], return_counts=True)
    if len(vals) == 0:
        return None
    gt_area_px = max(1, (px2 - px1) * (py2 - py1))
    # Merge components that make up significant parts of the existing GT interior
    nx1, ny1, nx2, ny2 = px1, py1, px2, py2
    used = 0
    for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1]):
        if c < 0.06 * gt_area_px and used > 0:
            continue
        cx1 = int(stats[v, cv2.CC_STAT_LEFT])
        cy1 = int(stats[v, cv2.CC_STAT_TOP])
        cw = int(stats[v, cv2.CC_STAT_WIDTH])
        ch = int(stats[v, cv2.CC_STAT_HEIGHT])
        cx2, cy2 = cx1 + cw, cy1 + ch
        # Safety: skip full-image border frames
        if cw > 0.98 * w and ch > 0.98 * h:
            continue
        nx1, ny1 = min(nx1, cx1), min(ny1, cy1)
        nx2, ny2 = max(nx2, cx2), max(ny2, cy2)
        used += 1
        if used >= 2:
            break
    if used == 0:
        return None

    # Validate expansion is genuine improvement and didn't explode onto entire catalog page
    new_area = ((nx2 - nx1) * (ny2 - ny1)) / float(w * h)
    old_area = (x2 - x1) * (y2 - y1)
    if new_area <= old_area * 1.06:
        return None  # negligible change
    if new_area > 0.88 or new_area > old_area * 2.6:
        return None  # too risky (might swallow catalog accessories/remote)
    return (nx1 / w, ny1 / h, nx2 / w, ny2 / h)


def main():
    t0 = time.time()
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    # Load auto_labeled stems from clean_dataset/relabel_audit_log.csv
    auto_stems = set()
    log_c1 = WORKSPACE / "clean_dataset" / "relabel_audit_log.csv"
    if log_c1.exists():
        with open(log_c1, encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[2] in ("auto_labeled", "clean_relabeled", "precision_relabeled_iter1"):
                    auto_stems.add(os.path.splitext(row[0])[0])
    print(f"Loaded {len(auto_stems)} auto_labeled stems from clean_dataset.", flush=True)

    # Load all flagged records
    flagged = []
    with open(FLG_P, encoding="utf-8") as f:
        for l in f:
            flagged.append(json.loads(l))
    print(f"Loaded {len(flagged)} flagged audit records.", flush=True)

    # Also collect ALL _hn_antiuav_ records from all_index_full.jsonl to sweep night LEDs & studio drones
    all_hn_antiuav = []
    with open(IDX_P, encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if "hn_antiuav" in d["src"]:
                all_hn_antiuav.append(d)
    print(f"Loaded {len(all_hn_antiuav)} _hn_antiuav_ records for Hard Negative purity sweep.", flush=True)

    stats = Counter()
    verify_samples = defaultdict(list)  # category -> list of (img_path, old_boxes, new_boxes, note)
    quarantined_set = set()  # (ds, sp, stem)
    modified_log = []

    # ---------------------------------------------------------
    # PASS 1: Hard Negative (_hn_antiuav_) Purity Sweep
    # ---------------------------------------------------------
    flagged_hn_keys = set()
    for r in flagged:
        if "hn_antiuav" in r["src"]:
            fkeys = {fl.split(":")[0] for fl in r["flags"]}
            if "hn_drone_detected" in fkeys or "hn_has_box" in fkeys or "flat_white_graphic" in fkeys:
                flagged_hn_keys.add((r["ds"], r["sp"], r["stem"]))

    for d in all_hn_antiuav:
        ds, sp, stem = d["ds"], d["sp"], d["stem"]
        ds_root = WORKSPACE / ds
        img_p = ds_root / "images" / sp / f"{stem}.jpg"
        lbl_p = ds_root / "labels" / sp / f"{stem}.txt"
        if not img_p.exists():
            continue
        reason = None
        if (ds, sp, stem) in flagged_hn_keys:
            reason = "hn_antiuav_real_drone_or_box"
        else:
            # Check night LED dot or non-AntiUAV image dimensions (Anti-UAV IR/RGB are 640x512 or 1920x1080)
            w, h = d.get("w", 0), d.get("h", 0)
            if (w, h) not in ((1920, 1080), (640, 512)):
                # Non-AntiUAV photo from old quarantine_iter1 (studio/grass drones!)
                reason = "hn_antiuav_non_antiuav_quarantine"
            else:
                im = cv2.imread(str(img_p))
                if im is not None and has_night_led_dot(im):
                    reason = "hn_antiuav_night_led_drone"

        if reason:
            quarantine_pair(ds, sp, stem, img_p, lbl_p, reason)
            quarantined_set.add((ds, sp, stem))
            stats[(ds, f"quar_{reason}")] += 1
            if len(verify_samples["hn_purged"]) < 12:
                q_p = QUARANTINE_V3 / ds / reason / "images" / sp / f"{stem}.jpg"
                verify_samples["hn_purged"].append((str(q_p), [], [], reason[-18:]))

    print(f"Pass 1 (_hn_antiuav_ purity) done: {sum(v for (_, k), v in stats.items() if 'hn_antiuav' in k)} purged.", flush=True)

    # ---------------------------------------------------------
    # PASS 2: Positive Label Repairs & Quarantines
    # ---------------------------------------------------------
    for r in flagged:
        ds, sp, stem, src = r["ds"], r["sp"], r["stem"], r["src"]
        if "_hn_" in stem or (ds, sp, stem) in quarantined_set:
            continue
        ds_root = WORKSPACE / ds
        img_p = Path(r["img"])
        lbl_p = Path(r["lbl"])
        if not img_p.exists() or not lbl_p.exists():
            continue

        fls = r["flags"]
        fkeys = {fl.split(":")[0] for fl in fls}
        d_preds = r.get("d_preds", [])
        c_preds = r.get("c_preds", [])
        orig_boxes = [tuple(b[4:8]) for b in r.get("boxes", [])]  # list of (x1,y1,x2,y2)
        if not orig_boxes:
            continue

        # --- Quarantine-first checks (whole image is non-drone / broken synthetic) ---
        # A. 2D Flowcharts / vest icons / shirt logos / wireframes
        if src in ("c1_orig", "c1_datatrain", "c1_tonghop"):
            if "flat_white_graphic" in fkeys and (
                "clean_v2_missed_all" in fkeys
                or any(f":clock:" in fl or f":tie:" in fl or f":book:" in fl for fl in fls)
            ):
                quarantine_pair(ds, sp, stem, img_p, lbl_p, "non_drone_2d_graphic")
                quarantined_set.add((ds, sp, stem))
                stats[(ds, "quar_non_drone_2d_graphic")] += 1
                if len(verify_samples["quar_2d_graphic"]) < 12:
                    qp = QUARANTINE_V3 / ds / "non_drone_2d_graphic" / "images" / sp / img_p.name
                    verify_samples["quar_2d_graphic"].append((str(qp), orig_boxes, [], stem[-10:]))
                continue

        # B. d2_rgb_dvb10k whole-image synthetic garbage (collages/swarms n>=5, extreme_ar airliners, huge unconfirmed helicopters)
        if src == "d2_rgb_dvb10k":
            if len(orig_boxes) >= 5:
                quarantine_pair(ds, sp, stem, img_p, lbl_p, "dvb10k_chaotic_swarm_collage")
                quarantined_set.add((ds, sp, stem))
                stats[(ds, "quar_dvb10k_swarm_collage")] += 1
                continue
            if "extreme_ar" in fkeys:
                quarantine_pair(ds, sp, stem, img_p, lbl_p, "dvb10k_extreme_ar_airliner")
                quarantined_set.add((ds, sp, stem))
                stats[(ds, "quar_dvb10k_extreme_ar")] += 1
                continue
            # Unlabeled foreground drone while GT labeled airplanes/birds
            if "clean_v2_missed_all" in fkeys and ("dvb_airplane_suspect" in fkeys or any(":bird:" in fl for fl in fls)):
                quarantine_pair(ds, sp, stem, img_p, lbl_p, "dvb10k_wrong_non_drone_only")
                quarantined_set.add((ds, sp, stem))
                stats[(ds, "quar_dvb10k_wrong_non_drone")] += 1
                if len(verify_samples["quar_dvb"]) < 12:
                    qp = QUARANTINE_V3 / ds / "dvb10k_wrong_non_drone_only" / "images" / sp / img_p.name
                    verify_samples["quar_dvb"].append((str(qp), orig_boxes, [], "dvb_bad"))
                continue

        # C. Shifted wrong boxes in c1_orig (telephone poles / balcony man while real drone sits elsewhere)
        if src == "c1_orig" and "clean_v2_missed_all" in fkeys and "unlabeled_clean_pred" in fkeys:
            strong_d = [dp for dp in d_preds if dp[4] >= 0.55]
            if strong_d:
                new_boxes = [tuple(dp[:4]) for dp in strong_d]
                backup_label(ds_root, sp, stem, lbl_p)
                with open(lbl_p, "w", encoding="utf-8") as f:
                    for b in new_boxes:
                        cx, cy, bw, bh = xyxy_to_yolo(*b)
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                stats[(ds, "fix_c1_orig_shifted_relabel")] += 1
                modified_log.append((ds, sp, stem, "c1_orig_shifted_relabel", len(orig_boxes), len(new_boxes)))
                if len(verify_samples["shifted_fixed"]) < 12:
                    verify_samples["shifted_fixed"].append((str(img_p), orig_boxes, new_boxes, "shifted->fix"))
                continue

        # --- Box-level surgical edits ---
        cur_boxes = list(orig_boxes)
        drop_idx = set()
        edit_reasons = set()
        im_bgr = None  # lazy load only if pixel check/edit needed

        # 1. Stadium light poles (c1_datatrain / c1_tonghop auto_labeled + outdoor sky check)
        if src in ("c1_datatrain", "c1_tonghop") and stem in auto_stems and "pole_below" in fkeys:
            im_bgr = cv2.imread(str(img_p))
            if im_bgr is not None:
                for fl in fls:
                    if fl.startswith("pole_below:"):
                        bi = int(fl.split(":")[1])
                        if bi < len(cur_boxes) and is_outdoor_sky_above(im_bgr, *cur_boxes[bi]):
                            drop_idx.add(bi)
                            edit_reasons.add("drop_stadium_pole")

        # 2. Human-body swallowed false boxes in c1_datatrain (Nike logo, sunglasses, shorts, audience head)
        if src == "c1_datatrain":
            for fl in fls:
                if fl.startswith("coco_conflict:") and ":person:" in fl:
                    ps = fl.split(":")
                    bi = int(ps[1])
                    if bi < len(cur_boxes):
                        bx = cur_boxes[bi]
                        area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                        dcf = float(fl.split("dcf=")[1])
                        # Swallowed tiny box (< 2.5% area) inside person OR large person overlap with zero clean_v2 support
                        if (area < 0.025 and dcf < 0.65) or dcf < 0.15:
                            drop_idx.add(bi)
                            edit_reasons.add("drop_person_body_fp")

        # 3. d2_rgb_dvb10k per-box non-drone pruning (birds, unconfirmed helicopters/airplanes, huge unconfirmed)
        if src == "d2_rgb_dvb10k":
            for fl in fls:
                if fl.startswith("coco_conflict:") and ":bird:" in fl:
                    bi = int(fl.split(":")[1])
                    dcf = float(fl.split("dcf=")[1])
                    if bi < len(cur_boxes) and dcf < 0.25:
                        drop_idx.add(bi)
                        edit_reasons.add("drop_dvb_bird")
                elif fl.startswith("dvb_airplane_suspect:"):
                    bi = int(fl.split(":")[1])
                    dcf = float(fl.split("dcf=")[1])
                    a = float(fl.split("a=")[1].split(":")[0])
                    if bi < len(cur_boxes) and (
                        (len(cur_boxes) >= 2 and dcf < 0.25)
                        or (a > 0.35 and dcf < 0.25)
                    ):
                        drop_idx.add(bi)
                        edit_reasons.add("drop_dvb_heli_plane")
            # Also drop unconfirmed tiny (< 0.0025 area) extra specks in multi-box dvb10k
            if len(cur_boxes) >= 2 and "clean_v2_partial" in fkeys:
                for bi, bx in enumerate(cur_boxes):
                    area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    if area < 0.0025 and best_clean_conf_for(bx, d_preds) < 0.18:
                        drop_idx.add(bi)
                        edit_reasons.add("drop_dvb_tiny_speck")

        # 4. Ghost/blank, full-width shifted dirt/strip boxes & RGB edge slivers
        if src == "c1_datatrain" and "huge_box" in fkeys and "clean_v2_missed_all" in fkeys:
            for bi, bx in enumerate(cur_boxes):
                if (bx[2] - bx[0]) >= 0.98 and (bx[3] - bx[1]) <= 0.55 and best_clean_conf_for(bx, d_preds) < 0.15:
                    drop_idx.add(bi)
                    edit_reasons.add("drop_fullwidth_strip_fp")

        if src not in ("d2_ir_2025", "d2_rgb_antiuav300") and "ghost_blank" in fkeys:
            for fl in fls:
                if fl.startswith("ghost_blank:"):
                    bi = int(fl.split(":")[1])
                    if bi < len(cur_boxes) and best_clean_conf_for(cur_boxes[bi], d_preds) < 0.25:
                        drop_idx.add(bi)
                        edit_reasons.add("drop_ghost_blank")

        if src in ("c1_datatrain", "c1_tonghop", "c1_orig", "d2_rgb_antiuav300", "d2_rgb_dvb10k", "d2_rgb_pathikg", "d2_ir_410") and "edge_sliver" in fkeys:
            for fl in fls:
                if fl.startswith("edge_sliver:"):
                    bi = int(fl.split(":")[1])
                    if bi < len(cur_boxes) and best_clean_conf_for(cur_boxes[bi], d_preds) < 0.35:
                        drop_idx.add(bi)
                        edit_reasons.add("drop_edge_sliver")

        # Special shoreline glare sliver in d2_rgb_mavvid (bottom-left waves with clean_v2=0)
        if src == "d2_rgb_mavvid" and "edge_sliver" in fkeys and "clean_v2_missed_all" in fkeys:
            for fl in fls:
                if fl.startswith("edge_sliver:"):
                    bi = int(fl.split(":")[1])
                    if bi < len(cur_boxes) and cur_boxes[bi][3] >= 0.99 and (cur_boxes[bi][3] - cur_boxes[bi][1]) < 0.03:
                        drop_idx.add(bi)
                        edit_reasons.add("drop_mavvid_shore_sliver")

        # 5. c1_tonghop yellow streetlamp false positives in dup_overlap dark frames
        if src == "c1_tonghop" and "dup_overlap" in fkeys:
            if im_bgr is None:
                im_bgr = cv2.imread(str(img_p))
            if im_bgr is not None and float(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY).mean()) < 35.0:
                for bi, bx in enumerate(cur_boxes):
                    area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    if area > 0.004 and is_bright_yellow_lamp(im_bgr, *bx):
                        drop_idx.add(bi)
                        edit_reasons.add("drop_night_streetlamp")

        # Apply box drops first
        surviving = [b for i, b in enumerate(cur_boxes) if i not in drop_idx]

        # 6. Duplicate / Nested / Night Split-LED merge & deduplication (skip d2_ir_2025 & d2_rgb_mavvid)
        if src not in ("d2_ir_2025", "d2_rgb_mavvid") and len(surviving) >= 2 and ("dup_overlap" in fkeys or "split_pair" in fkeys):
            if im_bgr is None and src in ("c1_tonghop", "c1_orig", "c1_datatrain"):
                im_bgr = cv2.imread(str(img_p))
            is_dark = (im_bgr is not None and float(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY).mean()) < 40.0)

            merged = []
            used = [False] * len(surviving)
            # Sort largest-area first so swallowed inner stickers/logos/pucks are easily detected
            order = sorted(range(len(surviving)), key=lambda k: -(surviving[k][2] - surviving[k][0]) * (surviving[k][3] - surviving[k][1]))
            for idx_a in order:
                if used[idx_a]:
                    continue
                ba = surviving[idx_a]
                aa = (ba[2] - ba[0]) * (ba[3] - ba[1])
                used[idx_a] = True
                for idx_b in order:
                    if used[idx_b]:
                        continue
                    bb = surviving[idx_b]
                    ab = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    iv = iou_xyxy(ba, bb)
                    iom = inter_over_min(ba, bb)
                    cx_a, cy_a = (ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2
                    cx_b, cy_b = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                    ux1, uy1 = min(ba[0], bb[0]), min(ba[1], bb[1])
                    ux2, uy2 = max(ba[2], bb[2]), max(ba[3], bb[3])

                    # Case 6a: Dark night split-LED drone -> MERGE into union box
                    if is_dark and (iv >= 0.15 or iom >= 0.72 or (dist < 0.07 and (ux2 - ux1) < 0.14 and (uy2 - uy1) < 0.12)):
                        ba = (ux1, uy1, ux2, uy2)
                        aa = (ba[2] - ba[0]) * (ba[3] - ba[1])
                        used[idx_b] = True
                        edit_reasons.add("merge_night_split_led")
                    # Case 6b: Tiny swallowed sticker/logo/manual inside larger box -> DROP tiny inner box
                    elif iom >= 0.72 and ab < 0.35 * aa:
                        used[idx_b] = True
                        edit_reasons.add("drop_nested_inner_dup")
                    # Case 6c: High-IoU duplicate box on same drone -> keep best/tighter single box
                    elif iv >= 0.50:
                        ca = best_clean_conf_for(ba, d_preds)
                        cb = best_clean_conf_for(bb, d_preds)
                        # In dvb10k, inner box is tighter than bloated outer box; pick higher conf or tighter
                        if src == "d2_rgb_dvb10k":
                            if cb >= ca - 0.05 and ab < aa:
                                ba = bb
                                aa = ab
                        elif cb > ca + 0.10:
                            ba = bb
                            aa = ab
                        used[idx_b] = True
                        edit_reasons.add("dedup_high_iou")
                merged.append(ba)
            if len(merged) != len(surviving) or "merge_night_split_led" in edit_reasons:
                surviving = merged

        # 7. White-background truncated drone expansion (c1_orig, c1_datatrain, d2_rgb_pathikg)
        if "white_bg_truncated" in fkeys and src in ("c1_orig", "c1_datatrain", "d2_rgb_pathikg") and 1 <= len(surviving) <= 2:
            if im_bgr is None:
                im_bgr = cv2.imread(str(img_p))
            if im_bgr is not None:
                new_surv = []
                expanded_any = False
                for bx in surviving:
                    exp = expand_white_bg_box(im_bgr, *bx)
                    if exp is not None:
                        new_surv.append(exp)
                        expanded_any = True
                    else:
                        new_surv.append(bx)
                if expanded_any:
                    surviving = new_surv
                    edit_reasons.add("expand_white_bg")

        # --- Commit changes if modified ---
        if not edit_reasons:
            continue

        reason_tag = "+".join(sorted(edit_reasons))
        if not surviving:
            # 0 boxes remain -> quarantine image (safer than turning unverified images into empty negatives)
            quarantine_pair(ds, sp, stem, img_p, lbl_p, f"empty_after_{reason_tag[:32]}")
            quarantined_set.add((ds, sp, stem))
            stats[(ds, f"quar_empty_{reason_tag}")] += 1
            if "drop_stadium_pole" in edit_reasons and len(verify_samples["pole_quar"]) < 12:
                qp = QUARANTINE_V3 / ds / f"empty_after_{reason_tag[:32]}" / "images" / sp / img_p.name
                verify_samples["pole_quar"].append((str(qp), orig_boxes, [], "all_pole_quar"))
        else:
            backup_label(ds_root, sp, stem, lbl_p)
            with open(lbl_p, "w", encoding="utf-8") as f:
                for b in surviving:
                    cx, cy, bw, bh = xyxy_to_yolo(*b)
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats[(ds, f"fixed_{reason_tag}")] += 1
            modified_log.append((ds, sp, stem, reason_tag, len(orig_boxes), len(surviving)))

            # Collect before/after verification grids
            for cat_key, check_str in [
                ("pole_fixed", "drop_stadium_pole"),
                ("white_expanded", "expand_white_bg"),
                ("dup_merged", "merge_night_split_led"),
                ("dup_dedup", "drop_nested_inner_dup"),
                ("dvb_pruned", "drop_dvb_"),
                ("person_pruned", "drop_person_body_fp"),
            ]:
                if check_str in reason_tag and len(verify_samples[cat_key]) < 12:
                    verify_samples[cat_key].append((str(img_p), orig_boxes, surviving, reason_tag[:22]))

    # Remove stale YOLO .cache files so training reads fresh labels
    for ds_name in ("clean_dataset", "data2"):
        for cp in (WORKSPACE / ds_name / "labels").glob("*.cache"):
            try:
                cp.unlink()
            except OSError:
                pass

    # Render before (RED dashed/thin) vs after (GREEN thick) verification grids
    for cat, items in verify_samples.items():
        cells = []
        for ip, old_bs, new_bs, note in items[:12]:
            im = cv2.imread(ip)
            if im is None:
                continue
            h, w = im.shape[:2]
            for x1, y1, x2, y2 in old_bs:
                cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 2)
            for x1, y1, x2, y2 in new_bs:
                cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 3)
            im = cv2.resize(im, (320, 320))
            cv2.putText(im, f"{Path(ip).stem[-10:]} {len(old_bs)}->{len(new_bs)}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cells.append(im)
        if cells:
            while len(cells) < 12:
                cells.append(np.zeros((320, 320, 3), dtype=np.uint8))
            grid = np.vstack([np.hstack(cells[i : i + 4]) for i in range(0, 12, 4)])
            cv2.imwrite(str(VERIFY_DIR / f"verify_{cat}.jpg"), grid)

    # Save report
    report = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "total_quarantined": len(quarantined_set),
        "total_modified_labels": len(modified_log),
        "stats": {f"{ds}|{k}": v for (ds, k), v in sorted(stats.items())},
        "verify_grids": sorted(os.listdir(VERIFY_DIR)),
    }
    REPORT_P.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SURGICAL FIX COMPLETE in {report['elapsed_seconds']}s ===", flush=True)
    print(f"Total modified labels : {len(modified_log)}", flush=True)
    print(f"Total quarantined imgs: {len(quarantined_set)}", flush=True)
    for k, v in sorted(report["stats"].items()):
        print(f"  {k:55s}: {v}", flush=True)
    print("Verify grids:", report["verify_grids"], flush=True)


if __name__ == "__main__":
    main()
