"""
Micro-polish script after visual verification of all 11 before/after grids from fix_all_label_errors.py:
  1. Restore 1-box close-up drones (area <= 0.30, no coco bird) from quar_dvb10k_wrong_non_drone.
  2. Tighten c1_datatrain person-body filter: restore images falsely dropped via loose iou >= 0.42
     and keep ONLY strictly swallowed tiny false boxes inside confident person (iom >= 0.85, area < 0.025, conf >= 0.55).
  3. Restore Anti-UAV 1920x1080 / 640x512 building-ridge drones from quar_empty_drop_stadium_pole.
  4. Drop bright bottom-half yellow streetlamps in c1_tonghop dark frames (009762, 011374, 014542, 015288).
  5. Quarantine palm-tree FP relabels (drone_002252, drone_004912) & fix inner tight box on 014229/015558.
  6. Deduplicate post-expand duplicate boxes (IoU >= 0.70, e.g., drone_000734) & revert 2-object white-bg expansions
     that jumped > 1.9x area (like remote controller merges 001111, 002312).
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

WORKSPACE = Path(r"e:\trend_AI")
QUAR_V3 = WORKSPACE / "quarantine_audit_v3"
FLG_P = Path(r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\scratch\audit_full\flagged_full.jsonl")


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


def read_yolo_xyxy(lbl_p: Path):
    out = []
    for l in lbl_p.read_text(encoding="utf-8", errors="replace").splitlines():
        ps = l.strip().split()
        if len(ps) == 5:
            _, cx, cy, bw, bh = map(float, ps)
            out.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def write_yolo_xyxy(lbl_p: Path, boxes):
    with open(lbl_p, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
            x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw > 1e-5 and bh > 1e-5:
                f.write(f"0 {x1+bw/2:.6f} {y1+bh/2:.6f} {bw:.6f} {bh:.6f}\n")


def restore_from_quar(ds_name: str, reason_dir: Path, sp: str, stem: str):
    q_img = reason_dir / "images" / sp / f"{stem}.jpg"
    q_lbl = reason_dir / "labels" / sp / f"{stem}.txt"
    dst_img = WORKSPACE / ds_name / "images" / sp / f"{stem}.jpg"
    dst_lbl = WORKSPACE / ds_name / "labels" / sp / f"{stem}.txt"
    if q_img.exists():
        os.replace(q_img, dst_img)
    if q_lbl.exists():
        os.replace(q_lbl, dst_lbl)
    return dst_img.exists() and dst_lbl.exists()


def main():
    stats = Counter()
    # Index flagged records by (ds, sp, stem)
    flg_map = {}
    with open(FLG_P, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            flg_map[(r["ds"], r["sp"], r["stem"])] = r

    # 1. Restore 1-box clean close-up drones (area <= 0.30, no coco bird) from dvb10k_wrong_non_drone_only
    dvb_q = QUAR_V3 / "data2" / "dvb10k_wrong_non_drone_only"
    if dvb_q.exists():
        for sp in ("train", "val", "test"):
            idir = dvb_q / "images" / sp
            if not idir.exists():
                continue
            for p in list(idir.glob("*.jpg")):
                stem = p.stem
                r = flg_map.get(("data2", sp, stem))
                if not r:
                    continue
                boxes = [b[4:8] for b in r.get("boxes", [])]
                has_bird = any(":bird:" in fl for fl in r["flags"])
                if len(boxes) == 1 and not has_bird:
                    area = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
                    if area <= 0.30:
                        if restore_from_quar("data2", dvb_q, sp, stem):
                            stats["restored_dvb10k_1box_drone"] += 1

    # 2. Tighten person_body_fp on c1_datatrain:
    #    Only drop a box if strictly swallowed inside a high-conf person box (iom >= 0.85, area < 0.025, coco person conf >= 0.55)
    def strict_person_drop_indices(r):
        boxes = [tuple(b[4:8]) for b in r.get("boxes", [])]
        c_preds = r.get("c_preds", [])
        d_preds = r.get("d_preds", [])
        drops = set()
        for bi, bx in enumerate(boxes):
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            if area >= 0.025:
                continue
            dcf = max((dp[4] for dp in d_preds if iou_xyxy(bx, dp[:4]) >= 0.25), default=0.0)
            if dcf >= 0.65:
                continue
            for cp in c_preds:
                if cp[4] == "person" and cp[5] >= 0.55 and inter_over_min(bx, cp[:4]) >= 0.85:
                    drops.add(bi)
                    break
        return drops

    # 2a. Re-evaluate quar_empty_drop_person_body_fp
    for q_reason in ("empty_after_drop_person_body_fp", "empty_after_drop_person_body_fp+drop_stadium"):
        qdir = QUAR_V3 / "clean_dataset" / q_reason
        if not qdir.exists():
            continue
        for sp in ("train", "val", "test"):
            idir = qdir / "images" / sp
            if not idir.exists():
                continue
            for p in list(idir.glob("*.jpg")):
                stem = p.stem
                r = flg_map.get(("clean_dataset", sp, stem))
                if not r:
                    continue
                drops = strict_person_drop_indices(r)
                orig_boxes = [tuple(b[4:8]) for b in r.get("boxes", [])]
                surv = [b for i, b in enumerate(orig_boxes) if i not in drops]
                if surv:
                    if restore_from_quar("clean_dataset", qdir, sp, stem):
                        lbl_p = WORKSPACE / "clean_dataset" / "labels" / sp / f"{stem}.txt"
                        write_yolo_xyxy(lbl_p, surv)
                        stats["restored_c1_person_false_quar"] += 1

    # 2b. Re-evaluate modified person_body_fp labels using labels_backup_v3
    bdir_c1 = WORKSPACE / "clean_dataset" / "labels_backup_v3"
    if bdir_c1.exists():
        for sp in ("train", "val", "test"):
            sdir = bdir_c1 / sp
            if not sdir.exists():
                continue
            for bp in sdir.glob("*.txt"):
                stem = bp.stem
                r = flg_map.get(("clean_dataset", sp, stem))
                if not r:
                    continue
                lbl_p = WORKSPACE / "clean_dataset" / "labels" / sp / f"{stem}.txt"
                if not lbl_p.exists():
                    continue
                # Fix loose person drops
                if r["src"] == "c1_datatrain" and any(":person:" in fl for fl in r["flags"]):
                    old_b = read_yolo_xyxy(bp)
                    cur_b = read_yolo_xyxy(lbl_p)
                    if len(cur_b) < len(old_b) and "pole_below" not in {f.split(":")[0] for f in r["flags"]}:
                        drops = strict_person_drop_indices(r)
                        surv = [b for i, b in enumerate(old_b) if i not in drops]
                        if len(surv) > len(cur_b):
                            write_yolo_xyxy(lbl_p, surv)
                            stats["restored_c1_person_false_boxdrop"] += 1

                # Fix white_bg_expanded over-expansion (> 1.85x area -> likely merged separate remote controller) + post-expand NMS
                if "white_bg_truncated" in {f.split(":")[0] for f in r["flags"]}:
                    old_b = read_yolo_xyxy(bp)
                    cur_b = read_yolo_xyxy(lbl_p)
                    changed = False
                    # Check over-expansion
                    if len(old_b) == len(cur_b):
                        reverted = []
                        for ob, cb in zip(old_b, cur_b):
                            oa = (ob[2] - ob[0]) * (ob[3] - ob[1])
                            ca = (cb[2] - cb[0]) * (cb[3] - cb[1])
                            if ca > oa * 1.85:
                                reverted.append(ob)
                                changed = True
                            else:
                                reverted.append(cb)
                        cur_b = reverted
                    # Post-expand NMS (e.g. drone_000734 2 identical green boxes)
                    if len(cur_b) >= 2:
                        dedup = []
                        for b in cur_b:
                            if not any(iou_xyxy(b, d) >= 0.70 for d in dedup):
                                dedup.append(b)
                            else:
                                changed = True
                        cur_b = dedup
                    if changed:
                        write_yolo_xyxy(lbl_p, cur_b)
                        stats["polished_white_bg_or_nms"] += 1

    # 3. Restore Anti-UAV 1920x1080 / 640x512 building-ridge drones from quar_empty_drop_stadium_pole
    pole_q = QUAR_V3 / "clean_dataset" / "empty_after_drop_stadium_pole"
    if pole_q.exists():
        for sp in ("train", "val", "test"):
            idir = pole_q / "images" / sp
            if not idir.exists():
                continue
            for p in list(idir.glob("*.jpg")):
                stem = p.stem
                r = flg_map.get(("clean_dataset", sp, stem))
                if not r:
                    continue
                if r["src"] == "c1_tonghop" and (r["w"], r["h"]) in ((1920, 1080), (640, 512)):
                    # Check if Anti-UAV HUD top-left white Chinese text exists (non-stadium!)
                    im = cv2.imread(str(p))
                    if im is not None:
                        # Stadium poles are colorful daytime 1920x1080 WITHOUT Anti-UAV white top-right timestamp!
                        # Check bright white timestamp digits in top-right corner of Anti-UAV frames
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        h, w = gray.shape
                        tr = gray[: int(h * 0.06), int(w * 0.78) :]
                        tl = gray[: int(h * 0.06), : int(w * 0.14)]
                        has_hud_text = (float((tr > 220).mean()) > 0.015 and float((tl > 220).mean()) > 0.015) or (w == 640 and h == 512)
                        if has_hud_text:
                            if restore_from_quar("clean_dataset", pole_q, sp, stem):
                                stats["restored_antiuav_building_drone"] += 1

    # 4. Drop bright bottom-half yellow streetlamps in c1_tonghop dark frames (009762, 011374, 014542, 015288 + any alike)
    for (ds, sp, stem), r in flg_map.items():
        if r["src"] != "c1_tonghop" or "dup_overlap" not in {f.split(":")[0] for f in r["flags"]}:
            continue
        lbl_p = WORKSPACE / ds / "labels" / sp / f"{stem}.txt"
        img_p = WORKSPACE / ds / "images" / sp / f"{stem}.jpg"
        if not lbl_p.exists() or not img_p.exists():
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) >= 35.0:
            continue
        boxes = read_yolo_xyxy(lbl_p)
        h, w = gray.shape
        kept = []
        dropped = 0
        for bx in boxes:
            cy = (bx[1] + bx[3]) / 2
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            px1, py1 = max(0, int(bx[0] * w)), max(0, int(bx[1] * h))
            px2, py2 = min(w, int(bx[2] * w)), min(h, int(bx[3] * h))
            crop = im[py1:py2, px1:px2]
            warm_bright = float(((crop[:, :, 2] > 180) & (crop[:, :, 1] > 150) & (crop[:, :, 2].astype(int) - crop[:, :, 0].astype(int) > 40)).mean()) if crop.size else 0.0
            if cy > 0.65 and area > 0.006 and warm_bright > 0.05:
                dropped += 1
            else:
                kept.append(bx)
        if dropped > 0:
            if kept:
                write_yolo_xyxy(lbl_p, kept)
                stats["fixed_c1_tonghop_streetlamp_drop"] += 1
            else:
                q_img_dir = QUAR_V3 / ds / "c1_tonghop_streetlamp_only" / "images" / sp
                q_lbl_dir = QUAR_V3 / ds / "c1_tonghop_streetlamp_only" / "labels" / sp
                q_img_dir.mkdir(parents=True, exist_ok=True)
                q_lbl_dir.mkdir(parents=True, exist_ok=True)
                os.replace(img_p, q_img_dir / img_p.name)
                os.replace(lbl_p, q_lbl_dir / lbl_p.name)
                stats["quar_c1_tonghop_streetlamp_only"] += 1

    # 5. Surgical fix of 4 specific visually verified edge cases:
    #    - drone_002252, drone_004912 (palm tree crowns) -> quarantine
    for stem in ("drone_002252", "drone_004912"):
        for sp in ("train", "val", "test"):
            ip = WORKSPACE / "clean_dataset" / "images" / sp / f"{stem}.jpg"
            lp = WORKSPACE / "clean_dataset" / "labels" / sp / f"{stem}.txt"
            if ip.exists():
                qd = QUAR_V3 / "clean_dataset" / "palm_tree_fp"
                (qd / "images" / sp).mkdir(parents=True, exist_ok=True)
                (qd / "labels" / sp).mkdir(parents=True, exist_ok=True)
                os.replace(ip, qd / "images" / sp / ip.name)
                if lp.exists():
                    os.replace(lp, qd / "labels" / sp / lp.name)
                stats["quar_palm_tree_fp"] += 1

    #    - drone_014229, drone_015558 (keep tighter inner drone box instead of outer building box)
    for stem in ("drone_014229", "drone_015558"):
        for sp in ("train", "val", "test"):
            bp = WORKSPACE / "clean_dataset" / "labels_backup_v3" / sp / f"{stem}.txt"
            lp = WORKSPACE / "clean_dataset" / "labels" / sp / f"{stem}.txt"
            if bp.exists() and lp.exists():
                bs = read_yolo_xyxy(bp)
                if len(bs) >= 2:
                    tightest = min(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    write_yolo_xyxy(lp, [tightest])
                    stats["fixed_inner_tight_drone"] += 1

    #    - d2_rgb_dvb10k_054205 (keep tight inner drone box without wire) & d2_rgb_dvb10k_053761 (quarantine mixed 3-object helicopter/drone)
    for sp in ("train", "val", "test"):
        bp = WORKSPACE / "data2" / "labels_backup_v3" / sp / "d2_rgb_dvb10k_054205.txt"
        lp = WORKSPACE / "data2" / "labels" / sp / "d2_rgb_dvb10k_054205.txt"
        if bp.exists() and lp.exists():
            bs = read_yolo_xyxy(bp)
            if len(bs) >= 2:
                tightest = min(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                write_yolo_xyxy(lp, [tightest])
                stats["fixed_dvb_054205_wire"] += 1
        ip = WORKSPACE / "data2" / "images" / sp / "d2_rgb_dvb10k_053761.jpg"
        lp2 = WORKSPACE / "data2" / "labels" / sp / "d2_rgb_dvb10k_053761.txt"
        if ip.exists():
            qd = QUAR_V3 / "data2" / "dvb_mixed_heli"
            (qd / "images" / sp).mkdir(parents=True, exist_ok=True)
            (qd / "labels" / sp).mkdir(parents=True, exist_ok=True)
            os.replace(ip, qd / "images" / sp / ip.name)
            if lp2.exists():
                os.replace(lp2, qd / "labels" / sp / lp2.name)
            stats["quar_dvb_053761_heli"] += 1

    print("=== POLISH V3 COMPLETE ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:40s}: {v}")

    # Verify exact remaining counts & 100% pairing in both clean_dataset and data2
    for ds in ("clean_dataset", "data2"):
        tot_i, tot_l, unp = 0, 0, 0
        for sp in ("train", "val", "test"):
            imgs = {p.stem for p in (WORKSPACE / ds / "images" / sp).iterdir() if p.is_file()}
            lbls = {p.stem for p in (WORKSPACE / ds / "labels" / sp).glob("*.txt")}
            tot_i += len(imgs)
            tot_l += len(lbls)
            unp += len(imgs ^ lbls)
        print(f"Post-polish [{ds}]: images={tot_i}, labels={tot_l}, unpaired={unp}")


if __name__ == "__main__":
    main()
