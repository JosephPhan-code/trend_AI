"""
Export YOLO26s-P2 (best.pt) -> FP32 ONNX (best.onnx) -> Static QDQ INT8 ONNX (best_int8.onnx).

Key safeguards implemented:
1. Stratified calibration & disjoint evaluation sets from E:\\trend_AI\\clean_dataset\\images\\val
   (prevents alphabetical slice bias where early files are ~90% empty hard negatives).
2. ONNX IR version capped <= 10 for onnxruntime 1.23.2 compatibility.
3. Quantizes only Conv/Gemm/MatMul while excluding final Detect-head output Conv layers
   (keeps box coordinates & class logits in float so tiny-drone accuracy is preserved).
4. Restores Ultralytics metadata_props (names, stride, task, imgsz, ...) onto best_int8.onnx.
5. Evaluates FP32 vs INT8 on the same 200-image disjoint stratified eval split and benchmarks CPU speed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

# Prevent Ultralytics ONNXBackend from trying to pip-install 'onnxruntime' when 'onnxruntime-gpu' is already installed
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

DEFAULT_WEIGHTS = Path(r"E:\trend_AI\yolo26s_p2_img640\yolo26s_p2_img640\weights\best.pt")
DEFAULT_VAL_IMAGES = Path(r"E:\trend_AI\clean_dataset\images\val")
DEFAULT_VAL_LABELS = Path(r"E:\trend_AI\clean_dataset\labels\val")
DEFAULT_TRACKER = Path(r"E:\trend_AI\clean_dataset\drone_tracker.yaml")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def cap_ir_version(model_path: Path, max_ir: int = 10) -> onnx.ModelProto:
    """Load an ONNX model and cap its IR version for ONNX Runtime compatibility."""
    model = onnx.load(str(model_path))
    if getattr(model, "ir_version", 0) > max_ir:
        print(f"  [IR] Capping ir_version {model.ir_version} -> {max_ir} on {model_path.name}")
        model.ir_version = max_ir
        onnx.save(model, str(model_path))
    return model


def extract_metadata(model: onnx.ModelProto) -> dict[str, str]:
    """Extract custom metadata_props from an ONNX ModelProto."""
    return {prop.key: prop.value for prop in model.metadata_props}


def apply_metadata(model_path: Path, metadata: dict[str, str], extra: dict[str, str] | None = None) -> None:
    """Restore metadata_props and ensure IR version <= 10 on an ONNX file."""
    model = onnx.load(str(model_path))
    if getattr(model, "ir_version", 0) > 10:
        model.ir_version = 10

    merged = dict(metadata)
    if extra:
        merged.update(extra)

    existing_keys = {prop.key for prop in model.metadata_props}
    for k, v in merged.items():
        if k in existing_keys:
            for prop in model.metadata_props:
                if prop.key == k:
                    prop.value = str(v)
        else:
            prop = model.metadata_props.add()
            prop.key = k
            prop.value = str(v)

    onnx.save(model, str(model_path))


def is_positive_label(label_path: Path) -> bool:
    """Return True if YOLO label file exists and contains at least one valid bbox line."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    try:
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(line.strip().split()) >= 5:
                    return True
    except OSError:
        return False
    return False


def _diversified_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    """Sample images spread across filename prefixes so varied scenes/sequences are represented."""
    if count <= 0 or not paths:
        return []
    if count >= len(paths):
        out = list(paths)
        random.Random(seed).shuffle(out)
        return out

    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        prefix = p.stem.split("_")[0]
        buckets[prefix].append(p)

    keys = sorted(buckets.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])

    selected: list[Path] = []
    idx = 0
    while len(selected) < count:
        added_any = False
        for k in keys:
            bucket = buckets[k]
            if idx < len(bucket):
                selected.append(bucket[idx])
                added_any = True
                if len(selected) >= count:
                    break
        if not added_any:
            break
        idx += 1
    return selected


def build_stratified_lists(
    val_img_dir: Path,
    val_lbl_dir: Path,
    calib_count: int,
    eval_count: int,
    seed: int = 42,
) -> tuple[list[Path], list[Path], dict[str, int]]:
    """Build disjoint stratified calibration and evaluation image lists (positives + hard negatives)."""
    positives: list[Path] = []
    negatives: list[Path] = []

    for entry in os.scandir(val_img_dir):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext not in VALID_IMAGE_EXTS:
            continue
        img_path = Path(entry.path)
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        if is_positive_label(lbl_path):
            positives.append(img_path)
        else:
            negatives.append(img_path)

    positives.sort()
    negatives.sort()

    stats = {
        "val_total": len(positives) + len(negatives),
        "val_positives": len(positives),
        "val_negatives": len(negatives),
    }

    if len(positives) == 0:
        raise RuntimeError(f"No positive drone images found in {val_img_dir} / {val_lbl_dir}")

    # Target ~87.5% positives + ~12.5% hard negatives for calibration, ~85% + ~15% for eval
    calib_neg_target = min(max(1, round(calib_count * 0.125)), len(negatives))
    calib_pos_target = min(calib_count - calib_neg_target, len(positives))

    calib_pos = _diversified_sample(positives, calib_pos_target, seed=seed)
    calib_neg = _diversified_sample(negatives, calib_neg_target, seed=seed + 1)
    calib_set = set(calib_pos) | set(calib_neg)

    rem_pos = [p for p in positives if p not in calib_set]
    rem_neg = [p for p in negatives if p not in calib_set]

    eval_neg_target = min(max(1, round(eval_count * 0.15)), len(rem_neg))
    eval_pos_target = min(eval_count - eval_neg_target, len(rem_pos))

    eval_pos = _diversified_sample(rem_pos, eval_pos_target, seed=seed + 2)
    eval_neg = _diversified_sample(rem_neg, eval_neg_target, seed=seed + 3)

    calib_list = calib_pos + calib_neg
    eval_list = eval_pos + eval_neg
    random.Random(seed + 10).shuffle(calib_list)
    random.Random(seed + 11).shuffle(eval_list)

    stats.update({
        "calib_pos": len(calib_pos),
        "calib_neg": len(calib_neg),
        "calib_total": len(calib_list),
        "eval_pos": len(eval_pos),
        "eval_neg": len(eval_neg),
        "eval_total": len(eval_list),
    })
    return calib_list, eval_list, stats


class DroneCalibrationDataReader(CalibrationDataReader):
    """ONNX Runtime CalibrationDataReader using Ultralytics LetterBox preprocessing."""

    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int = 640):
        self.image_paths = image_paths
        self.input_name = input_name
        self.imgsz = imgsz
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True)
        self.idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self.idx < len(self.image_paths):
            path = self.image_paths[self.idx]
            self.idx += 1
            if self.idx % 50 == 0 or self.idx == len(self.image_paths):
                print(f"  [Calib] {self.idx}/{len(self.image_paths)} images", flush=True)

            img = cv2.imread(str(path))
            if img is None:
                continue
            img = self.letterbox(image=img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
            blob = np.expand_dims(blob, axis=0)
            return {self.input_name: blob}
        return None

    def rewind(self) -> None:
        self.idx = 0


def find_nodes_to_exclude(onnx_path: Path) -> tuple[list[str], list[str]]:
    """
    Exclude all non-Conv/Gemm/MatMul ops PLUS strictly the final Detect-head output Conv layers
    (e.g., /model.29/cv2.*/cv2.*.2/Conv and /model.29/cv3.*/cv3.*.2/Conv).
    """
    model = onnx.load(str(onnx_path))
    head_conv_nodes: list[str] = []
    exclude: list[str] = []

    for node in model.graph.node:
        name = node.name or ""
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            exclude.append(name)
            continue

        if node.op_type == "Conv" and ("/cv2." in name or "/cv3." in name) and ".2/Conv" in name:
            exclude.append(name)
            head_conv_nodes.append(name)

    exclude = list(dict.fromkeys(exclude))
    head_conv_nodes = list(dict.fromkeys(head_conv_nodes))
    return exclude, head_conv_nodes


def count_onnx_ops(onnx_path: Path) -> dict[str, int]:
    """Count total nodes and quantization QDQ nodes in an ONNX model."""
    model = onnx.load(str(onnx_path))
    counts: dict[str, int] = defaultdict(int)
    for node in model.graph.node:
        counts[node.op_type] += 1
    return {
        "total_nodes": len(model.graph.node),
        "QuantizeLinear": counts["QuantizeLinear"],
        "DequantizeLinear": counts["DequantizeLinear"],
        "Conv": counts["Conv"],
        "QLinearConv": counts["QLinearConv"],
        "ir_version": int(model.ir_version),
    }


def run_quantization_variant(
    target_onnx: Path,
    output_onnx: Path,
    calib_imgs: list[Path],
    input_name: str,
    imgsz: int,
    nodes_to_exclude: list[str],
    metadata: dict[str, str],
    activation_type: QuantType,
    calibrate_method: CalibrationMethod,
    extra_options: dict,
    variant_name: str,
) -> dict[str, int]:
    """Run static QDQ quantization, restore metadata, and return node statistics."""
    print(f"\n[Quantize] Running variant '{variant_name}' -> {output_onnx.name}", flush=True)
    reader = DroneCalibrationDataReader(calib_imgs, input_name=input_name, imgsz=imgsz)
    quantize_static(
        model_input=str(target_onnx),
        model_output=str(output_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=calibrate_method,
        nodes_to_exclude=nodes_to_exclude,
        extra_options=extra_options,
    )
    apply_metadata(
        output_onnx,
        metadata,
        extra={
            "quantization": f"int8_qdq_{variant_name}",
            "calibration_samples": str(len(calib_imgs)),
        },
    )
    stats = count_onnx_ops(output_onnx)
    print(f"  [Quantize] {variant_name} stats: {stats}", flush=True)
    return stats


def evaluate_model_subset(model_path: Path, eval_yaml: Path, imgsz: int, device: str = "cpu") -> dict[str, float]:
    """Run Ultralytics val() on the disjoint eval YAML and return box metrics."""
    model = YOLO(str(model_path), task="detect")
    metrics = model.val(
        data=str(eval_yaml),
        split="val",
        imgsz=imgsz,
        batch=1,
        device=device,
        plots=False,
        verbose=False,
    )
    return {
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "mAP50": round(float(metrics.box.map50), 5),
        "mAP50_95": round(float(metrics.box.map), 5),
    }


def benchmark_ort_cpu(onnx_path: Path, sample_img: np.ndarray, imgsz: int, warmup: int = 3, runs: int = 15) -> dict[str, float]:
    """Benchmark pure ONNX Runtime CPU inference latency (ms) and FPS."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = min(6, os.cpu_count() or 6)
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True)
    im = lb(image=sample_img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    blob = np.expand_dims(np.ascontiguousarray(im.transpose(2, 0, 1)).astype(np.float32) / 255.0, 0)

    for _ in range(warmup):
        sess.run(out_names, {in_name: blob})

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(out_names, {in_name: blob})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    p50_ms = float(np.median(times_ms))
    return {
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(p50_ms, 2),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def benchmark_ultralytics_end2end(
    model_path: Path,
    sample_img: np.ndarray,
    imgsz: int,
    device: str = "cpu",
    warmup: int = 3,
    runs: int = 15,
) -> dict[str, float]:
    """Benchmark full Ultralytics YOLO.predict() pipeline (preprocess + inference + NMS)."""
    model = YOLO(str(model_path), task="detect")
    for _ in range(warmup):
        model.predict(sample_img, imgsz=imgsz, device=device, conf=0.25, verbose=False)

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(sample_img, imgsz=imgsz, device=device, conf=0.25, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    return {
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(float(np.median(times_ms)), 2),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def smoke_test_tracking(int8_onnx: Path, positive_imgs: list[Path], imgsz: int) -> dict[str, object]:
    """Verify YOLO(best_int8.onnx) loads names={0:'drone'} and runs TrackTrack cleanly."""
    model = YOLO(str(int8_onnx), task="detect")
    names: dict[int, str] = {}
    tracker_cfg = str(DEFAULT_TRACKER) if DEFAULT_TRACKER.exists() else "tracktrack.yaml"
    total_boxes = 0
    for img_path in positive_imgs[:5]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = model.track(
            frame,
            tracker=tracker_cfg,
            persist=True,
            imgsz=imgsz,
            conf=0.25,
            device="cpu",
            verbose=False,
        )
        if not names and res[0].names:
            names = dict(res[0].names)
        total_boxes += len(res[0].boxes)
    return {"names": names, "frames_tested": min(5, len(positive_imgs)), "detections_seen": total_boxes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and quantize YOLO26s-P2 to INT8 ONNX")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--val-images", type=Path, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--val-labels", type=Path, default=DEFAULT_VAL_LABELS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calib-count", type=int, default=320)
    parser.add_argument("--eval-count", type=int, default=200)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    best_pt: Path = args.weights.resolve()
    if not best_pt.exists():
        raise FileNotFoundError(f"Model weights not found: {best_pt}")

    weights_dir = best_pt.parent
    best_onnx = weights_dir / "best.onnx"
    best_int8 = weights_dir / "best_int8.onnx"
    prep_onnx = weights_dir / "_prep_int8_tmp.onnx"
    candidate_alt_onnx = weights_dir / "_alt_int8_tmp.onnx"
    report_json = weights_dir / "int8_quantization_report.json"
    eval_dir = weights_dir / "_eval_subset"

    print("=" * 72, flush=True)
    print(f"YOLO26s-P2 INT8 QUANTIZATION PIPELINE", flush=True)
    print(f"  Source PT  : {best_pt} ({best_pt.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print(f"  Target INT8: {best_int8}", flush=True)
    print(f"  imgsz      : {args.imgsz}", flush=True)
    print("=" * 72, flush=True)

    # 1. Stratified calibration and evaluation sets
    print("\n[1/5] Building stratified calibration & evaluation splits...", flush=True)
    calib_imgs, eval_imgs, split_stats = build_stratified_lists(
        args.val_images, args.val_labels, args.calib_count, args.eval_count
    )
    print(f"  Dataset val stats: {split_stats}", flush=True)

    # 2. Export FP32 ONNX
    print("\n[2/5] Exporting FP32 ONNX (opset=17, simplify=True, cpu)...", flush=True)
    pt_model = YOLO(str(best_pt))
    exported = pt_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=17,
        simplify=True,
        dynamic=False,
        device="cpu",
    )
    exported_path = Path(exported)
    if exported_path.resolve() != best_onnx.resolve():
        shutil.copy2(exported_path, best_onnx)

    fp32_proto = cap_ir_version(best_onnx, max_ir=10)
    metadata = extract_metadata(fp32_proto)
    print(f"  FP32 ONNX saved: {best_onnx.name} ({best_onnx.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print(f"  Metadata keys preserved: {sorted(metadata.keys())} | names={metadata.get('names')}", flush=True)

    # 3. Quantization pre-processing
    print("\n[3/5] Pre-processing ONNX graph for static INT8 quantization...", flush=True)
    target_onnx = best_onnx
    try:
        quant_pre_process(str(best_onnx), str(prep_onnx), skip_symbolic_shape=False)
        cap_ir_version(prep_onnx, max_ir=10)
        target_onnx = prep_onnx
        print("  quant_pre_process succeeded (symbolic shape inference ON).", flush=True)
    except Exception as e1:
        print(f"  quant_pre_process symbolic shape failed ({e1}), retrying with skip_symbolic_shape=True...", flush=True)
        try:
            quant_pre_process(str(best_onnx), str(prep_onnx), skip_symbolic_shape=True)
            cap_ir_version(prep_onnx, max_ir=10)
            target_onnx = prep_onnx
            print("  quant_pre_process succeeded (skip_symbolic_shape=True).", flush=True)
        except Exception as e2:
            print(f"  quant_pre_process skipped ({e2}); using best.onnx directly.", flush=True)
            target_onnx = best_onnx

    target_proto = onnx.load(str(target_onnx))
    input_name = target_proto.graph.input[0].name
    nodes_to_exclude, head_conv_nodes = find_nodes_to_exclude(target_onnx)
    print(f"  Input tensor: '{input_name}'", flush=True)
    print(f"  Excluded non-weighted + head nodes: {len(nodes_to_exclude)} total ({len(head_conv_nodes)} head tail Convs):", flush=True)
    for hn in head_conv_nodes:
        print(f"    - {hn}", flush=True)

    # 4. Run Primary Static QDQ INT8 Quantization (QUInt8 activations, QInt8 symmetric per-channel weights)
    print("\n[4/5] Running Static QDQ INT8 quantization...", flush=True)
    q_stats = run_quantization_variant(
        target_onnx=target_onnx,
        output_onnx=best_int8,
        calib_imgs=calib_imgs,
        input_name=input_name,
        imgsz=args.imgsz,
        nodes_to_exclude=nodes_to_exclude,
        metadata=metadata,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        variant_name="quint8_minmax_perchannel",
    )

    # 5. Smoke test, accuracy validation, and speed benchmark
    print("\n[5/5] Verifying INT8 model, accuracy (200 disjoint val images), and CPU speed...", flush=True)
    pos_sample_paths = [p for p in calib_imgs if is_positive_label(args.val_labels / f"{p.stem}.txt")]
    smoke = smoke_test_tracking(best_int8, pos_sample_paths, args.imgsz)
    print(f"  Smoke test (YOLO + TrackTrack): {smoke}", flush=True)

    eval_metrics: dict[str, object] = {}
    selected_variant = "quint8_minmax_perchannel"

    if not args.skip_eval and len(eval_imgs) > 0:
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_txt = eval_dir / "val_eval.txt"
        eval_yaml = eval_dir / "eval_data.yaml"
        eval_txt.write_text("\n".join(str(p) for p in eval_imgs) + "\n", encoding="utf-8")
        eval_yaml.write_text(
            f"path: {args.val_images.parent.parent.as_posix()}\n"
            f"train: {eval_txt.as_posix()}\n"
            f"val: {eval_txt.as_posix()}\n"
            "nc: 1\n"
            "names:\n"
            "  0: drone\n",
            encoding="utf-8",
        )

        print("  Evaluating FP32 best.pt on 200 disjoint images...", flush=True)
        fp32_pt_metrics = evaluate_model_subset(best_pt, eval_yaml, args.imgsz, device="cpu")
        print(f"  FP32 best.pt metrics: {fp32_pt_metrics}", flush=True)

        print("  Evaluating INT8 best_int8.onnx on 200 disjoint images...", flush=True)
        int8_metrics = evaluate_model_subset(best_int8, eval_yaml, args.imgsz, device="cpu")
        print(f"  INT8 (QUInt8 MinMax) metrics: {int8_metrics}", flush=True)

        drop_map50 = fp32_pt_metrics["mAP50"] - int8_metrics["mAP50"]
        if drop_map50 > 0.05:
            print(
                f"  WARNING: INT8 mAP50 dropped by {drop_map50:.4f} (> 0.05). "
                "Trying fallback QInt8 + Percentile(99.99%) calibration...",
                flush=True,
            )
            alt_stats = run_quantization_variant(
                target_onnx=target_onnx,
                output_onnx=candidate_alt_onnx,
                calib_imgs=calib_imgs,
                input_name=input_name,
                imgsz=args.imgsz,
                nodes_to_exclude=nodes_to_exclude,
                metadata=metadata,
                activation_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.Percentile,
                extra_options={"CalibTensorRangePercentile": 99.99, "WeightSymmetric": True},
                variant_name="qint8_percentile9999",
            )
            alt_metrics = evaluate_model_subset(candidate_alt_onnx, eval_yaml, args.imgsz, device="cpu")
            print(f"  INT8 fallback (QInt8 Percentile) metrics: {alt_metrics}", flush=True)
            if alt_metrics["mAP50"] > int8_metrics["mAP50"]:
                shutil.move(str(candidate_alt_onnx), str(best_int8))
                int8_metrics = alt_metrics
                q_stats = alt_stats
                selected_variant = "qint8_percentile9999"
                print(f"  Selected fallback '{selected_variant}' (higher mAP50).", flush=True)
            else:
                candidate_alt_onnx.unlink(missing_ok=True)
                print(f"  Kept primary '{selected_variant}'.", flush=True)

        eval_metrics = {
            "eval_images": len(eval_imgs),
            "fp32_pt": fp32_pt_metrics,
            "int8_onnx": int8_metrics,
            "delta_mAP50": round(int8_metrics["mAP50"] - fp32_pt_metrics["mAP50"], 5),
            "delta_mAP50_95": round(int8_metrics["mAP50_95"] - fp32_pt_metrics["mAP50_95"], 5),
        }

    # Cleanup temp dirs/files
    prep_onnx.unlink(missing_ok=True)
    candidate_alt_onnx.unlink(missing_ok=True)
    if eval_dir.exists():
        shutil.rmtree(eval_dir, ignore_errors=True)

    # Benchmark CPU speeds
    sample_bgr = cv2.imread(str(pos_sample_paths[0]))
    print("  Benchmarking pure ORT CPU & full Ultralytics CPU pipelines...", flush=True)
    ort_fp32_bench = benchmark_ort_cpu(best_onnx, sample_bgr, args.imgsz)
    ort_int8_bench = benchmark_ort_cpu(best_int8, sample_bgr, args.imgsz)
    e2e_pt_cpu_bench = benchmark_ultralytics_end2end(best_pt, sample_bgr, args.imgsz, device="cpu")
    e2e_fp32_onnx_bench = benchmark_ultralytics_end2end(best_onnx, sample_bgr, args.imgsz, device="cpu")
    e2e_int8_onnx_bench = benchmark_ultralytics_end2end(best_int8, sample_bgr, args.imgsz, device="cpu")

    pt_mb = round(best_pt.stat().st_size / (1024 * 1024), 2)
    fp32_mb = round(best_onnx.stat().st_size / (1024 * 1024), 2)
    int8_mb = round(best_int8.stat().st_size / (1024 * 1024), 2)

    report = {
        "model_name": "yolo26s_p2_img640",
        "imgsz": args.imgsz,
        "selected_variant": selected_variant,
        "weights": {
            "best_pt": str(best_pt),
            "best_onnx": str(best_onnx),
            "best_int8_onnx": str(best_int8),
        },
        "file_sizes_mb": {
            "best_pt": pt_mb,
            "best_onnx": fp32_mb,
            "best_int8_onnx": int8_mb,
            "reduction_vs_pt_pct": round((1.0 - int8_mb / pt_mb) * 100.0, 1),
            "reduction_vs_onnx_pct": round((1.0 - int8_mb / fp32_mb) * 100.0, 1),
        },
        "dataset_split": split_stats,
        "onnx_graph_stats": q_stats,
        "excluded_head_conv_nodes": head_conv_nodes,
        "smoke_test": smoke,
        "accuracy": eval_metrics,
        "benchmark_cpu": {
            "ort_raw_fp32": ort_fp32_bench,
            "ort_raw_int8": ort_int8_bench,
            "ultralytics_pt_cpu": e2e_pt_cpu_bench,
            "ultralytics_fp32_onnx_cpu": e2e_fp32_onnx_bench,
            "ultralytics_int8_onnx_cpu": e2e_int8_onnx_bench,
            "ort_speedup_x": round(ort_fp32_bench["mean_ms"] / ort_int8_bench["mean_ms"], 2)
            if ort_int8_bench["mean_ms"] > 0
            else 0.0,
        },
    }

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72, flush=True)
    print("INT8 QUANTIZATION SUMMARY", flush=True)
    print(f"  Filesize   : PT={pt_mb} MB | FP32 ONNX={fp32_mb} MB | INT8 ONNX={int8_mb} MB (-{report['file_sizes_mb']['reduction_vs_pt_pct']}% vs PT)", flush=True)
    print(f"  QDQ Nodes  : QuantizeLinear={q_stats['QuantizeLinear']} | DequantizeLinear={q_stats['DequantizeLinear']} | IR={q_stats['ir_version']}", flush=True)
    if eval_metrics:
        print(
            f"  mAP50      : FP32={eval_metrics['fp32_pt']['mAP50']:.4f} | "
            f"INT8={eval_metrics['int8_onnx']['mAP50']:.4f} (delta={eval_metrics['delta_mAP50']:+.4f})",
            flush=True,
        )
        print(
            f"  mAP50-95   : FP32={eval_metrics['fp32_pt']['mAP50_95']:.4f} | "
            f"INT8={eval_metrics['int8_onnx']['mAP50_95']:.4f} (delta={eval_metrics['delta_mAP50_95']:+.4f})",
            flush=True,
        )
    print(
        f"  ORT CPU    : FP32={ort_fp32_bench['mean_ms']} ms ({ort_fp32_bench['fps']} FPS) | "
        f"INT8={ort_int8_bench['mean_ms']} ms ({ort_int8_bench['fps']} FPS) "
        f"[{report['benchmark_cpu']['ort_speedup_x']}x]",
        flush=True,
    )
    print(
        f"  E2E CPU    : PT={e2e_pt_cpu_bench['mean_ms']} ms | "
        f"FP32_ONNX={e2e_fp32_onnx_bench['mean_ms']} ms | "
        f"INT8_ONNX={e2e_int8_onnx_bench['mean_ms']} ms",
        flush=True,
    )
    print(f"  Report     : {report_json}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
