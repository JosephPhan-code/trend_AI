"""
==============================================================================
Module: detector_engine.py
Mô tả : Các lớp OOP cho suy luận AI (BaseDetector, HailoDetector, CpuFallbackDetector).
        Tối ưu hóa:
          1. Lọc ngưỡng trực tiếp trên UINT8/UINT16 (postprocess_hailo_quant) trước
             khi dequantize để loại bỏ 99.8% phép tính dequantize nền trời trên CPU.
          2. Tận dụng luồng PiSP lores (640x360) của Pi 5 để bỏ hoàn toàn cv2.resize
             ở chế độ FULL frame (nhanh hơn 44x).
          3. Tự động tương thích với mọi biến thể model: YOLO26s-P2 (4 nhánh P2..P5,
             direct LTRB c=4), YOLOv8/YOLO11 (DFL c=64), và End-to-End NMS-free.
==============================================================================
"""

import math
import time
from queue import Empty
import cv2
import numpy as np

from config import (
    CONF_THRESHOLD,
    DETECTION_FLOOR,
    HEF_MODEL_PATH,
    INFER_SIZE,
    IOU_THRESHOLD,
    LORES_GAIN,
    LORES_HEIGHT,
    LORES_PAD_Y,
    LORES_WIDTH,
    MOTION_CUE_ENABLED,
    NUM_CLASSES,
    PT_MODEL_PATH,
    REG_MAX,
    ROI_HINT_TTL,
    STATE,
    STOP_EVENT,
    TARGET_GUIDE,
    RateMeter,
    put_latest,
    q_data,
    q_raw,
)
from tracker_ekf import (
    MotionDetector,
    RoiPlanner,
    _offset_detections,
    detect_with_roi,
)


def _make_anchors(input_size=640, strides=(4, 8, 16, 32)):
    anchors = {}
    for s in strides:
        g = input_size // s
        y, x = np.mgrid[0:g, 0:g]
        xy = (np.stack([x, y], -1).reshape(-1, 2) + 0.5) * s
        anchors[g] = (s, xy.astype(np.float32))
    return anchors


_STRIDES = [4, 8, 16, 32]
_ANCHORS_BY_GRID = _make_anchors(INFER_SIZE, _STRIDES)
_ANCHORS = [_ANCHORS_BY_GRID[INFER_SIZE // s][1] for s in _STRIDES]
_LOGIT_FLOOR = float(np.log(DETECTION_FLOOR / max(1.0 - DETECTION_FLOOR, 1e-6)))


def _dfl(pred):
    """DFL decode: (N, 4*REG_MAX) -> (N, 4) khoảng cách l,t,r,b."""
    N = pred.shape[0]
    p = pred.reshape(N, 4, REG_MAX)
    p = np.exp(p - p.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return (p * np.arange(REG_MAX, dtype=np.float32)).sum(-1)


def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def letterbox_image(frame, size=INFER_SIZE, color=114):
    """Resize without stretching; return image and (gain, pad_x, pad_y)."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty HxWxC array")
    oh, ow = frame.shape[:2]
    if ow == size and oh == size:
        return frame, (1.0, 0.0, 0.0)
    gain = min(size / max(ow, 1), size / max(oh, 1))
    nw, nh = max(1, int(round(ow * gain))), max(1, int(round(oh * gain)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, frame.shape[2]), color, dtype=frame.dtype)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, (float(gain), float(pad_x), float(pad_y))


def _restore_boxes(boxes, orig_hw, letterbox_meta):
    """Map boxes from 640 input coordinates back to the camera image."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    oh, ow = orig_hw
    if letterbox_meta is None:
        boxes[:, [0, 2]] *= ow / INFER_SIZE
        boxes[:, [1, 3]] *= oh / INFER_SIZE
    else:
        gain, pad_x, pad_y = letterbox_meta
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(gain, 1e-9)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(gain, 1e-9)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)
    return boxes


def _decode_result_dict(boxes, scores, orig_hw, letterbox_meta):
    boxes = _restore_boxes(boxes, orig_hw, letterbox_meta)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (scores > DETECTION_FLOOR)
    )
    if not valid.any():
        return []
    boxes, scores = boxes[valid], scores[valid]
    keep = _nms(boxes, scores, IOU_THRESHOLD)
    return [{
        "cx": float((boxes[i, 0] + boxes[i, 2]) / 2),
        "cy": float((boxes[i, 1] + boxes[i, 3]) / 2),
        "w": float(boxes[i, 2] - boxes[i, 0]),
        "h": float(boxes[i, 3] - boxes[i, 1]),
        "conf": float(scores[i]),
        "x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
        "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]),
    } for i in keep]


def _decode_end_to_end(raw_out, orig_hw, letterbox_meta):
    """Decode YOLO26 rows [x1,y1,x2,y2,score,class]."""
    for tensor in raw_out.values():
        arr = np.asarray(tensor)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            continue
        if arr.shape[1] != 6 and arr.shape[0] == 6:
            arr = arr.T
        if arr.shape[1] < 6 or arr.shape[1] > 8:
            continue
        rows = arr[:, :6].astype(np.float32, copy=False)
        finite_rows = np.isfinite(rows).all(axis=1)
        if not finite_rows.any():
            return []
        rows = rows[finite_rows]
        coord_max = float(np.max(np.abs(rows[:, :4]))) if rows.size else 0.0
        boxes = rows[:, :4] * (INFER_SIZE if coord_max <= 2.0 else 1.0)
        scores = rows[:, 4]
        if scores.min() < 0 or scores.max() > 1.01:
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
        return _decode_result_dict(boxes, scores, orig_hw, letterbox_meta)
    return None


def postprocess_hailo(raw_out, orig_hw, letterbox_meta=None):
    """Decode FLOAT32 YOLO26 end2end output or multi-scale YOLO heads from Hailo."""
    end_to_end = _decode_end_to_end(raw_out, orig_hw, letterbox_meta)
    if end_to_end is not None:
        return end_to_end

    cv2_out, cv3_out = {}, {}
    for name, tensor in raw_out.items():
        t = tensor[0] if tensor.ndim == 4 else tensor
        if t.ndim != 3:
            continue
        h, w, c = t.shape
        if h != w or h not in _ANCHORS_BY_GRID:
            continue
        if c == 4 or c == 4 * REG_MAX:
            cv2_out[h] = t
        elif c == NUM_CLASSES:
            cv3_out[h] = t

    all_boxes, all_scores = [], []
    for g in sorted(cv2_out.keys(), reverse=True):
        if g not in cv3_out:
            continue
        stride, grid_anchors = _ANCHORS_BY_GRID[g]
        bf = cv2_out[g].reshape(-1, cv2_out[g].shape[-1])
        cf = cv3_out[g].reshape(-1, NUM_CLASSES)

        raw_max = cf.max(-1)
        is_logits = bool(cf.min() < 0 or cf.max() > 1.01)
        if is_logits:
            mask = raw_max > _LOGIT_FLOOR
            if not mask.any():
                continue
            scores_sel = 1.0 / (1.0 + np.exp(-np.clip(raw_max[mask], -40.0, 40.0)))
        else:
            mask = raw_max > DETECTION_FLOOR
            if not mask.any():
                continue
            scores_sel = raw_max[mask]

        if bf.shape[-1] == 4 * REG_MAX:
            ltrb = _dfl(bf[mask]) * stride
        else:
            ltrb = bf[mask] * stride

        anc = grid_anchors[mask]
        boxes_640 = np.stack([
            anc[:, 0] - ltrb[:, 0], anc[:, 1] - ltrb[:, 1],
            anc[:, 0] + ltrb[:, 2], anc[:, 1] + ltrb[:, 3],
        ], -1)
        all_boxes.append(_restore_boxes(boxes_640, orig_hw, letterbox_meta))
        all_scores.append(scores_sel)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = _nms(boxes, scores, IOU_THRESHOLD)

    return [{
        "cx": float((boxes[i, 0] + boxes[i, 2]) / 2),
        "cy": float((boxes[i, 1] + boxes[i, 3]) / 2),
        "w": float(boxes[i, 2] - boxes[i, 0]),
        "h": float(boxes[i, 3] - boxes[i, 1]),
        "conf": float(scores[i]),
        "x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
        "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]),
    } for i in keep]


def postprocess_hailo_quant(quant_out, qmap, orig_hw, letterbox_meta=None):
    """Giải mã trực tiếp từ tensor UINT8/UINT16 của Hailo-8:
    Lọc ngưỡng trên miền số nguyên trước, chỉ dequantize đúng các ô vượt ngưỡng (tiết kiệm >99% phép tính dequantize).
    """
    cv2_out, cv3_out = {}, {}
    for name, tensor in quant_out.items():
        t = tensor[0] if tensor.ndim == 4 else tensor
        if t.ndim != 3:
            continue
        h, w, c = t.shape
        if h != w or h not in _ANCHORS_BY_GRID:
            continue
        if c == 4 or c == 4 * REG_MAX:
            cv2_out[h] = (name, t)
        elif c == NUM_CLASSES:
            cv3_out[h] = (name, t)

    if not cv2_out or not cv3_out:
        # Fallback cho mô hình End-to-End 2D nếu có
        f32_out = {
            name: ((arr.astype(np.float32) - qmap[name][1]) * qmap[name][0])[np.newaxis, ...]
            for name, arr in quant_out.items()
        }
        return postprocess_hailo(f32_out, orig_hw, letterbox_meta)

    all_boxes, all_scores = [], []
    for g in sorted(cv2_out.keys(), reverse=True):
        if g not in cv3_out:
            continue
        stride, grid_anchors = _ANCHORS_BY_GRID[g]
        b_name, b_t = cv2_out[g]
        c_name, c_t = cv3_out[g]
        b_scale, b_zp = qmap[b_name][0], qmap[b_name][1]
        c_scale, c_zp = qmap[c_name][0], qmap[c_name][1]

        cf_q = c_t.reshape(-1, NUM_CLASSES)
        raw_q = cf_q[:, 0] if NUM_CLASSES == 1 else cf_q.max(-1)

        # Kiểm tra miền giá trị đã lượng tử hóa là xác suất [0..1] hay logits [-inf..+inf]
        max_q_val = 65535.0 if raw_q.dtype == np.uint16 else 255.0
        min_deq = (0.0 - c_zp) * c_scale
        max_deq = (max_q_val - c_zp) * c_scale
        is_logits = bool(min_deq < -0.05 or max_deq > 1.05)

        thr_float = _LOGIT_FLOOR if is_logits else DETECTION_FLOOR
        thr_q = int(math.floor(thr_float / max(c_scale, 1e-9) + c_zp))
        mask = raw_q > thr_q
        if not mask.any():
            continue

        raw_sel = (raw_q[mask].astype(np.float32) - c_zp) * c_scale
        if is_logits:
            scores_sel = 1.0 / (1.0 + np.exp(-np.clip(raw_sel, -40.0, 40.0)))
        else:
            scores_sel = raw_sel

        valid_s = scores_sel > DETECTION_FLOOR
        if not valid_s.any():
            continue

        if not valid_s.all():
            idx = np.flatnonzero(mask)[valid_s]
            scores_sel = scores_sel[valid_s]
            bf_sel = (b_t.reshape(-1, b_t.shape[-1])[idx].astype(np.float32) - b_zp) * b_scale
            anc = grid_anchors[idx]
        else:
            bf_sel = (b_t.reshape(-1, b_t.shape[-1])[mask].astype(np.float32) - b_zp) * b_scale
            anc = grid_anchors[mask]

        if bf_sel.shape[-1] == 4 * REG_MAX:
            ltrb = _dfl(bf_sel) * stride
        else:
            ltrb = bf_sel * stride

        boxes_640 = np.stack([
            anc[:, 0] - ltrb[:, 0], anc[:, 1] - ltrb[:, 1],
            anc[:, 0] + ltrb[:, 2], anc[:, 1] + ltrb[:, 3],
        ], -1)
        all_boxes.append(_restore_boxes(boxes_640, orig_hw, letterbox_meta))
        all_scores.append(scores_sel)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = _nms(boxes, scores, IOU_THRESHOLD)

    return [{
        "cx": float((boxes[i, 0] + boxes[i, 2]) / 2),
        "cy": float((boxes[i, 1] + boxes[i, 3]) / 2),
        "w": float(boxes[i, 2] - boxes[i, 0]),
        "h": float(boxes[i, 3] - boxes[i, 1]),
        "conf": float(scores[i]),
        "x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
        "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]),
    } for i in keep]


def _push_results(frame, detections, ai_fps, timestamp, gimbal=None, data_queue=None):
    target_q = data_queue or q_data
    visible = [d for d in detections if d["conf"] >= CONF_THRESHOLD]
    mc = max((d["conf"] for d in visible), default=0.0)
    STATE.update_stats(ai_fps=ai_fps, drone_count=len(visible), max_conf=mc)
    if visible:
        STATE.add_detections(len(visible))
    payload = {
        "detections": detections,
        "drone_count": len(visible),
        "frame_w": frame.shape[1],
        "frame_h": frame.shape[0],
        "frame": frame,
        "timestamp": timestamp,
        "ai_fps": ai_fps,
        "gimbal": gimbal,
    }
    put_latest(target_q, payload)


class BaseDetector:
    """Giao diện trừu tượng (Abstract Interface) cho mọi AI Detector backend."""

    def run_loop(self):
        raise NotImplementedError


class HailoDetector(BaseDetector):
    """Đóng gói OOP trình suy luận Hailo-8 NPU:
    - Sử dụng trực tiếp output UINT8/UINT16 + postprocess_hailo_quant (lọc ngưỡng trước khi dequantize).
    - Tận dụng luồng PiSP lores (640x360) để bỏ cv2.resize khi chạy FULL frame.
    - Chồng lấp (Async Overlap) thời gian chạy NPU với MotionDetector trên CPU.
    """

    def __init__(self, hef_path=HEF_MODEL_PATH, raw_queue=None, data_queue=None,
                 target_guide=None, stop_event=None):
        self.hef_path = hef_path
        self.raw_queue = raw_queue or q_raw
        self.data_queue = data_queue or q_data
        self.target_guide = target_guide or TARGET_GUIDE
        self.stop_event = stop_event or STOP_EVENT

    def run_loop(self):
        from hailo_platform import HEF, VDevice, FormatType

        print(f"[AI]  Nap Hailo NPU: {self.hef_path}")
        hef_meta = HEF(self.hef_path)
        qmap = {}
        for vinfo in hef_meta.get_output_vstream_infos():
            qi = vinfo.quant_info
            qmap[vinfo.name] = (float(qi.qp_scale), float(qi.qp_zp), vinfo.format.type)

        with VDevice(VDevice.create_params()) as vd:
            if hasattr(vd, "create_infer_model"):
                infer_model = vd.create_infer_model(self.hef_path)
                infer_model.input().set_format_type(FormatType.UINT8)
                use_quant_out = True
                out_dtypes = {}
                for out_s in infer_model.outputs:
                    v_fmt = qmap.get(out_s.name, (1.0, 0.0, FormatType.FLOAT32))[2]
                    if v_fmt == FormatType.UINT8:
                        out_s.set_format_type(FormatType.UINT8)
                        out_dtypes[out_s.name] = np.uint8
                    elif v_fmt == FormatType.UINT16:
                        out_s.set_format_type(FormatType.UINT16)
                        out_dtypes[out_s.name] = np.uint16
                    else:
                        out_s.set_format_type(FormatType.FLOAT32)
                        out_dtypes[out_s.name] = np.float32
                        use_quant_out = False

                with infer_model.configure() as cfg_m:
                    cfg_m.activate()
                    try:
                        bindings = cfg_m.create_bindings()
                        buf_in = np.full((INFER_SIZE, INFER_SIZE, 3), 114, dtype=np.uint8)
                        bindings.input().set_buffer(buf_in)
                        out_bufs = {}
                        out_views = {}
                        for out_s in infer_model.outputs:
                            arr = np.empty(out_s.shape, dtype=out_dtypes[out_s.name])
                            bindings.output(out_s.name).set_buffer(arr)
                            out_bufs[out_s.name] = arr
                            out_views[out_s.name] = arr[np.newaxis, ...]

                        mode_tag = "Native UINT8/INT8 Zero-Dequant" if use_quant_out else "FLOAT32"
                        print(f"[AI]  OK Hailo-8 26 TOPS (Async Overlapped | {mode_tag})! Input: {infer_model.input().name}")
                        STATE.update_stats(backend="HAILO-8 NPU")
                        fps_meter = RateMeter(alpha=0.15)
                        roi_planner = RoiPlanner()
                        motion_detector = MotionDetector(enabled=MOTION_CUE_ENABLED)
                        cached_motion = None
                        cached_motion_t = 0.0
                        was_crop = True

                        while not self.stop_event.is_set():
                            try:
                                packet = self.raw_queue.get(timeout=1.0)
                            except Empty:
                                continue
                            frame = packet["frame"]
                            lores = packet.get("lores")
                            stamp = packet["timestamp"]

                            guide = self.target_guide.snapshot(stamp)
                            is_locked = guide["locked"]
                            has_hint = (
                                roi_planner.hint is not None
                                and stamp - roi_planner.hint_time <= ROI_HINT_TTL
                            )
                            motion_hint = (
                                cached_motion
                                if (
                                    not is_locked and not has_hint
                                    and cached_motion is not None
                                    and stamp - cached_motion_t <= 0.20
                                )
                                else None
                            )

                            region = roi_planner.next_region(
                                frame.shape, stamp, target_guide=self.target_guide, motion_hint=motion_hint
                            )
                            if region is None:
                                oh, ow = frame.shape[:2]
                                if (
                                    lores is not None
                                    and lores.shape[0] == LORES_HEIGHT
                                    and lores.shape[1] == LORES_WIDTH
                                ):
                                    if was_crop:
                                        buf_in[:LORES_PAD_Y, :] = 114
                                        buf_in[LORES_PAD_Y + LORES_HEIGHT:, :] = 114
                                        was_crop = False
                                    cv2.cvtColor(
                                        lores,
                                        cv2.COLOR_BGR2RGB,
                                        dst=buf_in[LORES_PAD_Y:LORES_PAD_Y + LORES_HEIGHT, 0:LORES_WIDTH],
                                    )
                                    letterbox_meta = (LORES_GAIN, 0.0, float(LORES_PAD_Y))
                                else:
                                    was_crop = True
                                    img_lb, letterbox_meta = letterbox_image(frame)
                                    cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB, dst=buf_in)
                            else:
                                was_crop = True
                                x0, y0, cw, ch = region
                                img_src = frame[y0:y0 + ch, x0:x0 + cw]
                                oh, ow = img_src.shape[:2]
                                if ow == INFER_SIZE and oh == INFER_SIZE:
                                    cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB, dst=buf_in)
                                    letterbox_meta = (1.0, 0.0, 0.0)
                                else:
                                    img_lb, letterbox_meta = letterbox_image(img_src)
                                    cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB, dst=buf_in)

                            job = cfg_m.run_async([bindings], lambda completion_info: None)
                            if is_locked or has_hint:
                                motion_detector.reset()
                                cached_motion = None
                            else:
                                cached_motion = motion_detector.detect_hint(frame, stamp, lores_frame=lores)
                                cached_motion_t = stamp
                            job.wait(2000)

                            if use_quant_out:
                                detections = postprocess_hailo_quant(out_bufs, qmap, (oh, ow), letterbox_meta)
                            else:
                                detections = postprocess_hailo(out_views, (oh, ow), letterbox_meta)

                            if region is not None:
                                _offset_detections(detections, region[0], region[1])
                            roi_planner.update(
                                detections, stamp, target_guide=self.target_guide, motion_detector=motion_detector
                            )
                            STATE.update_stats(roi_mode=roi_planner.last_mode)
                            ai_fps = fps_meter.update(time.monotonic())
                            _push_results(
                                frame, detections, ai_fps, stamp,
                                gimbal=packet.get("gimbal"), data_queue=self.data_queue,
                            )
                    finally:
                        try:
                            cfg_m.deactivate()
                        except Exception:
                            pass
                    return

        from hailo_platform import (
            ConfigureParams,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )
        target = VDevice()
        hef_obj = HEF(self.hef_path)
        cfg_p = ConfigureParams.create_from_hef(hef_obj, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef_obj, cfg_p)[0]
        ng_p = ng.create_params()
        inp_p = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        with InferVStreams(ng, inp_p, out_p) as pipeline:
            inp_name = hef_obj.get_input_vstream_infos()[0].name
            print(f"[AI]  OK Hailo-8 26 TOPS! Input: {inp_name}")
            STATE.update_stats(backend="HAILO-8 NPU")
            fps_meter = RateMeter(alpha=0.15)
            roi_planner = RoiPlanner()
            motion_detector = MotionDetector(enabled=MOTION_CUE_ENABLED)
            with ng.activate(ng_p):
                while not self.stop_event.is_set():
                    try:
                        packet = self.raw_queue.get(timeout=1.0)
                    except Empty:
                        continue
                    frame = packet["frame"]

                    def _hailo_detect(img_src):
                        oh, ow = img_src.shape[:2]
                        img, letterbox_meta = letterbox_image(img_src)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        raw = pipeline.infer({inp_name: np.expand_dims(img_rgb, 0)})
                        return postprocess_hailo(raw, (oh, ow), letterbox_meta)

                    detections = detect_with_roi(
                        frame, roi_planner, _hailo_detect, packet["timestamp"],
                        target_guide=self.target_guide, motion_detector=motion_detector,
                    )
                    ai_fps = fps_meter.update(time.monotonic())
                    _push_results(
                        frame, detections, ai_fps, packet["timestamp"],
                        gimbal=packet.get("gimbal"), data_queue=self.data_queue,
                    )


class CpuFallbackDetector(BaseDetector):
    """Đóng gói OOP trình suy luận CPU dự phòng bằng Ultralytics YOLO (best.pt)."""

    def __init__(self, pt_path=PT_MODEL_PATH, raw_queue=None, data_queue=None,
                 target_guide=None, stop_event=None):
        self.pt_path = pt_path
        self.raw_queue = raw_queue or q_raw
        self.data_queue = data_queue or q_data
        self.target_guide = target_guide or TARGET_GUIDE
        self.stop_event = stop_event or STOP_EVENT

    def run_loop(self):
        try:
            import torch
            _orig = torch.load

            def _p(*a, **kw):
                kw["weights_only"] = False
                return _orig(*a, **kw)

            torch.load = _p
        except Exception:
            pass

        from ultralytics import YOLO
        print(f"[AI]  Nap CPU model: {self.pt_path}")
        model = YOLO(self.pt_path)
        STATE.update_stats(backend="CPU (best.pt)")
        print("[AI]  OK CPU model (FPS thap hon Hailo)")
        fps_meter = RateMeter(alpha=0.15)
        roi_planner = RoiPlanner()
        motion_detector = MotionDetector(enabled=MOTION_CUE_ENABLED)

        while not self.stop_event.is_set():
            try:
                packet = self.raw_queue.get(timeout=1.0)
            except Empty:
                continue
            frame = packet["frame"]

            def _cpu_detect(img_src):
                results = model.predict(
                    img_src, imgsz=INFER_SIZE,
                    conf=DETECTION_FLOOR, iou=IOU_THRESHOLD, verbose=False,
                )
                dets = []
                if results[0].boxes:
                    for box, conf in zip(
                        results[0].boxes.xyxy.cpu().numpy(),
                        results[0].boxes.conf.cpu().tolist(),
                    ):
                        x1, y1, x2, y2 = box
                        dets.append({
                            "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2),
                            "w": float(x2 - x1), "h": float(y2 - y1),
                            "conf": float(conf),
                            "x1": float(x1), "y1": float(y1),
                            "x2": float(x2), "y2": float(y2),
                        })
                return dets

            detections = detect_with_roi(
                frame, roi_planner, _cpu_detect, packet["timestamp"],
                target_guide=self.target_guide, motion_detector=motion_detector,
            )
            ai_fps = fps_meter.update(time.monotonic())
            _push_results(
                frame, detections, ai_fps, packet["timestamp"],
                gimbal=packet.get("gimbal"), data_queue=self.data_queue,
            )


def _run_hailo_loop():
    HailoDetector().run_loop()


def _run_cpu_loop():
    CpuFallbackDetector().run_loop()


def thread_ai():
    try:
        _run_hailo_loop()
    except Exception as e:
        if STOP_EVENT.is_set():
            return
        print(f"[AI]  Hailo NPU loi ({e}) -> Fallback CPU")
        _run_cpu_loop()
