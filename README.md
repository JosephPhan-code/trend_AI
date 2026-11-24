# Hệ Thống AI Phát Hiện & Bám Bắt Drone Thời Gian Thực (Edge Pi 5 + Hailo-8 26 TOPS)

Hệ thống phát hiện, bám bắt (Lock-on Tracking) và điều khiển Gimbal 2 trục Pan/Tilt tự động cho mục tiêu bay kích thước nhỏ (Drone/UAV), tối ưu hóa chuyên sâu cho **Raspberry Pi 5 + NPU Hailo-8 (26 TOPS) + Pi Camera V3 (`IMX708`)**.

---

## 1. Kiến Trúc Tổng Thể Dự Án

Dự án được chia thành **4 phân hệ chính**:

```
trend_AI/
├── edge_pi/                  # [PRODUCTION] Bộ mã nguồn OOP chạy trên Raspberry Pi 5 + Hailo-8
│   ├── main.py               # Entry Point & Orchestrator (AegisTrackerApplication)
│   ├── config.py             # Cấu hình hệ thống, hằng số, SharedState, TargetRoiGuide, Buffered Logger
│   ├── camera_engine.py      # CameraEngine (PiSP Dual-Stream Hardware Resize + Dynamic AF + Spot AE)
│   ├── detector_engine.py    # HailoDetector (UINT8/UINT16 Threshold-First Zero-Dequant + Async NPU)
│   ├── tracker_ekf.py        # ExtendedKalmanTracker (8-state EKF + Smith-Predictor), DroneLockon, RoiPlanner
│   ├── controller_pid.py     # StableTrackingController (Dual PID + Velocity Feed-Forward + Coasting)
│   ├── servo_controller.py   # ServoController (RP1 Hardware PWM Sysfs 50Hz / Gpiozero fallback)
│   ├── web_dashboard.py      # WebDashboardServer (Flask MJPEG Telemetry) & LocalDisplayWorker (HDMI)
│   └── run_drone_tracker.sh  # Script khởi chạy hệ thống ở chế độ hiệu năng tối đa (120 FPS + CPU Performance)
│
├── Archive/                  # [COMPILER] Pipeline biên dịch & lượng tử hóa mô hình sang Hailo-8 (.hef)
│   ├── step1_pt_to_onnx.py   # Xuất PyTorch (.pt) -> ONNX
│   └── step2_onnx_to_hef.py  # Biên dịch ONNX -> HAR -> INT8 Quantization -> Single-Context HEF
│
├── docs/                     # [DOCUMENTATION] Tài liệu giải phẫu hiệu năng & cấu hình Best Practice cho YOLO26s
│   ├── PI5_HAILO8_OPTIMIZATION_GUIDE.md          # Hướng dẫn chi tiết tối ưu Pi 5 + Hailo-8 & 3 Best Practices cho YOLO26s-P2
│   ├── README_TRAIN_BP2_YOLO26S_P2_MAX512.md     # [BÀN GIAO TRAIN 640] BP2: YOLO26s-P2 (max_channels=512, 4 nhánh @ 640x640)
│   ├── README_TRAIN_BP3_YOLO26S_P2_P3_P4.md      # [BÀN GIAO TRAIN 640] BP3: YOLO26s-P2-P3-P4 (No P5, 3 nhánh @ 640x640)
│   ├── README_TRAIN_BP2_YOLO26S_P2_MAX512_960.md # [BÀN GIAO TRAIN 960] BP2: YOLO26s-P2 (max_channels=512, 4 nhánh @ 960x960)
│   ├── README_TRAIN_BP3_YOLO26S_P2_P3_P4_960.md  # [BÀN GIAO TRAIN 960] BP3: YOLO26s-P2-P3-P4 (No P5, 3 nhánh @ 960x960)
│   ├── yolo26s_p2_max512.yaml                    # Cấu hình YAML Best Practice 2 (~4.80M params)
│   └── yolo26s_p2_p3_p4.yaml                     # Cấu hình YAML Best Practice 3 (~2.95M params)
│
├── drone_test_web/           # [PC TESTER] Web Dashboard kiểm thử & đánh giá mô hình trên PC (Flask + ByteTrack)
│   └── app.py                # Giao diện test video/camera trên PC (PyTorch .pt / ONNX)
│
├── train_yolo26_p2.py        # [TRAINING] Huấn luyện kiến trúc YOLO26s-P2 (4-head P2/P3/P4/P5 cho drone ở xa)
├── finetune_anti_fp.py       # Fine-tune chống báo giả (Hard Negative Mining bầu trời/chim/mây)
├── clean_and_audit_dataset.py# Kiểm toán & làm sạch nhãn dataset tự động
├── requirements-pc.txt       # Thư viện cho máy trạm huấn luyện / kiểm thử (PC GPU)
└── requirements-pi.txt       # Thư viện cho bộ xử lý biên (Raspberry Pi 5 + Hailo-8)
```

---

## 2. Các Tối Ưu Hóa Phần Cứng Trọng Tâm Trên Raspberry Pi 5 (`edge_pi/`)

1. **Threshold-First `UINT8`/`UINT16` VStream Decoding (`postprocess_hailo_quant`)**:
   - Đọc trực tiếp luồng tensor lượng tử hóa (`UINT8` hoặc `UINT16`) từ DMA của Hailo-8 thay vì ép `HailoRT` chuyển đổi toàn bộ 8.400–34.000 anchor sang `FLOAT32` trên CPU.
   - Quy đổi ngưỡng điểm tự tin (`conf_thresh`) sang miền số nguyên `int_thresh = ceil(target_val / qp_scale + qp_zp)` và lọc bỏ **>99.8% anchor nền** bằng phép so sánh số nguyên trước khi giải lượng tử (dequantize) và tính `DFL Softmax` cho số ít bounding box vượt ngưỡng.
   - Giảm thời gian hậu xử lý trên CPU Cortex-A76 xuống **~0.49 ms – 0.67 ms/frame** (nhanh gấp **2.8 lần** với 8.400 anchors và tiết kiệm 75% băng thông bộ nhớ VStream).

2. **PiSP Hardware Dual-Stream (`main: 1280×720` + `lores: 640×360`)**:
   - Tận dụng bộ xử lý ảnh phần cứng **PiSP (`BCM2712_C0`)** trên Raspberry Pi 5 để xuất đồng thời luồng `main` (`1280×720`) cho chế độ cắt cận cảnh `ROI Crop 640×640` và luồng `lores` (`640×360`) đã thu nhỏ bằng phần cứng cho chế độ toàn khung `FULL`.
   - Loại bỏ hoàn toàn lệnh `cv2.resize(1280×720 -> 640×360)` trên CPU ở chế độ `FULL`, giảm thời gian tiền xử lý letterbox từ **`1.27 ms` xuống `0.028 ms` (nhanh gấp 44.8 lần)**.

3. **Bất Đồng Bộ Hóa NPU & CPU (`Async Overlapped Inference`)**:
   - Trong khi NPU Hailo-8 đang thực thi suy luận (`configured.run_async`), CPU đồng thời tính toán `cv2.phaseCorrelate` (`MotionDetector`) để phát hiện chuyển động bù rung nền mà không tốn thêm độ trễ tuần tự.

4. **8-State Smith-Predictor EKF & Dual-Loop PID + Feed-Forward**:
   - Bộ lọc Kalman mở rộng 8 trạng thái `[x, y, vx, vy, ax, ay, w, h]` kết hợp bù trễ quán tính servo (`Smith-Predictor`) và quy chiếu hệ tọa độ Thế giới (`World-Frame Ego-Motion Compensation`).
   - Hỗ trợ tự động điều chỉnh cửa sổ lấy nét `AfWindows` và đo sáng điểm `TargetSpotAeController` bám theo tọa độ hộp giới hạn của Drone.

---

## 3. Tiêu Chuẩn Định Dạng Mô Hình (Model Format Standard)

| Môi trường / Giai đoạn | Định dạng chuẩn | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **1. Huấn luyện & Fine-tune (PC GPU)** | **PyTorch (`.pt`)** | Lưu trọng số `FP32`/`FP16` để tiếp tục fine-tune (`train_yolo26_p2.py`, `finetune_anti_fp.py`). |
| **2. Trung gian xuất đồ thị** | **ONNX (`.onnx`)** | Xuất với `opset=11` hoặc `13`, `dynamic=False`, `simplify=True`, kích thước cố định `640×640`. |
| **3. Chạy thực tế trên Pi 5 + Hailo-8** | **Hailo Executable (`.hef`)** | Lượng tử hóa **Full INT8** cho trọng số/nhân chập, VStream đầu vào `UINT8 (640×640×3)`, VStream đầu ra `UINT8` (6 hoặc 8 nhánh: `reg_stride` `64ch` + `cls_stride` `1ch`). |

> **Khả năng tương thích khi thay đổi cách huấn luyện:**
> Module `HailoDetector` trong [`edge_pi/detector_engine.py`](edge_pi/detector_engine.py) tự động đọc cấu trúc nhánh đầu ra và tham số lượng tử (`qp_scale`, `qp_zp`) trực tiếp từ metadata của file `.hef`. Bạn có thể tự do thay đổi dataset, hàm loss, augmentation, hoặc chuyển đổi giữa **3-head (`P3/P4/P5` — 8.400 anchors)** và **4-head (`P2/P3/P4/P5` — 34.000 anchors)** mà **không cần sửa bất kỳ dòng code nào trên Pi 5** (miễn giữ `NUM_CLASSES = 1` và `REG_MAX = 16`).

---

## 4. Hướng Dẫn Cài Đặt & Vận Hành

### A. Trên Raspberry Pi 5 (Edge Runtime)
```bash
# 1. Tạo môi trường ảo kế thừa driver HailoRT và Picamera2 của hệ thống
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements-pi.txt

# 2. Đặt file model best.hef vào thư mục edge_pi/ và khởi chạy ở chế độ 120 FPS
cd edge_pi
chmod +x run_drone_tracker.sh
./run_drone_tracker.sh
```
- Truy cập giao diện điều khiển & giám sát trực tiếp tại: `http://<IP_CỦA_PI>:5000`

### B. Biên Dịch ONNX Sang HEF (Trên máy Linux x86_64 có Hailo DFC)
```bash
python Archive/step2_onnx_to_hef.py \
    --onnx yolo26s_p2_epoch14_640.onnx \
    --calib_dir calib_images \
    --output_hef models/best.hef \
    --target hailo8 \
    --single_context \
    --auto_4bit_ratio 0.50
```

---

## 5. Tài Liệu Chuyên Sâu: Tối Ưu Hóa `YOLO26s-P2` Trên Pi 5 + Hailo-8 (Không Hạ Xuống Nano)

Xem chi tiết phân tích giải phẫu từng layer (`9.71M params`), bảng đối chiếu đánh đổi (Trade-off) và 3 Best Practices nâng tốc độ **`YOLO26s-P2`** từ `29.8 FPS` lên `80 – 120+ FPS` tại:
- **[`docs/PI5_HAILO8_OPTIMIZATION_GUIDE.md`](docs/PI5_HAILO8_OPTIMIZATION_GUIDE.md)**
- **Tài liệu bàn giao huấn luyện lại ở độ phân giải `640×640` (`imgsz = 640`):**
  - **[`docs/README_TRAIN_BP2_YOLO26S_P2_MAX512.md`](docs/README_TRAIN_BP2_YOLO26S_P2_MAX512.md)** *(BP2 @ 640: `YOLO26s` 4 nhánh `P2..P5`, `max_channels=512`, `34.000` anchors, `~4.80M` params)*.
  - **[`docs/README_TRAIN_BP3_YOLO26S_P2_P3_P4.md`](docs/README_TRAIN_BP3_YOLO26S_P2_P3_P4.md)** *(BP3 Cấp độ 2 @ 640: `YOLO26s` 3 nhánh `P2/P3/P4`, bỏ `P5`, `33.600` anchors, `~2.95M` params)*.
- **Tài liệu bàn giao huấn luyện lại ở độ phân giải cao `960×960` (`imgsz = 960`):**
  - **[`docs/README_TRAIN_BP2_YOLO26S_P2_MAX512_960.md`](docs/README_TRAIN_BP2_YOLO26S_P2_MAX512_960.md)** *(BP2 @ 960: `YOLO26s` 4 nhánh `P2..P5`, lưới `240×240..30×30`, `76.500` anchors, `~4.80M` params)*.
  - **[`docs/README_TRAIN_BP3_YOLO26S_P2_P3_P4_960.md`](docs/README_TRAIN_BP3_YOLO26S_P2_P3_P4_960.md)** *(BP3 Cấp độ 2 @ 960: `YOLO26s` 3 nhánh `P2/P3/P4`, lưới `240×240..60×60`, `75.600` anchors, `~2.95M` params)*.
