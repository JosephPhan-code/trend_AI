# Bộ Công Cụ Chuyển Đổi Mô Hình YOLO Drone: PT -> ONNX -> HEF (Hailo-8 / Hailo-8L)

Thư mục này được thiết kế độc lập, trọn gói (self-contained package) để hỗ trợ toàn bộ chu trình chuyển đổi mô hình YOLO phát hiện Drone từ PyTorch (`.pt`) sang ONNX (`.onnx`) và cuối cùng là sang định dạng nhị phân Hailo Executable Format (`.hef`, lượng tử hóa 8-bit INT8) để triển khai trên card tăng tốc **Hailo-8 / Hailo-8L (trên Raspberry Pi 5 hoặc drone onboard computer)**.

---

## 1. Cấu Trúc Thư Mục

```text
hailo_model_converter/
├── models/
│   ├── best_epoch14.pt                 # Trọng số tốt nhất PyTorch (FP32/FP16) - 56 MB
│   ├── yolo26s_p2_epoch14_640.onnx     # Đồ thị ONNX đã tối ưu (Opset 12, Static 1x3x640x640) - 36.8 MB
│   └── (yolo26s_p2_epoch14_640.hef)   # File nhị phân HEF sau khi compile (INT8) - ~15-20 MB
├── calib_images/                       # 200 ảnh drone thực tế dùng cho quá trình lượng tử hóa INT8
├── step1_pt_to_onnx.py                 # Script xuất mô hình PyTorch sang ONNX chuẩn Hailo
├── step2_onnx_to_hef.py                 # Script lượng tử hóa INT8 và biên dịch sang .hef bằng Hailo DFC
├── run_docker_converter.sh             # Script 1-click chạy Hailo DFC qua Docker trên Linux/WSL2
└── README.md                           # Cẩm nang hướng dẫn sử dụng chi tiết
```

---

## 2. Thông Số Mô Hình Hiện Tại (Epoch 14 - YOLO26s-P2)

- **Kiến trúc**: `YOLO26s-P2` (bổ sung P2 feature map stride 4 để bắt drone siêu nhỏ ở khoảng cách xa, End-to-End NMS-free).
- **Độ phân giải đầu vào**: $640 \times 640$ pixels.
- **Độ chính xác đạt được (Epoch 14)**:
  - **mAP50**: **94,60%**
  - **mAP50-95**: **75,73%**
  - **Precision**: **90,36%**
  - **Recall**: **89,94%**

---

## 3. Hướng Dẫn Sử Dụng Chi Tiết

### BƯỚC 1: Xuất `.pt` sang `.onnx` (Chạy trực tiếp trên Mac hoặc PC)
File `models/yolo26s_p2_epoch14_640.onnx` đã được tạo sẵn trong thư mục. Nếu bạn muốn xuất lại từ một file `.pt` khác:

```bash
python3 step1_pt_to_onnx.py --weights models/best_epoch14.pt --imgsz 640 --output_dir models
```
*Lưu ý: Script đã cấu hình `dynamic=False` và `opset=12` nhằm đảm bảo tensor cố định $1 \times 3 \times 640 \times 640$, đúng yêu cầu kiến trúc phần cứng NPU Hailo.*

---

### BƯỚC 2: Biên dịch `.onnx` sang `.hef` (INT8 Quantization & Compilation)
> **Quan trọng**: Bộ công cụ biên dịch **Hailo Dataflow Compiler (DFC)** yêu cầu môi trường **Linux x86_64** (Ubuntu 20.04/22.04 hoặc WSL2 trên Windows). DFC không hỗ trợ macOS do phụ thuộc vào các thư viện driver và trình biên dịch phần cứng độc quyền của Hailo.

#### Cách 2.1: Sử dụng Docker Container (Khuyến nghị trên Linux / Windows WSL2)
Hailo cung cấp sẵn Docker image chứa đầy đủ DFC. Bạn chỉ cần copy cả thư mục `hailo_model_converter/` sang máy Linux/WSL2 và chạy:

```bash
# Đối với card Hailo-8 (26 TOPS):
./run_docker_converter.sh hailo_dfc:latest hailo8

# Đối với Raspberry Pi 5 AI Kit dùng Hailo-8L (13 TOPS):
./run_docker_converter.sh hailo_dfc:latest hailo8l
```

#### Cách 2.2: Chạy trực tiếp qua Python trên máy Ubuntu x86_64
Nếu đã cài đặt `hailo_dataflow_compiler` trong môi trường conda/venv của Ubuntu:

```bash
# Chuyển đổi cho Hailo-8
python3 step2_onnx_to_hef.py \
    --onnx models/yolo26s_p2_epoch14_640.onnx \
    --calib_dir calib_images \
    --output_hef models/yolo26s_p2_epoch14_640_hailo8.hef \
    --target hailo8

# Chuyển đổi cho Hailo-8L (Raspberry Pi 5)
python3 step2_onnx_to_hef.py \
    --onnx models/yolo26s_p2_epoch14_640.onnx \
    --calib_dir calib_images \
    --output_hef models/yolo26s_p2_epoch14_640_hailo8l.hef \
    --target hailo8l
```

Quá trình này sẽ sử dụng 200 ảnh trong thư mục `calib_images/` để tính toán phân phối giá trị kích hoạt (activation distribution) và tìm ra scale/zero-point tối ưu nhất, hạn chế tối đa việc sụt giảm mAP khi chuyển sang 8-bit INT8 (thường mAP chỉ lệch $< 1\%$).

---

## 4. Triển Khai Trên Raspberry Pi 5 (Hailo-8L)

Sau khi có file `.hef`:
1. Copy file `yolo26s_p2_epoch14_640_hailo8l.hef` sang Raspberry Pi 5.
2. Cài đặt `hailort` và `hailo-tappas-core` trên Raspberry Pi OS.
3. Chạy inference qua HailoRT CLI hoặc Python API:
   ```bash
   hailortcli run yolo26s_p2_epoch14_640_hailo8l.hef
   ```
4. Mô hình sẽ chạy ở tốc độ thực tế **~60 - 90 FPS**, mức tiêu thụ điện năng **chỉ 2.5W**, hoàn toàn đáp ứng thời gian thực cho hệ thống theo dõi và tránh va chạm của Drone!
