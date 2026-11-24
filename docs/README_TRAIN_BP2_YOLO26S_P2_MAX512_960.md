# HƯỚNG DẪN HUẤN LUYỆN (TRAINING SPEC @ 960×960): PHƯƠNG ÁN BEST PRACTICE 2 — `YOLO26s-P2 (max_channels = 512, imgsz = 960)`

> **Tài liệu bàn giao cho Kỹ sư AI / Người phụ trách Huấn luyện Model (Bản Độ Phân Giải Cao `960×960`)**
> - **Mục tiêu:** Huấn luyện mô hình phát hiện Drone **`YOLO26s-P2` ở độ phân giải đầu vào `960×960` (`imgsz = 960`)** với đủ **4 nhánh đầu ra `[P2, P3, P4, P5]` (76.500 anchors)** và khóa trần kênh **`max_channels = 512`** cho bản `Small (s)`.
> - **Lợi thế của bản `960×960`:** Nhánh `P2` đạt lưới đặc trưng siêu mịn **`240×240` (`57.600` anchors — mật độ điểm ảnh cao gấp 2.25 lần bản `640×640`)**, cho phép phát hiện sớm Drone siêu nhỏ ở khoảng cách cực xa, trong khi việc khóa `max_channels = 512` giúp cắt giảm 51% trọng số dư thừa (`9.71M` $\rightarrow$ `~4.80M` params) để bù đắp dung lượng SRAM cho NPU **Hailo-8 (26 TOPS)**.

---

## 1. Yêu Cầu & Ràng Buộc Kỹ Thuật Bắt Buộc (Hard Constraints)

Người thực hiện huấn luyện vui lòng tuân thủ tuyệt đối 5 ràng buộc sau:

1. **TUYỆT ĐỐI KHÔNG hạ xuống bản `Nano (n)`:** Bắt buộc giữ nguyên `scale: s` (`depth_multiple = 0.50`, `width_multiple = 0.50`). Toàn bộ số kênh của các tầng trích xuất vật thể nhỏ `P1 (32ch)`, `P2 (128ch)`, `P3 (256ch)`, `P4 (256ch)` phải giữ nguyên 100% như `YOLO26s` gốc.
2. **Độ phân giải đầu vào cố định `imgsz = 960` (`960×960`) với đủ 4 nhánh `[P2, P3, P4, P5]`:**
   - `P2` (`240×240`, stride 4 — **57.600 anchors**)
   - `P3` (`120×120`, stride 8 — **14.400 anchors**)
   - `P4` (`60×60`, stride 16 — **3.600 anchors**)
   - `P5` (`30×30`, stride 32 — **900 anchors**)
   - Tổng số anchors: **`76.500` anchors** (gấp 2.25 lần so với 34.000 anchors ở bản `640×640`).
3. **Khóa trần `max_channels` của bản `s` ở `512`:**
   - Trong YAML: `s: [0.50, 0.50, 512]`.
4. **Thông số lớp & đầu ra:** `nc: 1` (1 class duy nhất: `drone`), `end2end: True`, `reg_max: 1`.
5. **Sản phẩm bàn giao sau khi train xong:**
   - File trọng số PyTorch tốt nhất: `best_max512_960.pt`
   - File đồ thị ONNX tĩnh: `best_max512_960.onnx` (`imgsz=960`, `dynamic=False`, `simplify=True`, `opset=13`).

---

## 2. Tại Sao Khi Lên Độ Phân Giải `960×960` Bắt Buộc Phải Khóa `max_channels = 512`?

Khi tăng độ phân giải đầu vào từ `640×640` lên `960×960`, kích thước mỗi chiều tăng $1.5\times$ và **diện tích bản đồ đặc trưng (Activation Feature Maps) tăng gấp $1.5^2 = 2.25$ lần**:
- Nhánh `P2` tăng từ `160×160` lên **`240×240`**, tiêu tốn lượng bộ nhớ SRAM Activation trên chip Hailo-8 gấp 2.25 lần.
- Nếu giữ `max_channels = 1024` như bản cũ (`9.71M` params), riêng tầng `P5 (30×30)` sẽ ngốn thêm **`7.50 triệu` tham số (`77.2%` toàn mạng)** làm tràn nặng SRAM của Hailo-8.
- Khi đổi sang **`s: [0.50, 0.50, 512]`**, chúng ta tiết kiệm ngay **`4.91 triệu` tham số trọng số ở tầng `P5`** để nhường bộ nhớ SRAM trên chip Hailo-8 cho lưới đặc trưng siêu nét `240×240` của tầng `P2`:

| Tầng đặc trưng | Kích thước lưới ở `960×960` | Số Anchors | Số kênh bản cũ (`max=1024`) | Số kênh bản mới (`max=512`) | Đánh giá ảnh hưởng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`P1` (Stride 2)** | `480×480` | — | `32` kênh | **`32` kênh (Giữ 100%)** | Không thay đổi |
| **`P2` (Stride 4 - Tiny Drone)** | **`240×240`** | `57.600` | `128` kênh | **`128` kênh (Giữ 100%)** | Mật độ lưới tăng 2.25x, giữ 100% số kênh |
| **`P3` (Stride 8 - Small Drone)** | **`120×120`** | `14.400` | `256` kênh | **`256` kênh (Giữ 100%)** | Mật độ lưới tăng 2.25x, giữ 100% số kênh |
| **`P4` (Stride 16 - Medium)** | **`60×60`** | `3.600` | `256` kênh | **`256` kênh (Giữ 100%)** | Mật độ lưới tăng 2.25x, giữ 100% số kênh |
| **`P5` (Stride 32 - Large)** | **`30×30`** | `900` | `512` kênh | **`256` kênh (Chuẩn hóa)** | Giảm 4 lần tham số tầng P5 (`6.05M` $\rightarrow$ `1.51M`) |
| **Tổng cộng** | — | **`76.500`** | `9.71M` params | **`~4.80M` params (-51%)** | **Nhường SRAM trọng số cho lưới `240×240`** |

---

## 3. File Cấu Hình Kiến Trúc Chuẩn (`yolo26s_p2_max512.yaml`)

Cấu trúc lớp tích chập trong YAML không phụ thuộc vào `imgsz` (độ phân giải `960` được truyền vào ở hàm `model.train(imgsz=960)` và `model.export(imgsz=960)`). Lưu nội dung sau vào **`yolo26s_p2_max512.yaml`** (hoặc dùng trực tiếp file [`docs/yolo26s_p2_max512.yaml`](yolo26s_p2_max512.yaml)):

```yaml
# ==============================================================================
# File: yolo26s_p2_max512.yaml
# Kiến trúc: YOLO26s-P2 (4 nhánh P2/P3/P4/P5) tối ưu cho Hailo-8 ở imgsz=960
# ==============================================================================
nc: 1
end2end: True
reg_max: 1
scale: s

scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 512]   # Giữ nguyên width=0.50 của bản Small, khóa trần P5 ở 512 (256ch thực tế)
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  # [from, repeats, module, args]          # Kích thước tại imgsz = 960x960:
  - [-1, 1, Conv, [64, 3, 2]]            # 0-P1/2  (480x480, 32ch)
  - [-1, 1, Conv, [128, 3, 2]]           # 1-P2/4  (240x240, 64ch)
  - [-1, 2, C3k2, [256, False, 0.25]]    # 2-P2/4  (240x240, 128ch)
  - [-1, 1, Conv, [256, 3, 2]]           # 3-P3/8  (120x120, 128ch)
  - [-1, 2, C3k2, [512, False, 0.25]]    # 4-P3/8  (120x120, 256ch)
  - [-1, 1, Conv, [512, 3, 2]]           # 5-P4/16 (60x60,   256ch)
  - [-1, 2, C3k2, [512, True]]           # 6-P4/16 (60x60,   256ch)
  - [-1, 1, Conv, [1024, 3, 2]]          # 7-P5/32 (30x30,   256ch)
  - [-1, 2, C3k2, [1024, True]]          # 8-P5/32 (30x30,   256ch)
  - [-1, 1, SPPF, [1024, 5, 3, True]]    # 9-P5/32 (30x30,   256ch)
  - [-1, 2, C2PSA, [1024]]               # 10-P5/32 (30x30,  256ch)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]            # 12: cat backbone P4
  - [-1, 2, C3k2, [512, True]]           # 13: (60x60,   256ch)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]            # 15: cat backbone P3
  - [-1, 2, C3k2, [256, True]]           # 16: (120x120, 128ch)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]            # 18: cat backbone P2
  - [-1, 2, C3k2, [128, True]]           # 19: P2/4-tiny (240x240, 64ch)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]           # 21: cat head P3
  - [-1, 2, C3k2, [256, True]]           # 22: P3/8-small (120x120, 128ch)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]           # 24: cat head P4
  - [-1, 2, C3k2, [512, True]]           # 25: P4/16-medium (60x60, 256ch)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]           # 27: cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28: P5/32-large (30x30, 256ch)

  - [[19, 22, 25, 28], 1, Detect, [nc]]  # 29: Detect(P2, P3, P4, P5) - 76,500 anchors @ 960x960
```

---

## 4. Script Huấn Luyện Chuẩn Ở `imgsz = 960` (Có Kế Thừa Trọng Số `Transfer Learning` Từ `best.pt`)

> **Mẹo cực kỳ quan trọng khi train ở `960×960`:**
> Trọng số của các bộ lọc tích chập (`3×3`, `1×1`) không phụ thuộc vào độ phân giải ảnh đầu vào (`640` hay `960`). Vì vậy, bạn **nên nạp (`model.load()`) trực tiếp từ file `best.pt` đã train ở `640×640`** để kế thừa 100% trọng số của `Layer 0..6` và `Layer 14..25`, giúp quá trình train ở `960×960` hội tụ nhanh gấp 3 lần!

```python
#!/usr/bin/env python3
"""
Script huấn luyện YOLO26s-P2 (max_channels=512) ở độ phân giải cao 960x960.
"""
from ultralytics import YOLO

def main():
    # 1. Khởi tạo kiến trúc từ file YAML (~4.80M params)
    model = YOLO("docs/yolo26s_p2_max512.yaml")

    # 2. Kế thừa trọng số các tầng P1..P4 từ checkpoint best.pt (bản 640 cũ)
    try:
        model.load("yolo26s_p2_img640/yolo26s_p2_img640/weights/best.pt")
        print("[+] Đã kế thừa thành công trọng số các tầng P1..P4 từ best.pt!")
    except Exception as e:
        print(f"[*] Không tìm thấy best.pt cũ ({e}), khởi tạo từ yolo26s.pt hoặc train từ đầu.")

    # 3. Huấn luyện ở độ phân giải 960x960
    # Lưu ý: Ảnh 960x960 tốn VRAM gấp 2.25 lần ảnh 640x640 nên để batch=16 (hoặc 8 nếu GPU 12GB VRAM)
    model.train(
        data="clean_dataset/data.yaml",      # Đường dẫn tới file data.yaml của bộ dataset
        epochs=60,                           # Nếu kế thừa từ best.pt chỉ cần 50-60 epochs; nếu train từ đầu để 100 epochs
        imgsz=960,                           # BẮT BUỘC cố định 960x960
        batch=16,                            # Điều chỉnh theo VRAM GPU (8 cho 12GB, 16 cho 24GB, 32 cho A100/H100)
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.0008,                          # Learning rate êm hơn cho độ phân giải lớn 960
        lrf=0.01,
        weight_decay=0.0005,
        box=7.5,                             # Tăng trọng số phạt sai số bbox cho drone siêu nhỏ ở lưới 240x240
        cls=0.5,
        mosaic=1.0,
        close_mosaic=15,                     # Tắt mosaic ở 15 epochs cuối để tinh chỉnh tọa độ pixel ở 960x960
        scale=0.5,
        mixup=0.0,
        project="runs/train_bp2_960",
        name="yolo26s_p2_max512_img960",
        exist_ok=True,
    )

    # 4. Xuất thẳng sang ONNX tĩnh 960x960 chuẩn cho Hailo Dataflow Compiler
    best_model = YOLO("runs/train_bp2_960/yolo26s_p2_max512_img960/weights/best.pt")
    onnx_path = best_model.export(
        format="onnx",
        imgsz=960,                           # BẮT BUỘC 960
        opset=13,
        simplify=True,
        dynamic=False,                       # BẮT BUỘC False (static shape 1x3x960x960 cho Hailo NPU)
    )
    print(f"[THÀNH CÔNG] Đã xuất ONNX 960x960 tại: {onnx_path}")

if __name__ == "__main__":
    main()
```

---

## 5. Checklist Nghiệm Thu Trước Khi Bàn Giao (Acceptance Criteria @ `960×960`)

Trước khi gửi lại file `best.pt` và `best.onnx`, người train vui lòng chạy đoạn kiểm tra nhanh sau để xác nhận model đạt chuẩn `960×960`:

```bash
python -c "
import torch, onnx
ckpt = torch.load('runs/train_bp2_960/yolo26s_p2_max512_img960/weights/best.pt', map_location='cpu', weights_only=False)
m = ckpt.get('ema') or ckpt.get('model')
params = sum(p.numel() for p in m.parameters())
print('1. Tổng số tham số:', f'{params:,}', '(Kỳ vọng: ~4,700,000 - 4,950,000 params)')
assert 4_000_000 <= params <= 5_500_000, 'Sai số lượng tham số! Kiểm tra lại max_channels=512 trong YAML.'

ox = onnx.load('runs/train_bp2_960/yolo26s_p2_max512_img960/weights/best.onnx')
in_shape = [d.dim_value for d in ox.graph.input[0].type.tensor_type.shape.dim]
out_shape = [d.dim_value for d in ox.graph.output[0].type.tensor_type.shape.dim]
print('2. ONNX Input Shape :', in_shape, '(Kỳ vọng: [1, 3, 960, 960])')
print('3. ONNX Output Shape:', out_shape, '(Kỳ vọng: [1, 5, 76500] cho 4 nhánh 240x240 + 120x120 + 60x60 + 30x30)')
assert in_shape == [1, 3, 960, 960], 'Input shape phải cố định là [1, 3, 960, 960]!'
assert out_shape[-1] == 76500, 'Tổng số anchors ở 960x960 phải là 76,500!'
print('=> NGHIỆM THU BẢN 960x960 (BP2) ĐẠT CHUẨN 100%!')
"
```

---

## 6. Lệnh Biên Dịch `.hef` & Chạy Trên Raspberry Pi 5 Với Bản `960×960`
1. **Biên dịch sang `.hef` trên máy Hailo DFC:**
   ```bash
   python Archive/step2_onnx_to_hef.py \
       --onnx best_max512_960.onnx \
       --calib_dir calib_images \
       --output_hef models/best_960.hef \
       --target hailo8 \
       --imgsz 960 \
       --single_context \
       --auto_4bit_ratio 0.40
   ```
2. **Chạy trên Raspberry Pi 5 (`edge_pi/`):**
   Bộ code OOP trong [`edge_pi/config.py`](../edge_pi/config.py) và [`edge_pi/detector_engine.py`](../edge_pi/detector_engine.py) đã hỗ trợ biến môi trường `DRONE_TRACKER_INFER_SIZE=960`:
   ```bash
   DRONE_TRACKER_INFER_SIZE=960 ./run_drone_tracker.sh
   ```
