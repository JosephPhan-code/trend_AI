# HƯỚNG DẪN HUẤN LUYỆN (TRAINING SPEC): PHƯƠNG ÁN BEST PRACTICE 2 — `YOLO26s-P2 (max_channels = 512)`

> **Tài liệu bàn giao cho Kỹ sư AI / Người phụ trách Huấn luyện Model**
> - **Mục tiêu:** Huấn luyện lại mô hình phát hiện Drone **`YOLO26s-P2` (4 nhánh đầu ra `P2, P3, P4, P5`)** với cấu hình khóa trần kênh **`max_channels = 512`** cho bản `Small (s)`, giúp giảm hơn 50% tham số dư thừa ở tầng `P5` (`9.71M` $\rightarrow$ `~4.80M` params) để biên dịch vừa khít vào **1 Context (Single-Context, `80–105 FPS`)** trên chip NPU **Hailo-8 (26 TOPS)** của Raspberry Pi 5.

---

## 1. Yêu Cầu & Ràng Buộc Kỹ Thuật Bắt Buộc (Hard Constraints)

Người thực hiện huấn luyện vui lòng tuân thủ tuyệt đối 5 ràng buộc sau để đảm bảo file model đầu ra tương thích 100% với hệ thống nhúng trên Raspberry Pi 5 + Hailo-8:

1. **TUYỆT ĐỐI KHÔNG hạ xuống bản `Nano (n)`:** Bắt buộc giữ nguyên `scale: s` (`depth_multiple = 0.50`, `width_multiple = 0.50`). Toàn bộ số kênh của các tầng trích xuất vật thể nhỏ `P1 (32ch)`, `P2 (128ch)`, `P3 (256ch)`, `P4 (256ch)` phải giữ nguyên 100% như `YOLO26s` gốc.
2. **Giữ đủ 4 nhánh đầu ra `[P2, P3, P4, P5]`:**
   - `P2` (`160×160`, stride 4 — 25.600 anchors)
   - `P3` (`80×80`, stride 8 — 6.400 anchors)
   - `P4` (`40×40`, stride 16 — 1.600 anchors)
   - `P5` (`20×20`, stride 32 — 400 anchors)
   - Tổng số anchors: **34.000 anchors** ở độ phân giải đầu vào cố định **`imgsz = 640`**.
3. **Chỉ thay đổi duy nhất `max_channels` của bản `s` từ `1024` xuống `512`:**
   - Trong YAML: `s: [0.50, 0.50, 512]` (tương tự chuẩn `max_channels = 512` của các bản `m`, `l`, `x`).
4. **Thông số lớp & đầu ra:** `nc: 1` (1 class duy nhất: `drone`), `end2end: True`, `reg_max: 1`.
5. **Sản phẩm bàn giao sau khi train xong:**
   - File trọng số PyTorch tốt nhất: `best.pt`
   - File đồ thị ONNX tĩnh: `best_max512_640.onnx` (`imgsz=640x640`, `dynamic=False`, `simplify=True`, `opset=13`).

---

## 2. Tại Sao Phải Đổi `s: [0.50, 0.50, 1024]` $\rightarrow$ `s: [0.50, 0.50, 512]`?

Trong bản `YOLO26s-P2` cũ (`9.714.664` tham số), do `max_channels = 1024`, tại tầng sâu nhất `P5 (20×20, stride 32)`, số kênh sau khi nhân `width = 0.50` nhảy vọt lên `512` kênh. Điều này làm riêng tầng `P5` (`Layer 7, 8, 9, 10, 26, 28`) ngốn tới **`7.50 triệu` tham số (`77.2%` toàn mạng)**, khiến bộ nhớ SRAM trên chip Hailo-8 bị tràn và phải cắt mạng thành **3 Contexts** (làm tốc độ bị khóa ở `29.8 FPS`).

Khi đổi sang **`s: [0.50, 0.50, 512]`**:

| Tầng đặc trưng | Độ phân giải (`imgsz=640`) | Số kênh bản cũ (`max=1024`) | Số kênh bản mới (`max=512`) | Đánh giá ảnh hưởng |
| :--- | :---: | :---: | :---: | :--- |
| **`P1` (Stride 2)** | `320×320` | `32` kênh | **`32` kênh (Giữ 100%)** | Không thay đổi |
| **`P2` (Stride 4 - Tiny Drone)** | `160×160` | `128` kênh | **`128` kênh (Giữ 100%)** | Không thay đổi — giữ nguyên độ nhạy bắt drone ở xa |
| **`P3` (Stride 8 - Small Drone)** | `80×80` | `256` kênh | **`256` kênh (Giữ 100%)** | Không thay đổi |
| **`P4` (Stride 16 - Medium)** | `40×40` | `256` kênh | **`256` kênh (Giữ 100%)** | Không thay đổi |
| **`P5` (Stride 32 - Large)** | `20×20` | `512` kênh | **`256` kênh (Chuẩn hóa)** | Giảm 4 lần tham số tầng P5 (`6.05M` $\rightarrow$ `1.51M` params) |
| **Tổng số tham số toàn mạng** | — | `9.714.664` (`9.71M`) | **`~4.800.000` (`~4.80M`)** | **Giảm 51% dung lượng trọng số, lọt vào 1 Context Hailo-8** |

---

## 3. File Cấu Hình Kiến Trúc Chuẩn (`yolo26s_p2_max512.yaml`)

Lưu nội dung dưới đây vào file **`yolo26s_p2_max512.yaml`** (hoặc lấy trực tiếp từ [`docs/yolo26s_p2_max512.yaml`](yolo26s_p2_max512.yaml) trong repo):

```yaml
# ==============================================================================
# File: yolo26s_p2_max512.yaml
# Kiến trúc: YOLO26s-P2 (4 nhánh P2/P3/P4/P5) tối ưu Single-Context cho Hailo-8
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
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]            # 0-P1/2  (320x320, 32ch)
  - [-1, 1, Conv, [128, 3, 2]]           # 1-P2/4  (160x160, 64ch)
  - [-1, 2, C3k2, [256, False, 0.25]]    # 2-P2/4  (160x160, 128ch)
  - [-1, 1, Conv, [256, 3, 2]]           # 3-P3/8  (80x80,   128ch)
  - [-1, 2, C3k2, [512, False, 0.25]]    # 4-P3/8  (80x80,   256ch)
  - [-1, 1, Conv, [512, 3, 2]]           # 5-P4/16 (40x40,   256ch)
  - [-1, 2, C3k2, [512, True]]           # 6-P4/16 (40x40,   256ch)
  - [-1, 1, Conv, [1024, 3, 2]]          # 7-P5/32 (20x20,   256ch)
  - [-1, 2, C3k2, [1024, True]]          # 8-P5/32 (20x20,   256ch)
  - [-1, 1, SPPF, [1024, 5, 3, True]]    # 9-P5/32 (20x20,   256ch)
  - [-1, 2, C2PSA, [1024]]               # 10-P5/32 (20x20,  256ch)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]            # 12: cat backbone P4
  - [-1, 2, C3k2, [512, True]]           # 13: (40x40, 256ch)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]            # 15: cat backbone P3
  - [-1, 2, C3k2, [256, True]]           # 16: (80x80, 128ch)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]            # 18: cat backbone P2
  - [-1, 2, C3k2, [128, True]]           # 19: P2/4-tiny (160x160, 64ch)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]           # 21: cat head P3
  - [-1, 2, C3k2, [256, True]]           # 22: P3/8-small (80x80, 128ch)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]           # 24: cat head P4
  - [-1, 2, C3k2, [512, True]]           # 25: P4/16-medium (40x40, 256ch)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]           # 27: cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 28: P5/32-large (20x20, 256ch)

  - [[19, 22, 25, 28], 1, Detect, [nc]]  # 29: Detect(P2, P3, P4, P5) - 34,000 anchors
```

---

## 4. Script Huấn Luyện Chuẩn (Có Kế Thừa Trọng Số `Transfer Learning` Từ `best.pt` Cũ)

Vì toàn bộ `Layer 0..6` (Backbone `P1..P4`) và `Layer 14..25` (Neck `P2..P4`) của kiến trúc mới có kích thước tensor **khớp 100%** với file `best.pt` (`YOLO26s-P2` cũ), việc nạp (`model.load()`) từ `best.pt` cũ sẽ giúp model kế thừa toàn bộ bộ lọc đặc trưng vật thể nhỏ đã học và hội tụ cực nhanh:

```python
#!/usr/bin/env python3
"""
Script huấn luyện YOLO26s-P2 (max_channels=512) cho Hailo-8 Single-Context.
"""
from ultralytics import YOLO

def main():
    # 1. Khởi tạo kiến trúc từ file YAML mới (~4.80M params)
    model = YOLO("docs/yolo26s_p2_max512.yaml")

    # 2. Kế thừa trọng số các tầng P1..P4 từ checkpoint best.pt hiện tại (hoặc yolo26s.pt)
    # Ultralytics sẽ tự động nạp tất cả các layer khớp shape (Layer 0..6 và Layer 14..25)
    try:
        model.load("yolo26s_p2_img640/yolo26s_p2_img640/weights/best.pt")
        print("[+] Đã kế thừa thành công trọng số các tầng P1..P4 từ best.pt!")
    except Exception as e:
        print(f"[*] Không tìm thấy best.pt cũ ({e}), khởi tạo từ yolo26s.pt hoặc train từ đầu.")

    # 3. Huấn luyện với bộ tham số tối ưu cho vật thể bay nhỏ (Drone)
    model.train(
        data="clean_dataset/data.yaml",      # Đường dẫn tới file data.yaml của bộ dataset
        epochs=60,                           # Nếu nạp từ best.pt cũ chỉ cần 40-60 epochs; nếu train mới hoàn toàn để 100-120 epochs
        imgsz=640,                           # BẮT BUỘC cố định 640x640
        batch=32,                            # Điều chỉnh theo VRAM của GPU (16, 32, hoặc 64)
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        box=7.5,                             # Tăng trọng số phạt sai số hộp giới hạn cho vật thể nhỏ
        cls=0.5,
        mosaic=1.0,
        close_mosaic=15,                     # Tắt mosaic ở 15 epochs cuối để giữ biên dạng thật của drone nhỏ
        scale=0.5,
        mixup=0.0,                           # Không dùng mixup để tránh làm mờ drone kích thước < 10x10 px
        project="runs/train_bp2",
        name="yolo26s_p2_max512_img640",
        exist_ok=True,
    )

    # 4. Xuất thẳng sang ONNX chuẩn cho Hailo Dataflow Compiler
    best_model = YOLO("runs/train_bp2/yolo26s_p2_max512_img640/weights/best.pt")
    onnx_path = best_model.export(
        format="onnx",
        imgsz=640,
        opset=13,
        simplify=True,
        dynamic=False,                       # BẮT BUỘC False (static shape 1x3x640x640 cho Hailo NPU)
    )
    print(f"[THÀNH CÔNG] Đã xuất ONNX tại: {onnx_path}")

if __name__ == "__main__":
    main()
```

---

## 5. Checklist Nghiệm Thu Trước Khi Bàn Giao (Acceptance Criteria)

Trước khi gửi lại file `best.pt` và `best.onnx`, người train vui lòng chạy đoạn kiểm tra nhanh sau để xác nhận model đạt chuẩn:

```bash
python -c "
import torch, onnx
ckpt = torch.load('runs/train_bp2/yolo26s_p2_max512_img640/weights/best.pt', map_location='cpu', weights_only=False)
m = ckpt.get('ema') or ckpt.get('model')
params = sum(p.numel() for p in m.parameters())
print('1. Tổng số tham số:', f'{params:,}', '(Kỳ vọng: ~4,700,000 - 4,950,000 params)')
assert 4_000_000 <= params <= 5_500_000, 'Sai số lượng tham số! Kiểm tra lại max_channels=512 trong YAML.'

ox = onnx.load('runs/train_bp2/yolo26s_p2_max512_img640/weights/best.onnx')
in_shape = [d.dim_value for d in ox.graph.input[0].type.tensor_type.shape.dim]
print('2. ONNX Input Shape:', in_shape, '(Kỳ vọng: [1, 3, 640, 640])')
assert in_shape == [1, 3, 640, 640], 'Input shape phải cố định là [1, 3, 640, 640]!'
print('=> NGHIỆM THU ĐẠT CHUẨN 100%!')
"
```
