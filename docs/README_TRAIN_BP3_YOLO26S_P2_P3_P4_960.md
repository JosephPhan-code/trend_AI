# HƯỚNG DẪN HUẤN LUYỆN (TRAINING SPEC @ 960×960): PHƯƠNG ÁN BEST PRACTICE 3 (CẤP ĐỘ 2) — `YOLO26s-P2-P3-P4 (No P5, imgsz = 960)`

> **Tài liệu bàn giao cho Kỹ sư AI / Người phụ trách Huấn luyện Model (Bản Độ Phân Giải Cao `960×960`)**
> - **Mục tiêu:** Huấn luyện mô hình chuyên biệt cho phát hiện Drone tầm xa **`YOLO26s-P2-P3-P4` ở độ phân giải đầu vào cao `960×960` (`imgsz = 960`)** với **3 nhánh độ phân giải cao (`P2: 240×240`, `P3: 120×120`, `P4: 60×60` — loại bỏ hoàn toàn tầng `P5: 30×30`)**.
> - **Tại sao phương án này là lựa chọn số 1 khi lên `960×960` trên Hailo-8?**
>   Khi nâng độ phân giải từ `640×640` lên `960×960`, diện tích các bản đồ đặc trưng (`240×240`, `120×120`, `60×60`) tăng gấp **2.25 lần**, chiếm dụng rất nhiều SRAM Activation trên chip Hailo-8. Việc loại bỏ tầng `P5` giúp cắt bỏ **70% tổng lượng trọng số (`9.71M` $\rightarrow$ `~2.95M` params)** và rút ngắn 50% đường nối tắt `P2`, giải phóng đúng phần SRAM cần thiết để chip Hailo-8 chạy mượt mà mô hình `960×960` ở tốc độ cao!

---

## 1. Yêu Cầu & Ràng Buộc Kỹ Thuật Bắt Buộc (Hard Constraints)

Người thực hiện huấn luyện vui lòng tuân thủ tuyệt đối 5 ràng buộc sau:

1. **TUYỆT ĐỐI KHÔNG hạ xuống bản `Nano (n)`:** Bắt buộc giữ nguyên `scale: s` (`depth_multiple = 0.50`, `width_multiple = 0.50`). Toàn bộ số kênh của các tầng `P1 (32ch)`, `P2 (128ch)`, `P3 (256ch)`, `P4 (256ch)` phải giữ nguyên 100% như `YOLO26s` gốc.
2. **Độ phân giải đầu vào cố định `imgsz = 960` (`960×960`) với 3 nhánh `[P2, P3, P4]` (Bỏ tầng `P5`):**
   - `P2` (`240×240`, stride 4 — **57.600 anchors**: bắt Drone siêu nhỏ từ `3×3 px` ở cự ly rất xa)
   - `P3` (`120×120`, stride 8 — **14.400 anchors**: bắt Drone nhỏ và vừa)
   - `P4` (`60×60`, stride 16 — **3.600 anchors**: bắt Drone tầm trung và gần lên tới `380×380 px`)
   - Tổng số anchors: **`75.600` anchors**.
3. **Đặt khối `SPPF` và Self-Attention `C2PSA` ngay tại đỉnh tầng `P4 (60×60, 256 kênh)`:** Không hạ độ phân giải xuống `30×30` (`P5`), giúp khối Self-Attention `C2PSA` hoạt động trên lưới `60×60` sắc nét hơn gấp 4 lần so với `30×30`.
4. **Thông số lớp & đầu ra:** `nc: 1` (1 class duy nhất: `drone`), `end2end: True`, `reg_max: 1`.
5. **Sản phẩm bàn giao sau khi train xong:**
   - File trọng số PyTorch tốt nhất: `best_p2_p3_p4_960.pt`
   - File đồ thị ONNX tĩnh: `best_p2_p3_p4_960.onnx` (`imgsz=960`, `dynamic=False`, `simplify=True`, `opset=13`).

---

## 2. Bảng Đối Chiếu Thông Số Kiến Trúc Ở `960×960`

| Chỉ số kỹ thuật | Bản `YOLO26s-P2` (4 nhánh cũ @ 640) | Bản `YOLO26s-P2-P3-P4` (Best Practice 3 @ `960×960`) | Lợi ích thực tế cho bài toán Drone |
| :--- | :---: | :---: | :--- |
| **Độ phân giải đầu vào** | `640×640` | **`960×960`** | Chi tiết điểm ảnh đầu vào tăng **2.25 lần** |
| **Lưới đặc trưng `P2 / P3 / P4`** | `160×160 / 80×80 / 40×40` | **`240×240 / 120×120 / 60×60`** | Tổng số điểm neo tầm xa đạt **75.600 anchors** |
| **Số kênh `P1 / P2 / P3 / P4`** | `32 / 128 / 256 / 256` | **`32 / 128 / 256 / 256` (Giữ 100%)** | Giữ nguyên 100% năng lực trích xuất của `YOLO26s` |
| **Tổng số Layer** | `30` layers (`0..29`) | **`22` layers (`0..21`)** | Rút ngắn 8 layers sâu nhất, giảm độ trễ |
| **Tổng số tham số (Parameters)** | `9.714.664` (`9.71M`) | **`~2.950.000` (`~2.95M`)** | **Giảm 70% trọng số để bù SRAM cho ảnh `960×960`** |

---

## 3. File Cấu Hình Kiến Trúc Chuẩn (`yolo26s_p2_p3_p4.yaml`)

Lưu nội dung dưới đây vào file **`yolo26s_p2_p3_p4.yaml`** (hoặc dùng trực tiếp file [`docs/yolo26s_p2_p3_p4.yaml`](yolo26s_p2_p3_p4.yaml) trong repo):

```yaml
# ==============================================================================
# File: yolo26s_p2_p3_p4.yaml
# Kiến trúc: YOLO26s-P2-P3-P4 (3 nhánh 240x240, 120x120, 60x60 tại imgsz=960)
# ==============================================================================
nc: 1
end2end: True
reg_max: 1
scale: s

scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
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
  - [-1, 1, SPPF, [512, 5, 3, True]]     # 7-P4/16 (60x60,   256ch)
  - [-1, 2, C2PSA, [512]]                # 8-P4/16 (60x60,   256ch)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]            # 10: cat backbone P3
  - [-1, 2, C3k2, [256, True]]           # 11: (120x120, 128ch)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]            # 13: cat backbone P2
  - [-1, 2, C3k2, [128, True]]           # 14: P2/4-tiny (240x240, 64ch)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]           # 16: cat head P3
  - [-1, 2, C3k2, [256, True]]           # 17: P3/8-small (120x120, 128ch)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]            # 19: cat backbone P4 (C2PSA)
  - [-1, 2, C3k2, [512, True]]           # 20: P4/16-medium (60x60, 256ch)

  - [[14, 17, 20], 1, Detect, [nc]]      # 21: Detect(P2, P3, P4) - 75,600 anchors @ 960x960
```

---

## 4. Script Huấn Luyện Chuẩn Ở `imgsz = 960` (Kế Thừa Backbone `Layer 0..6` Từ `best.pt`)

Toàn bộ `Layer 0..6` (Backbone `P1, P2, P3, P4`) có kích thước bộ lọc tích chập khớp 100% với `Layer 0..6` của file `best.pt` (bản 640 cũ). Script dưới đây tự động nạp trọng số Backbone đã học để đẩy nhanh tốc độ hội tụ ở độ phân giải `960×960`:

```python
#!/usr/bin/env python3
"""
Script huấn luyện YOLO26s-P2-P3-P4 (No P5) ở độ phân giải cao 960x960 cho Hailo-8.
"""
from ultralytics import YOLO

def main():
    # 1. Khởi tạo kiến trúc 3-head P2/P3/P4 (~2.95M params)
    model = YOLO("docs/yolo26s_p2_p3_p4.yaml")

    # 2. Kế thừa trọng số Backbone P1..P4 (Layer 0..6) từ best.pt hiện tại
    try:
        model.load("yolo26s_p2_img640/yolo26s_p2_img640/weights/best.pt")
        print("[+] Đã kế thừa trọng số Backbone P1..P4 từ best.pt!")
    except Exception as e:
        print(f"[*] Không tìm thấy best.pt cũ ({e}), huấn luyện từ đầu.")

    # 3. Huấn luyện ở độ phân giải 960x960
    # Vì mô hình này chỉ nặng ~2.95M params (nhẹ hơn 70%), tốc độ train ở 960x960 rất nhanh
    model.train(
        data="clean_dataset/data.yaml",      # Đường dẫn tới file data.yaml của bộ dataset
        epochs=80,                           # Khuyến nghị 80-100 epochs để Neck P2/P3/P4 hội tụ tối đa ở 960x960
        imgsz=960,                           # BẮT BUỘC cố định 960x960
        batch=16,                            # Điều chỉnh theo VRAM GPU (16 cho 16-24GB VRAM, 32 cho A100/H100)
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.0008,                          # Learning rate tối ưu cho độ phân giải 960x960
        lrf=0.01,
        weight_decay=0.0005,
        box=7.5,                             # Tăng trọng số phạt sai số bbox cho vật thể siêu nhỏ trên lưới 240x240
        cls=0.5,
        mosaic=1.0,
        close_mosaic=15,                     # Tắt mosaic ở 15 epochs cuối để tinh chỉnh độ chính xác bbox
        scale=0.5,
        mixup=0.0,
        project="runs/train_bp3_960",
        name="yolo26s_p2_p3_p4_img960",
        exist_ok=True,
    )

    # 4. Xuất thẳng sang ONNX tĩnh 960x960 chuẩn cho Hailo Dataflow Compiler
    best_model = YOLO("runs/train_bp3_960/yolo26s_p2_p3_p4_img960/weights/best.pt")
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

## 5. Kiểm Tra Nghiệm Thu & Biên Dịch Sang `.hef` (`960×960`)

### A. Kiểm tra nghiệm thu sau khi train (Acceptance Criteria @ `960×960`)
Chạy lệnh sau để xác nhận file `best.pt` và `best.onnx` đạt chuẩn trước khi bàn giao:

```bash
python -c "
import torch, onnx
ckpt = torch.load('runs/train_bp3_960/yolo26s_p2_p3_p4_img960/weights/best.pt', map_location='cpu', weights_only=False)
m = ckpt.get('ema') or ckpt.get('model')
params = sum(p.numel() for p in m.parameters())
print('1. Tổng số tham số:', f'{params:,}', '(Kỳ vọng: ~2,800,000 - 3,150,000 params)')
assert 2_500_000 <= params <= 3_500_000, 'Sai số lượng tham số! Kiểm tra lại file yolo26s_p2_p3_p4.yaml.'

ox = onnx.load('runs/train_bp3_960/yolo26s_p2_p3_p4_img960/weights/best.onnx')
in_shape = [d.dim_value for d in ox.graph.input[0].type.tensor_type.shape.dim]
out_shape = [d.dim_value for d in ox.graph.output[0].type.tensor_type.shape.dim]
print('2. ONNX Input Shape :', in_shape, '(Kỳ vọng: [1, 3, 960, 960])')
print('3. ONNX Output Shape:', out_shape, '(Kỳ vọng: [1, 5, 75600] cho 3 nhánh 240x240 + 120x120 + 60x60)')
assert in_shape == [1, 3, 960, 960], 'Input shape phải cố định là [1, 3, 960, 960]!'
assert out_shape[-1] == 75600, 'Tổng số anchors ở 960x960 phải là 75,600 (57600 + 14400 + 3600)!'
print('=> NGHIỆM THU BẢN 960x960 (BP3) ĐẠT CHUẨN 100%!')
"
```

### B. Biên dịch sang `.hef` (`960×960`) và chạy trên Raspberry Pi 5
1. **Biên dịch sang `.hef` trên máy Hailo DFC:**
   ```bash
   python Archive/step2_onnx_to_hef.py \
       --onnx best_p2_p3_p4_960.onnx \
       --calib_dir calib_images \
       --output_hef models/best_p2_p3_p4_960.hef \
       --target hailo8 \
       --imgsz 960 \
       --single_context \
       --auto_4bit_ratio 0.25 \
       --end_nodes \
           /model.21/cv2.0/cv2.0.2/Conv /model.21/cv3.0/cv3.0.2/Conv \
           /model.21/cv2.1/cv2.1.2/Conv /model.21/cv3.1/cv3.1.2/Conv \
           /model.21/cv2.2/cv2.2.2/Conv /model.21/cv3.2/cv3.2.2/Conv
   ```
2. **Chạy trên Raspberry Pi 5 (`edge_pi/`):**
   ```bash
   DRONE_TRACKER_INFER_SIZE=960 ./run_drone_tracker.sh
   ```
   *(Khi đặt `DRONE_TRACKER_INFER_SIZE=960`, `CameraEngine` sẽ tự động cấu hình luồng PiSP `lores` ở `960×540` và `HailoDetector` tự động sinh bảng anchor `240×240`, `120×120`, `60×60` mà không cần sửa code).*
