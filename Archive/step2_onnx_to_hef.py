#!/usr/bin/env python3
"""
Hailo ONNX to HEF Converter (INT8 Quantization & Compilation)
------------------------------------------------------------
Script này sử dụng Hailo Dataflow Compiler (DFC) để:
1. Parse đồ thị ONNX vào Hailo Internal Representation (HAR).
2. Tối ưu hóa đồ thị (Model Optimization).
3. Lượng tử hóa 8-bit INT8 (Full Integer Post-Training Quantization) sử dụng tập ảnh calib.
4. Biên dịch (Compile) đồ thị ra file Hailo Executable Format (.hef) cho chip Hailo-8 hoặc Hailo-8L.

Lưu ý: Hailo Dataflow Compiler chỉ chạy trên Linux x86_64 (Ubuntu 20.04/22.04 hoặc Docker).
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

try:
    from hailo_sdk_client import ClientRunner, InferenceContext
except ImportError:
    print("[!] CẢNH BÁO: Không tìm thấy thư viện 'hailo_sdk_client'.")
    print("    Script này bắt buộc phải chạy trong môi trường cài đặt Hailo Dataflow Compiler (Linux x86_64 hoặc Hailo Docker container).")


def load_calibration_dataset(calib_dir, imgsz=640, max_images=200):
    """
    Đọc và tiền xử lý tập ảnh calibration:
    - Resize về (imgsz, imgsz)
    - Chuẩn hóa màu RGB
    - Giữ giá trị uint8 [0, 255] hoặc float32 tùy theo cấu hình DFC normalization.
    """
    image_paths = glob.glob(os.path.join(calib_dir, "*.jpg")) + glob.glob(os.path.join(calib_dir, "*.png"))
    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh calibration nào trong: {calib_dir}")
    
    image_paths = sorted(image_paths)[:max_images]
    print(f"[*] Đang nạp {len(image_paths)} ảnh calibration từ {calib_dir}...")
    
    calib_data = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        img = img.resize((imgsz, imgsz), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) # Shape: (640, 640, 3)
        calib_data.append(arr)
        
    calib_data = np.stack(calib_data, axis=0)
    print(f"[+] Nạp thành công calib data: Shape {calib_data.shape}, Type {calib_data.dtype}")
    return calib_data


def convert_onnx_to_hef(
    onnx_path, 
    calib_dir, 
    output_hef_path, 
    target_device="hailo8", # Hoặc "hailo8l"
    imgsz=640,
    network_name="yolo26s_drone",
    single_context=True,
    auto_4bit_ratio=0.25,
    end_node_names=None,
):
    print(f"\n========================================================")
    print(f" BẮT ĐẦU QUY TRÌNH CHUYỂN ĐỔI ONNX -> HEF (INT8) CHO {target_device.upper()}")
    print(f"========================================================")
    
    har_path = output_hef_path.replace(".hef", ".har")
    
    # 1. Khởi tạo ClientRunner
    print(f"\n[Bước 1/4] Khởi tạo ClientRunner cho target: {target_device}...")
    runner = ClientRunner(hw_arch=target_device)
    
    # 2. Parse ONNX model
    print(f"\n[Bước 2/4] Parsing đồ thị ONNX: {onnx_path}...")
    hn, npz = runner.translate_onnx_model(
        model=onnx_path,
        net_name=network_name,
        start_node_names=None,
        end_node_names=end_node_names
    )
    print(f"[+] Parse thành công ONNX vào Hailo HAR.")
    
    # Cấu hình tiền xử lý (Normalization: pixel / 255.0) + Best Practice Single-Context cho YOLO26s-P2
    # YOLO nhận đầu vào [0.0, 1.0], Hailo tích hợp trực tiếp phép chia 255 vào NPU
    alls_lines = [
        "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])",
        "performance_param(compiler_optimization_level=max)",
    ]
    if single_context:
        alls_lines.extend([
            "model_optimization_flavor(optimization_level=4, compression_level=4)",
            f"model_optimization_config(compression_params, auto_4bit_weights_ratio={auto_4bit_ratio:.2f})",
            "context_switch_param(max_control_utilization=0.90, max_compute_utilization=0.90, max_memory_utilization=0.90)",
        ])
    alls_script = "\n".join(alls_lines) + "\n"
    print(f"[*] Áp dụng Hailo Model Script (.alls):\n{alls_script}")
    try:
        runner.load_model_script(alls_script)
    except Exception as e:
        print(f"[*] Lưu ý cấu hình model script (fallback không ép context_switch_param): {e}")
        fallback_alls = (
            "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
            f"model_optimization_config(compression_params, auto_4bit_weights_ratio={auto_4bit_ratio:.2f})\n"
            "performance_param(compiler_optimization_level=max)\n"
        )
        try:
            runner.load_model_script(fallback_alls)
        except Exception as e2:
            print(f"[*] Lưu ý fallback model script: {e2}")
    
    # 3. Lượng tử hóa 8-bit INT8 (Full Integer Post-Training Quantization)
    print(f"\n[Bước 3/4] Lượng tử hóa INT8 với tập ảnh Calibration...")
    calib_dataset = load_calibration_dataset(calib_dir, imgsz=imgsz)
    runner.optimize(calib_dataset)
    print(f"[+] Lượng tử hóa INT8 hoàn tất!")
    
    # 4. Biên dịch sang HEF (Hailo Executable Format)
    print(f"\n[Bước 4/4] Biên dịch đồ thị mạng sang HEF...")
    hef = runner.compile()
    
    with open(output_hef_path, "wb") as f:
        f.write(hef)
    print(f"\n[THÀNH CÔNG] File HEF đã được tạo: {output_hef_path}")
    print(f"  -> Sẵn sàng copy vào Raspberry Pi 5 để chạy inference với HailoRT!\n")

    # Lưu HAR file dự phòng
    runner.save_har(har_path)
    print(f"[+] Đã lưu file archive HAR: {har_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi ONNX sang HEF (INT8) cho Hailo-8 / Hailo-8L")
    parser.add_argument("--onnx", type=str, default="models/yolo26s_p2_epoch14_640.onnx", help="File ONNX đầu vào")
    parser.add_argument("--calib_dir", type=str, default="calib_images", help="Thư mục chứa ảnh calibration")
    parser.add_argument("--output_hef", type=str, default="models/yolo26s_p2_epoch14_640.hef", help="File HEF đầu ra")
    parser.add_argument("--target", type=str, default="hailo8", choices=["hailo8", "hailo8l"], help="NPU đích: hailo8 hoặc hailo8l (Pi 5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh vuông")
    parser.add_argument("--single_context", action="store_true", default=True, help="Tối ưu ép model vào 1 context (tắt context switch)")
    parser.add_argument("--auto_4bit_ratio", type=float, default=0.25, help="Tỉ lệ nén trọng số 4-bit để vừa SRAM Hailo-8")
    parser.add_argument("--end_nodes", nargs="+", default=None, help="Danh sách tên các output node cuối trước post-process")
    args = parser.parse_args()

    convert_onnx_to_hef(
        onnx_path=args.onnx,
        calib_dir=args.calib_dir,
        output_hef_path=args.output_hef,
        target_device=args.target,
        imgsz=args.imgsz,
        single_context=args.single_context,
        auto_4bit_ratio=args.auto_4bit_ratio,
        end_node_names=args.end_nodes,
    )
