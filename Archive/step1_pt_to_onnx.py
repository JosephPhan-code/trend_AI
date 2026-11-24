import os
import argparse
from ultralytics import YOLO

def export_pt_to_onnx(pt_path, output_dir=None, imgsz=640, opset=12, simplify=True):
    """
    Chuyển đổi file trọng số YOLO PyTorch (.pt) sang ONNX chuẩn hóa cho Hailo-8 NPU.
    - imgsz: 640x640 (kích thước tối ưu cho Hailo-8)
    - opset: 12 (chuẩn tương thích tốt nhất với Hailo Dataflow Compiler)
    - dynamic: False (Hailo yêu cầu kích thước tensor cố định Static Shape)
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Không tìm thấy file: {pt_path}")
    
    print(f"[*] Bắt đầu load model PyTorch: {pt_path}")
    model = YOLO(pt_path)
    
    print(f"[*] Đang xuất sang ONNX (imgsz={imgsz}, opset={opset}, simplify={simplify})...")
    onnx_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False  # Hailo NPU bắt buộc static input shape: 1x3x640x640
    )
    
    print(f"[+] Hoàn tất chuyển đổi PyTorch -> ONNX: {onnx_file}")
    
    if output_dir and os.path.exists(output_dir):
        dest_path = os.path.join(output_dir, os.path.basename(onnx_file))
        if os.path.abspath(onnx_file) != os.path.abspath(dest_path):
            import shutil
            shutil.move(onnx_file, dest_path)
            print(f"[+] Đã di chuyển file ONNX vào: {dest_path}")
            return dest_path
            
    return onnx_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi mô hình YOLO .pt sang .onnx chuẩn Hailo")
    parser.add_argument("--weights", type=str, default="models/best_epoch14.pt", help="Đường dẫn tới file .pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh vuông (mặc định 640)")
    parser.add_argument("--output_dir", type=str, default="models", help="Thư mục lưu kết quả ONNX")
    args = parser.parse_args()

    export_pt_to_onnx(args.weights, output_dir=args.output_dir, imgsz=args.imgsz)
