# Dataset Changelog — Drone Detection

Tất cả các phiên bản dữ liệu đều tuân theo chuẩn Semantic Versioning (`MAJOR.MINOR.PATCH`):
- `MAJOR`: Gộp nguồn lớn / thay đổi cấu trúc chia tập hoặc quy mô bộ dữ liệu.
- `MINOR`: Thêm đợt ảnh mới, lọc sạch bằng Teacher mới, hoặc cập nhật Hard Negative.
- `PATCH`: Sửa lẻ nhãn/box nhỏ không thay đổi nguồn.

---

## [`v3.0.0`] — 2026-11-24 (ACTIVE PRODUCTION)
- **Thư mục:** `e:\trend_AI\drone_dataset_unified`
- **SHA256:** `34df6f9935237722a40fdea3a2b47740d04b642abdada95690a773c5702c3010`
- **Quy mô:** **`320,980` ảnh duy nhất** | **`867,389` boxes** (`train`: `263,332` / `val`: `28,762` / `test`: `28,886`).
- **Cải tiến chính:**
  - Hợp nhất sạch `v1.0.0` (`clean_dataset`) + `v2.0.0` (`data2`) + `10.000` ảnh Hard Negative bầu trời (`d2_hn_sky*`) bằng NTFS hardlink (không nhân đôi dung lượng).
  - Quét đa tín hiệu toàn bộ `333.009` ảnh + sửa/cách ly lỗi nhãn đợt 1 (`fix_all_label_errors.py` + `polish_fixes_v3.py`): làm sạch `3.027` ảnh drone lẫn trong `_hn_antiuav_`, xóa cột đèn sân vận động, gộp box tách đôi LED ban đêm, mở rộng box cắt hụt nền trắng, lọc chim/trực thăng/máy bay trong `dvb10k`, xóa đồ họa 2D.
  - Khử trùng lặp tuyệt đối **`4.510`** bản trùng MD5 + **`12.196`** bản trùng hình ảnh `dHash64` chéo đợt & HN mâu thuẫn -> **`0` rò rỉ MD5 và `0` rò rỉ `dHash64` entropy cao** giữa `train / val / test`.
  - Huấn luyện **Clean Teacher v3** (20K ảnh tinh sạch 100%, `mAP50=0.893`, FP cột điện chỉ `2.80%`) và tinh lọc thêm **`7.968`** ảnh (`4.477` xóa box giả phụ, `1.854` sửa box dịch sai, `1.637` cách ly ảnh auto-label rác).
  - Kiểm định mù 88 mẫu ngẫu nhiên đạt **`98.9%` zero-FP** (`72/72` ảnh positive sạch 100% không có box báo giả).
  - Gắn đầy đủ hệ thống Tag 10 chiều (`metadata/dataset_catalog_v3.0.0.parquet` + `.csv`) và công cụ CLI `manage_dataset.py`.

---

## [`v2.0.0`] — 2026-11-22 (ARCHIVED SOURCE — `e:\trend_AI\data2`)
- Tổng hợp đợt 2 quy mô `227.636` ảnh đa dạng màu RGB + nhiệt IR (`rgb_antiuav300`, `rgb_pathikg`, `rgb_mavvid`, `rgb_dvb10k`, `rgb_detfly`, `ir_2025`, `ir_410`, Hard Negative `_hn_bird_` / `_hn_antiuav_`) và bổ sung `10.000` ảnh bầu trời `d2_hn_sky*`.
- Chia phân tầng 80/10/10 chống học vẹt. Đã được sửa lỗi v3 và sáp nhập vào `v3.0.0`.

---

## [`v1.0.0`] — 2026-11-20 (ARCHIVED SOURCE — `e:\trend_AI\clean_dataset`)
- Tổng hợp đợt 1 (`105.373` ảnh gồm `c1_orig`, `c1_tonghop`, `c1_datatrain` và Hard Negative đợt 1).
- Đã được làm sạch lại toàn diện (loại cột đèn, bóng râm, icon 2D, hộp kép), khử trùng lặp chéo với đợt 2 và sáp nhập vào `v3.0.0`.
