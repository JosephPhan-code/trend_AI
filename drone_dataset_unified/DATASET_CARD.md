# Drone Detection Dataset Card — `v3.0.0`

> **Trạng thái:** `ACTIVE_PRODUCTION` | **Phát hành:** `2026-11-24T15:16:50`  
> **Thư mục chuẩn:** `e:\trend_AI\drone_dataset_unified`  
> **Mã kiểm toàn vẹn (`dataset_sha256`):** `34df6f9935237722a40fdea3a2b47740d04b642abdada95690a773c5702c3010`

## 1. Tổng quan nhanh (`v3.0.0`)
- **Tổng ảnh duy nhất:** **`320,980` ảnh** (`36.84 GB` dữ liệu ảnh gốc, hardlink không trùng lặp)
- **Tổng bounding boxes (`0: drone`):** **`867,389` boxes**
- **Chia tập (Zero-Leakage):**
  - `train`: **`263,332`** (`82.04%`)
  - `val`: **`28,762`** (`8.96%`)
  - `test`: **`28,886`** (`9.00%`)
- **Độ sạch & chống rò rỉ:**
  - Cấu trúc & ghép cặp (`unpaired / invalid / hn_with_box / pos_empty`): **`0 / 0 / 0 / 0` (`100%` sạch)**
  - Rò rỉ `MD5` & `dHash64` entropy cao giữa `train / val / test`: **`0 / 0 / 0`**
  - Chấm mù trực quan 88 mẫu ngẫu nhiên sau Clean Teacher v3: **`87/88` (`98.9%`) zero-FP**, riêng positive đạt **`72/72` (`100%`)** không có box báo giả.

---

## 2. Danh mục chi tiết theo `source_tag` (16 nguồn)
| `source_tag` | `modality_tag` | `domain_tag` | Tổng ảnh (%) | Train | Val | Test | Mô tả nguồn |
|---|---|---|---:|---:|---:|---:|---|
| `d2_rgb_antiuav300` | `rgb` | `tracking_hud_rgb` | 71,559 (22.29%) | 58,543 | 6,458 | 6,558 | Anti-UAV-RGBT-300 Visible RGB video tracking |
| `c1_datatrain` | `rgb` | `mixed_autolabel_purified` | 67,785 (21.12%) | 55,889 | 5,917 | 5,979 | Batch 1 merged open datasets (Teacher v3 purified) |
| `d2_rgb_pathikg` | `rgb` | `outdoor_youtube_park` | 53,722 (16.74%) | 43,153 | 5,292 | 5,277 | Pathikg outdoor RGB drone park/desert/handheld |
| `d2_rgb_mavvid` | `rgb` | `outdoor_video_multi` | 36,523 (11.38%) | 29,059 | 3,745 | 3,719 | MAV-VID outdoor multi-drone video sequences |
| `d2_ir_2025` | `thermal_ir` | `thermal_swarm_ir` | 25,000 (7.79%) | 22,300 | 1,358 | 1,342 | Anti-UAV 2025 thermal IR multi-drone swarms |
| `d2_ir_410` | `thermal_ir` | `thermal_single_ir` | 24,984 (7.78%) | 20,412 | 2,259 | 2,313 | Anti-UAV 410 thermal IR single-drone tracking |
| `c1_tonghop` | `rgb` | `tracking_hud_rgb` | 12,730 (3.97%) | 10,716 | 1,029 | 985 | Batch 1 consolidated Anti-UAV & night LED frames |
| `c1_orig` | `rgb` | `mixed_real_closeup` | 6,765 (2.11%) | 5,648 | 565 | 552 | Batch 1 original close-up / outdoor / handheld drones |
| `d2_rgb_dvb10k` | `rgb` | `synthetic_distractor_rgb` | 5,178 (1.61%) | 4,108 | 533 | 537 | Drone-vs-Bird synthetic RGB (purified drones near birds/planes) |
| `hn_bird` | `rgb` | `hn_birds_heli_planes` | 4,596 (1.43%) | 3,652 | 473 | 471 | Hard Negative: pure birds, helicopters, airliners (0 drone) |
| `hn_skyair` | `rgb` | `hn_airplanes` | 2,500 (0.78%) | 2,001 | 250 | 249 | Hard Negative: commercial airplanes in sky (0 drone) |
| `hn_skyant` | `rgb` | `hn_poles_antennas_towers` | 2,499 (0.78%) | 1,999 | 250 | 250 | Hard Negative: utility poles, pylons, antennas, towers (0 drone) |
| `hn_skycloud` | `rgb` | `hn_clouds_sky` | 2,499 (0.78%) | 1,999 | 248 | 252 | Hard Negative: clouds, empty skies, rooftops (0 drone) |
| `hn_skybird` | `rgb` | `hn_birds_sky_ir` | 2,495 (0.78%) | 2,030 | 227 | 238 | Hard Negative: birds & thermal sky/branches (0 drone) |
| `hn_antiuav` | `rgb` | `hn_empty_hud_buildings` | 1,147 (0.36%) | 949 | 98 | 100 | Hard Negative: purified empty HUD crosshairs/buildings/cranes |
| `d2_rgb_detfly` | `rgb` | `topdown_nadir_rgb` | 998 (0.31%) | 874 | 60 | 64 | Det-Fly top-down / nadir tiny RGB drones |

---

## 3. Thống kê theo hệ thống Tag quản lý

### 3.1. Theo loại cảm biến (`modality_tag`) & Nhãn (`label_type_tag`)
| Tag | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `rgb` | 270,996 | 84.43% |
| `thermal_ir` | 49,984 | 15.57% |
| `hard_negative_0box` | 15,736 | 4.90% |
| `positive_drone` | 305,244 | 95.10% |

### 3.2. Theo mật độ drone trong ảnh (`density_tag`)
| `density_tag` | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `single_drone` | 259,336 | 80.80% |
| `swarm_drone` | 24,630 | 7.67% |
| `multi_drone` | 21,278 | 6.63% |
| `empty_hn` | 15,736 | 4.90% |

### 3.3. Theo kích thước drone (`scale_tag`, tính theo diện tích trung vị box)
| `scale_tag` | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `tiny_small` | 174,347 | 54.32% |
| `medium` | 101,208 | 31.53% |
| `large_closeup` | 29,689 | 9.25% |
| `none` | 15,736 | 4.90% |

### 3.4. Theo lịch sử xử lý / làm sạch (`provenance_tag`)
| `provenance_tag` | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `original_verified` | 297,401 | 92.65% |
| `hard_negative_verified` | 15,736 | 4.90% |
| `purified_teacher_v3` | 6,331 | 1.97% |
| `fixed_v3_surgical` | 1,512 | 0.47% |

### 3.5. Theo đợt thu thập (`batch_origin`)
| `batch_origin` | Số ảnh | Tỷ lệ |
|---|---:|---:|
| `batch1_clean` | 87,280 | 27.19% |
| `batch2_data2` | 233,700 | 72.81% |

---

## 4. Hướng dẫn quản lý nhanh bằng CLI (`manage_dataset.py`)
Chạy tại thư mục `e:\trend_AI\drone_dataset_unified`:

```powershell
# 1. Xem nhanh phiên bản hiện tại và toàn bộ thống kê tag
python -X utf8 manage_dataset.py info

# 2. Kiểm tra toàn vẹn SHA256 (phát hiện ngay nếu có ảnh/nhãn bị sửa hoặc thiếu)
python -X utf8 manage_dataset.py verify

# 3. Truy vấn nhanh theo tag bất kỳ (vd: ảnh nhiệt IR trong tập val, hoặc drone nhỏ)
python -X utf8 manage_dataset.py query --modality thermal_ir --split val
python -X utf8 manage_dataset.py query --domain hn_poles_antennas_towers
python -X utf8 manage_dataset.py query --provenance purified_teacher_v3 --scale tiny_small

# 4. Xuất ngay tập con để train thử nghiệm (0 GB dung lượng phụ, chỉ tạo file .yaml + .txt)
python -X utf8 manage_dataset.py export-subset --name rgb_only_v3.0.0 --modality rgb
python -X utf8 manage_dataset.py export-subset --name ir_only_v3.0.0 --modality thermal_ir
python -X utf8 manage_dataset.py export-subset --name no_autolabel_v3.0.0 --exclude-source c1_datatrain

# 5. Đóng gói phiên bản mới (vd: v3.1.0) sau khi bổ sung/sửa dữ liệu
python -X utf8 manage_dataset.py snapshot --version v3.1.0 --note "Bo sung them 5000 anh dem"
```

## 5. File lưu trữ siêu dữ liệu (Metadata)
- `DATASET_VERSION.json`: Manifest phiên bản hiện tại (JSON đọc tự động bằng code).
- `CHANGELOG.md`: Lịch sử thay đổi từ `v1.0.0` -> `v2.0.0` -> `v3.0.0`.
- `metadata/dataset_catalog_v3.0.0.parquet` & `.csv`: Bảng gắn đầy đủ **10 cột tag** cho từng ảnh trong `320,980` ảnh.
- `e:\trend_AI\DATASET_REGISTRY.json`: Sổ đăng ký tổng các thư mục dataset trong workspace.
