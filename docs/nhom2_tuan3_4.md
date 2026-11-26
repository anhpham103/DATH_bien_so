# Khâu đọc ký tự – phương án B · Tuần 3–4 (Tuấn Anh)

Nhánh: `nhom2/ocr-training`. Chạy toàn bộ bằng `notebooks/tuan3_4_phuong_an_B.ipynb` trên Colab hoặc Kaggle có GPU.

## Đầu vào cần có (đầu ra tuần 2)

| File | Nội dung |
| --- | --- |
| `datasets/chars/images/{train,val,test}`, `datasets/chars/labels/...` | Ảnh biển cắt (nới 8%), nhãn YOLO 30 lớp, chia theo `splits.csv` |
| `datasets/plate_strings.csv` | Cột đường dẫn ảnh (`crop_path`/`image`/`file`...) và cột chuỗi (`text`/`plate`/`label`...), biển 2 dòng có `\|`, ví dụ `59X1\|12345` |

Script tự nhận tên cột phổ biến. Nếu tên khác, sửa `PATH_COLS` hoặc `TEXT_COLS` trong `scripts/compare_ab.py`.

## Tuần 3

| Việc | Lệnh | Sản phẩm |
| --- | --- | --- |
| Đưa ảnh qua `enhance` (CLAHE), kiểm tra nhãn, thống kê ký tự | `python scripts/enhance_chars_dataset.py` | `datasets/chars_enh/`, `datasets/data_chars_enh.yaml`, `results/char_stats.csv` |
| Huấn luyện YOLO26n 30 lớp, ảnh 320, `fliplr=0`, seed 42 | `python scripts/train_chars.py --project <Drive>/runs/detect --device 0` | `models/char_yolo26n.pt`, `results/chars_b_train.csv` |
| Phiên bị ngắt | thêm `--resume` | chạy tiếp từ `last.pt` |
| `sort_chars`, `read_chars` | `src/read_chars.py`, kiểm thử `pytest tests/` | bàn giao **H4** (thứ 6) |

## Tuần 4

| Việc | Lệnh | Sản phẩm |
| --- | --- | --- |
| So sánh A và B trên tập **kiểm định** | `python scripts/compare_ab.py --split val` | `results/ab_comparison_val.csv`, `results/ab_predictions_val.csv` |
| Sửa lỗi sắp xếp biển 2 dòng | đã sửa trong `sort_chars`; dòng `B_guide` trong bảng so sánh cho thấy mức cải thiện | |
| Xuất ONNX và kiểm tra ONNX đọc giống `.pt` | `python scripts/export_chars_onnx.py` | `models/char_yolo26n.onnx`, `results/chars_onnx_check.csv` → bàn giao **H6** |
| Ảnh từng dòng cho phương án C | `python scripts/prepare_lines_c.py` | `datasets/lines/{train,val,test}.txt`, `dict_chars.txt`, `results/lines_c_stats.csv` |

## Các lỗi sắp xếp đã sửa so với mã mẫu (mục 5.3 Phần 1)

| Trường hợp | Mã mẫu | Bản mới |
| --- | --- | --- |
| Biển 2 dòng chỉ phát hiện được dòng dưới (dòng trên mờ, bị che) | `1234\|5` (cắt giữa các ký tự cùng dòng) | `\|12345` |
| Khung nhiễu thấp (ốc vít, logo dập chìm) giữa hai dòng | `59X18\|12345` | `59X1\|12345` |
| Biển 2 dòng chỉ có 1 ký tự ở dòng trên | `\|5` | `5\|` |
| Cùng một ký tự bị nhận thành hai lớp (`8` và `B`) | giữ cả hai | `agnostic_nms=True`, giữ một |

Ngưỡng: khe giữa hai dòng phải lớn hơn 0,5 × chiều cao ký tự trung vị. Nếu không, cắt ở giữa ảnh. Khung thấp hơn 0,4 × chiều cao trung vị bị bỏ.

## Phụ thuộc vào mô-đun của Bảo

Các script nạp `rectify.py`, `postprocess.py` và `read_baseline` thật trong `src/`. Nếu chưa có (trước H4), script dùng bản mã mẫu Phần 1 trong `src/_guide_fallback.py` và in cảnh báo. Không sửa file đó thay cho mô-đun của Bảo; sau khi có H4, xóa nó trong một PR riêng.

## Bảng so sánh A – B (điền từ `results/ab_comparison_val.csv`)

| Phương án | CER | Biển đúng, trước hậu xử lý | Biển đúng, sau hậu xử lý | ms/biển (CPU) |
| --- | --- | --- | --- | --- |
| A. PaddleOCR | | | | |
| B. YOLO26n ký tự | | | | |

Mục tiêu M2: B đạt từ 85% biển đúng hoàn toàn trên tập kiểm định (Khá 92%), CER từ 3% trở xuống.
Bảng cuối cho báo cáo đo **một lần** trên tập kiểm thử ở tuần 5 (`--split test`), sau khi đóng băng H6.
