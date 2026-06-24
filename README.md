# Dự Án Phân Tích Hình Thái Nhiễm Sắc Thể (Medical Karyotyping Pipeline)

Dự án này là một quy trình hoàn chỉnh (End-to-End Pipeline) tự động hoá việc phân tích hình ảnh Nhiễm Sắc Thể (NST) y khoa. Từ một bức ảnh thô chụp dưới kính hiển vi, hệ thống sử dụng các thuật toán xử lý ảnh số và Trí Tuệ Nhân Tạo (Deep Learning) để tự động trích xuất, tách cụm, duỗi thẳng, phân loại và sắp xếp chúng thành một bản đồ Karyogram chuẩn y khoa (gồm 23 cặp NST).

## 🚀 Các Tính Năng Cốt Lõi (8 Bước Tự Động)

1. **Tiền xử lý & Làm nét (Deblurring):** Áp dụng Unsharp Masking và CLAHE để làm rõ vân sọc G-banding trên NST.
2. **Trích xuất cụm NST (Segmentation):** Áp dụng Thresholding và Morphology để bóc tách các NST ra khỏi nền tối.
3. **Xử lý cụm chạm/chồng (Separation):** Phát hiện các cụm có từ 2 NST trở lên dính nhau (Touching/Overlapping). Sử dụng thuật toán cắt điểm lõm hình học (Geometric Concavity) và mạng U-Net để tách rời.
4. **Duỗi thẳng (Straightening):** Xoay và làm thẳng các NST bị cong vẹo để chuẩn hóa hình thái.
5. **Phân loại bằng AI (Classification):** Sử dụng mô hình Swin Transformer phân loại NST thành 24 lớp (Từ 1 đến 22, X, Y).
6. **Xác định giới tính (Sex Determination):** Đếm số lượng NST X/Y để kết luận giới tính mẫu (XX hoặc XY).
7. **Ghép cặp (Pairing):** Phân bổ 46 NST vào đúng 23 cặp dựa trên Thuật toán Hungarian và độ tự tin của AI.
8. **Vẽ Karyogram (Visualization):** Trình bày NST lên lưới chuẩn y khoa 4 hàng, xuất ra file báo cáo cuối cùng.

## 📂 Cấu Trúc Thư Mục

```text
NST_new/
├── ai_training/          # Mã nguồn huấn luyện các mô hình AI (CCINet, U-Net, SwinKaryotype)
├── config/               # Chứa các tham số cấu hình hệ thống
├── medical_pipeline/     # Lõi xử lý chính của dự án (Pipeline 8 bước)
│   ├── enhancement/      # Tiền xử lý, khử nhiễu, làm nét
│   ├── segmentation/     # Bóc tách và tìm cụm
│   ├── morphology/       # Hình thái học, thuật toán cắt điểm lõm, tìm tâm
│   ├── karyogram/        # Vẽ đồ thị Karyogram
│   └── pipeline/         # Nơi nối ghép các bước tuần tự (buoc1 -> buoc8)
├── models/               # Các file định nghĩa kiến trúc Deep Learning
├── pwarp/                # Module warp/xoay ảnh NST chuyên dụng
├── utils/                # Các hàm hỗ trợ chung (File I/O, Logging)
├── main.py               # File chạy chính của toàn bộ dự án
└── project_notebook.ipynb # Notebook nghiên cứu gốc (Dùng làm tham khảo)
```

## 🛠️ Hướng dẫn cài đặt và sử dụng

Dự án sử dụng công cụ [uv](https://github.com/astral-sh/uv) để quản lý môi trường.

### 1. Cài đặt môi trường
Mở Terminal tại thư mục của dự án và chạy:
```bash
uv sync
```

### 2. Chạy Pipeline Xử Lý Một Ảnh
Để chạy toàn bộ quá trình từ ảnh gốc ra Karyogram:
```bash
uv run main.py --step full_pipeline --input-dir "duong_dan_den_anh_goc.JPG"
```
*Kết quả (bao gồm ảnh từng bước và Karyogram cuối cùng) sẽ được lưu trong thư mục `results/`.*

### 3. Huấn luyện mô hình AI (Dành cho Developer)
- Huấn luyện U-Net (Tách cụm chồng): `uv run train_dual_branch.py`
- Huấn luyện SwinKaryotype (Phân loại): `uv run train_karyotype.py`

