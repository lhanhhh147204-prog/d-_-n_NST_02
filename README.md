# 🧬 Dự Án Phân Tích Hình Thái Nhiễm Sắc Thể (Medical Karyotyping Pipeline)

Dự án này là một quy trình hoàn chỉnh (End-to-End Pipeline) nhằm tự động hoá việc phân tích hình ảnh Nhiễm Sắc Thể (NST) y khoa. Bắt đầu từ một bức ảnh thô chụp dưới kính hiển vi, hệ thống sử dụng các thuật toán xử lý ảnh số và Trí Tuệ Nhân Tạo (Deep Learning) để tự động trích xuất, tách cụm, duỗi thẳng, phân loại và sắp xếp chúng thành một bản đồ Karyogram chuẩn y khoa (gồm 23 cặp NST).

## 🚀 Các Tính Năng Cốt Lõi & Pipeline 5 Bước

Dự án được cấu thành từ 5 bước (Pipeline) xử lý cốt lõi, tuân thủ nghiêm ngặt nguyên tắc **không làm biến dạng cấu trúc sinh học** của Nhiễm Sắc Thể.

### 1. Bước 1: Tăng cường ảnh, giảm mờ (Deblurring)
- **Mục đích:** Làm rõ vân sọc G-banding trên NST, tiền xử lý để thuật toán dễ dàng nhận diện cấu trúc.
- **Phương pháp:** Áp dụng mô hình mạng **MPRNet** (thông qua class `Deblurrer`) để khử mờ hình ảnh từ weights tải trước. Đồng thời đánh giá chất lượng đầu ra qua SSIM và Delta E. Các thuật toán truyền thống như Unsharp Masking và CLAHE cũng có thể được áp dụng bổ trợ.

### 2. Bước 2: Tách từng cụm nhiễm sắc thể bên trong ảnh (Segmentation)
- **Phân loại (Classification):** Sử dụng các mô hình AI:
  - **CCINet:** Sử dụng kiến trúc SEResNet61Backbone để phân loại 4 lớp.
  - **DualBranchModel:** Mô hình lai (Hybrid) kết hợp giữa FasterRCNN_ResNet50_FPN_V2 và Swin_T (Swin Transformer) với cơ chế Attention Fusion, xử lý đầu vào 1 kênh màu.
- **Tách cụm (Clustering):** Sử dụng `PixelDensityHeatmap` và Gaussian Blur để dò tìm vùng mật độ pixel cao, tạo mask định vị (mask màu đỏ), cắt bounding box, và padding đưa ảnh về nền trắng chuẩn (kích thước 128x128 hoặc 224x224) nhưng hoàn toàn **giữ nguyên tỷ lệ sinh học**.

### 3. Bước 3: Xử lý các cụm chạm, chồng, chồng chạm (Core Resolution)
*Đây là bước xử lý cốt lõi và phức tạp nhất, dùng để tách các NST bị dính vào nhau (Overlapping/Touching).*
- **Phân tích hình học (Geometric Concavity):**
  - Dùng `find_concave_points` để tìm các điểm lõm trên đường viền ngoài (contour) của cụm NST.
  - Dùng `find_best_cutting_points` chọn ra các cặp điểm lõm tối ưu nhất để chia cắt.
  - Dùng `compute_separation_path` tính toán đường phân tách đi qua vùng có mật độ điểm ảnh thấp nhất giữa các nhánh.
- **Bóc tách mượt mà (Smooth Separation):** 
  - Không cắt thô bạo (hard cut). Thay vào đó, sử dụng Distance Transform (`cv2.distanceTransform`) trên cut_mask để tạo ra **Gradient Mask**.
  - Áp dụng **Alpha Blending** với Gradient Mask để quá trình tách hai phần mượt mà, không gãy gắt ở biên. Kết hợp inpaint (như TELEA) để tái tạo phần di truyền bị che khuất.
  - Hai phần NST sau đó được tách riêng, chuẩn hóa và đặt vào giữa nền trắng 224x224.

### 4. Bước 4: Duỗi thẳng nhiễm sắc thể (Straightening/Unbending)
- **Xử lý hình thái uốn cong (Bent):**
  - Trích xuất khung xương (Skeletonization) thông qua `extract_skeleton` để tìm trục trung tâm.
  - Dựa trên trục này, hệ thống sẽ tính toán góc độ cong và tiến hành duỗi thẳng (Unbending) mà **không làm thay đổi độ dày mỏng hay kích thước hạt/gai** của NST.

### 5. Bước 5: Phân loại định danh (Karyotyping/Final Classification)
- Phân loại bằng **Swin Transformer** (SwinKaryotype) cho 24 lớp bao gồm từ NST số 1 đến 22, và NST giới tính X, Y.
- **Xác định giới tính:** Đếm số lượng NST X/Y để kết luận giới tính mẫu (XX hoặc XY).
- **Ghép cặp (Pairing):** Phân bổ 46 NST vào 23 cặp dựa trên Thuật toán Hungarian kết hợp độ tự tin của AI.
- **Vẽ Karyogram (Visualization):** Trình bày NST lên lưới chuẩn y khoa, xuất ra file báo cáo cuối cùng.

---

## 📂 Cấu Trúc Thư Mục Chi Tiết

```text
c:\Users\lehoa\dự_án_NTS/
├── code_refactored/      # Cấu trúc mã nguồn đã được tối ưu hóa
│   ├── config/           # Các tham số cấu hình (settings.py)
│   ├── medical_pipeline/ # Lõi xử lý chính của dự án
│   │   ├── pipeline/     # Nơi nối ghép tuần tự 5 bước xử lý
│   │   └── ...           
├── data/                 # Dữ liệu dự án, ảnh raw, ảnh đã tiền xử lý
├── karyotype_voc_massive/# Dataset lớn chuẩn VOC phục vụ cho bài toán Karyotyping
├── nst_tach_models_logs_checkpoints/ # Nơi lưu trữ models (.pth, .keras), logs và checkpoints của quá trình huấn luyện
├── .env                  # Cấu hình biến môi trường cục bộ
├── .gitignore            # Khai báo file bị loại bỏ khỏi quản lý phiên bản
├── main.py               # File chạy chính của toàn bộ dự án
├── pyproject.toml        # Quản lý dependencies và meta-data dự án (uv / poetry)
├── uv.lock               # Lock dependencies phiên bản chính xác
└── AI_RULES.md           # Bộ quy tắc quản lý cho các Agent/Developer tham gia dự án
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Yêu cầu hệ thống
- **Ngôn ngữ:** Python 3.12+
- **Thư viện chính:** TensorFlow, Keras, PyTorch, torchvision, timm (Swin-T), OpenCV (cv2), numpy, scipy.
- Khuyến nghị sử dụng **uv** hoặc **pip** để quản lý môi trường ảo.

### 2. Cài đặt Dependencies
Bạn có thể cài đặt thông qua `uv` (công cụ quản lý Python tốc độ cao):
```bash
# Cài đặt uv nếu chưa có
pip install uv

# Đồng bộ dependencies
uv sync
```

Hoặc qua `requirements.txt` (nếu có):
```bash
pip install -r requirements.txt
```

### 3. Thực thi Pipeline
Chạy toàn bộ pipeline trên một ảnh hoặc một bộ dữ liệu thông qua file `main.py`:
```bash
uv run main.py
```
*(Tham số cấu hình cụ thể có thể tuỳ chỉnh tại `code_refactored/config/settings.py`)*

### 4. Huấn luyện mô hình (Dành cho Developer/Researcher)
Dự án cung cấp kịch bản huấn luyện riêng biệt cho các tác vụ khác nhau:
- **Huấn luyện mô hình bóc tách (U-Net/Dual Branch):**
  ```bash
  uv run train_dual_branch.py
  ```
- **Huấn luyện mô hình phân loại Karyotype (Swin Transformer):**
  ```bash
  uv run train_karyotype.py
  ```

---

## ⚠️ Những Lưu Ý Quan Trọng (Dành Cho Đóng Góp Code)

1. **Tuân thủ Nguyên Tắc Sinh Học:** Bất kỳ chỉnh sửa nào trong module Morphology (tiền xử lý ảnh) đều KHÔNG được phép làm biến dạng NST (thay đổi độ dày, làm mất cấu trúc hạt/gai).
2. **Xử Lý Chồng Chạm:** Không được dùng các phép cắt cứng (hard cut) làm đứt gãy NST. Bắt buộc phải sử dụng **Alpha Blending với Gradient Mask**.
3. **Tránh Lỗi Khởi Tạo Mô Hình Nặng:** Khi thiết kế suy luận (inference), cần nạp model một lần duy nhất (Singleton Pattern) vào bộ nhớ. Tránh việc tải `torch.load` bên trong các vòng lặp gây tràn RAM.
4. **Không Dùng Đường Dẫn Cứng (Hardcoded Paths):** Sử dụng biến môi trường (qua `.env`) hoặc đường dẫn tương đối để dễ dàng triển khai trên môi trường khác biệt.
