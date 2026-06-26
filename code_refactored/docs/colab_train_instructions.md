# Hướng dẫn chạy Train Mô Hình trên Google Colab

**Bước 1: Upload file nén lên Google Drive**
Đảm bảo bạn đã upload file `code_refactored_for_colab.zip` vào thư mục `dự án nhiễm sắc thể` trên Google Drive của bạn.

> ⚠️ **QUAN TRỌNG:** Hãy nén lại thư mục `code_refactored` MỚI NHẤT (sau khi đã sửa code) rồi upload đè lên file cũ trên Google Drive.

**Bước 2: Chạy lệnh trên Colab**
Mở Google Colab, tạo Notebook mới, bật GPU (Runtime > Change runtime type > Hardware accelerator > T4 GPU).
Sau đó copy ĐOẠN CODE DƯỚI ĐÂY vào Cell và bấm nút Play:

```python
# 1. Kết nối Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Giải nén file
!unzip -o -q "/content/drive/MyDrive/dự án nhiễm sắc thể/code_refactored_for_colab.zip" -d "/content/NTS_Project"

# 3. Di chuyển vào đúng thư mục code, cài thư viện và chạy Train
%cd /content/NTS_Project/code_refactored
!pip install torch torchvision opencv-python Pillow numpy tqdm scikit-learn

# 4. CHẠY HUẤN LUYỆN
# Chọn 1 trong 2 lệnh dưới đây tùy theo mục đích:
!python train_karyotype.py     # (Train model phân loại Karyotype 24 lớp)
# Hoặc:
!python train_dual_branch.py   # (Train mô hình Siêu phân loại Cụm NST Đơn/Chạm/Chồng)
```

**Bước 3: Theo dõi quá trình huấn luyện**
Quá trình huấn luyện sẽ gồm 2 giai đoạn:
- **Phase 1 (15 epoch):** Đóng băng backbone, chỉ train Head → Accuracy tăng rất nhanh
- **Phase 2 (135 epoch):** Mở băng toàn bộ, fine-tune tất cả → Accuracy tiếp tục tăng dần

Nếu 25 epoch liên tiếp không cải thiện, hệ thống sẽ tự dừng sớm (Early Stopping).

**Bước 4: Lấy file kết quả**
Khi train xong, hệ thống sẽ lưu model tốt nhất tại thư mục:
`/content/NTS_Project/code_refactored/ai_training/weights/karyotype_swint_best.pth`
Hãy tải file này về và ném vào thư mục `ai_training/weights/` trong dự án trên máy tính của bạn.
