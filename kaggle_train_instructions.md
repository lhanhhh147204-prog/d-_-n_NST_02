# Hướng Dẫn Huấn Luyện Bằng GPU Miễn Phí Trên Kaggle

Kaggle (nền tảng thuộc sở hữu của Google) là một sự thay thế tuyệt vời cho Colab với ưu điểm: GPU mạnh, ít bị văng (disconnect) và có thể cắm máy chạy ngầm 12 tiếng.

> [!IMPORTANT]
> **Về Vấn Đề Bảo Mật & Rò Rỉ Dữ Liệu:**
> Trên Kaggle, mọi Bộ Dữ liệu (Dataset) và Mã nguồn (Notebook) khi bạn tạo ra đều được **MẶC ĐỊNH LÀ PRIVATE (Riêng tư)**. Chỉ duy nhất tài khoản của bạn mới có thể nhìn thấy và truy cập. Nó hoàn toàn an toàn và không bị leak ra ngoài (trừ khi bạn tự tay ấn nút chuyển sang Public). Vì vậy, bạn hoàn toàn yên tâm dùng Kaggle cho dự án y khoa này.

---

## Các Bước Thực Hiện Chạy Kaggle

### Bước 1: Tạo Bộ Dữ Liệu (Private Dataset)
1. Truy cập [Kaggle.com](https://www.kaggle.com/) và đăng nhập/đăng ký tài khoản.
2. Ở menu bên trái, chọn dấu **`+ Create`** -> **`Dataset`**.
3. Khung upload hiện ra, bạn điền **Tên Dataset** (Ví dụ: `nts-project-data`).
4. Kéo thả file `code_refactored_for_colab.zip` của bạn vào đó.
5. Ở góc dưới cùng bên trái của khung upload, kiểm tra biểu tượng 🔒 ổ khóa. Nếu ghi **"Private"** là đã an toàn 100%.
6. Bấm **Create** và chờ Kaggle tải lên. *(Điểm hay của Kaggle là nó sẽ tự động giải nén file zip này cho bạn).*

### Bước 2: Tạo Máy Chủ Kéo GPU (Notebook)
1. Ở menu bên trái, chọn **`+ Create`** -> **`Notebook`**.
2. **Bật GPU:** Ở thanh Menu bên phải (mục *Session options*), tìm dòng **Accelerator**, bấm vào đó và chọn **`GPU P100`** hoặc **`GPU T4x2`** đều được.
3. **Thêm dữ liệu:** Cũng ở cột bên phải, bấm vào nút **`+ Add Data`** (ở phần Input). Chuyển sang tab **`Your Datasets`** và nhấn nút dấu cộng (+) ở dòng `nts-project-data` bạn vừa tạo ban nãy.

### Bước 3: Chạy Lệnh Huấn Luyện (Code)
Kaggle có một đặc thù: Thư mục Data đầu vào (`/kaggle/input/`) là **Read-only** (Chỉ đọc, không cho lưu file model vào đó). Do vậy, ta cần copy code sang thư mục làm việc `/kaggle/working/` trước khi chạy.

Bạn hãy copy đoạn code dưới đây vào Cell (ô trống) của Notebook và bấm nút **Play** (hoặc Shift+Enter):

```python
import os
import shutil

# 1. Tìm đúng thư mục chứa file train_karyotype.py bất chấp Kaggle giải nén ra sao
source_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train_karyotype.py' in files:
        source_dir = root
        break

if source_dir:
    print(f"✅ Đã tìm thấy mã nguồn tại: {source_dir}")
    
    # 2. Copy sang Working để có quyền lưu file model
    work_dir = '/kaggle/working/code_refactored'
    if not os.path.exists(work_dir):
        shutil.copytree(source_dir, work_dir)
        print("✅ Đã copy code sang thư mục làm việc.")
    
    # 3. Cài thư viện và chạy thẳng trong môi trường Python
    os.chdir(work_dir)
    print("⏳ Đang cài đặt thư viện...")
    os.system("pip install torch torchvision opencv-python Pillow numpy tqdm scikit-learn -q")
    
    print("🚀 BẮT ĐẦU HUẤN LUYỆN...")
    os.system("python train_dual_branch.py")
else:
    print("❌ LỖI: Không tìm thấy mã nguồn. Xin hãy kiểm tra lại mục Input xem đã Add Data chưa.")
```

> [!TIP]
> **Mẹo cắm máy chạy ngầm (Không cần bật màn hình):**
> Kaggle có nút **"Save Version"** (ở góc trên cùng bên phải). Khi bạn bấm vào nút này và chọn **"Save & Run All (Commit)"**, Kaggle sẽ đóng gói Notebook, đưa vào máy chủ chạy ngầm. Bạn có thể tắt máy tính đi ngủ. Sáng mai mở lại là có kết quả!

### Bước 4: Tải Trọng Số Xuống
Khi quá trình chạy báo `%cd /kaggle/working/code_refactored` và train xong 135 epoch, bạn nhìn sang thanh Menu bên phải, ở mục **Output** (hoặc trong chính thư mục `/kaggle/working/code_refactored/ai_training/weights/`). 
Sẽ có file `karyotype_swint_best.pth`. Bạn chỉ việc bấm biểu tượng `⋮` bên cạnh file đó và chọn **Download** về máy. Mọi thứ hoàn tất!
