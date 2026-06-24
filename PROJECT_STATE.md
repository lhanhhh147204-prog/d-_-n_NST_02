# PROJECT STATE & MEMORY (Hệ thống Nhớ của AI)

> **Mục đích:** File này lưu trữ trạng thái hiện tại của dự án, các vấn đề đã kiểm chứng, và lộ trình sửa lỗi. AI phải luôn đọc file này trước khi đề xuất giải pháp để đảm bảo bám sát ngữ cảnh, không bị lan man sang các phần không cần thiết.

## 1. Trạng thái Pipeline (End-to-End)
- **Đã thông luồng (Working):** Pipeline chạy hoàn chỉnh từ đầu đến cuối (Từ 1 bức ảnh nguyên vẹn -> Cắt cụm -> Tách chồng chéo -> Duỗi thẳng -> Phân loại -> Ghép cặp -> Karyogram).
- **Mô hình U-Net Tách Cụm (Bước 3):** Đã có file trọng số `best_teacher.keras`. AI đã có thể trích xuất NST, dù có thể sót 1-2 NST (VD test `9250100210.1.k.JPG` ra 45/46 NST).
- **Trọng tâm hiện tại:** Khâu phân loại (Classification - Bước 5) và Ghép cặp (Pairing - Bước 7).

## 2. Vấn đề cốt lõi cần giải quyết ngay (Blocking Issues)
- ❌ **Bước 5 (SwinKaryotype) chưa được train:** Hệ thống báo `[BYPASS]` do thiếu `best_model_epoch.pth`. Thuật toán hiện tại fallback về phân loại theo kích thước (diện tích pixel), khiến việc gán nhãn sai bét (chủ yếu ra nhãn 19, 20, 21, 22, Y).
- ❌ **Bước 7 (Hungarian Pairing) cứng nhắc:** Do đầu vào sai từ Bước 5, thuật toán ép gượng các NST vào các slot (1-22, X, Y). Đồng thời chưa có logic bắt Dị bội (thừa/thiếu NST) hay đưa ra khay Unclassified.

## 3. Lộ trình sửa chữa (Roadmap)
1. **[ĐANG THỰC HIỆN] Gói Source lên Colab để Train Bước 5:**
   - Đã update script `zip_for_colab.py` để loại bỏ các thư mục nặng (`source_data`, `prepared_single_chromosomes`) nhằm giảm nhẹ file zip.
   - Thư mục dữ liệu train `data_storage/dataset/karyotype` được giữ nguyên.
   - *Next action:* Chạy file zip, up lên Colab, chạy `train_karyotype.py` để ra được `best_model_epoch.pth`.
2. **[TO-DO] Bổ sung Trọng số và Cấu hình:**
   - Gắn `best_model_epoch.pth` vào thư mục `ai_training/weights/`. Cập nhật `config/settings.py` nếu cần.
3. **[TO-DO] Fix Thuật toán Bước 7:**
   - Viết lại hàm nhận kết quả phân loại, nếu xác suất (Probability) quá thấp hoặc số lượng NST khác 46 -> Thảy các NST bất định vào list `Unclassified` thay vì cố nhồi nhét.
4. **[TO-DO] Tối ưu Threshold Bước 2:**
   - Điều chỉnh lại diện tích để tránh lọc sót NST (nguyên nhân gây ra vụ 45/46 NST).

## 4. Ràng buộc (Rules to Remember)
- Dữ liệu train gốc KHÔNG BAO GIỜ ĐƯỢC XÓA (đặc biệt thư mục `dataset`).
- Khi sửa code liên quan đến NST, tuyệt đối không sử dụng thuật toán làm bóp méo, co giãn sai tỷ lệ sinh học của NST.
