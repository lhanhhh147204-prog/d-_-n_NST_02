# 🧬 Trực Quan Hóa Luồng Xử Lý NTS (Medical Karyotyping Pipeline)

Sơ đồ dưới đây mô tả chi tiết toàn bộ quy trình 5 bước (End-to-End Pipeline) của hệ thống phân tích hình thái Nhiễm Sắc Thể, từ lúc nạp ảnh cho đến khi xuất báo cáo Karyogram chuẩn y khoa. Sơ đồ này được thiết kế để giúp các nhà phát triển dễ dàng hình dung cấu trúc bên trong dự án.

```mermaid
graph TD
    %% Định dạng phong cách (Styles)
    classDef inputOutput fill:#2d3436,stroke:#0984e3,stroke-width:3px,color:#fff,font-weight:bold,border-radius:10px;
    classDef subTask fill:#f8f9fa,stroke:#b2bec3,stroke-width:1px,color:#2d3436;
    classDef aiModel fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff,font-style:italic,border-radius:5px;
    classDef condition fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    classDef subGraphStyle fill:#ecf0f1,stroke:#bdc3c7,stroke-width:2px;

    %% Nút bắt đầu
    Start([📸 Ảnh Gốc Từ Kính Hiển Vi]) ::: inputOutput

    %% BƯỚC 1: TIỀN XỬ LÝ & LÀM MỜ
    subgraph Bước 1: Tiền Xử Lý & Khử Mờ
        B1_1[Khử Mờ Bằng AI] ::: subTask
        B1_Model[Mô hình: MPRNet] ::: aiModel
        B1_2[Đánh Giá Chất Lượng: SSIM / Delta E] ::: subTask
        
        B1_1 -.-> B1_Model
        B1_Model -.-> B1_1
        B1_1 --> B1_2
    end

    %% BƯỚC 2: BÓC TÁCH & PHÂN CỤM
    subgraph Bước 2: Bóc Tách & Phân Cụm (Segmentation)
        B2_1[Nhận Diện & Phân Loại Cụm] ::: subTask
        B2_Model[AI: CCINet & DualBranchModel] ::: aiModel
        B2_2[Dò Tìm Heatmap Mật Độ Pixel] ::: subTask
        B2_3[Cắt Bounding Box & Padding Trắng] ::: subTask

        B2_1 -.-> B2_Model
        B2_Model -.-> B2_1
        B2_1 --> B2_2 --> B2_3
    end

    %% PHÂN LOẠI TRẠNG THÁI
    CheckState{Trạng Thái Của Cụm NST?} ::: condition

    %% BƯỚC 3: XỬ LÝ CHỒNG CHẠM
    subgraph Bước 3: Tách Chồng/Chạm (Core Resolution)
        B3_1[Phân Tích Hình Học: Tìm Điểm Lõm] ::: subTask
        B3_2[Tính Toán Đường Cắt Tối Ưu] ::: subTask
        B3_3[Distance Transform & Gradient Mask] ::: subTask
        B3_4[Alpha Blending & Inpaint (Không Cắt Cứng)] ::: subTask

        B3_1 --> B3_2 --> B3_3 --> B3_4
    end

    %% BƯỚC 4: DUỖI THẲNG
    subgraph Bước 4: Duỗi Thẳng (Unbending)
        B4_1[Skeletonization: Trích Xuất Trục Xương] ::: subTask
        B4_2[Tính Toán Góc Độ Cong] ::: subTask
        B4_3[Xoay & Kéo Thẳng Trữ Nguyên Kích Thước] ::: subTask

        B4_1 --> B4_2 --> B4_3
    end

    %% BƯỚC 5: PHÂN LOẠI ĐỊNH DANH
    subgraph Bước 5: Phân Loại (Karyotyping)
        B5_1[Định Danh 24 Lớp Lõi] ::: subTask
        B5_Model[AI: Swin Transformer Karyotype] ::: aiModel
        B5_2[Xác Định Giới Tính XX/XY] ::: subTask
        B5_3[Thuật Toán Hungarian Ghép 23 Cặp] ::: subTask

        B5_1 -.-> B5_Model
        B5_Model -.-> B5_1
        B5_1 --> B5_2 --> B5_3
    end

    %% Nút kết thúc
    End([📊 Xuất Báo Cáo Bản Đồ Karyogram]) ::: inputOutput

    %% KẾT NỐI LUỒNG CHÍNH
    Start ==>|Nạp Ảnh Raw| B1_1
    B1_2 ==>|Truyền Dữ Liệu Sạch| B2_1
    B2_3 ==> CheckState
    
    CheckState == Dính/Chạm (Overlapping) ==> B3_1
    CheckState == Uốn Cong (Bent) ==> B4_1
    CheckState == Tách Rời Tuyệt Đối ==> B5_1

    B3_4 ==>|Hoàn Tất Tách| B5_1
    B4_3 ==>|Hoàn Tất Duỗi| B5_1
    
    B5_3 ==>|Sắp Xếp Hoàn Chỉnh| End
```
