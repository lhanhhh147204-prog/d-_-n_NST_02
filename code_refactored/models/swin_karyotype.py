import torch
import torch.nn as nn
from torchvision import models

class SwinKaryotype(nn.Module):
    """
    Mô hình chuẩn Swin Transformer (Swin-T) dùng cho dự án NTS.
    Nhận đầu vào RGB (3 kênh) và phân loại ra 24 lớp NST.
    Sử dụng Transfer Learning từ ImageNet để hội tụ nhanh hơn.
    """
    def __init__(self, num_classes=24, dropout=0.3):
        super(SwinKaryotype, self).__init__()
        # Khởi tạo mô hình Swin-T với Transfer Learning (weights từ ImageNet)
        self.model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        
        # Thay đổi lớp cuối (head) để tương ứng với 24 class (1-22, X, Y)
        # Thêm Dropout trước Linear để chống overfitting trên dataset nhỏ (5474 ảnh)
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def freeze_backbone(self):
        """
        Đóng băng toàn bộ backbone, chỉ cho phép train lớp head.
        Dùng trong Phase 1 của huấn luyện 2 giai đoạn.
        """
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("✅ Backbone đã được đóng băng. Chỉ train lớp Head.")
    
    def unfreeze_backbone(self):
        """
        Mở băng toàn bộ mô hình để fine-tune tất cả các layer.
        Dùng trong Phase 2 của huấn luyện 2 giai đoạn.
        """
        for param in self.model.parameters():
            param.requires_grad = True
        print("✅ Toàn bộ mô hình đã được mở băng để fine-tune.")
        
    def forward(self, x):
        return self.model(x)
