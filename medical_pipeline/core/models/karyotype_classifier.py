# ============================================================
# FILE: core/models/karyotype_classifier.py
# CHỨC NĂNG: Mạng Swin-T phân loại 24 lớp NST — PyTorch
# MỚI HOÀN TOÀN: Backbone Swin Transformer cho Y tế
# GHI CHÚ: Dùng Singleton pattern + hỗ trợ fine-tune từ pretrained
# ============================================================

"""
Module Phân Loại 24 Lớp NST (Karyotype Classifier).

Mạng phân loại ảnh NST đơn lẻ thành 1 trong 24 lớp:
- 22 NST thường (autosome): 1, 2, 3, ..., 22
- 2 NST giới tính (sex chromosome): X, Y

Kiến trúc:
- Backbone: Swin Transformer Tiny (Swin-T) — pretrained ImageNet
- Head: Global Average Pooling → Dropout → FC(768 → 24)
- Input: Ảnh xám (1 kênh) → replicate thành 3 kênh cho Swin-T

Lý do chọn Swin-T:
1. State-of-the-art cho Medical Image Classification
2. Attention mechanism giúp tập trung vào dải băng G (G-bands)
3. Hierarchical feature → nhận biết cấu trúc đa tầng của NST
4. Có pretrained weights → fine-tune nhanh hơn train from scratch

LƯU Ý QUAN TRỌNG:
- Cần thư viện `timm` (PyTorch Image Models) để tải Swin-T pretrained.
- Nếu không có `timm`, sẽ dùng kiến trúc thay thế đơn giản hơn.
- Khi chưa có dữ liệu 24 lớp, module sẽ hoạt động ở chế độ "chưa train".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from config.settings import (
    KARYOTYPE_CLASSIFIER_WEIGHTS,
    NUM_KARYOTYPE_CLASSES,
    KARYOTYPE_LABELS,
    KARYOTYPE_IMG_SIZE,
)


# =========================================================
# THỬ IMPORT TORCHVISION (THƯ VIỆN MÔ HÌNH PRETRAINED)
# =========================================================
try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("⚠️  [KaryotypeClassifier] Thư viện `torchvision` chưa được cài đặt. "
          "Sử dụng backbone thay thế (ResNet đơn giản). "
          "Để có hiệu suất tốt nhất, hãy chạy: uv add torchvision")


# =========================================================
# BACKBONE THAY THẾ (DÙNG KHI KHÔNG CÓ TIMM)
# =========================================================

class SimpleResNetBackbone(nn.Module):
    """
    Backbone ResNet đơn giản dùng khi không có thư viện timm.

    Kiến trúc nhỏ gọn nhưng đủ mạnh cho bài toán 24 lớp NST.
    Output: Feature vector 512 chiều.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 → 56
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Block 2: 56 → 28
            self._make_block(64, 128, stride=2),

            # Block 3: 28 → 14
            self._make_block(128, 256, stride=2),

            # Block 4: 14 → 7
            self._make_block(256, 512, stride=2),
        )

        self.num_features = 512

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# =========================================================
# MẠNG CHÍNH: SWIN-T KARYOTYPE CLASSIFIER
# =========================================================

class SwinTClassifier(nn.Module):
    """
    Bộ phân loại 24 lớp NST dùng Swin Transformer Tiny từ torchvision.

    Cách hoạt động:
    1. Ảnh xám 1 kênh → replicate thành 3 kênh
    2. Swin-T phân loại → 24 logits

    Tham số:
        num_classes: Số lớp phân loại (mặc định 24).
        pretrained: Có dùng pretrained ImageNet không (mặc định True).
        dropout: Tỷ lệ dropout (chỉ dùng cho fallback).
    """

    def __init__(
        self,
        num_classes: int = NUM_KARYOTYPE_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if HAS_TORCHVISION:
            self._use_torchvision = True
            if pretrained:
                self.model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
                if num_classes != 1000:
                    self.model.head = nn.Linear(self.model.head.in_features, num_classes)
            else:
                self.model = models.swin_t(weights=None, num_classes=num_classes)
                
            print(f"✅ [KaryotypeClassifier] Sử dụng Swin-T từ torchvision (num_classes={num_classes})")
        else:
            self._use_torchvision = False
            # Fallback: ResNet đơn giản
            self.model = SimpleResNetBackbone(in_channels=1)
            
            # Classification head
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(self.model.num_features, num_classes),
            )
            print(f"⚠️  [KaryotypeClassifier] Sử dụng ResNet thay thế (dim={self.model.num_features})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tham số:
            x: Tensor (B, 1, 224, 224) — ảnh xám NST.

        Trả về:
            Logits (B, 24) — điểm cho mỗi lớp.
        """
        if self._use_torchvision:
            # Replicate 1 kênh → 3 kênh cho Swin-T
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.model(x)
        else:
            features = self.model(x)
            return self.head(features)


# =========================================================
# SINGLETON WRAPPER
# =========================================================

class KaryotypePredictor:
    """
    Bộ phân loại 24 lớp NST (Singleton pattern).

    Cách sử dụng:
        predictor = KaryotypePredictor.get_instance()
        class_id, label, probs = predictor.predict(chromosome_image)

    Trạng thái:
    - Nếu có weights → phân loại chính xác
    - Nếu chưa có weights → trả về "Chưa xác định" + xác suất đều

    LƯU Ý:
    - Cần dữ liệu 24 lớp để train (xem config/settings.py → KARYOTYPE_DATASET_DIR)
    - Weights mặc định: config/settings.py → KARYOTYPE_CLASSIFIER_WEIGHTS
    """

    _instance: Optional["KaryotypePredictor"] = None

    def __init__(self, weights_path: Optional[Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinTClassifier(pretrained=False)  # Không tải pretrained khi inference

        wp = weights_path or KARYOTYPE_CLASSIFIER_WEIGHTS
        if wp.exists():
            state = torch.load(str(wp), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self._has_weights = True
            print(f"✅ [KaryotypeClassifier] Đã tải weights từ: {wp}")
        else:
            self._has_weights = False
            print(f"⚠️  [KaryotypeClassifier] Chưa có weights tại {wp}. "
                  f"Module hoạt động ở chế độ 'chưa train'.")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, weights_path: Optional[Path] = None) -> "KaryotypePredictor":
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[int, str, np.ndarray]:
        """
        Phân loại 1 ảnh NST đơn thành 1 trong 24 lớp.

        Tham số:
            image: Ảnh xám numpy (H, W), uint8 [0,255] hoặc float [0,1].

        Trả về:
            Tuple (class_id, label, probabilities):
            - class_id: int (0-23)
            - label: str ("1", "2", ..., "22", "X", "Y")
            - probabilities: np.ndarray (24,) — xác suất mỗi lớp
        """
        if not self._has_weights:
            # Chưa có weights → trả về xác suất đều
            probs = np.ones(NUM_KARYOTYPE_CLASSES) / NUM_KARYOTYPE_CLASSES
            return 0, "Chưa xác định", probs

        # Chuẩn bị input
        import cv2
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)

        img = cv2.resize(img, (KARYOTYPE_IMG_SIZE, KARYOTYPE_IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        class_id = int(probs.argmax())

        return class_id, KARYOTYPE_LABELS[class_id], probs

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """
        Phân loại hàng loạt ảnh NST.

        Tham số:
            images: Danh sách ảnh numpy (H, W), uint8.

        Trả về:
            Danh sách Tuple (class_id, label, probabilities) cho mỗi ảnh.
        """
        return [self.predict(img) for img in images]
