# ============================================================
# FILE: core/models/ccinet.py
# CHỨC NĂNG: Mạng CCINet phân loại 4 lớp trạng thái NST — PyTorch
# NGUỒN GỐC: Chuyển từ extracted_code.py (CCINet, SEResNet61Backbone)
# GHI CHÚ: Phân loại: Tách rời | Trùng lấn | Uốn cong | Bình thường
# ============================================================

"""
Module Phân Loại Trạng Thái Hình Thái NST (Chromosome Classification Network).

CCINet = Chromosome Classification and Identification Network
- Backbone: SEResNet61 (ResNet + Squeeze-and-Excitation blocks)
- Feature Fusion: MFF Module (Multi-scale Feature Fusion)
- Output: 4 lớp — Separated, Overlapping, Bent, Normal

Lớp đầu ra:
    0: Tách rời (Separated)
    1: Trùng lấn (Overlapping)
    2: Uốn cong (Bent/Curved)
    3: Bình thường (Normal/Single)

LƯU Ý:
- Input ảnh 1 kênh (grayscale) kích thước 224x224.
- Dùng Singleton pattern giống Deblurrer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import CCINET_WEIGHTS


# =========================================================
# CÁC KHỐI XÂY DỰNG (BUILDING BLOCKS)
# =========================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.

    Học trọng số cho từng kênh (channel attention):
    - Squeeze: Global Average Pooling → vector 1D
    - Excitation: 2 lớp FC → tạo trọng số cho mỗi kênh
    - Scale: Nhân trọng số vào feature map gốc
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResidualUnit(nn.Module):
    """
    Residual Unit + SE Block.

    Conv → BN → ReLU → Conv → BN → SE → + Skip Connection → ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        # Skip connection (downsample nếu kích thước thay đổi)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity)


class MFFModule(nn.Module):
    """
    Multi-scale Feature Fusion Module.

    Kết hợp feature maps từ nhiều tầng (multi-scale) bằng cách:
    1. Resize tất cả về cùng kích thước
    2. Concatenate
    3. Conv 1x1 để giảm số kênh
    """

    def __init__(self, channel_list: list, out_channels: int = 128):
        super().__init__()
        total_channels = sum(channel_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list, target_size: tuple) -> torch.Tensor:
        """
        Tham số:
            features: Danh sách feature maps từ các tầng khác nhau.
            target_size: Kích thước (H, W) để resize về.
        """
        resized = []
        for f in features:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            resized.append(f)
        return self.fuse(torch.cat(resized, dim=1))


# =========================================================
# MẠNG CHÍNH: CCINet
# =========================================================

class CCINet(nn.Module):
    """
    Chromosome Classification and Identification Network.

    Kiến trúc:
    - Stem: Conv 7x7 → BN → ReLU → MaxPool (giảm từ 224 → 56)
    - 4 stage SEResidualUnit (giảm dần: 56→28→14→7)
    - MFF Module (kết hợp feature đa tầng)
    - Classifier: GAP → FC → 4 lớp

    Input: Ảnh xám (1, 1, 224, 224)
    Output: Logits (1, 4)
    """

    # Nhãn tiếng Việt cho 4 lớp đầu ra
    CLASS_NAMES = ["Tách rời", "Trùng lấn", "Uốn cong", "Bình thường"]

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()

        # Stem: đưa ảnh từ 224x224 xuống 56x56
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # 4 stage với SEResidualUnit
        self.stage1 = self._make_stage(64, 64, num_blocks=3, stride=1)    # 56x56
        self.stage2 = self._make_stage(64, 128, num_blocks=4, stride=2)   # 28x28
        self.stage3 = self._make_stage(128, 256, num_blocks=6, stride=2)  # 14x14
        self.stage4 = self._make_stage(256, 512, num_blocks=3, stride=2)  # 7x7

        # MFF: kết hợp feature từ stage 2, 3, 4
        self.mff = MFFModule([128, 256, 512], out_channels=128)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Tạo một stage gồm nhiều SEResidualUnit."""
        layers = [SEResidualUnit(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(SEResidualUnit(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        # MFF: kết hợp multi-scale features
        target_size = f2.shape[2:]  # 28x28
        fused = self.mff([f2, f3, f4], target_size)

        return self.classifier(fused)


# =========================================================
# SINGLETON WRAPPER
# =========================================================

class CCINetPredictor:
    """
    Bộ phân loại trạng thái NST dùng CCINet (Singleton pattern).

    Cách sử dụng:
        predictor = CCINetPredictor.get_instance()
        class_id, class_name, probs = predictor.predict(image_numpy)

    LƯU Ý:
    - Nếu không có weights, trả về class_id=3 (Bình thường) mặc định.
    """

    _instance: Optional["CCINetPredictor"] = None

    def __init__(self, weights_path: Optional[Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CCINet(in_channels=1, num_classes=4)

        wp = weights_path or CCINET_WEIGHTS
        if wp.exists():
            state = torch.load(str(wp), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self._has_weights = True
            print(f"✅ [CCINet] Đã tải weights từ: {wp}")
        else:
            self._has_weights = False
            print(f"⚠️  [CCINet] Không tìm thấy weights tại {wp}. "
                  f"Trả về mặc định 'Bình thường' cho mọi ảnh.")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, weights_path: Optional[Path] = None) -> "CCINetPredictor":
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple:
        """
        Phân loại trạng thái hình thái của 1 ảnh NST.

        Tham số:
            image: Ảnh xám numpy (H, W), uint8 [0,255] hoặc float [0,1].

        Trả về:
            Tuple (class_id, class_name, probabilities):
            - class_id: int (0-3)
            - class_name: str ("Tách rời" | "Trùng lấn" | "Uốn cong" | "Bình thường")
            - probabilities: np.ndarray (4,) — xác suất mỗi lớp
        """
        if not self._has_weights:
            probs = np.array([0.0, 0.0, 0.0, 1.0])
            return 3, CCINet.CLASS_NAMES[3], probs

        # Chuẩn bị input
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)

        # Resize về 224x224
        import cv2
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Đưa vào tensor (1, 1, 224, 224)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

        # Chạy model
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        class_id = int(probs.argmax())

        return class_id, CCINet.CLASS_NAMES[class_id], probs
