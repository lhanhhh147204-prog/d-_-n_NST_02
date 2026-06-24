# ============================================================
# FILE: core/models/deblurrer.py
# CHỨC NĂNG: Mạng MPRNet để khử mờ (deblur) ảnh NST — PyTorch
# NGUỒN GỐC: Chuyển từ extracted_code.py (class Deblurrer)
# GHI CHÚ: Dùng Singleton pattern — chỉ khởi tạo 1 lần duy nhất
# ============================================================

"""
Module Khử Mờ Ảnh NST (Chromosome Image Deblurring).

Sử dụng kiến trúc MPRNet (Multi-Stage Progressive Restoration Network)
để khử mờ ảnh NST trước khi đưa vào pipeline phân tích.

Thiết kế Singleton:
- Gọi `Deblurrer.get_instance()` thay vì `Deblurrer()` trực tiếp.
- Weights chỉ load 1 lần → tránh tràn RAM khi xử lý hàng loạt.

LƯU Ý:
- Module này CẦN file weights (.pth) để hoạt động.
- Nếu không có weights, module sẽ báo lỗi rõ ràng (không crash im lặng).
- Đường dẫn weights mặc định: config/settings.py → DEBLURRER_WEIGHTS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import DEBLURRER_WEIGHTS


# =========================================================
# KIẾN TRÚC MẠNG MPRNet (ĐƠN GIẢN HÓA CHO BÀI TOÁN NST)
# =========================================================

class CALayer(nn.Module):
    """Channel Attention Layer — tập trung vào kênh thông tin quan trọng."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class CAB(nn.Module):
    """Channel Attention Block — Conv + BN + ReLU + Channel Attention."""

    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca = CALayer(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.body(x))


class DownSample(nn.Module):
    """Giảm độ phân giải x2 bằng Pixel Unshuffle."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpSample(nn.Module):
    """Tăng độ phân giải x2 bằng Pixel Shuffle."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UNetEncoder(nn.Module):
    """Encoder đơn giản kiểu U-Net cho MPRNet stage."""

    def __init__(self, channels: int = 64, num_cab: int = 4):
        super().__init__()
        self.encoder1 = nn.Sequential(*[CAB(channels) for _ in range(num_cab)])
        self.down1 = DownSample(channels)

        self.encoder2 = nn.Sequential(*[CAB(channels * 2) for _ in range(num_cab)])
        self.down2 = DownSample(channels * 2)

        self.bottleneck = nn.Sequential(*[CAB(channels * 4) for _ in range(num_cab)])

        self.up2 = UpSample(channels * 4)
        self.decoder2 = nn.Sequential(*[CAB(channels * 2) for _ in range(num_cab)])

        self.up1 = UpSample(channels * 2)
        self.decoder1 = nn.Sequential(*[CAB(channels) for _ in range(num_cab)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        bot = self.bottleneck(self.down2(enc2))
        dec2 = self.decoder2(self.up2(bot) + enc2)
        dec1 = self.decoder1(self.up1(dec2) + enc1)
        return dec1


class SimpleMPRNet(nn.Module):
    """
    Phiên bản đơn giản hóa của MPRNet cho bài toán khử mờ NST.

    Kiến trúc:
    - 1 stage UNet Encoder-Decoder
    - Channel Attention ở mỗi block
    - Input: ảnh xám 1 kênh → Output: ảnh xám đã khử mờ
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            CAB(base_channels),
        )
        self.stage = UNetEncoder(base_channels, num_cab=4)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shallow_feat(x)
        out = self.stage(feat)
        return self.output_conv(out) + x  # Residual learning


# =========================================================
# SINGLETON WRAPPER — TRÁNH LOAD LẠI WEIGHTS MỖI LẦN GỌI
# =========================================================

class Deblurrer:
    """
    Bộ khử mờ ảnh NST dùng MPRNet (Singleton pattern).

    Cách sử dụng:
        deblurrer = Deblurrer.get_instance()
        sharp_image = deblurrer.process(blurry_image_numpy)

    LƯU Ý:
    - Gọi get_instance() thay vì Deblurrer() để tái sử dụng.
    - Nếu không có file weights, module sẽ chạy ở chế độ passthrough
      (trả về ảnh gốc không thay đổi) và in cảnh báo.
    """

    _instance: Optional["Deblurrer"] = None

    def __init__(self, weights_path: Optional[Path] = None):
        """Khởi tạo — KHÔNG gọi trực tiếp, dùng get_instance()."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleMPRNet(in_channels=1, out_channels=1)

        wp = weights_path or DEBLURRER_WEIGHTS
        if wp.exists():
            state = torch.load(str(wp), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self._has_weights = True
            print(f"✅ [Deblurrer] Đã tải weights từ: {wp}")
        else:
            self._has_weights = False
            print(f"⚠️  [Deblurrer] Không tìm thấy weights tại {wp}. "
                  f"Chạy ở chế độ passthrough (ảnh không thay đổi).")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, weights_path: Optional[Path] = None) -> "Deblurrer":
        """Trả về instance duy nhất (Singleton). Tạo mới nếu chưa có."""
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Xóa instance hiện tại (dùng khi cần load lại weights khác)."""
        cls._instance = None

    @property
    def is_ready(self) -> bool:
        """Kiểm tra xem đã có weights chưa."""
        return self._has_weights

    @torch.no_grad()
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Khử mờ một ảnh NST.

        Tham số:
            image: Ảnh xám numpy (H, W), giá trị [0, 255] uint8
                   hoặc [0.0, 1.0] float.

        Trả về:
            Ảnh đã khử mờ, cùng dtype và range với input.
        """
        if not self._has_weights:
            return image  # Passthrough — trả về ảnh gốc

        # Chuyển đổi input
        is_uint8 = image.dtype == np.uint8
        if is_uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        # Đưa vào tensor (1, 1, H, W)
        tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(self.device)

        # Pad nếu kích thước không chia hết cho 8 (yêu cầu của PixelShuffle)
        _, _, h, w = tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # Chạy model
        output = self.model(tensor)

        # Cắt bỏ phần padding
        output = output[:, :, :h, :w]

        # Chuyển về numpy
        result = output.squeeze().cpu().numpy()
        result = np.clip(result, 0.0, 1.0)

        if is_uint8:
            return (result * 255.0).astype(np.uint8)
        return result
