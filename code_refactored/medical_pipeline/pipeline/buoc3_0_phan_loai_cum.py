# ==========================================
# BƯỚC 3.0: PHÂN LOẠI CỤM NHIỄM SẮC THỂ
# CHỨC NĂNG: Phân loại cụm thành 0 (Đơn), 1 (Chạm), 2 (Chồng), 3 (Chồng Chạm)
# ==========================================

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import warnings
from pathlib import Path

from medical_pipeline.core.models.dual_branch import DualBranchModel


class ClusterClassifier:
    _instance = None

    @classmethod
    def get_instance(cls, model_path: Path, device: str = 'cpu') -> "ClusterClassifier":
        if cls._instance is None:
            cls._instance = cls(model_path, device)
        return cls._instance

    def __init__(self, model_path: Path, device: str = 'cpu'):
        if ClusterClassifier._instance is not None:
            raise Exception("Singleton class: Use get_instance() instead")
        self.device = torch.device(device)
        self.num_classes = 4 # 0: Đơn, 1: Chạm, 2: Chồng, 3: Chồng Chạm
        
        self.has_weights = model_path.exists()
        if not self.has_weights:
            warnings.warn(f"⚠️ [FALLBACK] Không tìm thấy {model_path}. Hệ thống sẽ giả định tất cả là NST Đơn (Class 0).")
            self.model = None
        else:
            self.model = DualBranchModel(num_classes=self.num_classes).to(self.device)
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                warnings.warn(f"⚠️ Lỗi load weights DualBranchModel: {e}. Fallback -> Class 0.")
                self.has_weights = False

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, img_bgr) -> int:
        """
        Dự đoán lớp của cụm NST.
        Input: cv2 image (BGR hoặc Grayscale)
        Output: int (0, 1, 2, 3)
        """
        if not self.has_weights:
            # Fallback an toàn: Coi như cụm đơn giản
            return 0
            
        if not isinstance(img_bgr, Image.Image):
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            # Chuyển OpenCV BGR sang RGB cho PIL
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        else:
            img_pil = img_bgr

        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(dim=1).item()
            
        return prediction
