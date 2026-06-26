# ==========================================
# BƯỚC 5: PHÂN LOẠI NHIỄM SẮC THỂ THEO LOẠI
# CẢI TIẾN: Đồng bộ hóa toàn diện với kiến trúc mô hình mới (SwinKaryotype)
# ==========================================

from typing import List
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as TF

# Thêm path dự án vào sys.path nếu cần thiết để import được models
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.swin_karyotype import SwinKaryotype
from config.settings import KARYOTYPE_CLASSIFIER_WEIGHTS


class PadToSquare:
    """
    Đệm thêm viền đen (pixel=0) vào ảnh để tạo thành hình vuông
    trước khi Resize, giúp NST không bị bóp méo tỷ lệ.
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return TF.pad(img, (p_left, p_top, p_right, p_bottom), self.fill, 'constant')


class ModelPredictor:
    _instance = None

    @classmethod
    def get_instance(cls, model_path, num_classes=24):
        if cls._instance is None:
            cls._instance = cls(model_path, num_classes)
        return cls._instance

    def __init__(self, model_path, num_classes=24):
        if ModelPredictor._instance is not None:
            raise Exception("Singleton class: Use get_instance() instead")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_weights = os.path.exists(model_path)
        
        if self.has_weights:
            # Khởi tạo SwinKaryotype (cấu trúc mới)
            # Dùng dropout=0.0 vì đang trong quá trình Inference
            self.model = SwinKaryotype(num_classes=num_classes, dropout=0.0).to(self.device)
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Đã load model phân loại từ {model_path}")
            except Exception as e:
                print(f"⚠️ Không thể load model: {e}. Chuyển sang Bypass.")
                self.has_weights = False
            
            if self.has_weights:
                self.model.eval()
        else:
            self.model = None
            print(f"⚠️ [BYPASS] Không tìm thấy model tại {model_path}.")
            print(f"   → Sẽ gán NST dựa trên kích thước hình thái (size-based classification).")

        # Đồng bộ hóa hoàn toàn Transform với lúc huấn luyện
        self.test_transforms = transforms.Compose([
            PadToSquare(fill=0),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Nhãn chuẩn y khoa
        self.class_names = [str(i) for i in range(1, 23)] + ["X", "Y"]

    def predict(self, img_segment):
        """Dự đoán lớp NST. Nếu không có model → dùng size-based heuristic."""
        
        if not self.has_weights:
            return self._size_based_classify(img_segment)
        
        # Tiền xử lý: Đảm bảo ảnh ở định dạng RGB (3 kênh) như lúc train
        if isinstance(img_segment, np.ndarray):
            if len(img_segment.shape) == 2:
                # Nếu là ảnh xám 1 kênh → chuyển sang RGB 3 kênh
                img_segment = cv2.cvtColor(img_segment, cv2.COLOR_GRAY2RGB)
            elif len(img_segment.shape) == 3:
                # Nếu là ảnh màu BGR (từ OpenCV) → chuyển sang RGB
                img_segment = cv2.cvtColor(img_segment, cv2.COLOR_BGR2RGB)
            img_segment = Image.fromarray(img_segment)
        elif isinstance(img_segment, Image.Image):
            if img_segment.mode != 'RGB':
                img_segment = img_segment.convert('RGB')
            
        img_tensor = self.test_transforms(img_segment).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy().tolist()
            _, predicted = torch.max(outputs, 1)

        predicted_label = self.class_names[predicted.item()]
        return predicted.item(), predicted_label, probs
    
    def _size_based_classify(self, img_segment):
        """
        Phân loại sơ bộ dựa trên KÍCH THƯỚC hình thái của NST.
        
        Quy tắc sinh học:
        - NST nhóm A (1-3): Rất lớn
        - NST nhóm B (4-5): Lớn  
        - NST nhóm C (6-12, X): Trung bình
        - NST nhóm D (13-15): Trung bình, có satellite
        - NST nhóm E (16-18): Nhỏ vừa
        - NST nhóm F (19-20): Nhỏ
        - NST nhóm G (21-22, Y): Rất nhỏ
        """
        if isinstance(img_segment, np.ndarray):
            if len(img_segment.shape) == 3:
                gray = cv2.cvtColor(img_segment, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_segment
        else:
            gray = np.array(img_segment.convert("L"))
        
        # Tính diện tích pixel tối (NST)
        mask = gray < 200
        area = np.sum(mask)
        
        # Tính chiều cao (bounding box)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            # Ảnh trắng → gán nhóm G
            dummy_probs = [0.0] * 24
            dummy_probs[20] = 1.0  # chr21
            return 20, "21", dummy_probs
        
        height = ys.max() - ys.min()
        width = xs.max() - xs.min()
        
        # Phân loại dựa trên diện tích
        # Bảng nhóm Denver theo kích thước giảm dần
        if area > 5000:
            # Nhóm A: chr 1, 2, 3
            candidates = [0, 1, 2]
        elif area > 3500:
            # Nhóm B: chr 4, 5
            candidates = [3, 4]
        elif area > 2000:
            # Nhóm C: chr 6-12, X
            candidates = [5, 6, 7, 8, 9, 10, 11, 22]
        elif area > 1200:
            # Nhóm D-E: chr 13-18
            candidates = [12, 13, 14, 15, 16, 17]
        elif area > 600:
            # Nhóm F: chr 19, 20
            candidates = [18, 19]
        else:
            # Nhóm G: chr 21, 22, Y
            candidates = [20, 21, 23]
        
        # Chọn ngẫu nhiên trong nhóm phù hợp (vì chưa có model chính xác)
        import random
        chosen = random.choice(candidates)
        
        probs = [0.0] * 24
        for c in candidates:
            probs[c] = 1.0 / len(candidates)
        
        label = self.class_names[chosen]
        return chosen, label, probs


def run_classification_by_type(
    chromosomes: List[dict], 
    model_path: str = str(KARYOTYPE_CLASSIFIER_WEIGHTS), 
    num_classes: int = 24
) -> List[dict]:
    """
    Chạy phân loại NST theo 24 lớp.
    Nếu không có model → bypass an toàn bằng size-based classification.
    """
    print("Chạy dự đoán phân loại mô hình SwinKaryotype...")
    
    predictor = ModelPredictor.get_instance(model_path=model_path, num_classes=num_classes)
    
    for chrom in chromosomes:
        # Ưu tiên ảnh đã duỗi thẳng
        img_to_predict = chrom.get("aligned_image", chrom.get("straight_image", chrom["image"]))
        
        # Dự đoán
        class_id, label, probs = predictor.predict(img_to_predict)
        
        chrom["class_id"] = class_id
        chrom["type_label"] = label
        chrom["label"] = label
        chrom["probabilities"] = probs

    return chromosomes
