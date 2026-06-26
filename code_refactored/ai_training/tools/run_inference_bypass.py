import os
import sys
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Để in tiếng Việt không lỗi trên cmd windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm đường dẫn thư mục cha để import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swin_karyotype import SwinKaryotype
from config import settings

VOC_DIR = r"c:\Users\lehoa\dự_án_NTS\data\24_chromosomes_object\24_chromosomes_object"
IMG_DIR = os.path.join(VOC_DIR, "JEPG")
ANN_DIR = os.path.join(VOC_DIR, "annotations")
OUTPUT_DIR = os.path.join(settings.PROJECT_ROOT, "results", "bypass_predictions")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class BypassPredictor:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading mô hình Swin-T lên thiết bị: {self.device}...")
        
        # Khởi tạo mô hình và load weights
        self.model = SwinKaryotype(num_classes=24).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Transform chuẩn của Swin-T (ImageNet RGB)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Nhãn lớp 1->22, X, Y
        self.classes = [str(i) for i in range(1, 23)] + ["X", "Y"]
        
    def predict(self, pil_img):
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        return self.classes[predicted.item()]

def clean_crop(img_crop_bgr):
    """
    Hàm dọn "sạn" bằng OpenCV:
    Chuyển xám -> Threshold -> Tìm contour lớn nhất (NST chính) -> Xóa nền
    """
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Otsu threshold (ngược lại vì nền trắng, NST đen)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm các đường viền
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_crop_bgr # Trả về ảnh gốc nếu không tìm thấy gì
        
    # Lấy đường viền có diện tích lớn nhất (chính là NST mục tiêu ở giữa)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Tạo mask trắng đen
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    
    # Áp mask: Xóa trắng những chỗ nằm ngoài contour chính
    result = img_crop_bgr.copy()
    result[mask == 0] = [255, 255, 255]
    
    return result

def run_inference_on_file(xml_filename, predictor):
    xml_path = os.path.join(ANN_DIR, xml_filename)
    if not os.path.exists(xml_path):
        print(f"Không tìm thấy file XML: {xml_path}")
        return
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    img_basename = root.find("filename").text
    img_path = os.path.join(IMG_DIR, img_basename)
    
    if not os.path.exists(img_path):
        name_part, ext_part = os.path.splitext(img_basename)
        img_path = os.path.join(IMG_DIR, name_part + ext_part.upper())
        if not os.path.exists(img_path):
            print(f"Lỗi: Không tìm thấy ảnh {img_basename}")
            return
            
    print(f"\nĐang xử lý ảnh: {img_basename}")
    
    # Đọc ảnh bằng OpenCV hỗ trợ đường dẫn tiếng Việt
    with open(img_path, "rb") as f:
        img_bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    img_canvas = img_bgr.copy()
    
    total_objects = 0
    correct_count = 0
    
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        raw_label = obj.find("name").text
        
        # Parse ground truth label (to verify accuracy optionally)
        gt_label = raw_label.strip().upper()
        import re
        if gt_label not in ["X", "Y"]:
            num_match = re.search(r'\d+', gt_label)
            gt_label = num_match.group() if num_match else "Unknown"
        
        xmin = max(0, int(float(bndbox.find("xmin").text)))
        ymin = max(0, int(float(bndbox.find("ymin").text)))
        xmax = min(img_bgr.shape[1], int(float(bndbox.find("xmax").text)))
        ymax = min(img_bgr.shape[0], int(float(bndbox.find("ymax").text)))
        
        if xmax <= xmin or ymax <= ymin:
            continue
            
        # 1. Cắt ảnh NST
        crop_bgr = img_bgr[ymin:ymax, xmin:xmax]
        
        # 2. Xóa sạn OpenCV
        clean_bgr = clean_crop(crop_bgr)
        
        # 3. Phân loại bằng AI Swin-T
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(clean_rgb)
        
        pred_label = predictor.predict(pil_img)
        
        # So sánh với nhãn gốc (GT)
        is_correct = (pred_label == gt_label)
        if is_correct: correct_count += 1
        total_objects += 1
        
        # 4. Vẽ Bounding Box và Nhãn Dự Đoán lên ảnh gốc
        color = (0, 255, 0) if is_correct else (0, 0, 255) # Xanh = Đúng, Đỏ = Sai
        cv2.rectangle(img_canvas, (xmin, ymin), (xmax, ymax), color, 2)
        
        text = f"Pred:{pred_label} (GT:{gt_label})"
        cv2.putText(img_canvas, text, (xmin, max(0, ymin - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tính độ chính xác của AI trên bức ảnh này
    acc = (correct_count / total_objects * 100) if total_objects > 0 else 0
    print(f"Tổng NST: {total_objects} | Đoán đúng: {correct_count} | Độ chính xác: {acc:.2f}%")
    
    # Lưu kết quả có hỗ trợ đường dẫn tiếng Việt
    save_path = os.path.join(OUTPUT_DIR, f"predicted_{img_basename}")
    cv2.imencode('.jpg', img_canvas)[1].tofile(save_path)
    print(f"-> Đã lưu ảnh kết quả tại: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Bypass Karyotype")
    parser.add_argument("--img", type=str, default="103064.xml", help="Tên file XML (ví dụ: 103064.xml)")
    args = parser.parse_args()
    
    # Load predictor 1 lần
    weights_path = settings.KARYOTYPE_CLASSIFIER_WEIGHTS
    if not os.path.exists(weights_path):
        print(f"LỖI: Không tìm thấy file trọng số {weights_path}")
        sys.exit(1)
        
    predictor = BypassPredictor(weights_path)
    
    # Chạy inference
    run_inference_on_file(args.img, predictor)
