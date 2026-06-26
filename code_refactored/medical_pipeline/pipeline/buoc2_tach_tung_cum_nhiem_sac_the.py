# ==========================================
# BƯỚC 2: TÁCH TỪNG CỤM NHIỄM SẮC THỂ
# CHỨC NĂNG: Segmentation - Tách cụm NST từ ảnh nguyên vẹn
# ==========================================

import cv2
import numpy as np

class ChromosomeClusterSegmenter:
    """
    Nhận diện và cắt cụm NST cực khít (Tight Crop).
    Loại bỏ hoàn toàn mảng trắng rác.
    """

    def __init__(self, image, min_area=80):
        self.image = image
        self.min_area = min_area

    def segment_clusters(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # === CẢI TIẾN BƯỚC 2: PIXEL DENSITY HEATMAP TỪ NOTEBOOK ===
        # 1. Khử nhiễu mạnh nhưng giữ viền (fastNlMeansDenoising)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Đảo ngược ảnh (Chromosomes tối -> Sáng)
        inverted = cv2.bitwise_not(denoised)
        
        # 3. Làm mờ nhẹ
        blur = cv2.GaussianBlur(inverted.astype(np.float32), (5,5), 0)
        
        # 4. Normalize và phủ Jet colormap
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        
        # 5. Tìm vùng đỏ (Pixel density cao = Chromosome)
        lower_red = np.array([0, 0, 100])
        upper_red = np.array([50, 50, 255])
        red_mask = cv2.inRange(heat, lower_red, upper_red)
        
        # 6. Gộp thêm với Otsu adaptive để không sót nét mờ
        _, otsu_mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        binary_mask = cv2.bitwise_or(red_mask, otsu_mask)
        # ==========================================================

        # 2. Xóa nhiễu nhỏ & Khôi phục nét đứt
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 3. Tìm contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_items = []
        visualization_img = self.image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            # Lọc hình thái học
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Loại bỏ các hạt nhân tế bào hình tròn/bầu dục đặc
            is_nucleus = (solidity > 0.95 and 0.5 < aspect_ratio < 2.0)
            # Tắt bỏ lọc cụm khổng lồ để AI cắt thử
            is_giant_blob = False
            # Loại bỏ dãy số, text hoặc vệt bẩn quá dài/quá dẹt
            is_watermark_text = (solidity > 0.8 and (aspect_ratio > 5.0 or aspect_ratio < 0.2))
            
            if is_nucleus or is_watermark_text:
                cv2.rectangle(visualization_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                continue

            # Cắt ảnh cực sát rạt (TIGHT CROP), bỏ nền
            mask_contour = np.zeros_like(gray)
            cv2.drawContours(mask_contour, [contour], -1, 255, thickness=cv2.FILLED)
            
            masked_part = cv2.bitwise_and(self.image, self.image, mask=mask_contour)
            masked_part[mask_contour == 0] = 255 
            
            cropped_part = masked_part[y:y+h, x:x+w]
            
            if cropped_part.size == 0:
                continue
            
            valid_items.append({
                "contour": contour,
                "bbox": (x, y, w, h),
                "cropped_part": cropped_part
            })

        # Sắp xếp các cụm nhận diện theo thứ tự từ trên xuống dưới, trái qua phải
        # Nhóm các cụm trên cùng một hàng (dung sai 50 pixels) để sort chuẩn hơn
        valid_items.sort(key=lambda item: (item["bbox"][1] // 50, item["bbox"][0]))

        cropped_images = []
        bboxes = []

        for idx, item in enumerate(valid_items):
            x, y, w, h = item["bbox"]
            contour = item["contour"]
            cropped_part = item["cropped_part"]

            # Vẽ xanh cho cụm NST hợp lệ
            cv2.rectangle(visualization_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Đánh số lên cụm NST nhận diện được một cách thẩm mỹ
            text = f"#{idx}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Vẽ nền chữ nhật mờ cho text để dễ nhìn và không thô
            cv2.rectangle(visualization_img, (x, max(0, y - text_h - 10)), (x + text_w + 10, max(0, y - text_h - 10) + text_h + 10), (0, 0, 0), cv2.FILLED)
            cv2.putText(visualization_img, text, (x + 5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # TUYỆT ĐỐI KHÔNG PADDING TRẮNG VÀO ĐÂY NỮA
            # Ghi chú: Ảnh crop truyền đi tiếp theo KHÔNG hề có đánh số, số chỉ vẽ trên ảnh visualization.
            cropped_images.append(cropped_part)
            bboxes.append((x, y, w, h))

        return cropped_images, visualization_img, binary_mask, bboxes


def stack_images(images, cols=5, bg_color=(255, 255, 255)):
    """Ghép ảnh sát rạt lại với nhau để trực quan"""
    if not images:
        return None
    
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    rows = (len(images) + cols - 1) // cols
    
    if channels > 1:
        stacked = np.ones((max_h * rows, max_w * cols, channels), dtype=np.uint8) * 255
    else:
        stacked = np.ones((max_h * rows, max_w * cols), dtype=np.uint8) * 255
        
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = row * max_h
        x = col * max_w
        
        h, w = img.shape[:2]
        # Canh giữa ảnh trong grid cell
        y_off = y + (max_h - h) // 2
        x_off = x + (max_w - w) // 2
        
        stacked[y_off:y_off+h, x_off:x_off+w] = img
        cv2.rectangle(stacked, (x_off, y_off), (x_off+w, y_off+h), (200, 200, 200), 1)
        
        # Đánh số đẹp lên góc trái trên của từng cell
        text = f"#{idx}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(stacked, (x, y), (x + text_w + 10, y + text_h + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(stacked, text, (x + 5, y + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return stacked


def run_segmentation(img_bgr):
    segmenter = ChromosomeClusterSegmenter(image=img_bgr, min_area=50) # Hạ min_area để không sót NST nhỏ
    cropped_images, visualization_img, binary_mask, bboxes = segmenter.segment_clusters()
    return cropped_images, visualization_img, binary_mask, bboxes
