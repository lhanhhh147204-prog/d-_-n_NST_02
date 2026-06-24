import cv2
import numpy as np

class PixelDensityHeatmap:
    """
    Trích xuất các cụm NST từ ảnh tế bào nguyên vẹn dựa trên mật độ pixel.
    (Chuyển từ PixelDensityHeatmap trong extracted_code.py)
    """
    def __init__(self, image, min_area=500):
        self.image = image
        self.min_area = min_area

    def plot_pixel_density_heatmap(self):
        """
        Tách tất cả các vùng nằm trong contour và chèn vào nền trắng kích thước 224x224.
        
        Trả về:
            Danh sách các ảnh (numpy array) chứa các cụm/NST đơn đã được cắt.
        """
        # Chuyển ảnh sang ảnh xám
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Tạo bản đồ mật độ sử dụng GaussianBlur
        density_map = cv2.GaussianBlur(gray_img.astype(np.float32), (5, 5), 0)
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Áp dụng màu sắc
        heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

        # Lọc các vùng có màu đỏ
        red_threshold = 170
        lower_red = np.array([0, 0, red_threshold])
        upper_red = np.array([10, 10, 255])
        red_mask = cv2.inRange(heatmap, lower_red, upper_red)

        # Tìm contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cropped_images = []
        bboxes = []

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                mask = np.zeros_like(gray_img)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

                masked_part = cv2.bitwise_and(self.image, self.image, mask=mask)
                masked_part[mask == 0] = 255

                x, y, w, h = cv2.boundingRect(contour)
                cropped_part = masked_part[y:y+h, x:x+w]
                bboxes.append((x, y, w, h))

                # Resize nếu cần
                if cropped_part.shape[0] > 224 or cropped_part.shape[1] > 224:
                    aspect_ratio = cropped_part.shape[1] / cropped_part.shape[0]
                    if aspect_ratio > 1:
                        new_w = 224
                        new_h = int(224 / aspect_ratio)
                    else:
                        new_h = 224
                        new_w = int(224 * aspect_ratio)

                    if new_w > 224 or new_h > 224:
                        scale_factor = min(224 / new_w, 224 / new_h)
                        new_w = int(new_w * scale_factor)
                        new_h = int(new_h * scale_factor)

                    resized_part = cv2.resize(cropped_part, (new_w, new_h))
                else:
                    resized_part = cropped_part

                white_background = np.ones((224, 224, 3), dtype=np.uint8) * 255
                y_offset = (224 - resized_part.shape[0]) // 2
                x_offset = (224 - resized_part.shape[1]) // 2
                white_background[y_offset:y_offset+resized_part.shape[0], x_offset:x_offset+resized_part.shape[1]] = resized_part

                cropped_images.append(white_background)

        return cropped_images, heatmap, red_mask, bboxes
