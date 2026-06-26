import os
import sys
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import re

# Đảm bảo in console tiếng Việt trên Windows không lỗi
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình đường dẫn
VOC_DIR = r"c:\Users\lehoa\dự_án_NTS\data\24_chromosomes_object\24_chromosomes_object"
IMG_DIR = os.path.join(VOC_DIR, "JEPG")
ANN_DIR = os.path.join(VOC_DIR, "annotations")

# Thư mục lưu dữ liệu mới (không ghi đè thư mục cũ)
OUTPUT_DIR = r"c:\Users\lehoa\dự_án_NTS\code_refactored\dataset\karyotype_voc_massive"

def get_standard_label(raw_name):
    """
    Chuyển đổi nhãn từ XML (VD: A1, B4, C12, X, Y) sang nhãn chuẩn (1, 4, 12, X, Y)
    """
    raw_name = raw_name.strip().upper()
    if raw_name in ["X", "Y"]:
        return raw_name
    
    # Rút trích phần số (1 đến 22)
    num_match = re.search(r'\d+', raw_name)
    if num_match:
        num_str = num_match.group()
        if 1 <= int(num_str) <= 22:
            return num_str
    
    return None # Bỏ qua nếu nhãn không hợp lệ

def extract_dataset(sample_mode=False):
    print("Bắt đầu trích xuất dữ liệu từ PASCAL VOC...")
    print(f"Nguồn ảnh: {IMG_DIR}")
    print(f"Nguồn XML: {ANN_DIR}")
    print(f"Đầu ra: {OUTPUT_DIR}")
    
    # Tạo thư mục đầu ra cho 24 class
    classes = [str(i) for i in range(1, 23)] + ["X", "Y"]
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
        
    xml_files = glob.glob(os.path.join(ANN_DIR, "*.xml"))
    print(f"Tìm thấy tổng cộng {len(xml_files)} file XML.")
    
    if sample_mode:
        xml_files = xml_files[:10]
        print("Chạy ở chế độ SAMPLE (10 file đầu tiên).")
        
    total_extracted = 0
    errors = 0
    
    for xml_path in tqdm(xml_files, desc="Đang xử lý"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Lấy tên file ảnh
            filename_node = root.find("filename")
            if filename_node is None:
                continue
            
            # Tìm file ảnh thực tế (có thể phần mở rộng .jpg hoặc .JPG)
            img_basename = filename_node.text
            img_path = os.path.join(IMG_DIR, img_basename)
            if not os.path.exists(img_path):
                # Thử viết hoa phần mở rộng nếu không tìm thấy
                name_part, ext_part = os.path.splitext(img_basename)
                img_path = os.path.join(IMG_DIR, name_part + ext_part.upper())
                if not os.path.exists(img_path):
                    errors += 1
                    continue
                    
            # Mở ảnh gốc
            with Image.open(img_path) as img:
                for obj in root.findall("object"):
                    name_node = obj.find("name")
                    bndbox = obj.find("bndbox")
                    
                    if name_node is None or bndbox is None:
                        continue
                        
                    raw_label = name_node.text
                    standard_label = get_standard_label(raw_label)
                    
                    if not standard_label:
                        continue
                        
                    # Lấy tọa độ bounding box
                    try:
                        xmin = int(float(bndbox.find("xmin").text))
                        ymin = int(float(bndbox.find("ymin").text))
                        xmax = int(float(bndbox.find("xmax").text))
                        ymax = int(float(bndbox.find("ymax").text))
                    except (ValueError, TypeError, AttributeError):
                        continue
                        
                    # Tránh cắt ra ngoài ảnh
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img.width, xmax)
                    ymax = min(img.height, ymax)
                    
                    # Tránh bounding box không hợp lệ (diện tích = 0)
                    if xmax <= xmin or ymax <= ymin:
                        continue
                        
                    # Cắt ảnh NST
                    chr_crop = img.crop((xmin, ymin, xmax, ymax))
                    
                    # Tạo tên file độc nhất để lưu
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_name = f"{base_name}_obj{total_extracted}_{standard_label}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, standard_label, save_name)
                    
                    chr_crop.save(save_path, "JPEG", quality=95)
                    total_extracted += 1
                    
        except Exception as e:
            errors += 1
            
    print("\n--- HOÀN TẤT ---")
    print(f"Tổng số Nhiễm sắc thể đã trích xuất thành công: {total_extracted}")
    print(f"Số file bị lỗi/không tìm thấy ảnh: {errors}")
    print(f"Dữ liệu đã được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trích xuất dataset Karyotype từ PASCAL VOC")
    parser.add_argument("--sample", action="store_true", help="Chỉ chạy thử 10 file đầu tiên")
    args = parser.parse_args()
    
    extract_dataset(sample_mode=args.sample)
