import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(r"c:\Users\lehoa\dự_án_NTS\code_refactored")

from medical_pipeline.pipeline.buoc2_tach_tung_cum_nhiem_sac_the import run_segmentation
from medical_pipeline.pipeline.buoc3_main import run_overlap_separation

# Setup paths
img_path = r"c:\Users\lehoa\dự_án_NTS\data\raw\2025\9250100210.1.k.JPG"
b2_out_dir = Path(r"C:\Users\lehoa\.gemini\antigravity-ide\brain\78e1a3d9-c973-43fe-b9ee-54e215b2b1e0\artifacts\test_buoc2_out")
b3_out_dir = Path(r"C:\Users\lehoa\.gemini\antigravity-ide\brain\78e1a3d9-c973-43fe-b9ee-54e215b2b1e0\artifacts\test_buoc3_out")

b2_out_dir.mkdir(parents=True, exist_ok=True)
b3_out_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Segment
print("Running Step 2...")
img_array = np.fromfile(img_path, np.uint8)
img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
if img_bgr is None:
    print("Cannot load image!")
    sys.exit(1)

cropped_images, visualization_img, binary_mask, bboxes = run_segmentation(img_bgr)
for idx, crop in enumerate(cropped_images):
    # Padding to white square 224x224 similar to the processing requirement if needed
    # but buoc3_main expects the cropped clusters.
    cv2.imwrite(str(b2_out_dir / f"cluster_{idx}.png"), crop)

print(f"Saved {len(cropped_images)} clusters to {b2_out_dir}")

# Step 3: Separate
print("Running Step 3...")
separated = run_overlap_separation(b2_out_dir, b3_out_dir)
print(f"Finished Step 3. Saved {len(separated)} final chromosomes to {b3_out_dir}")

