import cv2
import glob
import os
import numpy as np

out_dir = r"C:\Users\lehoa\.gemini\antigravity-ide\brain\78e1a3d9-c973-43fe-b9ee-54e215b2b1e0\artifacts\test_buoc3_out"
img_paths = glob.glob(os.path.join(out_dir, "*.png"))

images = []
for p in img_paths:
    img = cv2.imread(p)
    if img is not None:
        images.append(img)

# Tạo grid ảnh 6x6
cols = 6
rows = (len(images) + cols - 1) // cols
cell_size = 224

grid = np.ones((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) * 255

for idx, img in enumerate(images):
    r = idx // cols
    c = idx % cols
    y = r * cell_size
    x = c * cell_size
    # Resize nếu ảnh không phải 224x224 (dù thực tế nó đã 224x224)
    img_resized = cv2.resize(img, (cell_size, cell_size))
    grid[y:y+cell_size, x:x+cell_size] = img_resized
    # Viết tên file lên để dễ nhìn
    name = os.path.basename(img_paths[idx])
    cv2.putText(grid, f"{idx+1}", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

output_path = r"C:\Users\lehoa\.gemini\antigravity-ide\brain\78e1a3d9-c973-43fe-b9ee-54e215b2b1e0\artifacts\all_34_chromosomes.jpg"
cv2.imwrite(output_path, grid)
print("Grid saved!")
