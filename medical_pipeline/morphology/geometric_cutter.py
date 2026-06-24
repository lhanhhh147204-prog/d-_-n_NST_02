import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import label

def find_best_cutting_points(concavity_points, img):
    ideal_pairs = []
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    for i in range(len(concavity_points)):
        for j in range(i + 1, len(concavity_points)):
            p1, p2 = concavity_points[i], concavity_points[j]
            line_mask = np.zeros_like(image_gray, dtype=np.uint8)
            cv2.line(line_mask, p1, p2, 255, 1)
            line_pixels = image_gray[line_mask == 255]
            white_pixel_count = np.sum(line_pixels == 255)
            total_pixel_count = len(line_pixels)
            non_white_pixel_count = total_pixel_count - white_pixel_count

            if white_pixel_count <= non_white_pixel_count:
                ideal_pairs.append((p1, p2))
    return ideal_pairs

def find_concave_points(img):
    if img is None:
        return [], None, None, None

    img_blurred = cv2.GaussianBlur(img, (11, 11), 0)
    img_rgb = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2RGB)
    r, g, b = cv2.split(img_rgb)

    def create_color_heatmap(channel, color_map=cv2.COLORMAP_OCEAN):
        normalized_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(normalized_channel.astype(np.uint8), color_map)

    heatmap_r = create_color_heatmap(r)
    heatmap_g = create_color_heatmap(g)
    heatmap_b = create_color_heatmap(b)

    alpha = 1
    blended_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_r, alpha, 0)
    blended_img = cv2.addWeighted(blended_img, 1 - alpha, heatmap_g, alpha, 0)
    blended_img = cv2.addWeighted(blended_img, 1 - alpha, heatmap_b, alpha, 0)

    edges = cv2.Canny(blended_img.astype(np.uint8), 50, 240)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_concave(A, B, C):
        AB = np.array([B[0] - A[0], B[1] - A[1]])
        BC = np.array([C[0] - B[0], C[1] - B[1]])
        cross_product = AB[0] * BC[1] - AB[1] * BC[0]
        return cross_product > 0

    def find_nearest_point(point, contour):
        distances = np.sum((contour - point) ** 2, axis=2)
        nearest_idx = np.argmin(distances)
        return tuple(contour[nearest_idx][0])

    final_img = img_rgb.copy()
    concavity_points = []
    original_contour_img = img_rgb.copy()
    original_polygon = []

    for contour in contours:
        polygon_points = [tuple(point[0]) for point in contour]
        original_polygon.append(polygon_points)
        cv2.polylines(original_contour_img, [contour], isClosed=True, color=(255, 0, 0), thickness=2)

        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.polylines(final_img, [approx], isClosed=True, color=(255, 0, 0), thickness=2)

        n = len(approx)
        if n < 5:
            continue

        for i in range(n):
            A = approx[(i - 1) % n][0]
            B = approx[i][0]
            C = approx[(i + 1) % n][0]

            if is_concave(A, B, C):
                concavity_points.append(tuple(B))
                cv2.circle(final_img, tuple(B), 5, (0, 255, 0), -1)
                nearest_point = find_nearest_point(B, contour)
                cv2.circle(original_contour_img, nearest_point, 5, (0, 255, 0), -1)

    return concavity_points, final_img, original_contour_img, original_polygon

def process_chromosome_image(img):
    if img is None:
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)
    return opening

def compute_guide_line(point1, point2, shape):
    x1, y1 = point1
    x2, y2 = point2
    line = []
    dx, dy = x2 - x1, y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x1, y1)]
    x_step, y_step = dx / steps, dy / steps
    for i in range(int(steps) + 1):
        x = int(x1 + i * x_step)
        y = int(y1 + i * y_step)
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            line.append((x, y))
    return line

def count_white_pixels(path, binary_mask):
    white_pixel_count = 0
    for (x, y) in path:
        if 0 <= y < binary_mask.shape[0] and 0 <= x < binary_mask.shape[1]:
            if binary_mask[y, x] > 0:
                white_pixel_count += 1
    return white_pixel_count

def draw_and_count_white_pixels(path, binary_mask):
    mask_copy = np.zeros_like(binary_mask, dtype=np.uint8)
    for i in range(len(path) - 1):
        cv2.line(mask_copy, path[i], path[i + 1], 255, 1)
    white_count = np.sum((mask_copy == 255) & (binary_mask > 0))
    return white_count, mask_copy

def compute_separation_path(point1, point2, binary_mask, img, kernel_size=3, dilation_iters=1):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    guide_line = compute_guide_line(point1, point2, binary_mask.shape)
    if not guide_line:
        return [(point1[0], point1[1]), (point2[0], point2[1])]

    binary_img = np.zeros_like(binary_mask, dtype=np.uint8)
    for (x, y) in guide_line:
        binary_img[y, x] = 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=dilation_iters)

    junction_area = (dilated_img > 0)
    junction_pixels = img[junction_area]
    if len(junction_pixels) == 0:
        return guide_line
    V_A = np.mean(junction_pixels)

    candidate_points = []
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if dilated_img[y, x] > 0 and img[y, x] < V_A:
                candidate_points.append((x, y))

    decision_points = []
    if candidate_points:
        candidate_img = np.zeros_like(img, dtype=np.uint8)
        for (cx, cy) in candidate_points:
            candidate_img[cy, cx] = 255
        labeled_array, num_features = label(candidate_img, structure=np.ones((3, 3), np.int8))
        for i in range(1, num_features + 1):
            region = np.where(labeled_array == i)
            region_points = list(zip(region[1], region[0]))
            min_point = min(region_points, key=lambda p: img[p[1], p[0]], default=None)
            if min_point:
                decision_points.append(min_point)

    if point1 not in decision_points:
        decision_points.insert(0, point1)
    if point2 not in decision_points:
        decision_points.append(point2)

    candidate_paths = [guide_line]
    if len(decision_points) >= 3:
        x_coords, y_coords = zip(*decision_points)
        for s in range(9):
            k_spline = min(2, len(decision_points) - 1)
            try:
                tck, u = splprep([x_coords, y_coords], s=s, k=k_spline)
                spline_points = splev(np.linspace(0, 1, 100), tck)
                path = []
                for (xx, yy) in zip(spline_points[0], spline_points[1]):
                    ix, iy = int(round(xx)), int(round(yy))
                    if 0 <= ix < w and 0 <= iy < h:
                        path.append((ix, iy))
                candidate_paths.append(path)
            except:
                pass

    best_path = min(candidate_paths, key=lambda p: count_white_pixels(p, binary_mask))

    if count_white_pixels(best_path, binary_mask) > 0:
        optimized_path = list(best_path)
        for i, (x, y) in enumerate(optimized_path):
            if binary_mask[y, x] > 0:
                half = kernel_size // 2
                found = False
                for dy in range(-half, half+1):
                    for dx in range(-half, half+1):
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < w and 0 <= new_y < h:
                            if binary_mask[new_y, new_x] == 0:
                                optimized_path[i] = (new_x, new_y)
                                found = True
                                break
                    if found:
                        break
        best_path = optimized_path

    # Fix: Ensure point1 and point2 are exactly at the ends of the path
    # Because split_contour_with_path requires exact matches in the contour!
    if tuple(best_path[0]) != tuple(point1):
        best_path.insert(0, tuple(point1))
    if tuple(best_path[-1]) != tuple(point2):
        best_path.append(tuple(point2))

    return best_path

def split_contour_with_path(contour, path):
    intersection_points = []
    for p in path:
        if p in contour:
            intersection_points.append(p)

    if len(intersection_points) < 2:
        return None, None

    intersection_indices = [contour.index(p) for p in intersection_points]
    intersection_indices.sort()

    if len(intersection_indices) < 2:
        return None, None

    start_idx, end_idx = intersection_indices[0], intersection_indices[-1]

    contour1 = contour[:start_idx] + path[path.index(intersection_points[0]):path.index(intersection_points[-1])+1] + contour[end_idx:]
    contour2 = contour[start_idx:end_idx+1] + path[path.index(intersection_points[-1]):path.index(intersection_points[0])-1:-1]

    return contour1, contour2

def split_and_return_images(best_separation_path, original_polygon, img):
    if not best_separation_path or not original_polygon:
        return None, None

    # Tìm contour chứa điểm đầu và cuối của đường cắt
    original_contour = None
    contour1, contour2 = None, None
    for contour in original_polygon:
        c1, c2 = split_contour_with_path(contour, best_separation_path)
        if c1 is not None and c2 is not None:
            original_contour = contour
            contour1, contour2 = c1, c2
            break
            
    if contour1 is None or contour2 is None:
        return None, None

    original_img = img
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    mask1 = np.zeros_like(gray_img, dtype=np.uint8)
    mask2 = np.zeros_like(gray_img, dtype=np.uint8)
    cv2.fillPoly(mask1, [np.array(contour1, dtype=np.int32)], 255)
    cv2.fillPoly(mask2, [np.array(contour2, dtype=np.int32)], 255)

    cut_mask = np.zeros_like(mask1, dtype=np.uint8)
    for i in range(len(best_separation_path) - 1):
        cv2.line(cut_mask, best_separation_path[i], best_separation_path[i + 1], 255, 3)

    inv_cut = 255 - cut_mask
    dist_transform = cv2.distanceTransform(inv_cut, cv2.DIST_L2, 5)
    gradient_mask = np.clip(dist_transform / 5.0, 0, 1)
    gradient_mask = 1 - gradient_mask

    part1 = cv2.bitwise_and(original_img, original_img, mask=mask1)
    part2 = cv2.bitwise_and(original_img, original_img, mask=mask2)

    part1_clean = part1.copy()
    part2_clean = part2.copy()
    for c in range(3):
        part1_clean[:, :, c] = np.where(mask1 > 0, part1[:, :, c] * (1 - gradient_mask) + gradient_mask * 255, 255)
        part2_clean[:, :, c] = np.where(mask2 > 0, part2[:, :, c] * (1 - gradient_mask) + gradient_mask * 255, 255)

    x1, y1, w1, h1 = cv2.boundingRect(np.array(contour1, dtype=np.int32))
    x2, y2, w2, h2 = cv2.boundingRect(np.array(contour2, dtype=np.int32))

    part1_cut = part1_clean[y1:y1+h1, x1:x1+w1].copy()
    part2_cut = part2_clean[y2:y2+h2, x2:x2+w2].copy()

    background = np.ones((224, 224, 3), dtype=np.uint8) * 255

    part1_result = background.copy()
    h1_new, w1_new = part1_cut.shape[:2]
    y_offset1 = (224 - h1_new) // 2
    x_offset1 = (224 - w1_new) // 2
    if h1_new > 0 and w1_new > 0 and y_offset1 >= 0 and x_offset1 >= 0:
        mask1_cut = cv2.cvtColor(part1_cut, cv2.COLOR_BGR2GRAY) < 250
        part1_result[y_offset1:y_offset1+h1_new, x_offset1:x_offset1+w1_new][mask1_cut] = part1_cut[mask1_cut]

    part2_result = background.copy()
    h2_new, w2_new = part2_cut.shape[:2]
    y_offset2 = (224 - h2_new) // 2
    x_offset2 = (224 - w2_new) // 2
    if h2_new > 0 and w2_new > 0 and y_offset2 >= 0 and x_offset2 >= 0:
        mask2_cut = cv2.cvtColor(part2_cut, cv2.COLOR_BGR2GRAY) < 250
        part2_result[y_offset2:y_offset2+h2_new, x_offset2:x_offset2+w2_new][mask2_cut] = part2_cut[mask2_cut]

    return part1_result, part2_result
