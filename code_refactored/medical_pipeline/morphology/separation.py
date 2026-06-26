# ============================================================
# FILE: morphology/separation.py
# CHỨC NĂNG: Thuật toán tìm điểm lõm và cắt tách NST bị trùng lấn
# NGUỒN GỐC: Chuyển từ extracted_code.py (project_notebook.ipynb)
# RÀNG BUỘC: KHÔNG làm biến dạng cấu trúc sinh học NST
# ============================================================

"""
Module tách NST chồng lấn (Overlapping Chromosome Separation).

Thuật toán chính:
1. Tìm điểm lõm (concave points) trên đường viền vùng chồng
2. Chọn cặp điểm cắt tối ưu (cutting points) dựa trên khoảng cách + góc
3. Vẽ đường cắt giữa 2 điểm lõm → chia mask C thành 2 nửa
4. Gán mỗi nửa về NST A hoặc B dựa trên diện tích chồng

LƯU Ý CHO NGƯỜI ĐỌC:
- Hàm `find_concave_points()` là cốt lõi — tìm điểm mà đường viền
  "lõm vào trong" (negative curvature) dọc contour vùng overlap.
- Hàm `find_best_cutting_points()` chọn cặp điểm đối diện nhau nhất.
- Không dùng erode/dilate mạnh vì sẽ phá vỡ hình dạng NST.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
import math


def find_concave_points(
    contour: np.ndarray,
    angle_threshold: float = 60.0,
    k: int = 15,
) -> List[Tuple[int, int]]:
    """
    Tìm các điểm lõm (concave points) trên đường viền contour.

    Thuật toán:
    - Duyệt từng điểm P[i] trên contour
    - Tính góc giữa vector (P[i-k] → P[i]) và (P[i] → P[i+k])
    - Nếu góc < angle_threshold VÀ hướng lõm vào → đó là điểm lõm

    Tham số:
        contour: Mảng contour từ cv2.findContours(), shape (N, 1, 2).
        angle_threshold: Ngưỡng góc (độ) để xác định điểm lõm.
            Góc càng nhỏ = lõm càng sắc. Mặc định 60°.
        k: Khoảng cách (số điểm) giữa P[i] và các điểm lân cận.
            k lớn = bỏ qua noise nhỏ. Mặc định 15.

    Trả về:
        Danh sách tọa độ (x, y) của các điểm lõm.
    """
    if contour is None or len(contour) < 2 * k + 1:
        return []

    # Làm phẳng contour về dạng (N, 2)
    pts = contour.reshape(-1, 2)
    n = len(pts)
    concave_pts = []

    for i in range(n):
        # Lấy 2 điểm lân cận cách k bước (vòng tròn)
        prev = pts[(i - k) % n]
        curr = pts[i]
        next_pt = pts[(i + k) % n]

        # Vector từ prev → curr và curr → next
        v1 = curr - prev
        v2 = next_pt - curr

        # Tính góc giữa 2 vector
        dot = float(np.dot(v1, v2))
        mag1 = float(np.linalg.norm(v1))
        mag2 = float(np.linalg.norm(v2))

        if mag1 < 1e-6 or mag2 < 1e-6:
            continue

        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        # Tính cross product để xác định hướng lõm
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])

        # Điểm lõm: góc nhọn VÀ cross product < 0 (lõm vào bên phải)
        if angle_deg < angle_threshold and cross < 0:
            concave_pts.append((int(curr[0]), int(curr[1])))

    return concave_pts


def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Tính khoảng cách Euclid giữa 2 điểm."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_best_cutting_points(
    concave_points: List[Tuple[int, int]],
    min_distance: float = 10.0,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Chọn cặp điểm lõm tối ưu để vẽ đường cắt tách NST.

    Tiêu chí chọn:
    - Cặp điểm cách nhau >= min_distance (tránh cắt quá ngắn)
    - Ưu tiên cặp có khoảng cách NGẮN NHẤT (đường cắt tối thiểu)

    Tham số:
        concave_points: Danh sách điểm lõm từ find_concave_points().
        min_distance: Khoảng cách tối thiểu giữa 2 điểm cắt (pixel).

    Trả về:
        Tuple (điểm_1, điểm_2) hoặc None nếu không tìm được cặp hợp lệ.
    """
    if len(concave_points) < 2:
        return None

    best_pair = None
    best_dist = float("inf")

    for i in range(len(concave_points)):
        for j in range(i + 1, len(concave_points)):
            d = _distance(concave_points[i], concave_points[j])
            if d >= min_distance and d < best_dist:
                best_dist = d
                best_pair = (concave_points[i], concave_points[j])

    return best_pair


def draw_cutting_line(
    mask: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    Vẽ đường cắt lên mask vùng overlap C để chia thành 2 nửa.

    Tham số:
        mask: Mask nhị phân của vùng overlap (C), dtype bool hoặc uint8.
        pt1, pt2: Tọa độ 2 điểm cắt.
        thickness: Độ dày đường cắt (pixel).

    Trả về:
        Mask đã bị chia cắt (vùng đường cắt = 0).
    """
    result = mask.astype(np.uint8).copy()
    cv2.line(result, pt1, pt2, 0, thickness=thickness)
    return result.astype(bool)


def split_overlap_mask(
    mask_overlap: np.ndarray,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    angle_threshold: float = 60.0,
    k: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tách vùng overlap C thành 2 phần và gán về NST A hoặc B.

    Quy trình hoàn chỉnh:
    1. Tìm contour vùng C
    2. Phát hiện điểm lõm trên contour
    3. Chọn cặp cắt tối ưu → vẽ đường cắt
    4. Dùng connected components để chia C thành các vùng con
    5. Gán mỗi vùng con về A hoặc B (dựa trên diện tích chồng)

    Tham số:
        mask_overlap: Mask nhị phân vùng overlap C.
        mask_A: Mask nhị phân NST A.
        mask_B: Mask nhị phân NST B.
        angle_threshold: Ngưỡng góc cho find_concave_points().
        k: Tham số k cho find_concave_points().

    Trả về:
        Tuple (phần_gán_về_A, phần_gán_về_B) — cả 2 đều là mask bool.
    """
    # Tìm contour lớn nhất của vùng overlap
    overlap_u8 = mask_overlap.astype(np.uint8) * 255
    contours, _ = cv2.findContours(overlap_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        # Không có contour → chia đều (fallback)
        return mask_overlap, np.zeros_like(mask_overlap, dtype=bool)

    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)

    # Bước 1: Tìm điểm lõm
    concave_pts = find_concave_points(largest_contour, angle_threshold, k)

    # Bước 2: Chọn cặp cắt tối ưu
    cutting_pair = find_best_cutting_points(concave_pts)

    if cutting_pair is None:
        # Không tìm được cặp cắt → trả về toàn bộ C cho A (fallback an toàn)
        return mask_overlap, np.zeros_like(mask_overlap, dtype=bool)

    # Bước 3: Vẽ đường cắt lên mask overlap
    cut_mask = draw_cutting_line(mask_overlap, cutting_pair[0], cutting_pair[1])

    # Bước 4: Tách các vùng con bằng connected components
    cut_u8 = cut_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(cut_u8, connectivity=8)

    # Bước 5: Gán mỗi vùng con về A hoặc B
    part_A = np.zeros_like(mask_overlap, dtype=bool)
    part_B = np.zeros_like(mask_overlap, dtype=bool)

    for label_id in range(1, num_labels):
        region = labels == label_id
        # Đếm pixel chồng với A và B (bên ngoài vùng C)
        only_A = mask_A & (~mask_overlap)  # Vùng chỉ thuộc A
        only_B = mask_B & (~mask_overlap)  # Vùng chỉ thuộc B
        overlap_with_A = (region & only_A).sum()
        overlap_with_B = (region & only_B).sum()

        # Nếu chồng đều → kiểm tra vùng tiếp giáp (adjacency)
        if overlap_with_A == 0 and overlap_with_B == 0:
            # Dùng dilation 3x3 để tìm vùng lân cận
            dilated = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8))
            adj_A = (dilated.astype(bool) & only_A).sum()
            adj_B = (dilated.astype(bool) & only_B).sum()
            if adj_A >= adj_B:
                part_A |= region
            else:
                part_B |= region
        elif overlap_with_A >= overlap_with_B:
            part_A |= region
        else:
            part_B |= region

    return part_A, part_B
