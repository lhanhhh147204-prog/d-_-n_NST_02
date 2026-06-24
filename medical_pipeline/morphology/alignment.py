# ============================================================
# FILE: morphology/alignment.py
# CHỨC NĂNG: Xoay NST thẳng đứng + Định vị tâm động (Centromere)
# MỚI HOÀN TOÀN: Viết mới cho Clinical-Grade Karyotyping
# RÀNG BUỘC: KHÔNG làm biến dạng cấu trúc sinh học NST
# ============================================================

"""
Module Căn Chỉnh Hình Thái NST (Chromosome Alignment).

Hai chức năng chính:
1. **Xoay thẳng đứng (Vertical Alignment):**
   - Dùng PCA (Principal Component Analysis) hoặc Image Moments
     để tìm trục chính của NST → xoay sao cho trục dọc = 90°.
   - Phương pháp Moments ổn định hơn PCA khi NST bị gãy/nhỏ.

2. **Định vị Tâm động (Centromere Detection):**
   - Tìm vị trí eo hẹp nhất trên trục xương (skeleton).
   - Lật NST sao cho nhánh ngắn (p-arm) luôn ở trên.

LƯU Ý SINH HỌC:
- Phép xoay KHÔNG resize/crop/warp — chỉ rotate đúng góc.
- Padding bằng nền trắng (255) sau khi xoay.
- Tâm động là điểm thắt (constriction) tự nhiên, KHÔNG phải
  điểm giữa chiều dài → phải dùng skeleton width analysis.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional


def _find_orientation_angle(mask: np.ndarray) -> float:
    """
    Tìm góc nghiêng trục chính của NST bằng Image Moments.

    Trả về:
        Góc (độ) cần xoay để NST thẳng đứng.
        Giá trị dương = xoay ngược chiều kim đồng hồ.
    """
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] < 1:
        return 0.0  # Mask rỗng

    # Tính góc trục chính từ central moments (mu20, mu11, mu02)
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    # Góc trục chính (radians) — công thức PCA trên moments
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    angle_deg = math.degrees(theta)

    # Chuyển về góc cần xoay để trục dọc = 90°
    # Trục chính nằm ngang → xoay 90°
    # Trục chính nghiêng → xoay (90 - angle)°
    rotation_angle = 90.0 - angle_deg

    # Chuẩn hóa về [-180, 180]
    while rotation_angle > 180:
        rotation_angle -= 360
    while rotation_angle < -180:
        rotation_angle += 360

    return rotation_angle


def rotate_chromosome(
    image: np.ndarray,
    mask: np.ndarray,
    angle: Optional[float] = None,
    bg_value: int = 255,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Xoay ảnh NST và mask sao cho NST thẳng đứng.

    KHÔNG resize, KHÔNG crop — chỉ xoay + expand canvas.

    Tham số:
        image: Ảnh xám NST (H, W), uint8.
        mask: Mask nhị phân NST (H, W), bool hoặc uint8.
        angle: Góc xoay (độ). Nếu None → tự tính bằng Moments.
        bg_value: Giá trị pixel nền sau khi xoay (mặc định trắng = 255).

    Trả về:
        Tuple (ảnh_đã_xoay, mask_đã_xoay, góc_đã_dùng).
    """
    if angle is None:
        angle = _find_orientation_angle(mask)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Ma trận xoay với expand (mở rộng canvas để không bị cắt)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Tính kích thước canvas mới sau xoay
    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Điều chỉnh tâm xoay cho canvas mới
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Xoay ảnh (nội suy tuyến tính, nền trắng)
    rotated_img = cv2.warpAffine(
        image, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_value,
    )

    # Xoay mask (nearest neighbor — giữ biên sắc nét)
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8) * 255, rot_mat, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_img, rotated_mask > 127, angle


def _extract_skeleton_widths(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trích xuất trục xương và đo độ rộng tại mỗi điểm trên trục.

    Trả về:
        Tuple (skeleton_points, widths):
        - skeleton_points: Mảng (N, 2) tọa độ các điểm trên trục xương,
          được sắp xếp từ trên xuống dưới (theo trục y).
        - widths: Mảng (N,) — độ rộng NST tại mỗi điểm skeleton.
    """
    mask_u8 = mask.astype(np.uint8)

    # Tạo distance transform (khoảng cách từ mỗi pixel đến biên)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Thinning (skeletonization) bằng Zhang-Suen
    skeleton = cv2.ximgproc.thinning(mask_u8 * 255) if hasattr(cv2, 'ximgproc') else _fallback_skeleton(mask_u8)

    # Lấy tọa độ các điểm trên skeleton
    ys, xs = np.where(skeleton > 0)
    if len(ys) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    # Sắp xếp theo trục y (từ trên xuống dưới)
    order = np.argsort(ys)
    points = np.stack([xs[order], ys[order]], axis=1)  # (N, 2) — [x, y]

    # Độ rộng = 2 * distance_transform tại mỗi điểm skeleton
    widths = 2.0 * dist[ys[order], xs[order]]

    return points, widths


def _fallback_skeleton(mask_u8: np.ndarray) -> np.ndarray:
    """
    Skeleton bằng morphology nếu cv2.ximgproc không khả dụng.

    Dùng thuật toán erosion lặp lại cho đến khi mask biến mất,
    lưu lại các pixel biến mất ở mỗi bước.
    """
    skeleton = np.zeros_like(mask_u8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask_u8.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = temp - opened
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


def find_centromere(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Tìm vị trí Tâm Động (Centromere) của NST.

    Tâm động = điểm eo hẹp nhất trên trục xương.
    Đây là điểm mà chiều rộng NST giảm xuống mức tối thiểu cục bộ.

    Tham số:
        mask: Mask nhị phân NST đã được xoay thẳng đứng (H, W).

    Trả về:
        Tọa độ (x, y) của tâm động, hoặc None nếu không phát hiện được.
    """
    points, widths = _extract_skeleton_widths(mask)

    if len(widths) < 5:
        return None

    # Làm mịn đường width profile để tránh noise
    kernel_size = max(3, len(widths) // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(widths, np.ones(kernel_size) / kernel_size, mode="same")

    # Tìm điểm cực tiểu cục bộ (local minimum)
    # Bỏ qua 15% đầu và 15% cuối (vùng đầu mút NST thường hẹp tự nhiên)
    margin = int(len(smoothed) * 0.15)
    search_region = smoothed[margin:-margin] if margin > 0 else smoothed

    if len(search_region) < 3:
        return None

    # Tìm vị trí hẹp nhất trong vùng tìm kiếm
    min_idx = int(np.argmin(search_region)) + margin
    centromere_point = points[min_idx]

    return (int(centromere_point[0]), int(centromere_point[1]))


def orient_p_arm_up(
    image: np.ndarray,
    mask: np.ndarray,
    centromere: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lật NST sao cho nhánh ngắn (p-arm) luôn ở phía trên.

    Quy ước y khoa:
    - Nhánh ngắn (short arm = p-arm) → phía trên
    - Nhánh dài (long arm = q-arm) → phía dưới

    Thuật toán:
    1. Tìm tâm động (nếu chưa biết)
    2. Đo chiều dài pixel phía trên vs phía dưới tâm động
    3. Nếu phía trên DÀI hơn → lật 180°

    Tham số:
        image: Ảnh NST đã xoay thẳng đứng (H, W), uint8.
        mask: Mask NST đã xoay (H, W), bool.
        centromere: Tọa độ tâm động (x, y). Nếu None → tự tìm.

    Trả về:
        Tuple (ảnh_đã_lật, mask_đã_lật).
        Nếu không cần lật → trả về bản gốc (không copy thừa).
    """
    if centromere is None:
        centromere = find_centromere(mask)

    if centromere is None:
        # Không tìm được tâm động → giữ nguyên
        return image, mask

    _, cy = centromere

    # Đếm pixel phía trên và dưới tâm động
    pixels_above = mask[:cy, :].sum()
    pixels_below = mask[cy:, :].sum()

    if pixels_above > pixels_below:
        # Nhánh trên dài hơn → đang ngược → cần lật 180°
        return np.flip(image, axis=(0, 1)).copy(), np.flip(mask, axis=(0, 1)).copy()

    # Đã đúng hướng → giữ nguyên
    return image, mask


def align_chromosome(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Pipeline căn chỉnh hoàn chỉnh cho 1 NST đơn.

    Bước 1: Xoay thẳng đứng (PCA/Moments)
    Bước 2: Tìm tâm động
    Bước 3: Lật p-arm lên trên

    Tham số:
        image: Ảnh xám NST (H, W), uint8.
        mask: Mask nhị phân NST (H, W), bool hoặc uint8.

    Trả về:
        Tuple (ảnh_aligned, mask_aligned, metadata):
        - metadata chứa: rotation_angle, centromere_xy, was_flipped
    """
    # Bước 1: Xoay thẳng đứng
    rotated_img, rotated_mask, angle = rotate_chromosome(image, mask)

    # Bước 2: Tìm tâm động
    centromere = find_centromere(rotated_mask)

    # Bước 3: Lật p-arm lên trên
    was_flipped = False
    if centromere is not None:
        _, cy = centromere
        pixels_above = rotated_mask[:cy, :].sum()
        pixels_below = rotated_mask[cy:, :].sum()
        was_flipped = pixels_above > pixels_below

    final_img, final_mask = orient_p_arm_up(rotated_img, rotated_mask, centromere)

    metadata = {
        "góc_xoay": round(angle, 2),
        "tâm_động": centromere,
        "đã_lật_180": was_flipped,
    }

    return final_img, final_mask, metadata
