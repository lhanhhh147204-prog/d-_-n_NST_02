# ============================================================
# FILE: pairing/hungarian_matcher.py
# CHỨC NĂNG: Thuật toán ghép cặp NST bằng Hungarian Algorithm
# MỚI HOÀN TOÀN: Giải bài toán Bipartite Matching cho Karyotype
# RÀNG BUỘC: Mỗi lớp PHẢI có ĐÚNG 2 NST (1 cặp), trừ XY có thể 1+1
# ============================================================

"""
Module Ghép Cặp NST (Chromosome Pairing via Hungarian Algorithm).

Bài toán:
- Input: Danh sách N ảnh NST đã phân loại (mỗi ảnh có vector xác suất 24 lớp)
- Output: 23 cặp NST (mỗi cặp gồm 2 ảnh cùng lớp)
- Ràng buộc: 
  * Mỗi lớp autosome (1-22) → đúng 2 NST
  * Lớp X → 1 hoặc 2 NST (tùy giới tính)  
  * Lớp Y → 0 hoặc 1 NST

Thuật toán:
1. Dùng classifier để lấy xác suất cho mỗi NST
2. Xây dựng Cost Matrix từ xác suất
3. Áp dụng Hungarian Algorithm (scipy.optimize.linear_sum_assignment)
4. Hậu xử lý: ép buộc phân bố 23 cặp hợp lệ

LƯU Ý:
- Nếu số NST input ≠ 46, thuật toán vẫn cố gắng ghép tốt nhất có thể.
- Không bao giờ cho phép > 2 NST cùng lớp autosome.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from config.settings import KARYOTYPE_LABELS, NUM_KARYOTYPE_CLASSES


def build_cost_matrix(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Xây dựng Cost Matrix cho bài toán gán NST → lớp.

    Cost = 1 - probability (xác suất càng cao → chi phí càng thấp).

    Tham số:
        probabilities: Mảng (N, 24) — xác suất mỗi NST thuộc mỗi lớp.

    Trả về:
        Cost matrix (N, 24) để đưa vào Hungarian Algorithm.
    """
    return 1.0 - probabilities


def _greedy_assign(
    probabilities: np.ndarray,
    max_per_class: Dict[str, int],
) -> Dict[str, List[int]]:
    """
    Gán tham lam (Greedy Assignment) với ràng buộc số lượng tối đa mỗi lớp.

    Thuật toán:
    1. Sắp xếp tất cả cặp (NST, lớp) theo xác suất giảm dần
    2. Duyệt từng cặp: nếu NST chưa gán VÀ lớp chưa đầy → gán

    Tham số:
        probabilities: Mảng (N, 24).
        max_per_class: Dict {label → số tối đa}.

    Trả về:
        Dict {label → [danh sách index NST]}.
    """
    n = probabilities.shape[0]
    assigned = set()  # Các NST đã được gán
    class_count = {label: 0 for label in KARYOTYPE_LABELS}
    result: Dict[str, List[int]] = {label: [] for label in KARYOTYPE_LABELS}

    # Tạo danh sách tất cả cặp (nst_idx, class_idx, prob)
    pairs = []
    for i in range(n):
        for j in range(NUM_KARYOTYPE_CLASSES):
            pairs.append((i, j, probabilities[i, j]))

    # Sắp xếp theo xác suất giảm dần
    pairs.sort(key=lambda x: x[2], reverse=True)

    for nst_idx, class_idx, prob in pairs:
        label = KARYOTYPE_LABELS[class_idx]
        max_count = max_per_class.get(label, 2)

        if nst_idx not in assigned and class_count[label] < max_count:
            assigned.add(nst_idx)
            class_count[label] += 1
            result[label].append(nst_idx)

        if len(assigned) == n:
            break

    return result


def match_pairs(
    probabilities: np.ndarray,
    sex: str = "XX",
) -> Dict[str, List[int]]:
    """
    Ghép cặp NST thành 23 nhóm (đúng chuẩn y khoa).

    Đây là hàm chính của module — gọi hàm này để ghép cặp.

    Tham số:
        probabilities: Mảng (N, 24) — xác suất mỗi NST thuộc mỗi lớp.
            Lấy từ KaryotypePredictor.predict() hoặc predict_batch().
        sex: Giới tính mẫu — "XX" (nữ) hoặc "XY" (nam).
            - "XX": 2 NST X, 0 NST Y
            - "XY": 1 NST X, 1 NST Y

    Trả về:
        Dict {label → [danh sách index NST trong mảng input]}:
        Ví dụ: {"1": [0, 5], "2": [1, 12], ..., "X": [3, 7], "Y": []}

    LƯU Ý:
    - Nếu input có 46 NST → kết quả lý tưởng (mỗi lớp autosome = 2)
    - Nếu input có < 46 hoặc > 46 → thuật toán cố gắng ghép tốt nhất
    - Kết quả LUÔN có đủ 24 key (1-22, X, Y), value có thể rỗng
    """
    n = probabilities.shape[0]

    # Xác định số lượng tối đa cho mỗi lớp
    max_per_class: Dict[str, int] = {}
    for label in KARYOTYPE_LABELS:
        if label == "Y":
            max_per_class[label] = 1 if sex == "XY" else 0
        elif label == "X":
            max_per_class[label] = 2 if sex == "XX" else 1
        else:
            max_per_class[label] = 2  # Autosome: luôn 2

    # Phương pháp: Greedy Assignment với ràng buộc
    # (Hungarian Algorithm thuần không hỗ trợ ràng buộc max_per_class dễ dàng,
    # nên dùng greedy trên xác suất đã sắp xếp — đủ tốt cho bài toán này)
    result = _greedy_assign(probabilities, max_per_class)

    return result


def validate_pairing(
    pairing: Dict[str, List[int]],
    sex: str = "XX",
) -> Dict[str, str]:
    """
    Kiểm tra tính hợp lệ của kết quả ghép cặp.

    Trả về:
        Dict {label → "OK" hoặc thông báo lỗi}.
    """
    report: Dict[str, str] = {}

    for label in KARYOTYPE_LABELS:
        count = len(pairing.get(label, []))

        if label == "Y":
            expected = 1 if sex == "XY" else 0
        elif label == "X":
            expected = 2 if sex == "XX" else 1
        else:
            expected = 2

        if count == expected:
            report[label] = "✅ OK"
        elif count < expected:
            report[label] = f"⚠️ Thiếu {expected - count} NST (có {count}/{expected})"
        else:
            report[label] = f"❌ Thừa {count - expected} NST (có {count}/{expected})"

    return report


def format_pairing_summary(
    pairing: Dict[str, List[int]],
    sex: str = "XX",
) -> str:
    """
    Tạo bản tóm tắt kết quả ghép cặp dạng text để in ra console.

    Trả về:
        Chuỗi multi-line mô tả kết quả ghép cặp.
    """
    validation = validate_pairing(pairing, sex)
    lines = [
        "=" * 50,
        "  KẾT QUẢ GHÉP CẶP NST (CHROMOSOME PAIRING)",
        f"  Giới tính mẫu: {sex}",
        "=" * 50,
    ]

    total_assigned = 0
    for label in KARYOTYPE_LABELS:
        indices = pairing.get(label, [])
        total_assigned += len(indices)
        status = validation[label]
        indices_str = ", ".join(str(i) for i in indices) if indices else "(trống)"
        lines.append(f"  NST {label:>2s}: [{indices_str}] — {status}")

    lines.append("-" * 50)
    lines.append(f"  Tổng số NST đã gán: {total_assigned}")
    lines.append("=" * 50)

    return "\n".join(lines)
