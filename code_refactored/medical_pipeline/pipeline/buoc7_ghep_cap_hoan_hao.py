# ============================================================
# FILE: pipeline/buoc7_ghep_cap_hoan_hao.py
# (PHẦN CỦA BẠN - HỌC VIÊN)
# CHỨC NĂNG: Bước 7 - Áp dụng thuật toán Hungary để ghép cặp hoàn hảo
# ============================================================
from typing import List, Dict
import numpy as np
from medical_pipeline.pairing.hungarian_matcher import match_pairs, format_pairing_summary

def run_perfect_pairing(chromosomes: List[dict], sex: str = "XX") -> Dict[str, List[int]]:
    print("\\n🚀 [BƯỚC 7] BẮT ĐẦU CHẠY THUẬT TOÁN GHÉP CẶP HOÀN HẢO...")
    probs = np.array([chrom["probabilities"] for chrom in chromosomes])
    pairing = match_pairs(probs, sex=sex)
    summary = format_pairing_summary(pairing, sex=sex)
    print(summary)
    print("✅ [BƯỚC 7] ĐÃ GHÉP CẶP THÀNH CÔNG.")
    return pairing
