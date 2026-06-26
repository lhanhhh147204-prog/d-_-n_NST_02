import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean
from math import atan2, cos, sin, pi
from pwarp import triangular_mesh, graph_warp, graph_defined_warp

def _extract_skeleton(mask: np.ndarray) -> np.ndarray:
    sk = skeletonize(mask > 0)
    return sk.astype(np.uint8)

def _find_skeleton_endpoints(skel: np.ndarray):
    endpoints = []
    h, w = skel.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x] > 0:
                patch = skel[y-1:y+2, x-1:x+2]
                if np.count_nonzero(patch) == 2:
                    endpoints.append((x, y))
    return endpoints

def _select_main_branch_endpoints(endpoints):
    if len(endpoints) < 2:
        return None, None
    best_dist = -1
    best_pair = (None, None)
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            d = euclidean(endpoints[i], endpoints[j])
            if d > best_dist:
                best_dist = d
                best_pair = (endpoints[i], endpoints[j])
    return best_pair

def _find_max_deviation_point(skel, p1, p2):
    ys, xs = np.nonzero(skel)
    if len(ys) == 0: return None
    pts = np.stack((xs, ys), axis=1)
    x1, y1 = p1
    x2, y2 = p2
    denom = np.hypot(x2-x1, y2-y1) + 1e-6
    dists = np.abs((x2-x1)*(y1-pts[:,1]) - (x1-pts[:,0])*(y2-y1)) / denom
    idx = np.argmax(dists)
    max_dist = dists[idx]
    return (int(pts[idx,0]), int(pts[idx,1])), max_dist

def _classify_branches(red_pt, p1, p2):
    if euclidean(red_pt, p1) >= euclidean(red_pt, p2):
        return p1, p2
    return p2, p1

def _rotate_point(pt, center, angle):
    x, y = pt
    cx, cy = center
    dx, dy = x-cx, y-cy
    xr = dx*cos(angle) - dy*sin(angle) + cx
    yr = dx*sin(angle) + dy*cos(angle) + cy
    return (int(round(xr)), int(round(yr)))

def apply_arap_straightening(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Sử dụng thuật toán ARAP (As-Rigid-As-Possible) Puppet Warp để bẻ thẳng 
    nhiễm sắc thể bị cong (Bent Chromosome Correction).
    """
    skel = _extract_skeleton(mask)
    endpoints = _find_skeleton_endpoints(skel)
    pt1, pt2 = _select_main_branch_endpoints(endpoints)
    
    if pt1 is None or pt2 is None:
        return image, mask  # Không đủ endpoint để xác định trục
        
    dev_res = _find_max_deviation_point(skel, pt1, pt2)
    if dev_res is None: return image, mask
    
    red_point, max_dist = dev_res
    
    # Bỏ qua nếu NST gần như thẳng (độ cong nhỏ hơn 3 pixel)
    if max_dist < 3.0:
        return image, mask
        
    main_ep, sec_ep = _classify_branches(red_point, pt1, pt2)
    
    theta_main = atan2(main_ep[1]-red_point[1], main_ep[0]-red_point[0])
    theta_sec  = atan2(sec_ep[1]-red_point[1], sec_ep[0]-red_point[0])
    delta = theta_main + pi - theta_sec
    
    rot_sec_ep = _rotate_point(sec_ep, red_point, delta)
    
    ctrl_pts = np.array([pt1, pt2, red_point], dtype=int)
    new_locs = np.array([pt1, pt2, rot_sec_ep], dtype=int)
    
    h, w = image.shape[:2]
    
    # 1. Sinh lưới tam giác với delta = 15 (càng nhỏ càng chính xác, nhưng chậm hơn)
    r, f = triangular_mesh(width=w, height=h, delta=15)
    
    # 2. Map control point lên lưới
    ctrl_idx_mesh = [
        int(np.argmin(np.hypot(r[:,0] - x, r[:,1] - y)))
        for x, y in ctrl_pts
    ]
    
    # 3. Tính toán vị trí warp mới cho các đỉnh lưới
    new_r = graph_warp(
        vertices=r,
        faces=f,
        control_indices=np.array(ctrl_idx_mesh, dtype=int),
        shifted_locations=new_locs
    )
    
    # 4. Áp warp lên ảnh màu và mask
    # Thư viện pwarp yêu cầu input phải là ảnh 3 kênh (RGB/BGR), nếu truyền ảnh xám 2D sẽ sinh lỗi broadcast.
    if len(image.shape) == 2:
        img_3c = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_3c = image.copy()
        
    mask_3c = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    
    warped_img_3c = graph_defined_warp(
        img_3c,
        vertices_src=r, faces_src=f,
        vertices_dst=new_r, faces_dst=f
    )
    
    warped_mask_3c = graph_defined_warp(
        mask_3c,
        vertices_src=r, faces_src=f,
        vertices_dst=new_r, faces_dst=f
    )
    
    # Trích xuất lại ảnh 2D nếu input gốc là 2D
    if len(image.shape) == 2:
        warped_img = warped_img_3c[:, :, 0]
    else:
        warped_img = warped_img_3c
        
    warped_mask = warped_mask_3c[:, :, 0]
    
    # Fill background areas (where warp created black space) with 255
    bg_mask = (warped_mask == 0)
    if len(warped_img.shape) == 3:
        warped_img[bg_mask] = [255, 255, 255]
    else:
        warped_img[bg_mask] = 255
        
    return warped_img, warped_mask > 127
