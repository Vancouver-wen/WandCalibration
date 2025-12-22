import os
import sys
import json
import copy

import numpy as np
import cv2
import glob
from natsort import natsorted
from loguru import logger
import torch

def tetra_volume(p1, p2, p3, p4):
    """计算四面体体积"""
    return np.abs(np.dot(p2-p1, np.cross(p3-p1, p4-p1))) / 6.0

def estimate_scale_robust_final(source_points, target_points, coplanar_threshold=0.05):
    """
    【最终版本】稳健的尺度因子估计函数。
    自动适应点数（3、4或更多）和几何分布（共面/三维），选择最佳算法。

    参数:
        source_points (np.ndarray): 源点云，形状为 (N, 3)。
        target_points (np.ndarray): 目标点云，形状为 (N, 3)，与源点一一对应。
        coplanar_threshold (float): 判定四点"接近共面"的阈值（相对值）。默认0.05。
                                    值越小，对"不共面"的要求越严格。

    返回:
        dict: 包含以下键的结果字典：
            - 'scale_factor' (float): 估计的尺度因子。
            - 'method' (str): 使用的算法名称。
            - 'message' (str): 描述性的状态或警告信息。
            - 'coplanarity' (float, 可选): 仅当N=4时返回，共面性指标。
    """
    # ===== 1. 输入验证 =====
    assert isinstance(source_points, np.ndarray) and isinstance(target_points, np.ndarray), "输入必须为numpy数组"
    assert source_points.shape == target_points.shape, "两个点云的形状必须相同"
    assert source_points.shape[1] == 3, "点云必须是三维坐标 (N, 3)"
    N = source_points.shape[0]
    assert N >= 3, "至少需要3个点才能估计尺度"
    result = {'scale_factor': 1.0, 'method': '', 'message': ''}
    # ===== 2. 核心算法分支 =====
    if N == 3:
        # ---------- 针对3个点：三角形边长中位数法 ----------
        result['method'] = 'triangle_median'
        ratios = []
        for i in range(3):
            for j in range(i + 1, 3):
                len_src = np.linalg.norm(source_points[i] - source_points[j])
                len_tgt = np.linalg.norm(target_points[i] - target_points[j])
                if len_src > 1e-12:
                    ratios.append(len_tgt / len_src)
                else:
                    result['message'] = 'Warning: Extremely close point pair detected in source, ratio ignored.'
        if ratios:
            result['scale_factor'] = float(np.median(ratios))
        else:
            result['scale_factor'] = 1.0
            result['message'] = 'Error: No valid side length ratios could be computed.'
        logger.info(f"[Method] Triangle Side Length Median | [Factor] {result['scale_factor']:.6f}")
    elif N == 4:
        # ---------- 针对4个点：智能自适应法 ----------
        # 2.1 计算共面性指标
        v1 = source_points[1] - source_points[0]
        v2 = source_points[2] - source_points[0]
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        centroid = np.mean(source_points, axis=0)
        avg_distance = np.mean(np.linalg.norm(source_points - centroid, axis=1))

        if norm_normal < 1e-12 or avg_distance < 1e-12:
            # 前三点共线或所有点重合，高度退化，强制使用综合法
            coplanarity = 0.0
            result['method'] = 'quadrilateral_median_forced'
            result['message'] = 'Warning: Point cloud geometry highly degenerate (collinear or coincident), forcing quadrilateral median method.'
        else:
            d = np.abs(np.dot(normal, source_points[3] - source_points[0])) / norm_normal
            coplanarity = d / avg_distance
            result['coplanarity'] = float(coplanarity)

            # 2.2 根据阈值动态选择方法
            if coplanarity < coplanar_threshold:
                result['method'] = 'quadrilateral_median'
                result['message'] = f'Points near-coplanar (metric: {coplanarity:.3f} < {coplanar_threshold}), using quadrilateral median method.'
            else:
                result['method'] = 'tetrahedron_volume'
                result['message'] = f'Points well-distributed in 3D (metric: {coplanarity:.3f} >= {coplanar_threshold}), using tetrahedron volume method.'

        # 2.3 执行选定的方法
        if 'volume' in result['method']:
            # 四面体体积法（高精度路径）
            vol_src = np.abs(np.dot(v1, np.cross(v2, source_points[3] - source_points[0]))) / 6.0
            v1_t = target_points[1] - target_points[0]
            v2_t = target_points[2] - target_points[0]
            vol_tgt = np.abs(np.dot(v1_t, np.cross(v2_t, target_points[3] - target_points[0]))) / 6.0
            if vol_src > 1e-12:
                result['scale_factor'] = (vol_tgt / vol_src) ** (1 / 3.0)
            else:
                result['scale_factor'] = 1.0
                result['message'] += ' Volume too small, result may be unreliable.'
        else:
            # 四边形综合中位数法（高鲁棒性路径）
            ratios = []
            # 所有边（6条）
            for i in range(4):
                for j in range(i + 1, 4):
                    len_src = np.linalg.norm(source_points[i] - source_points[j])
                    len_tgt = np.linalg.norm(target_points[i] - target_points[j])
                    if len_src > 1e-12:
                        ratios.append(len_tgt / len_src)
            # 两条对角线
            for diag_pair in [(0, 2), (1, 3)]:
                len_src = np.linalg.norm(source_points[diag_pair[0]] - source_points[diag_pair[1]])
                len_tgt = np.linalg.norm(target_points[diag_pair[0]] - target_points[diag_pair[1]])
                if len_src > 1e-12:
                    ratios.append(len_tgt / len_src)
            if ratios:
                result['scale_factor'] = float(np.median(ratios))
                if len(ratios) < 8:
                    result['message'] += f' Some segments too short, only {len(ratios)} valid ratios used.'
            else:
                result['scale_factor'] = 1.0
                result['message'] = 'Error: Unable to compute any valid segment ratios.'
        logger.info(f"[Method] {result['method']} | [Factor] {result['scale_factor']:.6f} | [Coplanarity] {coplanarity if 'coplanarity' in result else 'N/A':.3f}")
    else:
        # ---------- 针对5个或更多点：对应点距离中位数法（最鲁棒） ----------
        result['method'] = 'point_pair_distance_median'
        # 计算每个点到点云质心的距离（利用所有对应点信息）
        centroid_src = np.mean(source_points, axis=0)
        centroid_tgt = np.mean(target_points, axis=0)
        dist_src = np.linalg.norm(source_points - centroid_src, axis=1)
        dist_tgt = np.linalg.norm(target_points - centroid_tgt, axis=1)
        # 过滤掉距离过小的点（避免噪声放大）
        valid = dist_src > np.percentile(dist_src, 25)
        if np.sum(valid) < max(3, N // 2):
            valid = np.ones(N, dtype=bool)  # 如果有效点太少，则使用全部
            result['message'] = 'Warning: Majority of points too clustered, using all points. Result may be noise-sensitive.'
        ratios = dist_tgt[valid] / dist_src[valid]
        result['scale_factor'] = float(np.median(ratios))
        logger.info(f"[Method] Point Pair Distance Median (N={N}) | [Factor] {result['scale_factor']:.6f}")
    # ===== 3. 最终验证与返回 =====
    # 快速验证：随机抽查几对点，确认尺度大致一致
    if result['scale_factor'] > 0 and 'message' not in result.get('message', ''):
        check_pairs = min(5, N)
        sample_ratios = []
        for i in range(1, check_pairs):
            d_src = np.linalg.norm(source_points[i] - source_points[0])
            d_tgt = np.linalg.norm(target_points[i] - target_points[0])
            if d_src > 1e-12:
                sample_ratios.append(d_tgt / d_src)
        if sample_ratios:
            median_sample = np.median(sample_ratios)
            if not np.isclose(median_sample, result['scale_factor'], rtol=0.1):  # 允许10%误差
                result['message'] += f' Verification warning: Sampled ratios ({median_sample:.3f}) differ significantly from final result. Check correspondences.'
    return result


def get_rescale_ratio(
    source:list,
    target:list
)->float:
    local_source=np.array(copy.deepcopy(source),dtype=np.float32)
    local_target=np.array(copy.deepcopy(target),dtype=np.float32)
    rescale_result=estimate_scale_robust_final(
        source_points=local_source,
        target_points=local_target
    )
    logger.warning(f"scale ratio finish with result : {rescale_result}")
    return float(rescale_result['scale_factor'])

def rescale_world_coord(
    rescale_ratio:float,
    cam_params:list,
    poles:list,
    pred_points:list
):
    """
    缩放 相机外参、重建T型杆、重建世界坐标系对齐角点
    """
    cam_params_bkp=copy.deepcopy(cam_params)
    poles_bkp=copy.deepcopy(poles)
    for i in range(len(cam_params)):
        cam_param=cam_params[i]
        cam_param['t']=(np.array(cam_param['t'],dtype=np.float32)*rescale_ratio).tolist()
    for i in range(len(poles)):
        pole=poles[i]
        for j in range(len(pole)):
            pole[j]=(np.array(pole[j],dtype=np.float32)*rescale_ratio).tolist()
    for i in range(len(pred_points)):
        for j in range(len(pred_points[i])):
            pred_points[i][j]*=rescale_ratio
        # pred_points[i]=(np.array(pred_points[i],dtype=np.float32)*rescale_ratio).tolist()
    logger.warning(f"rescale camera parameters, poles and pred_points with ratio : {rescale_ratio}")
    

if __name__ == "__main__":
    # 四点接近共面
    logger.info("--- Example 1: 4 Points Near-Coplanar ---")
    src1 = np.array([[0, 0, 0], [1, 0, 0.05], [1, 1, 0], [0, 1, 0]])  # 轻微偏离平面
    tgt1 = src1 * 2.0
    res1 = estimate_scale_robust_final(src1, tgt1, coplanar_threshold=0.05)
    logger.info(f"Result: {res1}\n")

    # 四点构成良好四面体
    logger.info("--- Example 2: 4 Points Forming Good Tetrahedron ---")
    src2 = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    tgt2 = src2 * 0.5
    res2 = estimate_scale_robust_final(src2, tgt2)
    logger.info(f"Result: {res2}\n")

    # 五个点
    logger.info("--- Example 3: 5 Points ---")
    np.random.seed(42)
    src3 = np.random.rand(5, 3) * 10
    tgt3 = src3 * 3.7 + np.random.randn(5, 3) * 0.02  # 缩放并加轻微噪声
    res3 = estimate_scale_robust_final(src3, tgt3)
    logger.info(f"Result: {res3}\n")

    # 应用尺度因子并准备ICP
    scale = res3['scale_factor']
    source_scaled = src3 * scale
    logger.info(f"Application: Multiply all source point coordinates by {scale:.6f}, then proceed to ICP registration.")