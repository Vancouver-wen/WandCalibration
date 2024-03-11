import os
import sys

import cv2
import numpy as np

from .normalizedImagePlane import get_undistort_points
from .multiViewTriangulate import normalized_pole_triangulate

def get_refine_pose(
        cam_num,
        pole_lists,
        intrinsics,
        pole_param,
        init_poses,
        save_path
    ):
    # 先用cv2.undistort 把像素坐标系转换成归一化图像坐标系
    undistorted_pole_lists=get_undistort_points(cam_num,pole_lists,intrinsics)
    # 在归一化平面上作三角测量
    normalized_pole_triangulate(
        cam_num=cam_num,
        normalized_pole_lists=undistorted_pole_lists,
        poses=init_poses
    )
    import pdb;pdb.set_trace()
    pass


if __name__=="__main__":
    pass