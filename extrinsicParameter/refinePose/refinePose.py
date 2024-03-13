import os
import sys

import cv2
import numpy as np
import torch

from .normalizedImagePlane import get_undistort_points
from .multiViewTriangulate import normalized_pole_triangulate
from .boundleAdjustment import BoundleAdjustment
from utils.verifyAccuracy import verify_accuracy

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
    # 在归一化平面上作三角测量获取3D点初值
    pole_3ds=normalized_pole_triangulate(
        cam_num=cam_num,
        normalized_pole_lists=undistorted_pole_lists,
        poses=init_poses,
        intrinsics=intrinsics
    )
    # 筛选用于 boundle adjustment 的 list
    # 要过滤到无法三角化的点,是因为 boundle adjustment的 优化参数中有 3d point
    mask=[pole_3d is not None for pole_3d in pole_3ds]
    masked_pole_lists=[
        item[1]
        for item in filter(lambda x:x[0],list(zip(mask,pole_lists)))
    ]
    masked_pole_3ds=[
        item[1]
        for item in filter(lambda x:x[0],list(zip(mask,pole_3ds)))
    ]
    # boundle adjustment
    myBoundAdjustment=BoundleAdjustment(
        pole_definition=pole_param,
        cam_num=cam_num,
        init_intrinsic=intrinsics,
        init_extrinsic=init_poses,
        image_num=np.array(mask).sum(),
        init_pole_3ds=masked_pole_3ds,
        detected_pole_2ds=masked_pole_lists,
        save_path=save_path
    )
    lr=5e-3 # lr=5e-3
    optimizer = torch.optim.Adam(
        params=myBoundAdjustment.parameters(),
        lr=lr
    )
    lrSchedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # ExponentialLR(optimizer, gamma=0.5)
    iteration = 1000 # iteration = 1000
    for step in range(iteration):
        loss=myBoundAdjustment()
        if step%10==0:
            output=myBoundAdjustment.get_dict() # 保存结果
            verify_accuracy(
                camera_params=output['calibration'],
                pole_3ds=output['poles'],
                pole_lists=pole_lists,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%100==0 and step!=0:
            lrSchedular.step()
        print({
            'step':f"{step}/{iteration}",
            'lr':lrSchedular.get_last_lr()[-1],
            'loss':loss.item()
        })
    


if __name__=="__main__":
    pass