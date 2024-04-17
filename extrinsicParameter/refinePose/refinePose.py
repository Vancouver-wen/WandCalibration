import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

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
    start=time.time()
    for step in tqdm(range(iteration)):
        loss=myBoundAdjustment(
            line_weight=0,
            length_weight=0,
            reproj_weight=1.0
        )
        if step%10==0:
            logger.info(f"lr:{lrSchedular.get_last_lr()[-1]:.5f}\t loss:{loss.item():.5f}")
            output=myBoundAdjustment.get_dict() # 保存结果
            # import pdb;pdb.set_trace() # p intrinsics -> 有 image_size 
            verify_accuracy(
                camera_params=output['calibration'],
                pole_3ds=output['poles'],
                pole_lists=pole_lists,
                time_consume=time.time()-start
            )
            start=time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%100==0 and step!=0:
            lrSchedular.step()
        
    


if __name__=="__main__":
    pass