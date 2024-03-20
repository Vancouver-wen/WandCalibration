import os
import sys

import numpy as np
import cv2
import glob
from natsort import natsorted

from .clickPoint import click_point

def get_cam0_extrinsic(
        cam_0_param,
        image_path,
        world_coord_param
    ):
    """
    给出 cam0 在 world coordinate 下的 Rotation 和 tran
    """
    if world_coord_param.type=="point":
        # 调用cam0的empty的第一张图片
        # 交互式给出 4 个点
        # yaml文件给出四个点的三维坐标
        image_empty=natsorted(glob.glob(os.path.join(image_path,"empty","cam1","*")))[0]
        cam0_R,cam0_t=click_point(
            cam_0_param=cam_0_param,
            image_path=image_empty,
            point_coordinates=world_coord_param.PointCoordinates
        )
        pass
    elif world_coord_param.type=="wand":
        # L型杆子
        # 检查是否有 wand 文件夹
        # blob detection
        # reconstruction with out id info
        # get id info according to 3d distance
        # solve icp problem using 
        pass # TODO
    else:
        support_list=["point","wand"]
        raise NotImplementedError(f"we only support {support_list}")
    return cam0_R,cam0_t



if __name__=="__main__":
    pass