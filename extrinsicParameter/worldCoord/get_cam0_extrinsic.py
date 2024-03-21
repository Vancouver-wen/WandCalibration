import os
import sys

import numpy as np
import cv2
import glob
from natsort import natsorted

from .clickPoint import click_point
from extrinsicParameter.poleDetection.wandDetection import get_wand
from .noIdReconstruction import no_id_reconstruct
from .get_id_with_distance import get_id_with_distance
from .solve_icp import solve_icp

def get_cam0_extrinsic(
        cam_num,
        cam_params,
        masks,
        image_path,
        world_coord_param,
        wand_blob_param
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
            cam_0_param=cam_params[0],
            image_path=image_empty,
            point_coordinates=world_coord_param.PointCoordinates
        )
    elif world_coord_param.type=="wand":
        # L型杆子
        # 检查是否有 wand 文件夹
        # blob detection
        # reconstruction without id info
        # get id info according to 3d distance
        # solve icp problem using open3d
        wand_folder=os.path.join(image_path,"wand")
        if not os.path.exists(wand_folder):
            assert False,f"can not find {wand_folder}"
        # import pdb;pdb.set_trace() # cam_params 有 image_size
        # ! 这里的  wand 不具备 id 信息 ! 
        wands=get_wand(
            cam_num=cam_num,
            resolutions=[cam_param['image_size'] for cam_param in cam_params],
            masks=masks,
            image_path=wand_folder,
            color=world_coord_param.color,
            wand_blob_param=wand_blob_param,
        )
        point_3ds=no_id_reconstruct(
            cam_num=cam_num,
            cam_params=cam_params,
            wands=wands,
        )
        point_3ds=get_id_with_distance(
            point_3ds=point_3ds,
            WandDefinition=world_coord_param['WandDefinition'],
        )
        R,t=solve_icp(
            target=point_3ds,
            source=world_coord_param['WandPointCoord']
        )
        cam0_R,cam0_t=R,t
    else:
        support_list=["point","wand"]
        raise NotImplementedError(f"we only support {support_list}")
    return cam0_R,cam0_t



if __name__=="__main__":
    pass