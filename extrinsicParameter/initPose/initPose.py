import os
import sys
import json

import numpy as np
from loguru import logger

from .mst import get_mst
from .recoverPose import get_unscaled_intrinsic,get_scaled_intrinsic,verify_scaled_intrinsic
from .integratePose import integrate_pose

def get_pole_len(pole_param):
    """
    the unit of length is meter
    """
    d=pole_param.d1+pole_param.d2
    if pole_param.length_unit=="mm":
        d=d/1000
    elif pole_param.length_unit=="m":
        d=d
    else:
        logger.error("the unit of length must be mm or m")
        assert False,"change length unit in pole config yaml file"
    return d

def get_pole_pairs(cam1_index,cam2_index,pole_lists):
    pole_pairs=[[],[]]
    for pole_list in pole_lists:
        temp=[pole_list[cam1_index],pole_list[cam2_index]]
        # print(temp)
        if None not in temp:
            pole_pairs[0].append(temp[0][0])
            pole_pairs[1].append(temp[1][0])
    return np.array(pole_pairs)

def get_init_pose(cam_num,pole_lists,intrinsics,pole_param,save_path):
    if os.path.exists(save_path):
        with open(save_path,'r') as f:
            poses=json.load(f)
        logger.info(f"find and load {save_path} successfully")
        return poses
    mst=get_mst(cam_num,pole_lists)
    logger.info(f"maximum spanning tree:{mst}")
    assert len(mst)+1==cam_num,f"mst_len:{len(mst)}\t cam_num:{cam_num}"
    pole_length=get_pole_len(pole_param)
    mst_extrinsic=[]
    for edge in mst:
        cam1_index=int(edge[1])
        cam2_index=int(edge[2])
        cam1_intrinsic=intrinsics[cam1_index]
        cam2_intrinsic=intrinsics[cam2_index]
        pole_pairs=get_pole_pairs(cam1_index,cam2_index,pole_lists)
        R21,t21=get_unscaled_intrinsic(pole_pairs,cam1_intrinsic,cam2_intrinsic)
        R21,t21=get_scaled_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic)
        # verify_scaled_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic)
        mst_extrinsic.append([
            cam1_index,
            cam2_index,
            R21,
            np.squeeze(t21)
        ])
    list_poses=integrate_pose(cam_num,mst_extrinsic)
    for pose in list_poses:
        pose['R']=pose['R'].tolist()
        pose['t']=pose['t'].tolist()
    poses=dict()
    for step,pose in enumerate(list_poses):
        poses[f'cam_{step}_0']=pose
    with open(save_path,'w') as f:
        json.dump(poses,f,indent=4)
    # import pdb;pdb.set_trace()
    return poses


if __name__=="__main__":
    pass