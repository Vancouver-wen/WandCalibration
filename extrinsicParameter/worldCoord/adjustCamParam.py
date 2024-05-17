import os
import sys
import json

import cv2
import numpy as np

def adjust_camera_params(
        cam_0_R,
        cam_0_t,
        camera_params,
        save_path
    ):
    """
    给定 cam0 的 R,t 
    返回所有 camera params 变换后的结果
    """
    world_camera_params=[]
    for camera_param in camera_params:
        world_camera_param=dict()
        world_camera_param['image_size']=camera_param['image_size']
        world_camera_param['K']=camera_param['K']
        world_camera_param['dist']=camera_param['dist']
        R=np.array(camera_param['R'])@cam_0_R
        t=np.array(camera_param['R'])@cam_0_t+np.array(camera_param['t'])
        world_camera_param['R']=R.tolist()
        world_camera_param['t']=t.tolist()
        world_camera_params.append(world_camera_param)
    with open(save_path,'w') as f:
        json.dump(world_camera_params,f)
    return world_camera_params

if __name__=="__main__":
    pass