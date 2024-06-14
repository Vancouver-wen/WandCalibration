import os
import sys
import json

import cv2
import numpy as np
from .get_cam0_extrinsic import vis_point3ds
from .handle_labelme import vis_points

def adjust_camera_params(
        cam_0_R,
        cam_0_t,
        camera_params,
        save_path,
        image_path,
        world_coord_param
    ):
    """
    给定 cam0 的 R,t 
    返回所有 camera params 变换后的结果
    """
    wand_folder=os.path.join(image_path,"wand")
    if not os.path.exists(wand_folder):
        assert False,f"can not find {wand_folder}"
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
    if world_coord_param.type=="wand":
        point_3ds=np.array(world_coord_param['WandPointCoord'])
        vis_point3ds(
            image_path=wand_folder,
            cam_num=len(world_camera_params),
            cam_params=world_camera_params,
            point_3ds=point_3ds,
            save_folder=os.path.join(wand_folder,'vis_wand_points')
        )
    elif world_coord_param.type=="labelme":
        point_3ds=np.array(world_coord_param['PointCoordinates'])
        vis_points(
            point_3ds=point_3ds,
            image_path=wand_folder,
            cam_num=len(world_camera_params),
            cam_params=world_camera_params,
            save_folder=os.path.join(wand_folder,'vis_wand_points')
        )
    elif world_coord_param.type=="board":
        point_3ds=np.array(world_coord_param['BoardPointCoord'])
        vis_points(
            point_3ds=point_3ds,
            image_path=wand_folder,
            cam_num=len(world_camera_params),
            cam_params=world_camera_params,
            save_folder=os.path.join(wand_folder,'vis_wand_points')
        )
    with open(save_path,'w') as f:
        json.dump(world_camera_params,f)
    return world_camera_params

if __name__=="__main__":
    pass