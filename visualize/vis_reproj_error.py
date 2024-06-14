import os
import sys
import random
from itertools import compress
import copy

import cv2
from tqdm import tqdm
import numpy as np
from loguru import logger
from joblib import Parallel,delayed

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from extrinsicParameter.refinePose.normalizedImagePlane import get_undistort_points
from extrinsicParameter.refinePose.multiViewTriangulate import multi_view_triangulate
from utils.imageConcat import show_multi_imgs

def point_2ds_list_convert(point_2ds_list):
    result=[[] for _ in range(3)]
    for pole_2ds in point_2ds_list:
        assert len(pole_2ds)==3,"an avail image must contain 3 marker"
        for step,pole_2d in enumerate(pole_2ds):
            result[step].append(pole_2d)
    result=np.array(result)
    return np.array(result)
            

def vis_each_reproj_error(
        step,
        frame_list,
        undistort_pole_list,
        reproj_folder,
        camera_params
    ):
    save_folder=os.path.join(reproj_folder,f"{step:06d}")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    mask=[undistort_pole is not None for undistort_pole in undistort_pole_list]
    if np.array(mask).sum()<2:
        for cam_index,frame_path in enumerate(frame_list):
            frame=cv2.imread(frame_path) 
            cv2.imwrite(os.path.join(save_folder,f"cam{cam_index+1}.jpg"),frame)
        return
    masked_pole_list=list(compress(undistort_pole_list,mask))
    point_2ds_list=np.squeeze(np.array(masked_pole_list),axis=2)
    point_2ds_list=point_2ds_list_convert(point_2ds_list)
    masked_camera_params=list(compress(camera_params,mask))
    point_3ds=[]
    for point_2ds in point_2ds_list:
        point_3d=multi_view_triangulate(
            point_2ds=point_2ds,
            poses=masked_camera_params
        )
        point_3ds.append(point_3d)
    for camera_index,(camera_param,frame_path) in enumerate(list(zip(camera_params,frame_list))):
        frame=cv2.imread(frame_path)
        origin=copy.deepcopy(frame)
        for id,point_3d in enumerate(point_3ds): # point_3ds 可能含有None
            if point_3d is None:
                continue
            point_2d,_=cv2.projectPoints(
                objectPoints=np.expand_dims(np.array(point_3d),axis=0),
                rvec=np.array(camera_param['R']),
                tvec=np.array(camera_param['t']),
                cameraMatrix=np.array(camera_param['K']),
                distCoeffs=np.array(camera_param['dist']),
            )
            point_2d=np.squeeze(point_2d)
            frame=cv2.circle(
                img=frame,
                center=point_2d.astype(np.int32),
                radius=10,
                color=(0,0,255),
                thickness=-1
            )
            frame=cv2.putText(
                img=frame,
                text=str(id),
                org=point_2d.astype(np.int32),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0,0,255),
                thickness=2
            )
        frame=cv2.addWeighted(origin,0.5,frame,0.5,0)
        cv2.imwrite(os.path.join(save_folder,f"cam{camera_index+1}.jpg"),frame)

def vis_reproj_error(
        cam_num,
        pole_lists,
        camera_params,
        image_path,
        vis_num=300,
        vis_folder="vis_reproj"
    ):
    # 遍历 挥杆 图片
    frame_lists=get_cam_list(image_path,cam_num)
    undistort_pole_lists=get_undistort_points(cam_num=cam_num,pole_lists=pole_lists,intrinsics=camera_params)
    iteration=random.sample(list(zip(frame_lists,undistort_pole_lists)),vis_num)
    reproj_folder=os.path.join(image_path,vis_folder)
    if not os.path.exists(reproj_folder):
        os.mkdir(reproj_folder)
    logger.info(f"visualize reprojection error in folder:{vis_folder}")
    Parallel(n_jobs=-1,backend="threading")(
        delayed(vis_each_reproj_error)(step,frame_list,undistort_pole_list,reproj_folder,camera_params)
        for step,(frame_list,undistort_pole_list) in enumerate(tqdm(iteration))
    )
        
        
        


if __name__=="__main__":
    pass