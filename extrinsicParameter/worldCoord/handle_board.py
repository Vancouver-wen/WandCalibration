import os
import sys
import json

import numpy as np
import cv2
import glob
from natsort import natsorted
from loguru import logger

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from extrinsicParameter.refinePose.multiViewTriangulate import easy_multi_view_triangulate
from utils.imageConcat import show_multi_imgs

def get_corner_map(
        image_path,
        cam_num,
        board_config,
        origin_point,
        cam_params
    ):
    frame_list=get_cam_list(image_path,cam_num)[0]
    # 生成 intrinsic.json
    board_type=board_config["type"].strip()
    # import pdb;pdb.set_trace()
    if board_type=="checkerboard":
        from intrinsicParameter.checkerboardCalibration.get_cam_calibration import IntrinsicCalibration
        height=board_config['height']
        width=board_config['width']
        square_length=board_config['square_length']
        intrinsicCalibrator=IntrinsicCalibration(height=height,width=width,image_path=image_path)
    elif board_type=="charucoboard":
        height=board_config['height']
        width=board_config['width']
        square_length=board_config['square_length']
        marker_length=board_config['marker_length']
        from intrinsicParameter.charucoboardCalibration.get_cam_calibration import IntrinsicCalibration
        intrinsicCalibrator=IntrinsicCalibration(height=height,width=width,square_length=square_length,markser_length=marker_length,image_path=image_path)
    else:
        support_list=[
            "checkerboard",
            "charucoboard"
        ]
        assert False,f"we only support {support_list}"
    corner_map=dict()
    corner_height,corner_width=height-1,width-1
    for i in range(corner_width):
        for j in range(corner_height):
            id=i*corner_height+j
            position=[i*square_length,j*square_length,0]
            corner_map.setdefault(id,dict())
            corner_map[id]['expect_point_3d']=np.array(position,dtype=np.float32)+np.squeeze(np.array(origin_point,dtype=np.float32))
    frames=[cv2.imread(frame_path) for frame_path in frame_list]
    for frame,frame_path,cam_param in list(zip(frames,frame_list,cam_params)):
        ret,ids,corners=intrinsicCalibrator.get_corners(frame_path)
        if ret is None:
            continue
        for id,corner in list(zip(np.squeeze(np.array(ids)),np.squeeze(np.array(corners)))):
            frame=cv2.circle(
                img=frame,
                center=corner.astype(np.int32),
                radius=3,
                color=(0,255,0),
                thickness=-1
            )
            frame=cv2.putText(
                img=frame,
                text=str(id),
                org=corner.astype(np.int32),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0,255,0),
                thickness=2
            )
            corner_map[id].setdefault('point_2ds',[])
            corner_map[id].setdefault('poses',[])
            corner_map[id]['point_2ds'].append(corner)
            corner_map[id]['poses'].append(cam_param)
    frame=show_multi_imgs(
        scale=1,
        imglist=frames,
        order=(int(cam_num/3+0.99),3)
    )
    return corner_map,frame

def triangulate_corner_map(corner_map):
    keys=natsorted(corner_map.keys())
    for key in keys:
        obj=corner_map[key]
        point_2ds=obj['point_2ds']
        poses=obj['poses']
        pred_point_3d=easy_multi_view_triangulate(
            point_2ds=point_2ds,
            poses=poses
        )
        obj['pred_point_3d']=np.squeeze(np.array(pred_point_3d))
        # 重投影
        pred_point_2ds=[]
        for camera_param in poses:
            pred_point_2d,_=cv2.projectPoints(
                objectPoints=np.expand_dims(np.array(pred_point_3d),axis=0),
                rvec=np.array(camera_param['R']),
                tvec=np.array(camera_param['t']),
                cameraMatrix=np.array(camera_param['K']),
                distCoeffs=np.array(camera_param['dist']),
            )
            pred_point_2ds.append(pred_point_2d)
        diff=np.squeeze(np.array(point_2ds))-np.squeeze(np.array(pred_point_2ds))
        diff=np.linalg.norm(diff,axis=1)
        logger.info(f"corner_id:{key} mean_pixel_error;{np.mean(diff).item()}")
    return corner_map

if __name__=="__main__":
    pass