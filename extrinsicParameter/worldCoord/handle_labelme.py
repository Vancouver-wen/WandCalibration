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

def get_labelme_json(json_path):
    with open(json_path,'r') as f:
        obj=json.load(f)
    shapes=obj['shapes']
    result=dict()
    for shape in shapes:
        label=int(shape['label'])
        point=np.squeeze(np.array(shape['points']))
        result[label]=point
    return result

def vis_objs(objs,image_path,cam_num):
    frame_list=get_cam_list(image_path,cam_num)[0]
    frames=[cv2.imread(frame_path) for frame_path in frame_list]
    for obj,frame in list(zip(objs,frames)):
        keys=obj.keys()
        for key in keys:
            point=obj[key]
            frame=cv2.circle(
                img=frame,
                center=point.astype(np.int32),
                radius=5,
                color=(0,255,0),
                thickness=-1
            )
            frame=cv2.putText(
                img=frame,
                text=f"{key}",
                org=point.astype(np.int32),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0,255,0),
                thickness=3
            )
    frame=show_multi_imgs(
        scale=1,
        imglist=frames,
        order=(int(cam_num/3+0.99),3)
    )
    return frame

def format_labelme_objs(objs,cam_params,point_coordinates):
    points=dict()
    for step,point_coordinate in enumerate(point_coordinates):
        points.setdefault(step,dict())
        points[step].setdefault('point_2ds',[])
        points[step].setdefault('cam_params',[])
        points[step]['expect_point_3d']=point_coordinate
    for obj,cam_param in list(zip(objs,cam_params)):
        keys=natsorted(obj.keys())
        for key in keys:
            point_2d=obj[key]
            points[key]['point_2ds'].append(point_2d.tolist())
            points[key]['cam_params'].append(cam_param)
    return points

def triangulate_point(key,point):
    poses=point['cam_params']
    point_2ds=point['point_2ds']
    point_3d=easy_multi_view_triangulate(point_2ds,poses)
    point['pred_point_3d']=point_3d
    # 计算 reproj pixel error
    pred_point_2ds=[]
    for pose in poses:
        camera_param=pose
        pred_point_2d,_=cv2.projectPoints(
            objectPoints=np.expand_dims(np.array(point_3d),axis=0),
            rvec=np.array(camera_param['R']),
            tvec=np.array(camera_param['t']),
            cameraMatrix=np.array(camera_param['K']),
            distCoeffs=np.array(camera_param['dist']),
        )
        pred_point_2ds.append(np.squeeze(pred_point_2d))
    pred_point_2ds=np.squeeze(np.array(pred_point_2ds))
    point['reproj_point_2ds']=pred_point_2ds
    diff=np.array(point_2ds)- pred_point_2ds
    diff=np.linalg.norm(diff,axis=1)
    logger.info(f"label:{key} reconstruction pixel error: {np.around(diff,3).tolist()}")
    return point_3d

def triangulate_points(points):
    keys=natsorted(points.keys())
    for key in keys:
        point_3d=triangulate_point(key,points[key])
    return points

def vis_points(
        point_3ds,
        image_path,
        cam_num,
        cam_params
    ):
    frame_list=get_cam_list(image_path,cam_num)[0]
    frames=[cv2.imread(frame_path) for frame_path in frame_list]
    for step,point_3d in enumerate(point_3ds):
        for frame,camera_param in list(zip(frames,cam_params)):
            pred_point_2d,_=cv2.projectPoints(
                objectPoints=np.expand_dims(np.array(point_3d),axis=0),
                rvec=np.array(camera_param['R']),
                tvec=np.array(camera_param['t']),
                cameraMatrix=np.array(camera_param['K']),
                distCoeffs=np.array(camera_param['dist']),
            )
            pred_point_2d=np.squeeze(pred_point_2d)
            frame=cv2.circle(
                img=frame,
                center=pred_point_2d.astype(np.int32),
                radius=5,
                color=(0,0,255),
                thickness=-1
            )
            frame=cv2.putText(
                img=frame,
                text=f"{step}",
                org=pred_point_2d.astype(np.int32),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0,0,255),
                thickness=3
            )
    frame=show_multi_imgs(
        scale=1,
        imglist=frames,
        order=(int(cam_num/3+0.99),3)
    )
    return frame

if __name__=="__main__":
    pass