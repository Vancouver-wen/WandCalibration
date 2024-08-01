import os
import sys
import random

import numpy as np
import cv2
from joblib import Parallel,delayed
from tqdm import tqdm
from loguru import logger
import cmap

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from utils.imageConcat import show_multi_imgs

def draw_point_2ds(frame,point_2ds,color):
    frame=cv2.line(
        img=frame,
        pt1=point_2ds[0].astype(np.int32),
        pt2=point_2ds[1].astype(np.int32),
        color=color.astype(np.uint8).tolist(),
        thickness=1
    )
    frame=cv2.line(
        img=frame,
        pt1=point_2ds[1].astype(np.int32),
        pt2=point_2ds[2].astype(np.int32),
        color=color.astype(np.uint8).tolist(),
        thickness=1
    )
    for point_2d in point_2ds:
        frame=cv2.circle(
            img=frame,
            center=point_2d.astype(np.int32),
            radius=3,
            color=color.astype(np.uint8).tolist(),
            thickness=-1
        )
    
def draw_pole_list(
        frames,
        pole_list,
        colors
    ):
    assert len(frames)==len(pole_list),f"len(frames)!=len(pole_list), check pole detection function"
    avail_num=np.array([pole is not None for pole in pole_list]).sum()
    color=colors[avail_num]
    for frame,pole in list(zip(frames,pole_list)):
        if pole is None:
            continue
        point_2ds,indexs=pole
        draw_point_2ds(frame,point_2ds,color)

def vis_spread(
        cam_num,
        image_path,
        pole_lists,
        save_path
    ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frame_list=get_cam_list(image_path,cam_num)[0]
    frames=[cv2.imread(frame_path) for frame_path in frame_list]
    color_bar = cmap.Colormap(["red","blue","green"]) # 以RGBA顺序给出
    colors=np.flip(color_bar(np.linspace(0,1,len(frames)+1))[:,:3]*255,axis=1) # 转换成BGR
    logger.info(f"visualize pole spread result")
    Parallel(n_jobs=-1,backend="threading")(
        delayed(draw_pole_list)(frames,pole_list,colors)
        for pole_list in tqdm(pole_lists)
    )
    for step,frame in enumerate(frames):
        cv2.imwrite(os.path.join(save_path,f'cam{step+1}.jpg'),frame)

def get_each_spread(
        step,
        frame_path,
        pole_lists
    ):
    frame=cv2.imread(frame_path)
    height,width,channel=frame.shape
    patch_num_h,patch_num_w=3,4
    spread_threshold=0.5
    point_num_threashold=300
    patch_h=height/patch_num_h
    patch_w=width/patch_num_w
    spread_map=np.zeros((patch_num_h,patch_num_w),dtype=np.int32)
    for pole_list in pole_lists:
        pole=pole_list[step]
        if pole is None:
            continue
        points=pole[0]
        for point in points:
            width,height=point
            spread_map[int(height/patch_h)][int(width/patch_w)]+=1
    spread_bool_map=spread_map>point_num_threashold
    spread=np.mean(spread_bool_map)
    if spread<spread_threshold:
        logger.warning(f"cam{step+1} spread too low!")
    return spread

def get_spread(
        cam_num,
        image_path,
        pole_lists,
    ):
    frame_list=get_cam_list(image_path,cam_num)[0]
    spreads=Parallel(n_jobs=-1,backend="threading")(
        delayed(get_each_spread)(step,frame_path,pole_lists)
        for step,frame_path in enumerate(frame_list)
    )
    return spreads

if __name__=="__main__":
    pass