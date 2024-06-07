import os
import sys
import random

import numpy as np
import cv2
from joblib import Parallel,delayed
from tqdm import tqdm
from loguru import logger

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from utils.imageConcat import show_multi_imgs

def draw_point_2ds(frame,point_2ds):
    frame=cv2.line(
        img=frame,
        pt1=point_2ds[0].astype(np.int32),
        pt2=point_2ds[1].astype(np.int32),
        color=(0,255,0),
        thickness=1
    )
    frame=cv2.line(
        img=frame,
        pt1=point_2ds[1].astype(np.int32),
        pt2=point_2ds[2].astype(np.int32),
        color=(0,255,0),
        thickness=1
    )
    for point_2d in point_2ds:
        frame=cv2.circle(
            img=frame,
            center=point_2d.astype(np.int32),
            radius=3,
            color=(0,0,255),
            thickness=-1
        )
    
def draw_pole_list(
        frames,
        pole_list
    ):
    assert len(frames)==len(pole_list),f"len(frames)!=len(pole_list), check pole detection function"
    for frame,pole in list(zip(frames,pole_list)):
        if pole is None:
            continue
        point_2ds,indexs=pole
        draw_point_2ds(frame,point_2ds)

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
    logger.info(f"visualize pole spread result")
    Parallel(n_jobs=-1,backend="threading")(
        delayed(draw_pole_list)(frames,pole_list)
        for pole_list in tqdm(pole_lists)
    )
    for step,frame in enumerate(frames):
        cv2.imwrite(os.path.join(save_path,f'cam{step+1}.jpg'),frame)

if __name__=="__main__":
    pass