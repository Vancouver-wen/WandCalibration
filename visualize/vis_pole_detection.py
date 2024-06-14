import os
import sys
import random
import copy

import numpy as np
import cv2
from joblib import Parallel,delayed
from tqdm import tqdm
from loguru import logger

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from utils.imageConcat import show_multi_imgs

def vis_one_frame(
        step,
        pole,
        frame_path,
        save_folder,
        color,
        min_threshold
    ):
    frame=cv2.imread(frame_path)
    origin=copy.deepcopy(frame)
    if pole is None:
        pass
    else:
        corners,ids=pole
        for corner,id in list(zip(corners,ids)):
            point=corner.astype(np.int32)
            # import pdb;pdb.set_trace()
            frame=cv2.circle(img=frame,center=point,radius=10,color=(0,255,0),thickness=-1)
            frame=cv2.putText(frame,str(id.item()),point,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
    frame=cv2.addWeighted(origin,0.5,frame,0.5,0)
    # 与Simple Blob Detection 的 frame_pre_process 函数处理保持一致
    if color=="white":
        middle=cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)
    elif color=="red":
        red_channel=origin[:,:,2].astype(np.int64)-np.maximum(origin[:,:,0].astype(np.int64),origin[:,:,1].astype(np.int64))
        red_channel=np.clip(red_channel,a_min=0,a_max=255).astype(np.uint8)
        # 形态学
        kernel = np.ones((10,10),np.uint8)
        red_channel=cv2.morphologyEx(red_channel, cv2.MORPH_CLOSE, kernel,iterations=2)
        middle=red_channel
    else:
        support_list=["white","red"]
        raise NotImplementedError(f"we only support {support_list}")
    middle=cv2.cvtColor(middle,cv2.COLOR_GRAY2BGR)
    ret,threshold=cv2.threshold(middle,min_threshold,255,cv2.THRESH_BINARY)
    result=show_multi_imgs(scale=1,imglist=[frame,middle,threshold],order=(3,1))
    cv2.imwrite(os.path.join(save_folder,f'cam{step+1}.jpg'),result)

def vis_each_pole(
        save_folder,
        pole_list,
        frame_list,
        color,
        threshold
    ):
    assert len(pole_list)==len(frame_list),"len(pole_list) != len(frame_list)"
    for step,(pole,frame_path) in enumerate(list(zip(pole_list,frame_list))):
        vis_one_frame(step,pole,frame_path,save_folder,color,threshold)

def save_each_pole(
        pole_list,
        frame_list,
        debug_path,
        step,
        color,
        threshold
    ):
    save_folder=os.path.join(debug_path,f"{step:06d}")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    vis_each_pole(
        save_folder,
        pole_list,
        frame_list,
        color,
        threshold
    )

def vis_pole(
        cam_num,
        image_path,
        pole_lists,
        vis_num=300,
        color='red',
        threshold=100
    ):
    debug_path=os.path.join(image_path,"vis_poles")
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
    frame_lists=get_cam_list(image_path,cam_num)
    assert len(pole_lists)==len(frame_lists),"len(pole_lists) != len(frame_lists)"
    iteration=random.sample(list(zip(pole_lists,frame_lists)),vis_num)
    logger.info(f"visualize pole detection result ")
    Parallel(n_jobs=-1,backend="threading")(
        delayed(save_each_pole)(pole_list,frame_list,debug_path,step,color,threshold)
        for step,(pole_list,frame_list) in enumerate(tqdm(iteration))
    )


if __name__=="__main__":
    pass