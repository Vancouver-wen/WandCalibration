import time
import os

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from joblib import Parallel,delayed
from loguru import logger

from .blobDetection import get_cam_list,get_image_list,SimpleBlobDetection

def get_mask(
        cam_num,
        resolutions,
        maskBlobParam,
        image_path,
        mask_path,
        fastBlob,
        color
    )->list:
    assert cam_num==len(resolutions),"camera num quantity is ambiguous"
    exist_masks=[]
    for step in range(cam_num):
        exist_mask_path=os.path.join(mask_path,f"cam{step+1}.jpg")
        if os.path.exists(exist_mask_path):
            exist_mask=cv2.imread(exist_mask_path,cv2.IMREAD_GRAYSCALE)
            exist_masks.append(exist_mask)
    if len(exist_masks)==cam_num:
        logger.info("find exist mask and load ..")
        return exist_masks
    else:
        logger.info("generating mask ..")
    masks=[]
    expand_size=maskBlobParam.expandSize
    detectors=[]
    for resolution in resolutions:
        # 生成 mask
        mask=np.zeros(resolution,np.uint8).T.copy()
        masks.append(mask)
        # 生成 detector
        minAreaRatio=maskBlobParam.minAreaRatio
        maxAreaRatio=maskBlobParam.maxAreaRatio
        minArea=resolution[0]*resolution[1]*minAreaRatio
        maxArea=resolution[0]*resolution[1]*maxAreaRatio
        detector=SimpleBlobDetection(
            minThreshold  =   maskBlobParam.minThreshold,
            maxThreshold  =   255,
            thresholdStep = 1,
            filterByColor=True,
            blobColor=255,
            minRepeatability=2,
            minDistBetweenBlobs=10,
            filterByArea  =   True,
            minArea  =  minArea,
            maxArea = maxArea,
            filterByCircularity  =   True,
            minCircularity  =   0.01,
            filterByConvexity  =   True,
            minConvexity  =   0.01,
            filterByInertia  =   True,
            minInertiaRatio  =   0.01,
            fastBlob=fastBlob,
            color=color
        )
        detectors.append(detector)
    # 遍历 空场 图片
    frame_lists=get_cam_list(image_path,cam_num)
    for frame_list in tqdm(frame_lists):
        temps=Parallel(n_jobs=-1,backend="threading")(
            delayed(get_image_list)(frame_path,detector)
            for frame_path,detector in list(zip(frame_list,detectors))
        )
        temps=list(zip(*temps))
        all_keypoints=list(temps[0])
        # 补充mask
        for keypoints,mask in list(zip(all_keypoints,masks)):
            for keypoint in keypoints:
                pt=(int(keypoint.pt[0]),int(keypoint.pt[1]))
                # import pdb;pdb.set_trace()
                size=int(keypoint.size)+expand_size
                cv2.circle(mask,pt,size,(255,255,255),-1) 
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    for step,mask in enumerate(masks):
        cv2.imwrite(os.path.join(mask_path,f"cam{step+1}.jpg"),mask)
    return masks

if __name__=="__main__":
    # 根据分辨率计算 area
    resolution=(1606,2856)
    minAreaRatio=1.0e-7
    maxAreaRatio=1.0e-2
    minArea=resolution[0]*resolution[1]*minAreaRatio
    maxArea=resolution[0]*resolution[1]*maxAreaRatio

    # 生成mask的时候，阈值要放宽
    myBackgroundBlobDetection=SimpleBlobDetection(
        minThreshold  =   220,
        maxThreshold  =   256,
        thresholdStep = 1,
        filterByColor=True,
        blobColor=255,
        minRepeatability=2,
        minDistBetweenBlobs=10,
        filterByArea  =   True,
        minArea  =  minArea,
        maxArea=maxArea,
        filterByCircularity  =   True,
        minCircularity  =   0.01,
        filterByConvexity  =   True,
        minConvexity  =   0.01,
        filterByInertia  =   True,
        minInertiaRatio  =   0.01
    )

    root_path="./wtt/0306/empty"
    num_cam=6

    # 生成mask
    masks=[np.zeros(resolution,np.uint8) for _ in range(num_cam)]
    # cv2.namedWindow("mask",cv2.WINDOW_GUI_NORMAL)
    expand_size=3
    # import pdb;pdb.set_trace()

    frame_lists=get_cam_list(root_path,num_cam)

    for frame_list in frame_lists:
        temps=Parallel(n_jobs=len(frame_list),backend="threading")(
            delayed(get_image_list)(frame_path,myBackgroundBlobDetection)
            for frame_path in frame_list
        )
        temps=list(zip(*temps))
        # import pdb;pdb.set_trace()
        all_keypoints=list(temps[0])
        all_frames=list(temps[1])
        # 补充mask
        for keypoints,mask in list(zip(all_keypoints,masks)):
            for keypoint in keypoints:
                pt=(int(keypoint.pt[0]),int(keypoint.pt[1]))
                # import pdb;pdb.set_trace()
                size=int(keypoint.size)+expand_size
                cv2.circle(mask,pt,size,(255,255,255),-1) 
    # 保存masks图片
    # mask_save_path="./archery/0306/mask"
    mask_save_path="./wtt/0306/mask"
    for step,mask in enumerate(masks):
        cv2.imwrite(os.path.join(mask_save_path,f"cam{step+1}.jpg"),mask)
