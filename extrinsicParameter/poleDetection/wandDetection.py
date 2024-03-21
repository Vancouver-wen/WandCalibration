import time
import os
import math
import pickle
import copy

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import glob
from joblib import Parallel,delayed
from loguru import logger
from tqdm import tqdm

from .blobDetection import get_cam_list,get_image_list,SimpleBlobDetection

class WandDetection(SimpleBlobDetection):
    def __init__(
            self, 
            minThreshold=235, 
            maxThreshold=255, 
            thresholdStep=1, 
            filterByColor=True, 
            blobColor=255, 
            minRepeatability=2, 
            minDistBetweenBlobs=10, 
            filterByArea=True, 
            minArea=10, 
            maxArea=50, 
            filterByCircularity=True, 
            minCircularity=0.8, 
            filterByConvexity=True, 
            minConvexity=0.4, 
            filterByInertia=True, 
            minInertiaRatio=0.1,
            color="white"
        ):
        super().__init__(
            minThreshold, 
            maxThreshold, 
            thresholdStep, 
            filterByColor, 
            blobColor, 
            minRepeatability, 
            minDistBetweenBlobs, 
            filterByArea, 
            minArea, 
            maxArea, 
            filterByCircularity, 
            minCircularity, 
            filterByConvexity, 
            minConvexity, 
            filterByInertia, 
            minInertiaRatio,
            color
        )
    
    def __call__(self, frame,mask=None):
        # import pdb;pdb.set_trace()
        if mask is None:
            mask=np.zeros_like(frame)
        frame=self.frame_pre_process(frame)
        keypoints = self.detector.detect(frame)
        points=[keypoint.pt for keypoint in keypoints]
        # 去除被 mask 掉的点
        masked_points=[]
        for point in points:
            if mask[int(point[1])][int(point[0])]>0: # 被 mask 掉的点
                continue
            masked_points.append(point)
        points=masked_points
        points=np.array(points)
        return points
    
def get_each_wand(frame_list,detectors,masks):
    results=[]
    for frame_path,detector,mask in list(zip(frame_list,detectors,masks)):
        frame=cv2.imread(frame_path)
        result=detector(
            frame,
            mask
        )
        results.append(result)
    return results

def merge_output(cam_num,output):
    results=[None for _ in range(cam_num)]
    for frame_index,wands in enumerate(output):
        for cam_index,wand in enumerate(wands):
            if results[cam_index] is None:
                results[cam_index]=copy.deepcopy(wand)
            else:
                for point_new in wand:
                    match=False
                    for i,point_exist in enumerate(results[cam_index]):
                        equal=np.allclose(
                            a=point_exist,
                            b=point_new,
                            atol=3 # 如果误差不超过 3 个像素, 就认为是相同
                        )
                        if equal:
                            match=True
                            # print("equal")
                            # import pdb;pdb.set_trace()
                            results[cam_index][i]=(frame_index+1)/len(output)*point_exist+(len(output)-1-frame_index)/len(output)*point_new
                    if match==False:
                        logger.warning(f"find unmatch point in wand folder")
                        results[cam_index]=results[cam_index].tolist()
                        results[cam_index].append(point_new.tolist())
                        results[cam_index]=np.array(results[cam_index])
    # import pdb;pdb.set_trace()
    return results
        
def get_wand(
        cam_num,
        resolutions,
        masks,
        image_path,
        color,
        wand_blob_param
    ):
    assert cam_num==len(resolutions),"camera num quantity is ambiguous"
    logger.info("detecting wands ...")
    detectors=[]
    for resolution in resolutions:
        # 生成 detector
        minAreaRatio=wand_blob_param.minAreaRatio
        maxAreaRatio=wand_blob_param.maxAreaRatio
        minArea=resolution[0]*resolution[1]*minAreaRatio
        maxArea=resolution[0]*resolution[1]*maxAreaRatio
        detector=WandDetection(
            minThreshold  =   wand_blob_param.minThreshold,
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
            minCircularity  = wand_blob_param.minCircularity,
            filterByConvexity  =   True,
            minConvexity  = wand_blob_param.minConvexity,
            filterByInertia  =   True,
            minInertiaRatio  =  wand_blob_param.minInertiaRatio,
            color=color
        )
        detectors.append(detector)
    # 遍历 L型直角杆 图片
    frame_lists=get_cam_list(image_path,cam_num)
    output=Parallel(n_jobs=-1,backend="threading")(
            delayed(get_each_wand)(frame_list,detectors,masks)
            for frame_list in tqdm(frame_lists)
        )
    output=merge_output(cam_num,output)
    return output