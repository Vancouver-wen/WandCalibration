import time
import os
import math
import pickle

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import glob
from joblib import Parallel,delayed
from loguru import logger
from tqdm import tqdm

from .blobDetection import get_cam_list,get_image_list,SimpleBlobDetection

class PoleDetection(SimpleBlobDetection):
    def __init__(
            self, 
            lineAngle=2,
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
        self.lineAngle=lineAngle
    
    def isLine(self, points) -> bool:
        if points[0] == points[1] or points[0] == points[2] or points[2] == points[1]:
            return False
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        x3, y3 = points[2][0], points[2][1]
        angle1 = math.atan((y2 - y1) / (x2 - x1)) if x2 != x1 else float('inf') 
        angle2 = math.atan((y3 - y2) / (x3 - x2)) if x2 != x3 else float('inf')
        angle1=angle1*180/math.pi
        angle2=angle2*180/math.pi
        if abs(angle1-angle2)<self.lineAngle: # 不超过lineAngle度算一条直线
            return True
        else:
            return False
    def __call__(self, frame,mask=None):
        # import pdb;pdb.set_trace()
        if mask is None:
            mask=np.zeros_like(frame)
        frame=self.frame_pre_process(frame)
        keypoints = self.detector.detect(frame)
        points=[keypoint.pt for keypoint in keypoints]
        masked_points=[]
        for point in points:
            if mask[int(point[1])][int(point[0])]>0: # 被 mask 掉的点
                continue
            masked_points.append(point)
        points=masked_points
        # print(points)
        if len(points)!=3:
            return None
        # 判断在一条直线
        is_line=self.isLine(points)
        if is_line==False:
            return None
        # 判断id
        points.sort()
        if points[1][0]-points[0][0]>points[2][0]-points[1][0]:
            points[0],points[2]=points[2],points[0]
        points=np.array(points)
        ids=np.array([[0],[1],[2]])
        return points,ids

def get_each_pole(
        frame_list,
        detectors,
        masks
    ):
    results=[]
    for frame_path,detector,mask in list(zip(frame_list,detectors,masks)):
        frame=cv2.imread(frame_path)
        result=detector(
            frame,
            mask
        )
        results.append(result)
    return results
        
def get_pole(
        cam_num,
        resolutions,
        poleBlobParam,
        image_path,
        masks,
        color="white"
    ):
    assert cam_num==len(resolutions),"camera num quantity is ambiguous"
    if not os.path.exists(os.path.join(image_path,'pole.pkl')):
        logger.info("detecting poles ...")
        detectors=[]
        for resolution in resolutions:
            # 生成 detector
            minAreaRatio=poleBlobParam.minAreaRatio
            maxAreaRatio=poleBlobParam.maxAreaRatio
            minArea=resolution[0]*resolution[1]*minAreaRatio
            maxArea=resolution[0]*resolution[1]*maxAreaRatio
            detector=PoleDetection(
                lineAngle = poleBlobParam.lineAngle,
                minThreshold  =   poleBlobParam.minThreshold,
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
                minCircularity  = poleBlobParam.minCircularity,
                filterByConvexity  =   True,
                minConvexity  = poleBlobParam.minConvexity,
                filterByInertia  =   True,
                minInertiaRatio  =  poleBlobParam.minInertiaRatio,
                color=color
            )
            detectors.append(detector)
        # 遍历 挥杆 图片
        frame_lists=get_cam_list(image_path,cam_num)
        output=Parallel(n_jobs=-1,backend="threading")(
            delayed(get_each_pole)(frame_list,detectors,masks)
            for frame_list in tqdm(frame_lists)
        )
        with open(os.path.join(image_path,'pole.pkl'),'wb') as f:
            pickle.dump(output,f)
        logger.info(f"pole detection complete! save result in {os.path.join(image_path,'pole.pkl')}")
    else:
        with open(os.path.join(image_path,'pole.pkl'),'rb') as f:
            output=pickle.load(f)
        logger.info(f"find and load existed pole.pkl successfully")
    # 统计每个视角下可用的pole数量
    pole_sum=[0 for _ in range(cam_num)]
    for result in output:
        for step,item in enumerate(result):
            if item is None:
                continue
            pole_sum[step]+=1
    logger.info(f"available pole number: {pole_sum}/{len(output)}")
    return output


if __name__=="__main__":
    # 根据分辨率计算 area
    resolution=(2856,1606)
    minAreaRatio=5.0e-6
    maxAreaRatio=6.5e-4
    minArea=resolution[0]*resolution[1]*minAreaRatio
    maxArea=resolution[0]*resolution[1]*maxAreaRatio

    myPoleDetection=PoleDetection(
        minThreshold  =   240,
        maxThreshold  =   255,
        thresholdStep = 1,
        filterByColor=True,
        blobColor=255,
        minRepeatability=2,
        minDistBetweenBlobs=10,
        filterByArea  =   True,
        minArea  =  minArea,
        maxArea=maxArea,
        filterByCircularity  =   True,
        minCircularity  =   0.2,
        filterByConvexity  =   True,
        minConvexity  =   0.2,
        filterByInertia  =   True,
        minInertiaRatio  =   0.1
    )
    # cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)

    root_path="../img_collect/blob"
    num_cam=6
    frame_lists=get_cam_list(root_path,num_cam)
    # 获取 masks
    masks_path="../img_collect/mask"
    masks=[cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in natsorted(glob.glob(os.path.join(masks_path,'*')))]
