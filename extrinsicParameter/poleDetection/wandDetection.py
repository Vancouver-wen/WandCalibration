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
from utils.imageConcat import show_multi_imgs

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
            fastBlob=True,
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
            fastBlob,
            color
        )
    
    def __call__(self, frame,mask=None):
        # import pdb;pdb.set_trace()
        if mask is None:
            mask=np.zeros_like(frame)
        frame=self.frame_pre_process(frame)
        # 用于debug
        # cv2.imwrite("./test.jpg",frame)
        # import pdb;pdb.set_trace()
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
    logger.info(f"merge all wand detection to one")
    results=[None for _ in range(cam_num)]
    for frame_index,wands in enumerate(tqdm(output)):
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
                            atol=5 # 如果误差不超过 5 个像素, 就认为是相同
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
        wand_blob_param,
        fastBlob
    ):
    assert cam_num==len(resolutions),"camera num quantity is ambiguous"
    pickle_path=os.path.join(image_path,'wand.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path,'rb') as f:
            output=pickle.load(f)
            logger.info(f"find exist {pickle_path} and load successfully")
    else:
        logger.info(f"not find exist {pickle_path}, detecting wands ...")
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
                fastBlob=fastBlob,
                color=color
            )
            detectors.append(detector)
        # 遍历 L型直角杆 图片
        frame_lists=get_cam_list(image_path,cam_num)
        output=Parallel(n_jobs=-1,backend="threading")(
                delayed(get_each_wand)(frame_list,detectors,masks)
                for frame_list in tqdm(frame_lists)
            )
        with open(pickle_path,'wb') as f:
            pickle.dump(output,f)
        logger.info(f"dump wand detection result in {pickle_path}")
    output=merge_output(cam_num,output)
    output_num=0
    for item in output:
        for _ in item:
            output_num+=1
    # import pdb;pdb.set_trace()
    if output_num==3*cam_num:
        logger.info(f"共检测出{output_num}个点,应为3*cam_num: {3*cam_num}")
    else:
        logger.warning(f"共检测出{output_num}个点,应为3*cam_num: {3*cam_num}")
    # draw_output(output)
    return output

def draw_output(output):
    cam0_path="/home/wenzihao/Desktop/WandCalibration/imageCollectRedWtt/wand/cam1/0-1711025876107726901.jpeg"
    cam1_path="/home/wenzihao/Desktop/WandCalibration/imageCollectRedWtt/wand/cam2/0-1711025876107726901.jpeg"
    cam2_path="/home/wenzihao/Desktop/WandCalibration/imageCollectRedWtt/wand/cam3/0-1711025876107726901.jpeg"
    cam3_path="/home/wenzihao/Desktop/WandCalibration/imageCollectRedWtt/wand/cam4/0-1711025876107726901.jpeg"
    cam0_image=cv2.imread(cam0_path)
    cam1_image=cv2.imread(cam1_path)
    cam2_image=cv2.imread(cam2_path)
    cam3_image=cv2.imread(cam3_path)
    cam_images=[
        cam0_image,
        cam1_image,
        cam2_image,
        cam3_image
    ]
    wand_vis_folder="/home/wenzihao/Desktop/WandCalibration/imageCollectRedWtt/wand_vis"
    for cam_index,points in enumerate(output):
        for point in points:
            cv2.circle(
                img=cam_images[cam_index],
                center=point.astype(np.int64).tolist(),
                radius=3,
                color=(0,255,0),
                thickness=-1
            )
    frame=show_multi_imgs(scale=1,imglist=cam_images,order=(2,2))
    save_path=cv2.imwrite(
        filename="/home/wenzihao/Desktop/WandCalibration/imageCollectRedWtt/wand/vis.jpg",
        img=frame
    )
    
