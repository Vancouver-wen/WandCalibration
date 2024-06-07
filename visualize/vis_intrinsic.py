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

def vis_intrinsic(
        cam_num,
        intrinsics,
        image_path,
        save_path
    ):
    image_lists=get_cam_list(
        root_path=image_path,
        cam_num=cam_num
    )
    image_list=image_lists[0]
    images=[cv2.imread(image_path) for image_path in image_list]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for step,(image,intrinsic) in enumerate(list(zip(images,intrinsics))):
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=np.array(intrinsic['K']),
            distCoeffs=np.array(intrinsic['dist']),
            imageSize=(image.shape[1],image.shape[0]),
            alpha=0, # 选择0，那么就意味着保留最少的黑边，使用1的话，保留全部像素，那么所有黑边都包含进去了
            newImgSize=(image.shape[1],image.shape[0])
        )
        undistort_image_alpha0=cv2.undistort(
            src=image,
            cameraMatrix=np.array(intrinsic['K']),
            distCoeffs=np.array(intrinsic['dist']),
            newCameraMatrix=new_camera_matrix
        )
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=np.array(intrinsic['K']),
            distCoeffs=np.array(intrinsic['dist']),
            imageSize=(image.shape[1],image.shape[0]),
            alpha=1, # 选择0，那么就意味着保留最少的黑边，使用1的话，保留全部像素，那么所有黑边都包含进去了
            newImgSize=(image.shape[1],image.shape[0])
        )
        undistort_image_alpha1=cv2.undistort(
            src=image,
            cameraMatrix=np.array(intrinsic['K']),
            distCoeffs=np.array(intrinsic['dist']),
            newCameraMatrix=new_camera_matrix
        )
        frame=show_multi_imgs(
            scale=1,
            imglist=[image,undistort_image_alpha0,undistort_image_alpha1],
            order=(1,3)
        )
        cv2.imwrite(os.path.join(save_path,f"cam{step+1}.jpg"),frame)
        