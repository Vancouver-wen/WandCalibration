import os
import sys

import cv2
import numpy as np
from loguru import logger

def get_available_pole_lists(pole_lists):
    available_pole_lists=[]
    for pole_list in pole_lists:
        mask=[pole is not None for pole in pole_list]
        if np.array(mask).sum()>=2:
            available_pole_lists.append(pole_list)
    return available_pole_lists

def verify_accuracy(
        camera_params,
        pole_3ds,
        pole_lists
    ):
    available_pole_lists=get_available_pole_lists(pole_lists)
    errors=[]
    diffs=[]
    for pole_3d,available_pole_list in list(zip(pole_3ds,available_pole_lists)):
        for available_pole,camera_param in list(zip(available_pole_list,camera_params)):
            if available_pole is None:
                continue
            available_pole=available_pole[0]
            for blob_2d,blob_3d in list(zip(available_pole,pole_3d)):
                expect_blob_2d,_=cv2.projectPoints(
                    objectPoints=np.expand_dims(np.array(blob_3d),axis=0),
                    rvec=np.array(camera_param['R']),
                    tvec=np.array(camera_param['t']),
                    cameraMatrix=np.array(camera_param['K']),
                    distCoeffs=np.array(camera_param['dist']),
                )
                expect_blob_2d=np.squeeze(expect_blob_2d)
                diff=blob_2d-expect_blob_2d
                diffs.append(np.abs(diff))
                error=np.linalg.norm(diff)
                errors.append(error)
    mean_error=np.mean(errors)
    mean_diff=np.mean(np.array(diffs),axis=0).tolist()
    logger.info(f"mean_pixel_error: {mean_error}    mean_coord_error; {mean_diff}")

if __name__=="__main__":
    pass