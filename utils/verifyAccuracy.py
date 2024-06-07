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
        pole_lists,
        time_consume=None
    ):
    available_pole_lists=get_available_pole_lists(pole_lists)
    errors=[]
    diffs=[]
    each_errors=[[] for _ in camera_params]
    for pole_3d,available_pole_list in list(zip(pole_3ds,available_pole_lists)):
        for step,(available_pole,camera_param) in enumerate(list(zip(available_pole_list,camera_params))):
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
                each_errors[step].append(error)
    mean_error=np.mean(errors)
    mean_diff=np.mean(np.array(diffs),axis=0).tolist()
    for i in range(len(each_errors)):
        each_errors[i]=np.around(np.mean(np.array(each_errors[i])),2)
    if time_consume is None:
        logger.info(f"mean_pixel_error: {mean_error:.2f}    mean_coord_error: {mean_diff}")
    else:
        mean_diff = [float('{:.2f}'.format(i)) for i in mean_diff]
        logger.info(f"mean_pixel_error: {mean_error:.2f}    mean_coord_error: {mean_diff}   second_consume: {time_consume:.2f}")
    logger.info(f"each_camera_pixel_error:{each_errors}")
    return mean_error

def vis_accuracy(
        
    ):
    pass

if __name__=="__main__":
    pass