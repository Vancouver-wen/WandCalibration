import os
import sys
import json

import numpy as np
import cv2

def calculate_homo(R, t, K):
    pre_ex = R[:, :2]
    pos_ex = (t * 100).reshape(3, 1)
    pos_ex2 = np.concatenate((pre_ex, pos_ex), axis=1)
    
    homo_transform_inv = np.dot(K, pos_ex2)
    homo_transform = np.linalg.inv(homo_transform_inv)
    homo_transform = homo_transform / homo_transform[-1,-1]
    return homo_transform

def format_one_camera(camera_param):
    width,height=camera_param['image_size']
    Ko=np.array(camera_param['K'],dtype=np.float32)
    dist=np.array(camera_param['dist'],dtype=np.float32)
    R=np.array(camera_param['R'],dtype=np.float32)
    t=np.array(camera_param['t'],dtype=np.float32)
    output=dict()
    alpha=0
    retcal,validPixROI=cv2.getOptimalNewCameraMatrix(
        cameraMatrix=Ko,
        distCoeffs=dist,
        imageSize=[width,height],
        alpha=alpha # alpha=0，则去除所有黑色区域，alpha=1，则保留所有原始图像像素，其他值则得到介于两者之间的效果
    )
    H = calculate_homo(R,t,retcal)
    output['K']=retcal.flatten().tolist()
    output['Ko']=Ko.flatten().tolist()
    output['R']=R.flatten().tolist()
    output['T']=t.tolist()
    output['H']=H.flatten().tolist()
    output['distCoeff']=dist.tolist()
    output['imgSize']=[width,height]
    output['rectifyAlpha']=alpha
    output['validPixROI']={
        "x":validPixROI[0],
        "y":validPixROI[1],
        "width":validPixROI[2],
        "height":validPixROI[3],
    }
    return output

def format_cameras(camera_params):
    return [
        format_one_camera(camera_param)
        for camera_param in camera_params
    ]

def main():
    # source = json.load(open("./world_pose.json",'r'))
    with open("./source.json",'r') as f:
        source=json.load(f)
    results=[]
    for obj in source:
        results.append(
            format_one_camera(camera_param=obj)
        )
    with open("./test.json",'w') as f:
        json.dump(results,f)

if __name__=="__main__":
    main()