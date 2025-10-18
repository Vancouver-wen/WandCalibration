import os
import sys

import cv2
import numpy as np

def get_unscaled_intrinsic(pole_pairs,cam1_intrinsic,cam2_intrinsic):
    blob_pairs=pole_pairs.reshape(2,-1,2)
    K1=np.array(cam1_intrinsic.K)
    K2=np.array(cam2_intrinsic.K)
    dist1=np.squeeze(np.array(cam1_intrinsic.dist))
    dist2=np.squeeze(np.array(cam2_intrinsic.dist))
    retval, E, R, t, mask=cv2.recoverPose(
        # revocerPose的重载函数非常多
        points1=blob_pairs[0],
        points2=blob_pairs[1],
        cameraMatrix1=K1,
        cameraMatrix2=K2,
        distCoeffs1=dist1,
        distCoeffs2=dist2
    )
    return R,t

def get_scale_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic,state="init"):
    """
    两个相机的三角化，求出杆子的平均长度
    根据杆子的平均长度,对t进行scale
    """
    # cv2.triangulatePoints的流程
    # https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate
    K1=np.array(cam1_intrinsic.K)
    dist1=np.squeeze(np.array(cam1_intrinsic.dist))
    cam1_Rt=np.array(
        object=[[1,1e-5,1e-5,1e-5],
        [1e-5,1,1e-5,1e-5],
        [1e-5,1e-5,1,1e-5]],
        dtype=np.float64
    )
    cam1_proj=K1@cam1_Rt

    K2=np.array(cam2_intrinsic.K)
    dist2=np.squeeze(np.array(cam2_intrinsic.dist))
    cam2_Rt=np.concatenate((R21,t21),axis=1)
    cam2_proj=K2@cam2_Rt

    blob_pairs=pole_pairs.reshape(2,-1,2)
    cam1_blobs=cv2.undistortPoints(
        src=blob_pairs[0],
        cameraMatrix=K1,
        distCoeffs=dist1,
        R=K1
    )
    cam2_blobs=cv2.undistortPoints(
        src=blob_pairs[1],
        cameraMatrix=K2,
        distCoeffs=dist2,
        R=K2
    )
    blob3ds=cv2.triangulatePoints(
        projMatr1=cam1_proj,
        projMatr2=cam2_proj,
        projPoints1=np.squeeze(cam1_blobs).T,
        projPoints2=np.squeeze(cam2_blobs).T
    )
    blob3ds=blob3ds[:3]/np.repeat(np.expand_dims(blob3ds[3],axis=0),3,axis=0)
    blob3ds=blob3ds.T.reshape(-1,3,3)
    d_average=0
    for blob3d in blob3ds:
        d1=np.linalg.norm(blob3d[0]-blob3d[1])
        d2=np.linalg.norm(blob3d[1]-blob3d[2])
        # d1/d2 的比例应该与标定杆保持一致
        d=d1+d2
        if state == "init":
            pass
        else:
            print({'state':state,'d1':d1,'d2':d2})
        # 从这个地方发现, 误差实际上是非常大的
        d_average+=d/len(blob3ds)
    t_ratio=pole_length/d_average
    return t_ratio

def get_scaled_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic):
    t_ratio=get_scale_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic,"init")
    return R21,t21*t_ratio,t_ratio

def verify_scaled_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic):
    t_ratio=get_scale_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic,"verify_init")
    import pdb;pdb.set_trace()

if __name__=="__main__":
    pass