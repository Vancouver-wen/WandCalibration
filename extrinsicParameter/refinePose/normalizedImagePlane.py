import os
import sys

import cv2
import numpy as np

def verify_undistort_points(
        blobs,
        undistorted_blobs,
        intrinsic
    ):
    """
    证明 undistorted_blobs 确实在 归一化图像坐标系下
    """
    # blobs 在 像素坐标系 下
    # undistorted_blobs 在 归一化图像坐标系 下
    # blobs 应该与  K@undistorted_blobs  差不多
    K=np.array(intrinsic['K'])
    new_blobs=K@np.concatenate(
        (np.squeeze(undistorted_blobs),np.array([[1],[1],[1]])),
        axis=1
    ).T
    new_blobs=new_blobs[:2].T
    # 观察 new_blobs 与 blobs
    # import pdb;pdb.set_trace()

def get_undistort_points(
        cam_num,
        pole_lists,
        intrinsics,
    ):
    undistorted_pole_lists=[]
    for pole_list in pole_lists:
        undistorted_pole_list=[]
        assert cam_num==len(pole_list),"cam_num != len(pole_list)"
        assert cam_num==len(intrinsics),"cam_num != len(intrinsics)"
        for pole,intrinsic in list(zip(pole_list,intrinsics)):
            K=np.array(intrinsic.K)
            dist=np.squeeze(np.array(intrinsic.dist))
            if pole is None:
                undistorted_pole_list.append(None)
                continue
            blobs=pole[0]
            undistorted_blobs=cv2.undistortPoints(
                src=np.copy(blobs),
                cameraMatrix=K,
                distCoeffs=dist,
                # P=K
            )
            verify_undistort_points(
                blobs=blobs,
                undistorted_blobs=undistorted_blobs,
                intrinsic=intrinsic
            )
            undistorted_pole_list.append(undistorted_blobs)
        undistorted_pole_lists.append(undistorted_pole_list)
    return undistorted_pole_lists



if __name__=="__main__":
    pass