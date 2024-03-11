import os
import sys

import cv2
import numpy as np

def integrate_pose(cam_num,mst_extrinsic):
    """
    找到一条从 第一个相机cam0/cam1 通向其他所有相机的路径
    """
    # poses 里面的参数都是相对于 cam0的
    # cam0的参数被初始化为单位阵R与零向量t
    poses=[None for _ in range(cam_num)]
    poses[0]={
        'R':np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ]),
        't':np.array(
            [0,0,0]
        )
    }
    for i in range(cam_num):
        for each_mst_extrinsic in mst_extrinsic:
            start=each_mst_extrinsic[0]
            end=each_mst_extrinsic[1]
            R_end_start=each_mst_extrinsic[2] # 表示从cam_start坐标系变换到cam_end坐标系
            t_end_start=each_mst_extrinsic[3] # 表示从cam_start坐标系变换到cam_end坐标系
            R_start_end=np.linalg.inv(R_end_start)
            t_start_end=-R_start_end@t_end_start
            # import pdb;pdb.set_trace()
            if poses[start] is not None and poses[end] is None:
                R_start_0=poses[start]['R']
                t_start_0=poses[start]['t']
                R_end_0=R_end_start@R_start_0
                t_end_0=R_end_start@t_start_0 + t_end_start
                poses[end]={
                    'R':R_end_0,
                    't':t_end_0
                }
            elif poses[end] is not None and poses[start] is None:
                R_end_0=poses[end]['R']
                t_end_0=poses[end]['t']
                R_start_0=R_start_end@R_end_0
                t_start_0=R_start_end@t_end_0 + t_start_end
                poses[start]={
                    'R':R_start_0,
                    't':t_start_0
                }
            else:
                pass
    return poses


if __name__=="__main__":
    pass