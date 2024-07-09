import os
import sys
import math

import numpy as np
import cv2
from sklearn.cluster import DBSCAN,KMeans

from extrinsicParameter.refinePose.multiViewTriangulate import multi_view_triangulate
from .cluster import sort_cluster,mcp_cluster

def vis_epipolar_line():
    pass # TODO

def epipolar_distance(line,point):
    A,B,C=line[0],line[1],line[2]
    x,y=point[0],point[1]
    distance=abs(A*x+B*y+C)/math.sqrt(A**2+B**2)
    return distance

def symmetrix_epipolar_distance(point2,F,point1):
    # 对极约束 p2.T@F@p1=0 p(3,1) F(3,3)
    # 点在直线上
    line2=np.squeeze(cv2.computeCorrespondEpilines(
        points=np.expand_dims(point1,axis=0),
        whichImage=1,
        F=F
    ))
    line1=np.squeeze(cv2.computeCorrespondEpilines(
        points=np.expand_dims(point2,axis=0),
        whichImage=2,
        F=F
    ))
    # import pdb;pdb.set_trace()
    distance1=epipolar_distance(line1,point1)
    distance2=epipolar_distance(line2,point2)
    distance=distance1+distance2
    return distance

def get_fundamental_matrix(camera2,camera1):
    "从cam2到cam1的 Fundamental Matrix"
    K1,R1,t1=np.array(camera1['K']),np.array(camera1['R']),np.array(camera1['t'])
    K2,R2,t2=np.array(camera2['K']),np.array(camera2['R']),np.array(camera2['t'])
    R21=R2@np.linalg.inv(R1)
    t21=(t2.T-R21@t1.T).T
    t21_antisymmetric=np.array([
        [      0,  -t21[2],    t21[1]],
        [ t21[2],        0,   -t21[0]],
        [-t21[1],   t21[0],         0]
    ])
    F21=np.linalg.inv(K2).T@t21_antisymmetric@R21@np.linalg.inv(K1)
    # import pdb;pdb.set_trace()
    return F21

def get_undistort_points(point1,point2,camera1,camera2):
    """
    返回归一化平面坐标系上的一点
    """
    point1=cv2.undistortPoints(
        src=np.expand_dims(np.array(point1),axis=0),
        cameraMatrix=np.array(camera1['K']),
        distCoeffs=np.squeeze(np.array(camera1['dist'])),
        P=np.array(camera1['K'])
    )
    point2=cv2.undistortPoints(
        src=np.expand_dims(np.array(point2),axis=0),
        cameraMatrix=np.array(camera2['K']),
        distCoeffs=np.squeeze(np.array(camera2['dist'])),
        P=np.array(camera2['K'])
    )
    # import pdb;pdb.set_trace()
    return np.squeeze(point1).tolist(),np.squeeze(point2).tolist()

def get_normalized_points(point1,point2,camera1,camera2):
    """
    返回归一化平面坐标系上的一点
    """
    point1=cv2.undistortPoints(
        src=np.expand_dims(np.array(point1),axis=0),
        cameraMatrix=np.array(camera1['K']),
        distCoeffs=np.squeeze(np.array(camera1['dist'])),
        # P=np.array(camera1['K'])
    )
    point2=cv2.undistortPoints(
        src=np.expand_dims(np.array(point2),axis=0),
        cameraMatrix=np.array(camera2['K']),
        distCoeffs=np.squeeze(np.array(camera2['dist'])),
        # P=np.array(camera2['K'])
    )
    # import pdb;pdb.set_trace()
    return np.squeeze(point1).tolist(),np.squeeze(point2).tolist()

def get_cost(
        point1,
        point2,
        cam_params
    ):
    if point1['cam_index']==point2['cam_index']:
        # 同一个相机内的两个点不可能在同一个簇内 -> cost 很大
        return 1e8
    camera1=cam_params[point1['cam_index']]
    camera2=cam_params[point2['cam_index']]
    # 如果 point1 与 point2 属于两个相机 -> 计算对称极限距离
    point1['point_undistorted'],point2['point_undistorted']=get_undistort_points(
        point1=point1['point_2d'],
        point2=point2['point_2d'],
        camera1=camera1,
        camera2=camera2
    )
    point1['point_normalized'],point2['point_normalized']=get_normalized_points(
        point1=point1['point_2d'],
        point2=point2['point_2d'],
        camera1=camera1,
        camera2=camera2
    )
    F=get_fundamental_matrix(
        camera2=camera2,
        camera1=camera1
    )
    distance=symmetrix_epipolar_distance(
        point2=np.array(point2['point_undistorted']),
        F=F,
        point1=np.array(point1['point_undistorted'])
    )
    # import pdb;pdb.set_trace()
    return distance

def check_clump_legality(clump):
    camera_indexs=[
        point['camera_index']
        for point in clump
    ]
    if len(set(camera_indexs))!=len(camera_indexs):
        assert False,"cluster algorithm fail! encounting conflicts! "

def triangulate_clump(clump):
    # 检查 clump 合法性
    # import pdb;pdb.set_trace()
    check_clump_legality(clump)
    # 构建三角化输入
    point_2ds=[]
    poses=[]
    for point in clump:
        camera_param=point['camera_param']
        point_2d=point['point_2d']
        point_2ds.append(point_2d)
        poses.append(camera_param)
    point_3d=multi_view_triangulate(
        point_2ds=point_2ds,
        poses=poses,
    )
    return point_3d

def no_id_reconstruct(
        cam_num,
        cam_params,
        wands,  
        mode:str
    ):
    """
    先构建一个n*n的矩阵, n为所有相机所有点的个数
    """
    points=dict()
    point_index=0
    init_cluster=None
    for cam_index,wand in enumerate(wands):
        if len(wand)==3 and init_cluster is None: # 说明正好有3个点,属于三个类
            init_cluster=[point_index+i for i in range(3)]
        for point in wand:
            points[point_index]={
                'point_2d': point.tolist(),
                'cam_index': cam_index,
            }
            point_index+=1
    # import pdb;pdb.set_trace() # 检查 init_cluster 是否属于同一个相机
    cost_matrix=np.zeros(
        shape=(len(points),len(points)),
        dtype=np.float64
    )
    for i in range(len(points)):
        for j in range(len(points)):
            cost_matrix[i][j]=get_cost(
                point1=points[i],
                point2=points[j],
                cam_params=cam_params
            )
    # 对 cost_matrix 进行聚类 -> 获取 clumps
    # 从points总获取 初始类中心点
    assert init_cluster is not None,"no camera can detect complete wand" # 至少应该有一个相机能看到完整的L型标定杆
    support_list=['dbscan','sort','mcp']
    assert mode in support_list,f'noIdReconstruction only support {support_list}'
    if mode == "dbscan":
        labels=DBSCAN(eps=10, min_samples=2, metric="precomputed").fit_predict(cost_matrix) # DBSCAN作为密度聚类,并不适合这个任务场景
    elif mode == "sort":
        labels=sort_cluster(
            init_cluster=init_cluster,
            cost_matrix=cost_matrix
        )
    elif mode == "mcp":
        labels=mcp_cluster(
            cost_matrix=cost_matrix,
            eps=1e1,
            min_samples=2
        )
    else:
        raise NotImplementedError
    # np.set_printoptions(
    #     threshold=5000,
    #     linewidth=5000
    # )
    # print(cost_matrix.astype(np.int32))
    # import pdb;pdb.set_trace()
    max_label=max(labels)+1
    clumps=[[] for _ in range(max_label)]
    for label,point in list(zip(labels,points.items())):
        point_index=point[0]
        cam_index=point[1]['cam_index']
        cam_param=cam_params[cam_index]
        point_normalized=point[1]['point_normalized']
        clumps[label].append({
            'camera_index':cam_index,
            'camera_param':cam_param,
            'point_2d':point_normalized
        })
    # 对每一个 clumps 进行三角测量
    point_3ds=[
        triangulate_clump(clump)
        for clump in clumps
    ]
    return point_3ds
    


if __name__=="__main__":
    pass