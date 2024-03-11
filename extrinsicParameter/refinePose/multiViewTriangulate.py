import os
import sys

import numpy as np
import cv2

def multi_view_triangulate(
        point_2ds,
        poses
    ):
    assert len(point_2ds)==len(poses),"illegal reconstruction parameters"
    if len(poses)<2:
        # triangulation need atleast 2 camera views
        return None
    D=[]
    for point_2d,pose in list(zip(point_2ds,poses)):
        P_matrix=np.concatenate(
            (np.array(pose['R']),np.expand_dims(np.array(pose['t']).T,axis=1)),
            axis=1
        )
        # import pdb;pdb.set_trace()
        D.append(point_2d[0]*P_matrix[2]-P_matrix[0])
        D.append(point_2d[1]*P_matrix[2]-P_matrix[1])
    D=np.array(D)
    # 此时 D 满足 D@point_2ds=0
    # 对 D.T@D 进行SVD分解, 最小特征值对应的特征向量就是 超定方程的最小二乘解
    eigen_value,eigen_vector=np.linalg.eig(D.T@D)
    # print(f'eigen_value:{eigen_value}')
    # import pdb;pdb.set_trace()
    # error=eigen_value[np.argmin(eigen_value)]
    vector=eigen_vector[np.argmin(eigen_value)]
    point_3d=vector[:3]/vector[3]
    return point_3d

def normalized_pole_triangulate(
        cam_num,
        normalized_pole_lists,
        poses    
    ):
    assert cam_num==len(poses),'cam_num != len(poses)'
    pole_3ds=[]
    for normalized_pole_list in normalized_pole_lists:
        assert cam_num==len(normalized_pole_list),'cam_num != len(normalized_pole_list)'
        mask=[temp is not None for temp in normalized_pole_list]
        if np.array(mask).sum()<2:
            pole_3ds.append(None)
            continue
        masked_pole_list=np.squeeze(np.array([
            temp[1]
            for temp in filter(lambda x:x[0],list(zip(mask,normalized_pole_list)))
        ]))
        masked_pole_list=masked_pole_list.transpose(1,0,2)
        masked_pose_list=[
            poses[temp[1]]
            for temp in filter(lambda x:x[0],list(zip(mask,poses)))
        ]
        point_3ds=[]
        for point_2d_list in masked_pole_list:
            point_3d=multi_view_triangulate(
                point_2ds=point_2d_list,
                poses=masked_pose_list
            )
            point_3ds.append(point_3d)
        d1=np.linalg.norm(point_3ds[0]-point_3ds[1])
        d2=np.linalg.norm(point_3ds[1]-point_3ds[2])
        print({'d1':d1,'d2':d2})
    import pdb;pdb.set_trace()
    return pole_3ds
        


if __name__=="__main__":
    pass