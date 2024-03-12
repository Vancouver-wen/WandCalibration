import os
import sys
import time

import numpy as np
import cv2
from natsort import natsorted
import glob
from matplotlib import pyplot as plt

from utils.imageConcat import show_multi_imgs

def get_cam_id_from_mask_and_step(
        mask,
        step,
    ):
    sum=0
    for i,each_mask in enumerate(mask):
        if each_mask:
            if step==sum:
                return i
            sum+=1

def ray_multi_view_triangulation(
        point_2ds,
        poses,
        point_3d,
        intrinsics,
        frame_id,
        mask,
    ):
    # 射线可视化验证三角化
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(point_3d[0],point_3d[1],point_3d[2],color='red')
    ax.set_xlim(-10, 10) 
    ax.set_ylim(-10, 10) 
    ax.set_zlim(-10, 10) 
    # 获取相机的位置 与 关键点位置
    cam_positions=[]
    point_positions=[]
    for pose,point_2d in list(zip(poses,point_2ds)):
        R=np.array(pose['R'])
        t=np.array(pose['t'])
        cam_position=np.linalg.inv(R)@(-t)
        cam_positions.append(cam_position)
        point_2d_position=np.append(point_2d,[1])*10
        point_2d_position=np.linalg.inv(R)@(point_2d_position-t)
        point_positions.append(point_2d_position)
    for cam_position in cam_positions:
        ax.scatter3D(cam_position[0],cam_position[1],cam_position[2],s=3,color='blue')
    for point_position in point_positions:
        ax.scatter3D(point_position[0],point_position[1],point_position[2],s=3,color='green')
    for cam_position,point_position in list(zip(cam_positions,point_positions)):
        x_data = np.linspace(cam_position[0],point_position[0], 100)
        y_data = np.linspace(cam_position[1],point_position[1], 100)
        z_data = np.linspace(cam_position[2],point_position[2], 100)
        # import pdb;pdb.set_trace()
        ax.scatter3D(x_data,y_data,z_data,s=0.2,color="gray")
    plt.savefig("./ray.jpg",dpi=1000)
    # plt.show()
    plt.close()

def reproj_multi_view_triangulation(
        point_2ds,
        poses,
        point_3d,
        intrinsics,
        frame_id,
        mask,
        image_root_path="/home/wenzihao/Desktop/WandCalibration/imageCollect/pole"
    ):
    # 重投影验证三角化
    images=[]
    for step,(point_2d,pose,intrinsic) in enumerate(list(zip(point_2ds,poses,intrinsics))):
        # 获取 K dist R t
        K=np.array(intrinsic['K'])
        dist=np.squeeze(np.array(intrinsic['dist']))
        R=np.array(pose['R']).astype(np.float64)
        t=np.squeeze(np.array(pose['t'])).astype(np.float64)
        # 验证 归一化平面坐标 -> 正确
        f_x,f_y,u_0,v_0=K[0][0],K[1][1],K[0,2],K[1,2]
        u=f_x*point_2d[0] + u_0
        v=f_y*point_2d[1] + v_0
        gt_point_2d=np.stack((u,v),axis=0)
        # 验证 3d point 坐标
        imagePoints, _ = cv2.projectPoints(
            objectPoints=np.expand_dims(point_3d,axis=0),
            rvec=R,
            tvec=t,
            cameraMatrix=K,
            distCoeffs=dist
        )
        expect_point_2d=np.squeeze(imagePoints)
        # 获取 cam_id
        cam_id=get_cam_id_from_mask_and_step(mask,step)
        image_path=natsorted(glob.glob(os.path.join(image_root_path,f'cam{cam_id+1}','*')))[frame_id]
        image=cv2.imread(image_path)
        image=cv2.circle(img=image,center=gt_point_2d.astype(np.int32),radius=10,color=(0,255,0),thickness=-1)
        image=cv2.circle(img=image,center=expect_point_2d.astype(np.int32),radius=10,color=(0,0,255),thickness=-1)
        images.append(image)
    frame = show_multi_imgs(scale=0.5,imglist=images,order=(int(len(images)/2+0.99),2))
    return frame

def multi_view_triangulate(
        point_2ds,
        poses,
        solve_method="SVD"
    ):
    assert len(point_2ds)==len(poses),"illegal reconstruction parameters"
    if len(poses)<2:
        # triangulation need atleast 2 camera views
        return None
    A=[]
    for point_2d,pose in list(zip(point_2ds,poses)):
        P_matrix=np.concatenate(
            (np.array(pose['R']),np.expand_dims(np.array(pose['t']).T,axis=1)),
            axis=1
        )
        x=point_2d[0]
        y=point_2d[1]
        A.append(x*P_matrix[2]-P_matrix[0])
        A.append(y*P_matrix[2]-P_matrix[1])
    A=np.array(A).astype(np.float32)
    if solve_method=="SVD":
        U,sigma,VH = np.linalg.svd(A,full_matrices=True)
        vector=VH[-1]
        point_3d=vector[:3]/vector[3]
    else:
        # 这个解法不好
        eigen_value,eigen_vector = np.linalg.eig(A.T@A)
        vector=eigen_vector[np.argmin(eigen_value)]
        point_3d=vector[:3]/vector[3]
    return point_3d

def normalized_pole_triangulate(
        cam_num,
        normalized_pole_lists,
        poses,
        intrinsics # 用于验证重投影
    ):
    assert cam_num==len(poses),'cam_num != len(poses)'
    pole_3ds=[]
    for step,normalized_pole_list in enumerate(normalized_pole_lists): # 遍历 帧号
        assert cam_num==len(normalized_pole_list),'cam_num != len(normalized_pole_list)'
        mask=[temp is not None for temp in normalized_pole_list]
        if np.array(mask).sum()<2:
            pole_3ds.append(None)
            continue
        masked_pole_list=np.squeeze(np.array([ 
            temp[1]
            for temp in filter(lambda x:x[0],list(zip(mask,normalized_pole_list)))
        ]))
        masked_pole_list=masked_pole_list.transpose(1,0,2) # (相机号,marker号,坐标) -> (marker号,相机号,坐标)
        masked_pose_list=[
            poses[temp[1]]
            for temp in filter(lambda x:x[0],list(zip(mask,poses)))
        ]
        masked_intrinsic_list=[
            temp[1]
            for temp in filter(lambda x:x[0],list(zip(mask,intrinsics)))
        ]
        point_3ds=[]
        # import pdb;pdb.set_trace() 
        for point_2d_list in masked_pole_list: 
            point_3d=multi_view_triangulate(
                point_2ds=point_2d_list,
                poses=masked_pose_list
            )
            if point_3d is not None and False: # 使用 True 与 False 控制是否可视化
                # 使用重投影验证误差
                frame=reproj_multi_view_triangulation(
                    point_2ds=point_2d_list,
                    poses=masked_pose_list,
                    point_3d=point_3d,
                    intrinsics=masked_intrinsic_list,
                    frame_id=step,
                    mask=mask
                )
                cv2.imwrite("/home/wenzihao/Desktop/WandCalibration/reproj.jpg",frame)
                # 使用射线可视化验证误差
                ray_multi_view_triangulation(
                    point_2ds=point_2d_list,
                    poses=masked_pose_list,
                    point_3d=point_3d,
                    intrinsics=masked_intrinsic_list,
                    frame_id=step,
                    mask=mask
                )
            point_3ds.append(point_3d)
        d1=np.linalg.norm(point_3ds[0]-point_3ds[1])
        d2=np.linalg.norm(point_3ds[1]-point_3ds[2])
        # print({'cam_num':np.array(mask).sum(),'d1':d1,'d2':d2})
        pole_3ds.append(point_3ds)
    return pole_3ds
        


if __name__=="__main__":
    pass