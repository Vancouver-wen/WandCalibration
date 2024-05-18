import os
import sys

import numpy as np
import cv2
import glob
from natsort import natsorted

from .clickPoint import click_point
from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from extrinsicParameter.poleDetection.wandDetection import get_wand
from .noIdReconstruction import no_id_reconstruct
from .get_id_with_distance import get_id_with_distance
from .solve_icp import solve_icp
from utils.imageConcat import show_multi_imgs

def vis_point3ds(
        image_path,
        cam_num,
        cam_params,
        point_3ds
    ):
    frame_list=get_cam_list(image_path,cam_num)[0]
    frames=[]
    for frame_path,camera_param in list(zip(frame_list,cam_params)):
        frame=cv2.imread(frame_path)
        point_names=['A','B','C']
        assert len(point_names)==len(point_3ds),f"len(point_names)!=len(point_3ds)\t check point_3ds definition"
        for point_name,point_3d in list(zip(point_names,point_3ds)):
            point_2d,_=cv2.projectPoints(
                objectPoints=np.expand_dims(np.array(point_3d),axis=0),
                rvec=np.array(camera_param['R']),
                tvec=np.array(camera_param['t']),
                cameraMatrix=np.array(camera_param['K']),
                distCoeffs=np.array(camera_param['dist']),
            )
            point_2d=np.squeeze(point_2d)
            frame=cv2.circle(
                img=frame,
                center=point_2d.astype(np.int32),
                radius=10,
                color=(0,0,255),
                thickness=-1
            )
            frame=cv2.putText(
                img=frame,
                text=point_name,
                org=point_2d.astype(np.int32),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0,0,255),
                thickness=3
            )
        frames.append(frame)
    frame=show_multi_imgs(
        scale=1,
        imglist=frames,
        order=(int(cam_num/3+0.99),3)
    )
    return frame
        
def vis_wand_detection(
        image_path,
        cam_num,
        wands,
    ):
    frame_list=get_cam_list(image_path,cam_num)[0]
    frames=[]
    for frame_path,wand in list(zip(frame_list,wands)):
        frame=cv2.imread(frame_path)
        for point_2d in wand:
            frame=cv2.circle(
                img=frame,
                center=np.array(point_2d).astype(np.int32),
                radius=10,
                color=(0,255,0),
                thickness=-1
            )
        frames.append(frame)
    frame=show_multi_imgs(
        scale=1,
        imglist=frames,
        order=(int(cam_num/3+0.99),3)
    )
    return frame

def get_cam0_extrinsic(
        cam_num,
        cam_params,
        masks,
        image_path,
        world_coord_param,
        wand_blob_param
    ):
    """
    给出 cam0 在 world coordinate 下的 Rotation 和 tran
    """
    if world_coord_param.type=="point":
        # 调用cam0的empty的第一张图片
        # 交互式给出 4 个点
        # yaml文件给出四个点的三维坐标
        image_empty=natsorted(glob.glob(os.path.join(image_path,"wand","cam1","*")))[0]
        cam0_R,cam0_t=click_point(
            cam_0_param=cam_params[0],
            image_path=image_empty,
            point_coordinates=world_coord_param.PointCoordinates
        )
    elif world_coord_param.type=="wand":
        # L型杆子
        # 检查是否有 wand 文件夹
        # blob detection
        # reconstruction without id info
        # get id info according to 3d distance
        # solve icp problem using open3d
        wand_folder=os.path.join(image_path,"wand")
        if not os.path.exists(wand_folder):
            assert False,f"can not find {wand_folder}"
        # import pdb;pdb.set_trace() # cam_params 有 image_size
        # ! 这里的  wand 不具备 id 信息 ! 
        wands=get_wand(
            cam_num=cam_num,
            resolutions=[cam_param['image_size'] for cam_param in cam_params],
            masks=masks,
            image_path=wand_folder,
            color=world_coord_param.color,
            wand_blob_param=wand_blob_param,
        )
        # vis wands
        frame=vis_wand_detection(
            image_path=wand_folder,
            cam_num=cam_num,
            wands=wands
        )
        cv2.imwrite(os.path.join(wand_folder,'vis_wand_detection.jpg'),frame)
        point_3ds=no_id_reconstruct(
            cam_num=cam_num,
            cam_params=cam_params,
            wands=wands,
        )
        point_3ds=get_id_with_distance(
            point_3ds=point_3ds,
            WandDefinition=world_coord_param['WandDefinition'],
        )
        # 可视化 point_3ds 的 reproj
        frame=vis_point3ds(
            image_path=wand_folder,
            cam_num=cam_num,
            cam_params=cam_params,
            point_3ds=point_3ds,
        )
        cv2.imwrite(os.path.join(wand_folder,'vis_reconstruct_points.jpg'),frame) 
        R,t=solve_icp(
            target=point_3ds,
            source=world_coord_param['WandPointCoord']
        )
        cam0_R,cam0_t=R,t
    else:
        support_list=["point","wand"]
        raise NotImplementedError(f"we only support {support_list}")
    return cam0_R,cam0_t



if __name__=="__main__":
    pass