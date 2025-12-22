import os
import sys
import json
import copy

import numpy as np
import cv2
import glob
from natsort import natsorted
from loguru import logger
import torch

from extrinsicParameter.refinePose.so3_exp_map import so3_exp_map
from .clickPoint import click_point
from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from extrinsicParameter.poleDetection.wandDetection import get_wand
from .noIdReconstruction import no_id_reconstruct
from .get_id_with_distance import get_id_with_distance
from .solve_icp import solve_icp
from utils.imageConcat import show_multi_imgs
from .handle_labelme import get_labelme_json,vis_objs,format_labelme_objs,triangulate_points,vis_points
from .handle_board import get_corner_map,triangulate_corner_map
from .enhanced_labelme import EnhancedLabelme,fit_model,vis_labels
from .rescale import get_rescale_ratio,rescale_world_coord

def vis_point3ds(
        image_path,
        cam_num,
        cam_params,
        point_3ds,
        save_folder
    ):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    frame_list=get_cam_list(image_path,cam_num)[0]
    for step,(frame_path,camera_param) in enumerate(list(zip(frame_list,cam_params))):
        frame=cv2.imread(frame_path)
        origin=copy.deepcopy(frame)
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
        frame=cv2.addWeighted(frame,0.5,origin,0.5,0)
        cv2.imwrite(os.path.join(save_folder,f"cam{step+1}.jpg"),frame)
        
def vis_wand_detection(
        image_path,
        cam_num,
        wands,
        save_folder
    ):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    frame_list=get_cam_list(image_path,cam_num)[0]
    for step,(frame_path,wand) in enumerate(list(zip(frame_list,wands))):
        frame=cv2.imread(frame_path)
        origin=copy.deepcopy(frame)
        for point_2d in wand:
            frame=cv2.circle(
                img=frame,
                center=np.array(point_2d).astype(np.int32),
                radius=10,
                color=(0,255,0),
                thickness=-1
            )
        frame=cv2.addWeighted(frame,0.5,origin,0.5,0)
        cv2.imwrite(os.path.join(save_folder,f"cam{step+1}.jpg"),frame)

def transfer_point_3ds(
        point_3ds,
        R,t
    ):
    transfered_point_3ds=[]
    message="\n"
    for step,point_3d in enumerate(point_3ds):
        i=np.linalg.inv(R)@(point_3d-t)
        transfered_point_3ds.append(i)
        i=np.around(i,3).tolist()
        message=message+f"{step}:\t"+str(i)+"\n"
    logger.info(f"trans reconstruct points coord: {message}")
    return np.array(transfered_point_3ds)

def get_cam0_extrinsic(
        cam_num,
        cam_params,
        poles,
        masks,
        image_path,
        world_coord_param,
        wand_blob_param,
        fastBlob=True,
    ):
    """
    给出 cam0 在 world coordinate 下的 Rotation 和 tran
    """
    wand_folder=os.path.join(image_path,"wand")
    if world_coord_param['type']=="point":
        # 调用cam0的empty的第一张图片
        # 交互式给出 4 个点, 使用 pnp 进行求解
        # yaml文件给出四个点的三维坐标
        image_empty=natsorted(glob.glob(os.path.join(image_path,"wand","cam1","*")))[0]
        cam0_R,cam0_t=click_point(
            cam_0_param=cam_params[0],
            image_path=image_empty,
            point_coordinates=world_coord_param['PointCoordinates']
        )
    elif world_coord_param['type']=="labelme":
        if world_coord_param['mode']=="norm":
            json_paths=natsorted(glob.glob(os.path.join(image_path,'wand','labelme','*.json')))
            assert cam_num==len(json_paths),f"wand/labelme/*.json file number not equal to camera number"
            objs=[get_labelme_json(json_path) for json_path in json_paths]
            vis_objs(
                objs=objs,
                image_path=wand_folder,
                cam_num=cam_num,
                save_folder=os.path.join(wand_folder,"vis_wand_detection")
            )
            points=format_labelme_objs(objs,cam_params,world_coord_param['PointCoordinates'])
            # 排除 重建点数量不够的点
            for key in list(points.keys()):
                if len(points[key]['point_2ds'])<world_coord_param['min_vis_num']:
                    points.pop(key)
                    logger.warning(f"discard 3D point {key} for less than threshold:{world_coord_param['min_vis_num']} cameras")
            # import pdb;pdb.set_trace()
            points=triangulate_points(points)
            # 排除 重建误差过大的点，认为是异常点，不应该参与ICP匹配
            # for key in list(points.keys()):
            #     if points[key]['reconstruction_mean_pixel_error']>50:
            #         points.pop(key)
            #         logger.warning(f"discard 3D point {key} for larger than threshold:{50} reconstruction mean pixel error")
            # import pdb;pdb.set_trace()
            vis_points(
                point_3ds={key:points[key]['pred_point_3d'] for key in points.keys()},
                image_path=wand_folder,
                cam_num=cam_num,
                cam_params=cam_params,
                save_folder=os.path.join(wand_folder,"vis_reconstruct_points")
            )
            if world_coord_param.rescale:
                rescale_ratio=get_rescale_ratio(
                    source=[points[key]['pred_point_3d'] for key in points.keys()],
                    target=[world_coord_param['PointCoordinates'][key] for key in points.keys()]
                )
                # print(cam_params)
                # print(poles[:5])
                # print([points[key]['pred_point_3d'] for key in points.keys()][:5])
                rescale_world_coord(
                    rescale_ratio=rescale_ratio,
                    cam_params=cam_params,
                    poles=poles,
                    pred_points=[points[key]['pred_point_3d'] for key in points.keys()]
                )
                # print('-----------------')
                # print(cam_params)
                # print(poles[:5])
                # print([points[key]['pred_point_3d'] for key in points.keys()][:5])
                # import pdb;pdb.set_trace()
            R,t=solve_icp(
                target=[points[key]['pred_point_3d'] for key in points.keys()],
                source=[world_coord_param['PointCoordinates'][key] for key in points.keys()]
            )
            transfered_point_3ds=transfer_point_3ds(
                point_3ds=[points[key]['pred_point_3d'] for key in points.keys()],
                R=R,
                t=t
            )
            cam0_R,cam0_t=R,t
            # print(R,t)
        elif world_coord_param['mode']=="enhance":
            json_paths=natsorted(glob.glob(os.path.join(image_path,'wand','labelme','*.json')))
            labels=dict()
            for json_path in json_paths:
                key=json_path.split('/')[-1].split('.')[0]
                try:
                    obj=get_labelme_json(json_path)
                    labels[key]=obj
                except Exception as e:
                    logger.warning(f"file {json_path} load fail, check format")
            keys=labels.keys()
            no_keys=[]
            for i in range(cam_num):
                if f'cam{i+1}' not in keys:
                    lack_json_path=os.path.join(image_path,'wand','labelme',f'cam{i+1}.json')
                    no_keys.append(f'cam{i+1}')
                    logger.warning(f"can not fild {lack_json_path}")
            if no_keys:
                logger.warning(f"{no_keys} without label! this may cause calculation error!")
            vis_labels(
                cam_num=cam_num,
                image_path=wand_folder,
                labels=labels,
                save_folder=os.path.join(wand_folder,"vis_wand_detection")
            )
            model=EnhancedLabelme(
                cam_params=cam_params,
                labels=labels,
                point_3ds=world_coord_param['PointCoordinates']
            )
            cam0_R,cam0_t=fit_model(
                model=model,
                iteration=int(5e4),
                lr=2e0,
                interval=int(1e4),
                print_frequence=int(1e3),
                gamma=0.2
            )
        else:
            support_list=['norm','labelme']
            logger.info(f"only support mode in {support_list}")
    elif world_coord_param['type']=="wand":
        # L型杆子
        # 检查是否有 wand 文件夹
        # blob detection
        # reconstruction without id info
        # get id info according to 3d distance
        # solve icp problem using open3d
        if not os.path.exists(wand_folder):
            assert False,f"can not find {wand_folder}"
        # import pdb;pdb.set_trace() # cam_params 有 image_size
        # ! 这里的  wand 不具备 id 信息 ! 
        wands=get_wand(
            cam_num=cam_num,
            resolutions=[cam_param['image_size'] for cam_param in cam_params],
            masks=masks,
            image_path=wand_folder,
            color=world_coord_param['color'],
            wand_blob_param=wand_blob_param,
            fastBlob=fastBlob
        )
        # vis wands
        vis_wand_detection(
            image_path=wand_folder,
            cam_num=cam_num,
            wands=wands,
            save_folder=os.path.join(wand_folder,'vis_wand_detection')
        )
        point_3ds=no_id_reconstruct(
            cam_num=cam_num,
            cam_params=cam_params,
            wands=wands,
            mode=world_coord_param['mode']
        )
        point_3ds=get_id_with_distance(
            point_3ds=point_3ds,
            WandDefinition=world_coord_param['WandDefinition'],
        )
        # 可视化 point_3ds 的 reproj
        vis_point3ds(
            image_path=wand_folder,
            cam_num=cam_num,
            cam_params=cam_params,
            point_3ds=point_3ds,
            save_folder=os.path.join(wand_folder,'vis_reconstruct_points')
        )
        if world_coord_param.rescale:
            rescale_ratio=get_rescale_ratio(
                source=point_3ds,
                target=world_coord_param['WandPointCoord']
            )
            rescale_world_coord(
                rescale_ratio=rescale_ratio,
                cam_params=cam_params,
                poles=poles,
                pred_points=point_3ds
            )
        R,t=solve_icp(
            target=point_3ds,
            source=world_coord_param['WandPointCoord']
        )
        transfered_point_3ds=transfer_point_3ds(
            point_3ds,R,t
        )
        cam0_R,cam0_t=R,t
    # elif world_coord_param['type']=="board":
    #     corner_map,frame=get_corner_map(
    #         image_path=wand_folder,
    #         cam_num=cam_num,
    #         board_config=world_coord_param['BoardDefinition'],
    #         origin_point=world_coord_param['ZeroPointCoord'],
    #         cam_params=cam_params
    #     )
    #     cv2.imwrite(os.path.join(wand_folder,'vis_wand_detection.jpg'),frame)
    #     corner_map=triangulate_corner_map(corner_map)
    #     frame=vis_points(
    #         point_3ds=[corner_map[key]['pred_point_3d'] for key in natsorted(corner_map.keys())],
    #         image_path=wand_folder,
    #         cam_num=cam_num,
    #         cam_params=cam_params
    #     )
    #     cv2.imwrite(os.path.join(wand_folder,'vis_reconstruct_points.jpg'),frame)
    #     R,t=solve_icp(
    #         target=[corner_map[key]['pred_point_3d'] for key in natsorted(corner_map.keys())],
    #         source=[corner_map[key]['expect_point_3d'] for key in natsorted(corner_map.keys())]
    #     )
    #     if world_coord_param['BoardPointCoord'] is None:
    #         world_coord_param['BoardPointCoord']=np.array([
    #             corner_map[key]['expect_point_3d'] 
    #             for key in natsorted(corner_map.keys())
    #         ]).tolist()
    #     # transfered_point_3ds=transfer_point_3ds(
    #     #     point_3ds,R,t
    #     # )
    #     cam0_R,cam0_t=R,t
    else:
        support_list=["point","labelme","board","wand"]
        raise NotImplementedError(f"we only support {support_list}")
    return cam0_R,cam0_t



if __name__=="__main__":
    pass