import os
import sys
import json
import argparse

import cv2
import yaml
from easydict import EasyDict
from loguru import logger

from utils.yamlLoader import get_yaml_data
from intrinsicParameter.intrinsicCalibration.get_intrinsic import get_intrinsic
from extrinsicParameter.poleDetection.maskGeneration import get_mask
from extrinsicParameter.poleDetection.poleDetection import get_pole
from visualize.vis_pole_detection import vis_pole
from extrinsicParameter.initPose.initPose import get_init_pose
from visualize.get_init_camera_params import get_init_camera_params
from extrinsicParameter.refinePose.refinePose import get_refine_pose
from utils.verifyAccuracy import verify_accuracy
from extrinsicParameter.worldCoord.get_cam0_extrinsic import get_cam0_extrinsic
from extrinsicParameter.worldCoord.adjustCamParam import adjust_camera_params
from visualize.visCameraParams import vis_camera_params
from visualize.vis_reproj_error import vis_reproj_error

class OptiTrack(object):
    def __init__(self,config_path):
        self.config=EasyDict(get_yaml_data(config_path))
        logger.info(self.config)
    def add_intrinsic(self):
        self.intrinsic=get_intrinsic(
            cam_num=self.config.cam_num,
            board_config=self.config.board,
            image_path=os.path.join(self.config.image_path,"board")
        )
    def add_mask(self):
        self.mask=get_mask(
            cam_num=self.config.cam_num,
            resolutions=[each_intrinsic.image_size for each_intrinsic in self.intrinsic],
            maskBlobParam=self.config.maskBlobParam,
            image_path=os.path.join(self.config.image_path,'empty'),
            mask_path=os.path.join(self.config.image_path,'mask'),
            color=self.config.pole.color
        )
    def add_pole(self):
        self.pole=get_pole(
            cam_num=self.config.cam_num,
            resolutions=[each_intrinsic.image_size for each_intrinsic in self.intrinsic],
            poleBlobParam=self.config.poleBlobParam,
            image_path=os.path.join(self.config.image_path,'pole'),
            masks=self.mask,
            color=self.config.pole.color
        )
        try:
            vis_pole(
                cam_num=self.config.cam_num,
                image_path=os.path.join(self.config.image_path,'pole'),
                pole_lists=self.pole,
                vis_num=self.config.vis_num
            )
        except:
            logger.info(f"early stop pole detection visualizer")
        # import pdb;pdb.set_trace()
    def init_pose(self):
        self.pose=get_init_pose(
            cam_num=self.config.cam_num,
            pole_lists=self.pole,
            intrinsics=self.intrinsic,
            pole_param=self.config.pole,
            save_path=os.path.join(self.config.image_path,'init_pose.json')
        )
        init_camera_params=get_init_camera_params(
            cam_num=self.config.cam_num,
            intrinsics=self.intrinsic,
            extrinsics=self.pose
        )
        try:
            vis_reproj_error(
                cam_num=self.config.cam_num,
                pole_lists=self.pole,
                camera_params=init_camera_params,
                image_path=os.path.join(self.config.image_path,'pole'),
                vis_num=self.config.vis_num,
                vis_folder="vis_init_reproj"
            )
        except:
            logger.info(f"early stop reprojection error visualizer")
    def refine_pose(self,early_stop=True):
        save_path=os.path.join(self.config.image_path,'refine_pose.json')
        if early_stop:
            try:
                get_refine_pose(
                    cam_num=self.config.cam_num,
                    pole_lists=self.pole,
                    intrinsics=self.intrinsic,
                    pole_param=self.config.pole,
                    init_poses=self.pose,
                    save_path=save_path,
                    multiprocessing=True
                )
            except:
                logger.info("early stop pose refiner")
        else:
            get_refine_pose( 
                cam_num=self.config.cam_num,
                pole_lists=self.pole,
                intrinsics=self.intrinsic,
                pole_param=self.config.pole,
                init_poses=self.pose,
                save_path=save_path,
                multiprocessing=True
            )
        with open(save_path,'r') as f:
            self.output=json.load(f)
    def verify_accuracy(self):
        verify_accuracy(
            camera_params=self.output['calibration'],
            pole_3ds=self.output['poles'],
            pole_lists=self.pole,
        )
        try:
            vis_reproj_error(
                cam_num=self.config.cam_num,
                pole_lists=self.pole,
                camera_params=self.output['calibration'],
                image_path=os.path.join(self.config.image_path,'pole'),
                vis_num=self.config.vis_num,
                vis_folder="vis_reproj"
            )
        except:
            logger.info(f"early stop reprojection error visualizer")
    def world_pose(self):
        save_path=os.path.join(self.config.image_path,'world_pose.json')
        # import pdb;pdb.set_trace() # p self.output
        cam0_R,cam0_t=get_cam0_extrinsic(
            cam_num=self.config.cam_num,
            cam_params=self.output['calibration'],
            masks=self.mask,
            image_path=self.config.image_path,
            world_coord_param=self.config.worldCoordParam,
            wand_blob_param=self.config.wandBlobParam
        )
        self.world_camera_params=adjust_camera_params(
            cam_0_R=cam0_R,
            cam_0_t=cam0_t,
            camera_params=self.output['calibration'],
            save_path=save_path
        )
    def visualize(self):
        save_path=os.path.join(self.config.image_path,'world_pose.json')
        with open(save_path,'r') as f:
            self.world_camera_params=json.load(f)
        vis_camera_params(
            camera_params=self.world_camera_params
        )
    def run(self,vis=True):
        self.add_intrinsic()
        self.add_mask()
        self.add_pole()
        self.init_pose()
        self.refine_pose()
        self.verify_accuracy()
        self.world_pose()
        if vis:
            self.visualize()
        return self.world_camera_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',default="./config/cfg_wtt.yaml",type=str)
    args = parser.parse_args()
    myOptitrack=OptiTrack(config_path=args.config_path)
    myOptitrack.run()


if __name__=="__main__":
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger.add("./logs/{time}.log")
    main()