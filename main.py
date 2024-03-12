import os
import sys
import json

import cv2
import yaml
from easydict import EasyDict
from loguru import logger
logger.add("./logs/{time}.log")

from utils.yamlLoader import get_yaml_data
from intrinsicParameter.intrinsicCalibration.get_intrinsic import get_intrinsic
from extrinsicParameter.poleDetection.maskGeneration import get_mask
from extrinsicParameter.poleDetection.poleDetection import get_pole
from visualize.vis_pole_detection import vis_pole
from extrinsicParameter.initPose.initPose import get_init_pose
from extrinsicParameter.refinePose.refinePose import get_refine_pose
from utils.verifyAccuracy import verify_accuracy

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
            mask_path=os.path.join(self.config.image_path,'mask')
        )
    def add_pole(self):
        self.pole=get_pole(
            cam_num=self.config.cam_num,
            resolutions=[each_intrinsic.image_size for each_intrinsic in self.intrinsic],
            poleBlobParam=self.config.poleBlobParam,
            image_path=os.path.join(self.config.image_path,'pole'),
            masks=self.mask
        )
        # vis_pole(
        #     cam_num=self.config.cam_num,
        #     image_path=os.path.join(self.config.image_path,'pole'),
        #     pole_lists=self.pole
        # )
        # import pdb;pdb.set_trace()
    def init_pose(self):
        self.pose=get_init_pose(
            cam_num=self.config.cam_num,
            pole_lists=self.pole,
            intrinsics=self.intrinsic,
            pole_param=self.config.pole,
            save_path=os.path.join(self.config.image_path,'init_pose.json')
        )
    def refine_pose(self):
        save_path=os.path.join(self.config.image_path,'refine_pose.json')
        get_refine_pose(
            cam_num=self.config.cam_num,
            pole_lists=self.pole,
            intrinsics=self.intrinsic,
            pole_param=self.config.pole,
            init_poses=self.pose,
            save_path=save_path
        )
        with open(save_path,'r') as f:
            self.output=json.load(f)
    def verify_accuracy(self):
        verify_accuracy(
            camera_params=self.output['calibration'],
            pole_3ds=self.output['poles'],
            pole_lists=self.pole,
        )
    def visualize(self):
        pass
    def run(self,vis=False):
        self.add_intrinsic()
        self.add_mask()
        self.add_pole()
        self.init_pose()
        self.refine_pose()
        self.verify_accuracy()
        if vis:
            self.visualize()


def main():
    myOptitrack=OptiTrack(config_path="./config/config.yaml")
    myOptitrack.run()


if __name__=="__main__":
    main()