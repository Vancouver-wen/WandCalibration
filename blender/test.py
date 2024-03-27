import os
import sys

from loguru import logger
import bpy

sys.path.append("/home/wenzihao/Desktop/WandCalibration")
from main import OptiTrack

myOptitrack=OptiTrack(config_path="/home/wenzihao/Desktop/WandCalibration/config/cfg_wtt.yaml")
myOptitrack.run()

if __name__=="__main__":
    pass