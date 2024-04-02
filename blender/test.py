import os
import sys

from loguru import logger
import bpy

sys.path.append("/home/wenzihao/Desktop/WandCalibration")
from main import OptiTrack

config_path="/home/wenzihao/Desktop/WandCalibration/config/cfg_wtt.yaml"

myOptitrack=OptiTrack(config_path=config_path)
params=myOptitrack.run()




logger.info("done successfully !")


if __name__=="__main__":
    # main区域基本无用, 打印无输出
    pass