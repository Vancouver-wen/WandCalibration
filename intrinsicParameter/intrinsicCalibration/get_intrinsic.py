import os
import sys
import json

from loguru import logger
from natsort import natsorted
from easydict import EasyDict

def get_intrinsic(cam_num,board_config,image_root_path):
    image_path=os.path.join(image_root_path,"board")
    # 检查是否有 intrinsic.json
    if os.path.exists(os.path.join(image_path,"intrinsic.json")):
        intrinsic=[]
        with open(os.path.join(image_path,"intrinsic.json"),'r',encoding="utf-8") as f:
            temp=json.load(f)
            for key in natsorted(temp.keys()):
                intrinsic.append(EasyDict(temp[key]))
        if cam_num==len(intrinsic):
            logger.info('find existed intrinsic.json file and load it successfully')
            return intrinsic
        else:
            logger.warning('the format of intrinsic.json is wrong')


if __name__=="__main__":
    pass