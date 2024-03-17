import os
import sys
import json

import glob
from loguru import logger
from natsort import natsorted
from easydict import EasyDict

def get_intrinsic(cam_num,board_config,image_path):
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
    # 生成 intrinsic.json
    board_type=board_config.pop("type")
    if board_type=="checkerboard":
        from intrinsicParameter.checkboardCalibration.get_cam_calibration import IntrinsicCalibration
        height=board_config['height']
        width=board_config['width']
        intrinsicCalibrator=IntrinsicCalibration(height=height,width=width)
    elif board_config=="charucoboard":
        height=board_config['height']
        width=board_config['width']
        suqare_length=board_config['square_length']
        marker_length=board_config['marker_length']
        from intrinsicParameter.charucoboardCalibration.get_cam_calibration import IntrinsicCalibration
        intrinsicCalibrator=IntrinsicCalibration(height=height,width=width,square_length=suqare_length,markser_length=marker_length)
    else:
        support_list=[
            "checkerboard",
            "charucoboard"
        ]
        assert False,f"we only support {support_list}"
    image_path_lists=[
        glob.glob(os.path.join(image_path,f"cam{i+1}",'*'))
        for i in range(cam_num)
    ]
    intrinsics=dict()
    for step,image_path_list in enumerate(image_path_lists):
        intrinsic=intrinsicCalibrator(image_path_list=image_path_list)
        intrinsics[f"cam{step+1}"]=intrinsic
    # save
    with open(os.path.join(image_path,"intrinsic.json"),'w') as f:
        json.dump(intrinsics,f)
    return intrinsics
             

if __name__=="__main__":
    pass