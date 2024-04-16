import os
import sys
import json

import glob
from loguru import logger
from natsort import natsorted
from easydict import EasyDict
from tqdm import tqdm
from joblib import Parallel,delayed

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
    board_type=board_config.pop("type").strip()
    # import pdb;pdb.set_trace()
    if board_type=="checkerboard":
        from intrinsicParameter.checkerboardCalibration.get_cam_calibration import IntrinsicCalibration
        height=board_config['height']
        width=board_config['width']
        intrinsicCalibrator=IntrinsicCalibration(height=height,width=width,image_path=image_path)
    elif board_type=="charucoboard":
        height=board_config['height']
        width=board_config['width']
        suqare_length=board_config['square_length']
        marker_length=board_config['marker_length']
        from intrinsicParameter.charucoboardCalibration.get_cam_calibration import IntrinsicCalibration
        intrinsicCalibrator=IntrinsicCalibration(height=height,width=width,square_length=suqare_length,markser_length=marker_length,image_path=image_path)
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
    # n_jobs = len(image_path_lists)
    Parallel(n_jobs=len(image_path_lists),backend="threading")(
        delayed(get_each_intrinsic)(
            intrinsicCalibrator,
            image_path_list,
            intrinsics,
            step
        )
        for step,image_path_list in enumerate(tqdm(image_path_lists))
    )
    # for step,image_path_list in enumerate(tqdm(image_path_lists)):
    #     intrinsic=intrinsicCalibrator(image_path_list=image_path_list)
    #     intrinsics[f"cam{step+1}"]=intrinsic
    # import pdb;pdb.set_trace()
    intrinsics={k:v for k,v in sorted(intrinsics.items())} # 对字典进行排序
    # save
    with open(os.path.join(image_path,"intrinsic.json"),'w') as f:
        json.dump(intrinsics,f)
    # format intrinsic result
    result=[]
    for key in natsorted(intrinsics.keys()):
        result.append(EasyDict(intrinsics[key]))
    return result

def get_each_intrinsic(intrinsicCalibrator,image_path_list,intrinsics,step):
    intrinsic=intrinsicCalibrator(image_path_list=image_path_list,cam_index=step)
    intrinsics[f"cam{step+1}"]=intrinsic


if __name__=="__main__":
    pass
