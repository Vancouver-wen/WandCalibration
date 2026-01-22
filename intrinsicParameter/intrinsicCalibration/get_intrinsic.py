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
    intrinsic=dict()
    # 检查是否有 intrinsic.json
    if os.path.exists(os.path.join(image_path,"intrinsic.json")):
        with open(os.path.join(image_path,"intrinsic.json"),'r',encoding="utf-8") as f:
            temp=json.load(f)
            for key in natsorted(temp.keys()):
                intrinsic[key]=EasyDict(temp[key])
        if cam_num==len(intrinsic):
            logger.info('find existed intrinsic.json file and load it successfully')
            return [intrinsic[key] for key in natsorted(intrinsic.keys())]
        else:
            not_indexs=[]
            for i in range(cam_num):
                if f'cam{i+1}' not in intrinsic.keys():
                    not_indexs.append(i)
            message=[f'cam{not_index+1}' for not_index in not_indexs]
            logger.warning(f'the format of intrinsic.json is wrong, lacking {message}')
    else:
        not_indexs=[i for i in range(cam_num)]
        message=[f'cam{not_index+1}' for not_index in not_indexs]
        logger.warning(f'can not find intrinsic.json, lacking {message}')
    # 生成 intrinsic.json
    board_type=board_config["type"].strip()
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
        for i in not_indexs
    ]
    intrinsics=dict()
    # n_jobs = len(image_path_lists)
    Parallel(n_jobs=1,backend="threading")(
        delayed(get_each_intrinsic)(
            intrinsicCalibrator,
            image_path_list,
            intrinsics,
            step,
            image_path
        )
        for step,image_path_list in tqdm(list(zip(not_indexs,image_path_lists)))
    )
    intrinsic.update(intrinsics)
    intrinsic={k:v for k,v in sorted(intrinsic.items())} # 对字典进行排序
    # save
    with open(os.path.join(image_path,"intrinsic.json"),'w') as f:
        json.dump(intrinsic,f)
    # format intrinsic result
    return [EasyDict(intrinsic[key]) for key in natsorted(intrinsic.keys())]

def get_each_intrinsic(intrinsicCalibrator,image_path_list,intrinsics,step,image_path):
    intrinsic,report=intrinsicCalibrator(image_path_list=image_path_list,cam_index=step)
    intrinsics[f"cam{step+1}"]=intrinsic
    with open(os.path.join(image_path,f'cam{step+1}_quality.md'),'w',encoding='utf-8') as f:
        f.write(report)

if __name__=="__main__":
    pass
