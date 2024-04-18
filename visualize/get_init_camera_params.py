import os
import sys

from natsort import natsorted

def get_init_camera_params(cam_num,intrinsics,extrinsics):
    assert cam_num==len(intrinsics)
    assert cam_num==len(extrinsics)
    camera_params=[]
    keys=natsorted(extrinsics.keys())
    for intrinsic,key in list(zip(intrinsics,keys)):
        extrinsic=extrinsics[key]
        camera_param=dict()
        camera_param.update(intrinsic)
        camera_param.update(extrinsic)
        camera_params.append(camera_param)
    return camera_params

if __name__=="__main__":
    pass