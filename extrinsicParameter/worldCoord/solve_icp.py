import os
import sys

import numpy as np
import cv2
from loguru import logger

from .icp import best_fit_transform
def solve_icp(source,target):
    # 每一行是一个3D点
    T, R1, t1=best_fit_transform(
        A=np.array(source,dtype=np.float64),
        B=np.array(target,dtype=np.float64)
    )
    R=R1
    t=t1
    # 验证
    pred=R@np.array(source,dtype=np.float64).T+np.repeat(np.expand_dims(t,axis=0),len(source),axis=0).T
    pred=pred.T
    diff=pred-np.array(target,dtype=np.float64)
    diff=np.linalg.norm(diff,axis=1)*1000
    logger.info(f"icp solver error:{diff.tolist()}mm")
    return R,t

if __name__=="__main__":
    pass