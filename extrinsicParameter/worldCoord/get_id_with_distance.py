import os
import sys

import cv2
import numpy as np
from loguru import logger

def conform_definition(
        A,B,C,
        d1,d2,d3
    ):
    calculate_d1=np.linalg.norm(A-B)
    calculate_d2=np.linalg.norm(B-C)
    calculate_d3=np.linalg.norm(A-C)
    expect=np.array([d1,d2,d3])
    calculate=np.array([calculate_d1,calculate_d2,calculate_d3])
    conform=np.allclose(expect,calculate,atol=0.05) # 不能超过 50mm
    return conform

def verify_distance(
        A,B,C,
        WandDefinition
    ):
    calculate_d1=np.linalg.norm(A-B)
    calculate_d2=np.linalg.norm(B-C)
    calculate_d3=np.linalg.norm(A-C)
    calculate=np.array([calculate_d1,calculate_d2,calculate_d3])
    d1=WandDefinition[0]
    d2=WandDefinition[1]
    d3=WandDefinition[2]
    expect=np.array([d1,d2,d3])
    diff=np.abs(calculate-expect)*1000
    logger.info(f"check distance\t calculate:{calculate.tolist()}m\t expect:{expect.tolist()}m\t diff:{diff.tolist()}mm")

def get_id_with_distance(
        point_3ds,
        WandDefinition
    ):
    """
    直接通过三重循环找符合 WandDefinition 定义的点
    按照顺序返回 A B C 三点，满足
    AB为短边 BC为长边 AC为斜边
    """
    A,B,C=None,None,None
    for a,point_3d_a in enumerate(point_3ds):
        for b,point_3d_b in enumerate(point_3ds):
            for c,point_3d_c in enumerate(point_3ds):
                if any([a==b,b==c,a==c]): # a b c 必须各不相同
                    continue
                conform=conform_definition(
                    point_3d_a,point_3d_b,point_3d_c,
                    d1=WandDefinition[0],
                    d2=WandDefinition[1],
                    d3=WandDefinition[2]
                )
                if conform:
                    A,B,C=a,b,c
                    break
    if A is None:
        logger.error("can not find L-shape wand, length can not conform")
    else:
        logger.info(f"get id with distance, sequence:{[A,B,C]}")
        verify_distance(point_3ds[A],point_3ds[B],point_3ds[C],WandDefinition)
        return [point_3ds[A],point_3ds[B],point_3ds[C]]




if __name__=="__main__":
    pass