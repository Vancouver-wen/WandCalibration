import os
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

class BoundleAdjustment(nn.Module):
    def __init__(
            self,
            pole_definition,
            cam_num,
            init_intrinsic,
            init_extrinsic,
            image_num,
            init_pole_3ds,
            detected_pole_2ds
        ):
        super().__init__()
        # 常量: pole
        if pole_definition.length_unit=="mm":
            pole_data=torch.tensor(
                data=[
                    pole_definition.d1/1000,
                    pole_definition.d2/1000
                ],
                dtype=torch.float32
            )
        elif pole_definition.length_unit=="m":
            pole_data=torch.tensor(
                data=[
                    pole_definition.d1,
                    pole_definition.d2
                ],
                dtype=torch.float32
            )
        self.pole=pole_data
        # 常量: 2d检测点 -> 含有None
        self.pole_2d_lists=detected_pole_2ds
        # 优化量: camera_param 
        self.camera_params=[]
        for i in range(cam_num):
            K=nn.Parameter(
                data=torch.tensor(
                    data=init_intrinsic[i]['K'],
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            dist=nn.Parameter(
                data=torch.tensor(
                    data=init_intrinsic[i]['dist'],
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            R=nn.Parameter(
                data=torch.tensor(
                    data=init_extrinsic[f'cam_{i}_0']['R'],
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            t=nn.Parameter(
                data=torch.tensor(
                    data=init_extrinsic[f'cam_{i}_0']['t'],
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            self.camera_params.append({
                'K':K,
                'dist':dist,
                'R':R,
                't':t
            })
        # 优化量: 3d points
        self.pole3d_posotions=[]
        for i in range(image_num):
            position=nn.Parameter(
                data=torch.tensor(
                    data=init_pole_3ds[i], # 一行是一个3D point
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            self.pole3d_posotions.append(position)
    
    def projectPoints(self,X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """
        # import pdb;pdb.set_trace()
        x = R@X + t.repeat(X.shape[1],1).T
        x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)
        r = x[0, :] * x[0, :] + x[1, :] * x[1, :]
        x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                                r + 2 * x[0, :] * x[0, :])
        x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                                r + 2 * x[1, :] * x[1, :])
        x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
        x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]
        return x
        
    def forward(
            self,
            line_weight=1,
            length_weight=1,
            
        ):
        loss=0
        for pole_2d_list,pole_3d in list(zip(self.pole_2d_lists,self.pole3d_posotions)):
            # 三个点在一条直线上
            except_pole_3d_1=self.pole[1]/self.pole.sum()*pole_3d[0]+self.pole[0]/self.pole.sum()*pole_3d[2]
            loss_line=torch.norm(pole_3d[1]-except_pole_3d_1)
            # 三点长度为760
            loss_length=torch.abs(torch.norm(pole_3d[0]-pole_3d[2])-self.pole.sum())
            # 3 marker wand loss
            loss_wand=loss_line+loss_length
            # 重投影误差
            loss_reproj=0
            masks=[pole_2d is not None for pole_2d in pole_2d_list]
            for mask,pole_2d,cam_param in list(zip(masks,pole_2d_list,self.camera_params)):
                if mask is False:
                    continue
                pole_2d=torch.tensor(
                    data=pole_2d[0],
                    dtype=torch.float32
                )
                expect_pole_2d=self.projectPoints(
                    X=pole_3d.T,
                    K=cam_param['K'],
                    R=cam_param['R'],
                    t=cam_param['t'],
                    Kd=cam_param['dist']
                )
                import pdb;pdb.set_trace()

            
            return self.camera_params # 先骗一下



if __name__=="__main__":
    pass