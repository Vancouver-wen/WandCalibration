import os
import sys
import json

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from joblib import Parallel,delayed
import  torch.multiprocessing as mp

class BoundleAdjustment(nn.Module):
    def __init__(
            self,
            pole_definition,
            cam_num,
            init_intrinsic,
            init_extrinsic,
            image_num,
            init_pole_3ds,
            detected_pole_2ds,
            save_path,
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
        self.camera_params=nn.ParameterList()
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
                    data=init_intrinsic[i]['dist'][0],
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
            camera_param=nn.ParameterDict({
                'K':K,
                'dist':dist,
                'R':R,
                't':t
            })
            
            self.camera_params.append(camera_param)
        # 优化量: 3d points
        self.pole3d_posotions=nn.ParameterList()
        for i in range(image_num):
            position=nn.Parameter(
                data=torch.tensor(
                    data=init_pole_3ds[i], # 一行是一个3D point
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            self.pole3d_posotions.append(position)
        self.save_path=save_path
    
    def get_dict(self):
        camera_params=[]
        for camera_param in self.camera_params:
            K=camera_param['K'].tolist()
            dist=camera_param['dist'].tolist()
            R=camera_param['R'].tolist()
            t=camera_param['t'].tolist()
            camera_params.append({
                'K':K,'dist':dist,'R':R,'t':t
            })
        pole_3ds=[]
        for pole_3d in self.pole3d_posotions:
            pole_3ds.append(pole_3d.tolist())
        output={
            'calibration':camera_params,
            'poles':pole_3ds
        }
        with open(self.save_path,'w') as f:
            json.dump(output,f)
        return output
    
    def projectPoints(self,X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """
        # import pdb;pdb.set_trace()
        N=X.shape[1]
        k1,k2,p1,p2,k3=Kd[0],Kd[1],Kd[2],Kd[3],Kd[4]
        pixels=[]
        for i in range(N):
            x = R@X[:,i] + t
            x=torch.divide(x,x[-1])
            r=torch.norm(x[:2])
            x_undistorted=x[0]*(1+k1*pow(r,2)+k2*pow(r,4)+k3*pow(r,6))+2*p1*x[0]*x[1]+p2*(pow(r,2)+2*pow(x[0],2))
            y_undistorted=x[1]*(1+k1*pow(r,2)+k2*pow(r,4)+k3*pow(r,6))+p1*(pow(r,2)+2*pow(x[1],2))+2*p2*x[0]*x[1]
            f_x,f_y,u_0,v_0=K[0][0],K[1][1],K[0,2],K[1,2]
            u=f_x*x_undistorted + u_0
            v=f_y*y_undistorted + v_0
            pixel=torch.cat((torch.unsqueeze(u,dim=0),torch.unsqueeze(v,dim=0)))
            pixels.append(pixel)
        output=torch.stack(pixels,dim=0)
        return output.T
    
    def forward_iter(
            self,
            pole_2d_list,
            pole_3d
        ):
        # 三个点在一条直线上
        except_pole_3d_1=(self.pole[1]*pole_3d[0]+self.pole[0]*pole_3d[2])/self.pole.sum()
        loss_line=torch.norm(pole_3d[1]-except_pole_3d_1)
        # 三点长度为760
        loss_length=torch.abs(torch.norm(pole_3d[0]-pole_3d[2])-self.pole.sum())
        # 3 marker wand loss
        loss_wand=self.line_weight*loss_line+self.length_weight*loss_length
        # 重投影误差
        loss_reproj=[]
        masks=[pole_2d is not None for pole_2d in pole_2d_list]
        for mask,pole_2d,cam_param in list(zip(masks,pole_2d_list,self.camera_params)):
            if mask is False: # 此时 pole_2d 是 None
                continue
            pole_2d=torch.tensor(
                data=pole_2d[0],
                dtype=torch.float32
            )
            expect_pole_2d=self.projectPoints(
                X=pole_3d.T, # 转换成每一列是一点3d点
                K=cam_param['K'],
                R=cam_param['R'],
                t=cam_param['t'],
                Kd=cam_param['dist']
            )
            diff=pole_2d-expect_pole_2d.T
            loss_reproj.append(
                torch.unsqueeze(
                    self.reproj_weight*torch.norm(diff,dim=1).mean(),
                    dim=0
                )
            )
        loss_reproj=torch.concat(loss_reproj,dim=0).mean()
        loss=loss_wand+loss_reproj
        return torch.unsqueeze(loss,dim=0)

    @torch.compile
    def forward(
            self,
            line_weight=0,
            length_weight=0,
            reproj_weight=1.0
        ):
        self.line_weight=line_weight
        self.length_weight=length_weight
        self.reproj_weight=reproj_weight
        # autograd does not support crossing process boundaries
        # with mp.Pool(processes=6) as pool:
        #     handle=pool.starmap_async(
        #         func=self.forward_iter,
        #         iterable=list(zip(self.pole_2d_lists,self.pole3d_posotions))
        #     )
        #     losses=handle.get()

        losses=Parallel(n_jobs=1,backend="threading")(
            delayed(self.forward_iter)(pole_2d_list,pole_3d)
            for pole_2d_list,pole_3d in list(zip(self.pole_2d_lists,self.pole3d_posotions))
        )
        loss=torch.mean(torch.concat(losses))
        return loss



if __name__=="__main__":
    pass