import os
import sys
import json
from functools import cache
import copy
import random
from itertools import compress

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from joblib import Parallel,delayed
import torch.multiprocessing as mp
from loguru import logger
from extrinsicParameter.refinePose.so3_exp_map import so3_exp_map

class BoundleAdjustment(nn.Module):
    def __init__(
            self,
            max_process,
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
        self.avails,self.pole_2d_lists=self.get_pole_2d_lists(detected_pole_2ds)
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
                    data=np.squeeze(np.array(init_intrinsic[i]['dist'],dtype=np.float32)),
                    dtype=torch.float32
                ),
                requires_grad=True
            )
            R=nn.Parameter(
                data=torch.tensor(
                    data=self.matrix_to_vector(init_extrinsic[f'cam_{i}_0']['R']),
                    dtype=torch.float32
                ),
                requires_grad=True # if i==0 else False # 第一个相机不需要反传,反传的效果会更好
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
        self.pole3d_posotions=nn.Parameter(data=torch.tensor(init_pole_3ds,dtype=torch.float32),requires_grad=True)
        # self.pole3d_posotions=nn.ParameterList()
        # for i in range(image_num):
        #     position=nn.Parameter(
        #         data=torch.tensor(
        #             data=init_pole_3ds[i], # 一行是一个3D point
        #             dtype=torch.float32
        #         ),
        #         requires_grad=True
        #     )
        #     self.pole3d_posotions.append(position)
        
        assert len(self.pole_2d_lists)==len(self.pole3d_posotions)
        self.list_len=len(self.pole_2d_lists)
        if self.list_len<100:
            logger.warning(f"pole num:{self.list_len}<100, too low")
        self.cpu_count=min(os.cpu_count(),max(int(self.list_len/100),1),max_process) # 每个进程至少100组图片
        self.cpu_count=max(self.cpu_count,1) # 防止max_process值异常
        self.save_path=save_path
        self.resolutions=[intrinsic['image_size'] for intrinsic in init_intrinsic]
        self.has_vmap=False

    def matrix_to_vector(self,rotation_matrix):
        # 只需要支持numpy就可以了, 不需要反向传播!
        rotation_matrix=np.array(rotation_matrix,dtype=np.float32)
        rvec,_=cv2.Rodrigues(rotation_matrix)
        return np.squeeze(rvec)
    
    def vector_to_matrix(self,rotation_vector,batch=False):
        # 需要支持tensor反向传播
        if batch:
            return so3_exp_map(rotation_vector)
        else:
            rotation_matrix=so3_exp_map(torch.unsqueeze(rotation_vector,dim=0))
            return torch.squeeze(rotation_matrix)
    
    def get_vmap_func(self):
        self.vmap_projectPoint=torch.vmap(self.projectPoint,in_dims=(1,None,None,None,None,None,None,None,None))
        self.vmap_projectIter=torch.vmap(self.projectIter,in_dims=(0,None,0,0,0,0))
        self.vmap_forward_iter=torch.vmap(self.forward_iter,in_dims=(0,0,0,None,None,None,None))
    
    def get_pole_2d_lists(self,detected_pole_2d_lists):
        size=(3,2)
        avails=[]
        pole_2d_lists=[]
        for detected_pole_2d_list in detected_pole_2d_lists:
            avail=[]
            pole_2d_list=[]
            for detected_pole_2d in detected_pole_2d_list:
                if detected_pole_2d is None:
                    avail.append(False)
                    pole_2d_list.append(torch.zeros(size,dtype=torch.float32).numpy())
                else:
                    avail.append(True)
                    pole_2d=torch.tensor(
                        data=detected_pole_2d[0],
                        dtype=torch.float32
                    ).numpy()
                    pole_2d_list.append(pole_2d)
                    assert pole_2d.shape==size,f"pole_2d_list.shape should be {size}"
            avails.append(avail)
            pole_2d_lists.append(pole_2d_list)
        avails=torch.tensor(
            data=avails,
            dtype=torch.bool,
            requires_grad=False
        )
        # import pdb;pdb.set_trace()
        pole_2d_lists=torch.tensor(
            data=pole_2d_lists,
            dtype=torch.float32,
            requires_grad=False
        )
        return avails,pole_2d_lists

    def get_dict(self):
        camera_params=[]
        for camera_param,resolution in list(zip(self.camera_params,self.resolutions)):
            K=camera_param['K'].tolist()
            dist=camera_param['dist'].tolist()
            R=self.vector_to_matrix(camera_param['R'],batch=False).tolist()
            t=camera_param['t'].tolist()
            camera_params.append({
                'image_size': resolution,
                'K':K,'dist':dist,
                'R':R,'t':t
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
    
    # @torch.compile
    def projectPoint(self,X,R,t,K,k1,k2,k3,p1,p2):
        x = R@X+ t
        x=torch.divide(x,x[-1])
        r=torch.norm(x[:2])
        r_2,r_4,r_6=pow(r,2),pow(r,4),pow(r,6)
        x_undistorted=x[0]*(1+k1*r_2+k2*r_4+k3*r_6)+2*p1*x[0]*x[1]+p2*(r_2+2*pow(x[0],2))
        y_undistorted=x[1]*(1+k1*r_2+k2*r_4+k3*r_6)+p1*(r_2+2*pow(x[1],2))+2*p2*x[0]*x[1]
        f_x,f_y,u_0,v_0=K[0][0],K[1][1],K[0,2],K[1,2]
        u=f_x*x_undistorted + u_0
        v=f_y*y_undistorted + v_0
        pixel=torch.cat((torch.unsqueeze(u,dim=0),torch.unsqueeze(v,dim=0)))
        return pixel

    def projectPoints(self,X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """
        k1,k2,p1,p2,k3=Kd[0],Kd[1],Kd[2],Kd[3],Kd[4]
        result=self.vmap_projectPoint(X,R,t,K,k1,k2,k3,p1,p2)
        return result.T
    
    def projectIter(self,pole_2d,X,K,R,t,Kd):
        expect_pole_2d=self.projectPoints(X,K,R,t,Kd)
        diff=pole_2d-expect_pole_2d.T
        loss_reproj=self.reproj_weight*torch.norm(diff,dim=1).mean()
        return loss_reproj
    
    def orthogonal(self,camera_params):
        losses=[]
        for camera_param in camera_params:
            rotation_matrix=camera_param['R']
            diff=rotation_matrix@rotation_matrix.T-torch.eye(rotation_matrix.shape[0])
            loss=torch.norm(diff)
            losses.append(loss)
        orthogonal_loss=torch.stack(losses).sum()
        return self.orthogonal_weight*orthogonal_loss

    def forward_iter(
            self,
            pole_2d_list,
            pole_3d,
            avail,
            Ks,Rs,ts,Kds
        ):
        # 三个点在一条直线上
        except_pole_3d_1=(self.pole[1]*pole_3d[0]+self.pole[0]*pole_3d[2])/self.pole.sum()
        loss_line=torch.norm(pole_3d[1]-except_pole_3d_1)
        # 三点长度为760
        loss_length=torch.abs(torch.norm(pole_3d[0]-pole_3d[2])-self.pole.sum())
        # 3 marker wand loss
        loss_wand=self.line_weight*loss_line+self.length_weight*loss_length
        # 重投影误差
        # 只支持 rotation_vector 作为中间表示
        Rs=self.vector_to_matrix(Rs,batch=True)
        loss_reproj=self.vmap_projectIter(pole_2d_list,pole_3d.T,Ks,Rs,ts,Kds)
        # RuntimeError: vmap: We do not support batching operators that can support dynamic shape. Attempting to batch over indexing with a boolean mask.
        loss_reproj=loss_reproj*avail  # 起到mask的作用 
        # loss_reproj=loss_reproj/loss_reproj.detach() # * 的效果不明显; /的效果很差
        loss_reproj=loss_reproj.mean()
        loss=loss_wand+loss_reproj
        # print({
        #     'loss': loss.item(),
        #     'loss_wand':loss_wand.item(),
        #     'loss_reproj':loss_reproj.item()
        # })
        return torch.unsqueeze(loss,dim=0)

    def forward(
            self,
            mask, # 用于多进程中筛选 mini-batch
            line_weight=1.0,
            length_weight=1.0,
            reproj_weight=1.0,
            orthogonal_weight=10.0
        ):
        if not self.has_vmap:
            self.get_vmap_func()
        self.line_weight=line_weight
        self.length_weight=length_weight
        self.reproj_weight=reproj_weight
        self.orthogonal_weight=orthogonal_weight
        # 转换 self.camera_params
        Ks,Rs,ts,Kds=[],[],[],[]
        for cam_param in self.camera_params:
            Ks.append(cam_param['K'])
            Rs.append(cam_param['R'])
            ts.append(cam_param['t'])
            Kds.append(cam_param['dist'])
        Ks,Rs,ts,Kds=torch.stack(Ks),torch.stack(Rs),torch.stack(ts),torch.stack(Kds)

        sequential_losses=self.vmap_forward_iter(
            self.pole_2d_lists[mask],
            self.pole3d_posotions[mask],
            self.avails[mask],
            Ks,Rs,ts,Kds
        )

        sequential_loss=torch.mean(sequential_losses)
        loss = sequential_loss
        return loss



if __name__=="__main__":
    pass