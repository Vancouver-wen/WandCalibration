import os
import sys
import json
import copy

import numpy as np
import cv2
import glob
from natsort import natsorted
from loguru import logger
import torch
from torch import nn
from tqdm import tqdm

from extrinsicParameter.refinePose.so3_exp_map import so3_exp_map

class EnhancedLabelme(nn.Module):
    def __init__(
            self,
            cam_params,
            labels,
            point_3ds
        ):
        super().__init__()
        self.cam_0_R=nn.Parameter(
            data=torch.tensor(
                data=self.matrix_to_vector(np.eye(3,3,dtype=np.float32)),
                # data=self.matrix_to_vector(np.array([[-0.49185724,-0.87067203,-0.00258318],[-0.117702,0.06943089,-0.99061879],[ 0.86268343,-0.48693898,-0.13662992]])),
                dtype=torch.float32
            ),
            requires_grad=True 
        )
        self.cam_0_t=nn.Parameter(
            data=torch.tensor(
                data=np.zeros(3,dtype=np.float32),
                # data=np.array([0.38126645,1.05384956,7.04792231]),
                dtype=torch.float32
            ),
            requires_grad=True 
        )
        Ks,dists,Rs,ts=[],[],[],[]
        for cam_param in cam_params:
            Ks.append(cam_param['K'])
            dists.append(cam_param['dist'])
            Rs.append(cam_param['R'])
            ts.append(cam_param['t'])
        self.Ks,self.dists,self.Rs,self.ts=torch.tensor(Ks),torch.tensor(dists),torch.tensor(Rs),torch.tensor(ts)
        self.labels=dict()
        for key in labels:
            self.labels[int(key.replace('cam',''))-1]={k:torch.tensor(v,dtype=torch.float32) for (k,v) in labels[key].items()}
        self.point_3ds=torch.tensor(point_3ds,dtype=torch.float32)
        self.formated_labels,self.formated_mask=self.format_labels(self.labels,len(self.point_3ds),len(cam_params))
        self.get_vmap_func()
    
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
    
    def format_labels(self,labels,point_num,cam_num):
        new_labels=[[np.array([0,0],dtype=np.float32) for _ in range(point_num)] for _ in range(cam_num)]
        new_mask=[[False for _ in range(point_num)] for _ in range(point_num)]
        for key in labels:
            for sub_key in labels[key]:
                point_2d=labels[key][sub_key]
                new_labels[key][sub_key]=point_2d.numpy()
                new_mask[key][sub_key]=True
        new_labels=torch.tensor(new_labels,dtype=torch.float32)
        new_mask=torch.tensor(new_mask,dtype=torch.bool)
        return new_labels,new_mask
    
    def get_new_R(self,old_R,cam_0_R):
        return old_R@self.vector_to_matrix(cam_0_R,batch=False)
    
    def get_new_t(self,old_R,old_t,cam_0_t):
        return old_R@cam_0_t+old_t
    
    def get_each_camera_loss(self,K,dist,R,t,point_2d,point_3d):
        k1,k2,p1,p2,k3=dist # dist的顺序不能错!
        pred_point_2d=self.projectPoint(point_3d,R,t,K,k1,k2,k3,p1,p2)
        loss=torch.norm(point_2d-pred_point_2d)
        return loss
    
    def get_all_camera_loss(self,K,dist,R,t,point_2ds,point_3ds,mask):
        losses=self.vmap_get_each_camera_loss(K,dist,R,t,point_2ds,point_3ds)
        loss=(losses*mask).mean()
        return loss
    
    def get_vmap_func(self):
        self.vmap_get_new_R=torch.vmap(self.get_new_R,in_dims=(0,None))
        self.vmap_get_new_t=torch.vmap(self.get_new_t,in_dims=(0,0,None))
        self.vmap_get_each_camera_loss=torch.vmap(self.get_each_camera_loss,in_dims=(None,None,None,None,0,0))
        self.vmap_get_all_camera_loss=torch.vmap(self.get_all_camera_loss,in_dims=(0,0,0,0,0,None,0))
  
    def forward(self,fast=True):
        if fast:
            # 优化 forward 速度
            new_Rs=self.vmap_get_new_R(self.Rs,self.cam_0_R)
            new_ts=self.vmap_get_new_t(self.Rs,self.ts,self.cam_0_t)
            losses=self.vmap_get_all_camera_loss(self.Ks,self.dists,new_Rs,new_ts,self.formated_labels,self.point_3ds,self.formated_mask)
            loss=losses.mean()
        else:
            new_Rs=[]
            new_ts=[]
            for R,t in list(zip(self.Rs,self.ts)):
                new_R=R@self.vector_to_matrix(self.cam_0_R,batch=False)
                new_t=R@self.cam_0_t+t
                new_Rs.append(new_R)
                new_ts.append(new_t)
            losses=[]
            for key in self.labels: # key代表相机编号
                K,dist,R,t=self.Ks[key],self.dists[key],new_Rs[key],new_ts[key]
                for sub_key in self.labels[key]: # sub_key代表point编号
                    point_3d=self.point_3ds[sub_key]
                    point_2d=self.labels[key][sub_key]
                    k1,k2,p1,p2,k3=dist # dist的顺序不能错!
                    pred_point_2d=self.projectPoint(point_3d,R,t,K,k1,k2,k3,p1,p2)
                    loss=torch.norm(point_2d-pred_point_2d)
                    losses.append(loss)
            loss=torch.stack(losses).mean()
        return loss

    def get_cam_0(self):
        cam_0_R=self.vector_to_matrix(self.cam_0_R,batch=False).detach().numpy()
        cam_0_t=self.cam_0_t.detach().numpy()
        return np.squeeze(cam_0_R),np.squeeze(cam_0_t)
    
def fit_model(
        model:EnhancedLabelme,
        iteration=1000,
        lr=1e-1,
        interval=100,
        print_frequence=10,
        gamma=0.5
    ):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    lrSchedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    model.train()
    try:
        for step in tqdm(range(iteration)):
            loss=model.forward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%print_frequence==0:
                logger.info(f"lr:{lrSchedular.get_last_lr()[-1]:.5f}\t loss is pixel_error:{loss:.5f}")
            if step%interval==0 and step!=0:
                lrSchedular.step()
    except KeyboardInterrupt:
        logger.info(f"early stop enhanced labelme fit")
    except Exception as e:
        logger.info(f"enhanced labelme fit encounter error {e}")
    return model.get_cam_0()

