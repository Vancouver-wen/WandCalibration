import os
import sys

import numpy as np
import cv2
from natsort import natsorted
import glob
from matplotlib import pyplot as plt

def vis_camera_params(
        camera_params
    ):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-10, 10) 
    ax.set_ylim(-10, 10) 
    ax.set_zlim(-10, 10) 
    # 获取相机位置
    for camera_param in camera_params:
        R=np.array(camera_param['R'])
        t=np.array(camera_param['t'])
        cam_position=np.linalg.inv(R)@(-t)
        x_position=np.linalg.inv(R)@(np.array([1,0,0])-t)
        y_position=np.linalg.inv(R)@(np.array([0,1,0])-t)
        z_position=np.linalg.inv(R)@(np.array([0,0,1])-t)
        ax.scatter3D(cam_position[0],cam_position[1],cam_position[2],s=3,color='yellow')
        ax.scatter3D(x_position[0],x_position[1],x_position[2],s=3,color='red')
        ax.scatter3D(y_position[0],y_position[1],y_position[2],s=3,color='green')
        ax.scatter3D(z_position[0],z_position[1],z_position[2],s=3,color='blue')
        # import pdb;pdb.set_trace()
        ax.plot(
            np.array([cam_position[0],x_position[0]]), 
            np.array([cam_position[1],x_position[1]]), 
            np.array([cam_position[2],x_position[2]]),
            color='gray'
        )
        ax.plot(
            np.array([cam_position[0],y_position[0]]), 
            np.array([cam_position[1],y_position[1]]), 
            np.array([cam_position[2],y_position[2]]),
            color='gray'
        )
        ax.plot(
            np.array([cam_position[0],z_position[0]]), 
            np.array([cam_position[1],z_position[1]]), 
            np.array([cam_position[2],z_position[2]]),
            color='gray'
        )
    # plt.show()
    plt.savefig("./world.jpg",dpi=1000)

if __name__=="__main__":
    pass