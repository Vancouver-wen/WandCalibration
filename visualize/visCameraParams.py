import os
import sys
import io
import gc
import random

import numpy as np
import cv2
from natsort import natsorted
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import animation 
from tqdm import tqdm
from loguru import logger
from joblib import Parallel,delayed

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def draw_pole(pole,ax):
    for point in pole:
        ax.scatter3D(point[0],point[1],point[2],s=1,color='lightpink')
        A,B,C=pole
        ax.plot(
            np.array([A[0],B[0]]),
            np.array([A[1],B[1]]),
            np.array([A[2],B[2]]),
            color='lightgray'
        )
        ax.plot(
            np.array([B[0],C[0]]),
            np.array([B[1],C[1]]),
            np.array([B[2],C[2]]),
            color='lightgray'
        )

def vis_camera_params(
        camera_params,
        poles,
        world_coord_param,
        convex_hull,
        save_path
    ):
    if world_coord_param.type=="wand":
        PointCoords=world_coord_param.WandPointCoord 
    elif world_coord_param.type=="labelme":
        PointCoords=world_coord_param.PointCoordinates
    else:
        support_list=["wand","labelme"]
        raise NotImplementedError(f"only support worldCoordParam.type in {support_list}")
    PointCoords=np.array(PointCoords,dtype=np.float32)
    fig = plt.figure()    
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(convex_hull.length[0],convex_hull.length[1]) 
    ax.set_ylim(convex_hull.width[0],convex_hull.width[1]) 
    ax.set_zlim(convex_hull.height[0],convex_hull.height[1]) 
    def init():
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
        # 获取 pole 位置
        # Parallel(n_jobs=-1,backend="threading")(
        #     delayed(draw_pole)(pole,ax)
        #     for pole in poles
        # )
        for pole in random.sample(poles,10):
            for point in pole:
                ax.scatter3D(point[0],point[1],point[2],s=1,color='lightpink')
            A,B,C=pole
            ax.plot(
                np.array([A[0],B[0]]),
                np.array([A[1],B[1]]),
                np.array([A[2],B[2]]),
                color='lightgray'
            )
            ax.plot(
                np.array([B[0],C[0]]),
                np.array([B[1],C[1]]),
                np.array([B[2],C[2]]),
                color='lightgray'
            )
            
        # 绘制 标定场景标志物
        for PointCoord in PointCoords:
            ax.scatter3D(PointCoord[0],PointCoord[1],PointCoord[2],s=3,color='gray')
        for i in range(0,len(PointCoords)):
            ax.plot(
                np.array([PointCoords[i-1][0],PointCoords[i][0]]), 
                np.array([PointCoords[i-1][1],PointCoords[i][1]]), 
                np.array([PointCoords[i-1][2],PointCoords[i][2]]),
                color='gray'
            )

    pbar = tqdm(total=180)
    def rotate(angle): 
        pbar.update(1)
        ax.view_init(elev=30.0, azim=angle)
    # animate, frames=values will be passed to rotate, interval means the the delay between frames in milliseconds
    rot_animation = animation.FuncAnimation(fig, rotate, init_func=init,frames=np.arange(0,362,2), interval=50) # 50ms的间隔
    # save the animat
    if 'gif' not in save_path:
        logger.warning(f"gif not in save_path:{save_path}, replace it!")
        save_path=save_path.split('.')[0]+'.gif'
    rot_animation.save(save_path, dpi=300, writer='imagemagick') 
    # save image
    # image=get_img_from_fig(
    #     fig=fig,
    #     dpi=1000
    # )
    # handle memory leak
    fig.clear()
    fig.clf()
    fig=None
    plt.clf()
    plt.close('all')
    gc.collect()
    # plt.savefig(save_path,dpi=1000)

if __name__=="__main__":
    pass