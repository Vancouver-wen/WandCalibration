import os
import sys
import itertools

import numpy as np
import scipy
import cv2
from joblib import Parallel,delayed
from loguru import logger

def get_sampled_points(convex_hull):
    logger.info(f"sampling voxel in space ..")
    length,width,height=convex_hull.length,convex_hull.width,convex_hull.height
    sample_interval=convex_hull.sample
    length_grid,width_grid,height_grid=np.meshgrid(
        np.linspace(length[0],length[1],int(abs(length[0]-length[1])/sample_interval)),
        np.linspace(width[0],width[1],int(abs(width[0]-width[1])/sample_interval)),
        np.linspace(height[0],height[1],int(abs(height[0]-height[1])/sample_interval))
    )
    points=np.stack([
        length_grid.flatten(),
        width_grid.flatten(),
        height_grid.flatten()
    ])
    return points.T

def can_vis(camera_param,sampled_points):
    image_size=camera_param['image_size']
    K=camera_param['K']
    dist=camera_param['dist']
    R=camera_param['R']
    t=camera_param['t']
    image_points,_=cv2.projectPoints(
        objectPoints=np.array(sampled_points),
        rvec=np.array(R),
        tvec=np.array(t),
        cameraMatrix=np.array(K),
        distCoeffs=np.array(dist)
    )
    image_points=image_points.squeeze()
    vis=np.logical_and.reduce([
        image_points[:,0]>=0,
        image_points[:,0]<image_size[0],
        image_points[:,1]>=0,
        image_points[:,1]<image_size[1]
    ])
    return vis

def get_point_cloud(
        camera_params,
        convex_hull
    ):
    sampled_points=get_sampled_points(convex_hull)
    logger.info(f"project sampled points to judge whether in image")
    vis=Parallel(n_jobs=1,backend="threading")( # min(len(camera_params),os.cpu_count())
        delayed(can_vis)(camera_param,sampled_points)
        for camera_param in camera_params
    )
    vis=np.array(vis).sum(axis=0)
    sampled_points=sampled_points[vis>=2]
    import pdb;pdb.set_trace()
    return sampled_points