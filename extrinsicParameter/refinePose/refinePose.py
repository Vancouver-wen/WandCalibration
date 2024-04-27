import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import math
import resource

import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import torch.multiprocessing as mp

from .normalizedImagePlane import get_undistort_points
from .multiViewTriangulate import normalized_pole_triangulate
from .boundleAdjustment import BoundleAdjustment
from utils.verifyAccuracy import verify_accuracy

def multi_thread_train(
        init_error:float,
        model:BoundleAdjustment,
        pole_lists,
    ):
    list_len=model.list_len
    mask=torch.ones(list_len,dtype=torch.bool)
    model.train()
    lr=5e-4*init_error # lr=5e-3 是比较合适的数值
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    lrSchedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # ExponentialLR(optimizer, gamma=0.5)
    iteration = 1000 # iteration = 1000
    start=time.time()
    for step in tqdm(range(iteration)):
        loss=model.forward(
            mask,
            line_weight=1.0,
            length_weight=1.0,
            reproj_weight=1.0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%10==0:
            logger.info(f"lr:{lrSchedular.get_last_lr()[-1]:.5f}\t loss:{loss:.5f}")
            output=model.get_dict() # 保存结果
            # import pdb;pdb.set_trace() # p intrinsics -> 有 image_size 
            verify_accuracy(
                camera_params=output['calibration'],
                pole_3ds=output['poles'],
                pole_lists=pole_lists,
                time_consume=time.time()-start
            )
            start=time.time()
        if step%100==0 and step!=0:
            lrSchedular.step()

def sub_process_train(
        rank:int,
        barrier,
        lr:float,
        model:BoundleAdjustment,
        losses:mp.Queue,
        pole_lists,
        iteration:int
    ):
    logger.info(f"sub process rank:{rank} has been started")
    cpu_count=model.cpu_count
    list_len=model.list_len
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    lrSchedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # ExponentialLR(optimizer, gamma=0.5)
    step_frequence=100 # int(iteration/math.sqrt(cpu_count))
    start=time.time()
    if rank==0:
        loop=tqdm(range(iteration))
    else:
        loop=range(iteration)
    for step in loop:
        try:
            torch.manual_seed(step)
            mask_index=torch.multinomial(input=torch.ones(cpu_count),num_samples=list_len,replacement=True)
            loss=model.forward(
                mask=(mask_index==rank),
                line_weight=1.0,
                length_weight=1.0,
                reproj_weight=1.0,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%10==0:
                losses.put(loss.item())
            barrier.wait() # 同步
            if rank==0 and step%10==0:
                avg_loss=[]
                while not losses.empty():
                    avg_loss.append(losses.get())
                if len(avg_loss)!=cpu_count:
                    logger.warning(f"expect {cpu_count} num losses but get {len(avg_loss)} num losses")
                avg_loss=np.array(avg_loss)
                logger.info(f"lr:{lrSchedular.get_last_lr()[-1]:.5f}\t avg_loss:{avg_loss.mean():.5f}\t each_losses:{np.round(avg_loss,1).tolist()}")
                output=model.get_dict() # 保存结果
                # import pdb;pdb.set_trace() # p intrinsics -> 有 image_size 
                verify_accuracy(
                    camera_params=output['calibration'],
                    pole_3ds=output['poles'],
                    pole_lists=pole_lists,
                    time_consume=time.time()-start
                )
                start=time.time()
            if step%step_frequence==0 and step!=0:
                lrSchedular.step()
        except KeyboardInterrupt as e:
            logger.info(f"rank {rank} daemon early stop")
            break

def multi_process_train(
        init_error:float,
        model:BoundleAdjustment,
        pole_lists,
    ):
    mp.get_all_start_methods() ['fork', 'spawn', 'forkserver']
    """
    使用fork启动多进程, 会导致无法创建大tensor,优化器会fail
    使用spawn启动多进程,虽然启动慢,但能够创建爱你较大的tensor
    """
    # 设置子进程的内存限制（单位：字节）
    # memory_limit_bytes = 64 * 1024 * 1024 * 1024  # 4GB
    # resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    # resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
    mp_start_method=mp.get_start_method() 
    mp_use_method='spawn'
    if mp_start_method!=mp_use_method:
        logger.warning(f"multiprocessing start method change to {mp_use_method}")
        mp.set_start_method(mp_use_method,force=True)
    mp.set_sharing_strategy('file_system') 
    cpu_count=model.cpu_count
    barrier = mp.Barrier(cpu_count)
    model.share_memory() # 在不同的进程间同步梯度
    model.train()
    lr=5e-4*init_error/cpu_count # lr=5e-3 是比较合适的数值
    iteration = max(int(1000/math.sqrt(cpu_count)),500) # iteration = 1000
    losses=mp.Queue() # put get empty
    processes=[]
    for rank in range(cpu_count):
        p=mp.Process(
            target=sub_process_train,
            args=(rank,barrier,lr,model,losses,pole_lists,iteration),
            name=f"train{rank}",
            daemon=True
        )
        p.daemon=True
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def ddp_train(
        init_error,
        model,
        pole_lists
    ):
    raise NotImplementedError(f"multiprocess can work very well, ddp train will be developed later")

def get_refine_pose(
        cam_num,
        pole_lists,
        intrinsics,
        pole_param,
        init_poses,
        save_path,
        refine_mode
    ):
    # 先用cv2.undistort 把像素坐标系转换成归一化图像坐标系
    undistorted_pole_lists=get_undistort_points(cam_num,pole_lists,intrinsics)
    # 在归一化平面上作三角测量获取3D点初值
    pole_3ds=normalized_pole_triangulate(
        cam_num=cam_num,
        normalized_pole_lists=undistorted_pole_lists,
        poses=init_poses,
        intrinsics=intrinsics
    )
    # 筛选用于 boundle adjustment 的 list
    # 要过滤到无法三角化的点,是因为 boundle adjustment的 优化参数中有 3d point
    mask=[pole_3d is not None for pole_3d in pole_3ds]
    masked_pole_lists=[
        item[1]
        for item in filter(lambda x:x[0],list(zip(mask,pole_lists)))
    ]
    masked_pole_3ds=[
        item[1]
        for item in filter(lambda x:x[0],list(zip(mask,pole_3ds)))
    ]
    # boundle adjustment
    myBoundAdjustment=BoundleAdjustment(
        pole_definition=pole_param,
        cam_num=cam_num,
        init_intrinsic=intrinsics,
        init_extrinsic=init_poses,
        image_num=np.array(mask).sum(),
        init_pole_3ds=masked_pole_3ds,
        detected_pole_2ds=masked_pole_lists,
        save_path=save_path
    )
    # 计算初始化精度
    logger.info(f"calculate boundle adjustment init pixel error to set init learning rate")
    output=myBoundAdjustment.get_dict() # 保存结果
    init_error=verify_accuracy(
        camera_params=output['calibration'],
        pole_3ds=output['poles'],
        pole_lists=pole_lists
    )
    if init_error>=20:
        logger.warning(f"init pixel error too large, please check intrinsic calibration!")

    # 训练
    if refine_mode=='thread':
        multi_thread_train(
            init_error=init_error,
            model=myBoundAdjustment,
            pole_lists=pole_lists
        )
    elif refine_mode=='process':
        multi_process_train(
            init_error=init_error,
            model=myBoundAdjustment,
            pole_lists=pole_lists
        )
    elif refine_mode=='distributed':
        ddp_train(
            init_error=init_error,
            model=myBoundAdjustment,
            pole_lists=pole_lists
        )
    else:
        support_list=['thread','process','distributed']
        raise NotImplementedError(f"refine mode only support {support_list}")

        
    


if __name__=="__main__":
    pass