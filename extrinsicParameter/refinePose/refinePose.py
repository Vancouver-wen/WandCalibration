import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import math
import threading

import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler

from .normalizedImagePlane import get_undistort_points
from .multiViewTriangulate import normalized_pole_triangulate
from .boundleAdjustment import BoundleAdjustment
from .dataLoader import BoundAdjustmentDataset
from .linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from utils.verifyAccuracy import verify_accuracy

def multi_thread_train(
        init_error:float,
        model:BoundleAdjustment,
        pole_lists,
    ):
    list_len=model.list_len
    myDataset=BoundAdjustmentDataset(list_len)
    myDataLoader=DataLoader(
        dataset=myDataset,
        batch_size=100,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    model.train()
    lr=min(5e-4*init_error,1e-1) # lr=5e-3 是比较合适的数值
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    lrSchedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # ExponentialLR(optimizer, gamma=0.5)
    iteration = 1000 # iteration = 1000
    start=time.time()
    for step in tqdm(range(iteration)):
        for batch in myDataLoader:
            time1=time.time()
            loss=model.forward(
                mask=batch,
                line_weight=1.0,
                length_weight=1.0,
                reproj_weight=1.0,
                orthogonal_weight=10.0
            )
            time2=time.time()
            optimizer.zero_grad()
            time3=time.time()
            loss.backward()
            time4=time.time()
            optimizer.step()
            # logger.info(f"backward time consume:{time.time()-time1} forward:{time2-time1} zero_grad:{time3-time2} backward:{time4-time3} step:{time.time()-time4}")
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

def setup(rank, world_size):
    master_addr='127.0.0.1'
    master_port='14514'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    # initialize the process group
    dist.init_process_group(
        "gloo", 
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank, 
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def sub_process_train(
        refine_mode:str,
        rank:int,
        barrier,
        verify_message:mp.Queue,
        lr:float,
        model:BoundleAdjustment,
        dataset:Dataset,
        losses:mp.Queue,
        pole_lists,
        iteration:int,
        thread_count=1
    ):
    torch.set_num_threads(thread_count)
    assert refine_mode in ['process','distributed']
    logger.info(f"{'ddp' if refine_mode=='distributed' else 'sub'} process rank:{rank} has been started")
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
    if refine_mode=="process":
        myDataLoader=DataLoader(
            dataset=dataset,
            batch_size=100,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
    elif refine_mode=="distributed":
        setup(rank, cpu_count)
        model = DDP(model, device_ids=None)
        mySampler=DistributedSampler(
            dataset=dataset,
            shuffle=True,
            drop_last=True
        )
        myDataLoader=DataLoader(
            dataset=dataset,
            batch_size=100,
            num_workers=0,
            shuffle=True,
            sampler=mySampler,
            drop_last=True
        )
    else:
        pass
    for step in loop:
        try:
            time0=time.time()
            torch.manual_seed(step)
            mask_index=torch.multinomial(input=torch.ones(cpu_count),num_samples=list_len,replacement=True)
            # logger.info(f"shuffle time consume:{time.time()-time0}")
            # for batch in myDataLoader:
            time1=time.time()
            loss=model.forward(
                mask=torch.tensor((mask_index==rank),dtype=torch.bool,requires_grad=False),
                # mask=batch,
                line_weight=1.0,
                length_weight=1.0,
                reproj_weight=1.0,
                orthogonal_weight=10.0
            )
            time2=time.time()
            optimizer.zero_grad()
            time3=time.time()
            loss.backward()
            time4=time.time()
            optimizer.step()
            # logger.info(f"backward time consume:{time.time()-time1} forward:{time2-time1} zero_grad:{time3-time2} backward:{time4-time3} step:{time.time()-time4}")
            if step%10==0:
                losses.put((rank,loss.item()))
            # if step%10==0: # 不是同步导致的性能障碍
            if refine_mode=="process":
                barrier.wait() # 同步
            else: # 'distributed'
                dist.barrier()
            if rank==0 and step%10==0:
                avg_loss=dict()
                while not losses.empty():
                    key,value=losses.get()
                    avg_loss[key]=value
                avg_loss={k:v for k,v in sorted(avg_loss.items(),key=lambda x:x[0])}
                avg_loss=list(avg_loss.values())
                if len(avg_loss)!=cpu_count:
                    logger.warning(f"expect {cpu_count} num losses but get {len(avg_loss)} num losses")
                avg_loss=np.array(avg_loss)
                logger.info(f"lr:{lrSchedular.get_last_lr()[-1]:.5f}\t avg_loss:{avg_loss.mean():.5f}")
                logger.info(f"each_losses:{np.round(avg_loss,1).tolist()}")
                if refine_mode=="distributed":
                    output=model.module.get_dict()
                else:
                    output=model.get_dict() # 保存结果
                verify_message.put({
                    'camera_params':output['calibration'],
                    'pole_3ds':output['poles'],
                    'pole_lists':pole_lists,
                    'time_consume':time.time()-start
                })
                start=time.time()
            if step%step_frequence==0 and step!=0:
                lrSchedular.step()
        except KeyboardInterrupt as e:
            logger.info(f"{'ddp' if refine_mode=='distributed' else 'sub'} process rank {rank} daemon early stop")
            verify_message.put(None)
            break
    verify_message.put(None)

def verify_process(verify_message:mp.Queue):
    logger.info(f'verify process has been started')
    message=dict()
    while True:
        overstock=0
        while not verify_message.empty():
            message=verify_message.get() # 读取积压的message,只保存最新的
            overstock+=1
            time.sleep(0.1)
        if message is None:
            logger.info(f'verify process receieve None, stop ..')
            break
        if message: # 如果message不是空向量
            verify_accuracy(**message)
            logger.warning(f"clean overstock message number: {overstock}")

def multi_process_train(
        refine_mode:str,
        init_error:float,
        model:BoundleAdjustment,
        pole_lists,
    ):
    # mp.get_all_start_methods() ['fork', 'spawn', 'forkserver']
    """
    使用fork启动多进程, 会导致无法创建大tensor,优化器会fail
    使用spawn启动多进程,虽然启动慢,但能够创建较大的tensor
    """
    mp_start_method=mp.get_start_method() 
    mp_use_method='spawn'
    if mp_start_method!=mp_use_method:
        logger.warning(f"multiprocessing start method change to {mp_use_method}")
        mp.set_start_method(mp_use_method,force=True)
    # 获取模型以外的训练超参
    cpu_count=model.cpu_count
    thread_count=int(os.cpu_count()/cpu_count+0.5) # define the num threads used in current sub-processes
    list_len=model.list_len
    myDataset=BoundAdjustmentDataset(list_len)
    lr=min(5e-3*init_error/cpu_count,1e-2) # lr=5e-3 是比较合适的数值
    iteration = max(int(1000/math.sqrt(cpu_count)),1000) # iteration = 1000
    losses=mp.Queue() # put get empty
    verify_message=mp.Queue()
    mp.set_sharing_strategy('file_system')
    if refine_mode=="process":
        barrier = mp.Barrier(cpu_count)
        model.share_memory() # 在不同的进程间同步梯度
        model.train()
    elif refine_mode=="distributed":
        barrier=None
    else:
        support_list=['process','distributed']
        raise NotImplementedError(f"multi process train only support {support_list}")

    processes=[]
    p=mp.Process(
        target=verify_process,
        args=(verify_message,),
        name=f'verify',
        daemon=True
    )
    p.start()
    processes.append(p)
    for rank in range(cpu_count):
        p=mp.Process(
            target=sub_process_train,
            args=(refine_mode,rank,barrier,verify_message,lr,model,myDataset,losses,pole_lists,iteration,thread_count),
            name=f"train{rank}",
            daemon=True
        )
        p.daemon=True
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def get_refine_pose(
        max_process,
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
        max_process=max_process,
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
    elif refine_mode=='process' or refine_mode=='distributed':
        multi_process_train(
            refine_mode=refine_mode,
            init_error=init_error,
            model=myBoundAdjustment,
            pole_lists=pole_lists
        )
    else:
        support_list=['thread','process','distributed']
        raise NotImplementedError(f"refine mode only support {support_list}")

        
    


if __name__=="__main__":
    pass