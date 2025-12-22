import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import math
import copy
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
from torch.nn.utils import clip_grad_norm_

from .normalizedImagePlane import get_undistort_points
from .multiViewTriangulate import normalized_pole_triangulate
from .boundleAdjustment import BoundleAdjustment
from .dataLoader import BoundAdjustmentDataset
from .linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from utils.verifyAccuracy import verify_accuracy

def update(model:torch.nn.Module,lr=None,param_names=[])->list:
    param_list=[]
    for name,p in model.named_parameters():
        param_dict=dict()
        param_dict['params']=p
        if ('K' in name) or ('dist' in name):
            param_dict['lr']=lr if lr is not None else 0.0001#  min(5e-4*init_error,1e-1) min(5e-3*init_error/cpu_count,1e-2) # lr=5e-3 是比较合适的数值
            # continue
        elif 'R' in name:
            param_dict['lr']=lr if lr is not None else 0.001
        elif 't' in name:
            param_dict['lr']=lr if lr is not None else 0.005
        else:
            param_dict['lr']=lr if lr is not None else 0.005
        param_names.append(name)
        param_list.append(param_dict)
    return param_list

def nan_grad_to_zero(model:torch.nn.Module,nan_to_num=True):
    """
    检查所有参数的梯度是否存在NaN 或 Inf
    将梯度中的NaN和Inf替换为0
    """
    parameters=model.named_parameters()
    nan_params,inf_params = [],[]
    for name, param in parameters:
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_params.append(name)
                # logger.warning(f"NaN gradient detected in parameter {name}, shape: {param.shape}")
            if torch.isinf(param.grad).any():
                inf_params.append(name)
            if nan_to_num:
                param.grad.data=torch.nan_to_num(param.grad.data,nan=0.0,posinf=0.0, neginf=0.0)
    # logger.warning(f"NaN gradient detected in parameter {nan_params}")
    # logger.warning(f"Inf gradient detected in parameter {inf_params}")
    return nan_params,inf_params

def apply_parameter_constraints(model:torch.nn.Module):
    # with torch.no_grad():
    for name,p in model.named_parameters():
        if 'pole' in name:
            p.data=torch.clamp(p.data,min=-250.0,max=250.0)
        if 'dist' in name:
            p.data=torch.clamp(p.data,min=-0.5, max=0.5)

def multi_thread_train(
        init_error:float,
        model:BoundleAdjustment,
        pole_lists,
        weights
    ):
    list_len=model.list_len
    myDataset=BoundAdjustmentDataset(list_len)
    myDataLoader=DataLoader(
        dataset=myDataset,
        batch_size=100, # batch size不要开太大，随机梯度下降 随机才是灵魂
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    model.train()
    param_names=[]
    lr=min(5e-4*init_error,1e-1)
    # optimizer = torch.optim.Adam(model.parameters(),lr=min(5e-4*init_error,1e-1))
    optimizer = torch.optim.Adam(update(model,lr,param_names))
    lrSchedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # ExponentialLR(optimizer, gamma=0.5)
    iteration = 1000 # iteration = 1000
    start=time.time()
    loss=0.0
    for step in tqdm(range(iteration)):
        for batch in myDataLoader:
            loss=model.forward(
                mask=batch,
                line_weight=weights.line_weight,
                length_weight=weights.length_weight,
                reproj_weight=weights.reproj_weight,
                orthogonal_weight=weights.orthogonal_weight,
                quaternion_weight=weights.quaternion_weight
            )
            optimizer.zero_grad()
            loss.backward()
            # clip the grad 梯度裁剪没有效果
            # if weights.max_norm:
            #     clip_grad_norm_(model.parameters(), max_norm=weights.max_norm, norm_type=2)
            nan_params,inf_params=nan_grad_to_zero(model,nan_to_num=True)
            if nan_params or inf_params:
                # import pdb;pdb.set_trace()
                if set(nan_params+inf_params) & set(param_names): # 集合交集
                    logger.warning(f"detect nan:{len(nan_params)}:{nan_params} and inf:{len(inf_params)}:{inf_params} in gradient, convert nan to zero")
                    # continue
            # if loss==0.0:
            #     import pdb;pdb.set_trace()
            model.get_dict()
            # model_bkp=copy.deepcopy({**dict(model.named_parameters()),**{k+'.grad':v.grad for k,v in model.named_parameters()}})
            optimizer.step()
            # apply_parameter_constraints(model)
            # logger.info(f"lr:{lrSchedular.get_last_lr()[-1]:.5f}\t loss:{loss:.5f}")
            
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
        init_error:float,
        model:BoundleAdjustment,
        weights,
        dataset:Dataset,
        losses:mp.Queue,
        pole_lists,
        iteration:int,
        thread_count=1,
    ):
    torch.set_num_threads(thread_count)
    assert refine_mode in ['process','distributed']
    logger.info(f"{'ddp' if refine_mode=='distributed' else 'sub'} process rank:{rank} has been started")
    cpu_count=model.cpu_count
    list_len=model.list_len
    param_names=[]
    lr = min(5e-3*init_error/cpu_count,1e-2)
    optimizer = torch.optim.Adam(update(model,lr,param_names))
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
            torch.manual_seed(step)
            mask_index=torch.multinomial(input=torch.ones(cpu_count),num_samples=list_len,replacement=True)
            # logger.info(f"shuffle time consume:{time.time()-time0}")
            # for batch in myDataLoader:
            loss=model.forward(
                mask=torch.tensor((mask_index==rank),dtype=torch.bool,requires_grad=False),
                # mask=batch,
                line_weight=weights.line_weight,
                length_weight=weights.length_weight,
                reproj_weight=weights.reproj_weight,
                orthogonal_weight=weights.orthogonal_weight,
                quaternion_weight=weights.quaternion_weight
            )
            optimizer.zero_grad()
            loss.backward()
            # clip the grad
            # if weights.max_norm:
            #     clip_grad_norm_(model.parameters(), max_norm=weights.max_norm, norm_type=2)
            nan_params,inf_params=nan_grad_to_zero(model,nan_to_num=True)
            if nan_params or inf_params:
                # import pdb;pdb.set_trace()
                if set(nan_params+inf_params) & set(param_names): # 集合交集
                    logger.warning(f"process rank:{rank} detect nan:{len(nan_params)}:{nan_params} and inf:{len(inf_params)}:{inf_params} in gradient, convert nan to zero")
                    # continue
            optimizer.step()
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
        weights
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
    iteration = 1000
    iteration = max(int(iteration/math.sqrt(cpu_count)),iteration) 
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
            args=(refine_mode,rank,barrier,verify_message,init_error,model,weights,myDataset,losses,pole_lists,iteration,thread_count),
            name=f"train{rank}",
            daemon=True
        )
        p.daemon=True
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def comprehensive_nan_check(model:torch.nn.Module, check_gradients=True):
    """全面检查模型参数和梯度"""
    issues_found = False
    # 检查参数
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN in parameter: {name}")
            issues_found = True
        elif torch.isinf(param).any():
            logger.error(f"Inf in parameter: {name}")
            issues_found = True
    # 检查梯度
    if check_gradients:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error(f"NaN in gradient: {name}")
                    issues_found = True
                elif torch.isinf(param.grad).any():
                    logger.error(f"Inf in gradient: {name}")
                    issues_found = True
    # if not issues_found:
    #     print("✅ 模型参数和梯度正常")
    return issues_found

def get_refine_pose(
        max_process,
        cam_num,
        pole_lists,
        intrinsics,
        rotation_representation,
        pole_param,
        init_poses,
        save_path,
        refine_mode,
        weights
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
        rotation_representation=rotation_representation,
        init_extrinsic=init_poses,
        image_num=np.array(mask).sum(),
        init_pole_3ds=masked_pole_3ds,
        detected_pole_2ds=masked_pole_lists,
        save_path=save_path
    )
    if comprehensive_nan_check(myBoundAdjustment):
        import pdb;pdb.set_trace()
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
            pole_lists=pole_lists,
            weights=weights
        )
    elif refine_mode=='process' or refine_mode=='distributed':
        multi_process_train(
            refine_mode=refine_mode,
            init_error=init_error,
            model=myBoundAdjustment,
            pole_lists=pole_lists,
            weights=weights
        )
    else:
        support_list=['thread','process','distributed']
        raise NotImplementedError(f"refine mode only support {support_list}")

        
    


if __name__=="__main__":
    pass