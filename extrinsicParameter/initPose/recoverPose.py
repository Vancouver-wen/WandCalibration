import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram(data:np.ndarray, bins=30, title="数据分布直方图"):
    """
    绘制直方图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图
    axes[0].hist(data, bins=bins, edgecolor='black', alpha=0.7, color='skyblue', density=True)
    axes[0].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {data.mean():.2f}')
    axes[0].axvline(np.median(data), color='blue', linestyle='--', linewidth=2, label=f'中位数: {np.median(data):.2f}')
    axes[0].axvline(find_peak_by_kde(data), color='green', linestyle='--', linewidth=2, label=f'核密度最大点: {find_peak_by_kde(data):.2f}')
    axes[0].set_xlabel('数值')
    axes[0].set_ylabel('频率/密度')
    axes[0].set_title(f'{title} - 直方图')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 累积分布直方图
    axes[1].hist(data, bins=bins, edgecolor='black', alpha=0.7,
                 color='lightcoral', cumulative=True, density=True)
    axes[1].set_xlabel('数值')
    axes[1].set_ylabel('累积频率')
    axes[1].set_title(f'{title} - 累积分布')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./0.jpg')
    plt.clf()
    
    # 打印统计信息
    print(f"数据统计信息:")
    print(f"样本数量: {len(data)}")
    print(f"均值: {data.mean():.4f}")
    print(f"标准差: {data.std():.4f}")
    print(f"最小值: {data.min():.4f}")
    print(f"25%分位数: {np.percentile(data, 25):.4f}")
    print(f"中位数: {np.median(data):.4f}")
    print(f"75%分位数: {np.percentile(data, 75):.4f}")
    print(f"最大值: {data.max():.4f}")

def find_peak_by_kde(data, grid_points=1000, bandwidth=None, draw=False):
    """
    使用核密度估计找到密度最高点
    """
    # 创建KDE估计
    kde = stats.gaussian_kde(data, bw_method=bandwidth)
    # 在数据范围内创建密集的网格点
    x_grid = np.linspace(data.min() - 1, data.max() + 1, grid_points)
    kde_values = kde(x_grid)
    # 找到KDE最大值对应的点
    peak_idx = np.argmax(kde_values)
    peak_value = x_grid[peak_idx]
    peak_density = kde_values[peak_idx]
    if draw:
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # KDE图
        axes[0].plot(x_grid, kde_values, 'b-', linewidth=2, label='KDE')
        axes[0].fill_between(x_grid, kde_values, alpha=0.3, color='skyblue')
        axes[0].axvline(peak_value, color='red', linestyle='--', linewidth=2,
                    label=f'最高密度点: {peak_value:.2f}')
        axes[0].set_xlabel('数值')
        axes[0].set_ylabel('概率密度')
        axes[0].set_title('核密度估计（KDE）')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # 直方图+KDE
        axes[1].hist(data, bins=30, density=True, alpha=0.5, 
                    color='lightblue', edgecolor='black', label='直方图')
        axes[1].plot(x_grid, kde_values, 'b-', linewidth=2, label='KDE')
        axes[1].axvline(peak_value, color='red', linestyle='--', linewidth=2,
                    label=f'最高密度点: {peak_value:.2f}')
        axes[1].set_xlabel('数值')
        axes[1].set_ylabel('密度')
        axes[1].set_title('直方图与KDE对比')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./0.jpg')
        plt.clf()
    kde_result={
        'global_peak': peak_value,
        'global_peak_density': peak_density,
        'kde': kde,
        'x_grid': x_grid,
        'kde_values': kde_values
    }
    if draw:
        print("\nKDE方法结果:")
        print(f"全局最高密度点: {kde_result['global_peak']:.4f}")
        print(f"最高密度值: {kde_result['global_peak_density']:.4f}")
    return peak_value

def get_unscaled_intrinsic(pole_pairs,cam1_intrinsic,cam2_intrinsic):
    blob_pairs=pole_pairs.reshape(2,-1,2)
    K1=np.array(cam1_intrinsic.K)
    K2=np.array(cam2_intrinsic.K)
    dist1=np.squeeze(np.array(cam1_intrinsic.dist))
    dist2=np.squeeze(np.array(cam2_intrinsic.dist))
    retval, E, R, t, mask=cv2.recoverPose(
        # revocerPose的重载函数非常多
        points1=blob_pairs[0],
        points2=blob_pairs[1],
        cameraMatrix1=K1,
        cameraMatrix2=K2,
        distCoeffs1=dist1,
        distCoeffs2=dist2
    )
    return R,t

def get_scale_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic,state="init"):
    """
    两个相机的三角化，求出杆子的平均长度
    根据杆子的平均长度,对t进行scale
    """
    # cv2.triangulatePoints的流程
    # https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate
    K1=np.array(cam1_intrinsic.K)
    dist1=np.squeeze(np.array(cam1_intrinsic.dist))
    cam1_Rt=np.array(
        object=[[1,1e-5,1e-5,1e-5],
        [1e-5,1,1e-5,1e-5],
        [1e-5,1e-5,1,1e-5]],
        dtype=np.float64
    )
    cam1_proj=K1@cam1_Rt

    K2=np.array(cam2_intrinsic.K)
    dist2=np.squeeze(np.array(cam2_intrinsic.dist))
    cam2_Rt=np.concatenate((R21,t21),axis=1)
    cam2_proj=K2@cam2_Rt

    blob_pairs=pole_pairs.reshape(2,-1,2)
    cam1_blobs=cv2.undistortPoints(
        src=blob_pairs[0],
        cameraMatrix=K1,
        distCoeffs=dist1,
        R=K1
    )
    cam2_blobs=cv2.undistortPoints(
        src=blob_pairs[1],
        cameraMatrix=K2,
        distCoeffs=dist2,
        R=K2
    )
    blob3ds=cv2.triangulatePoints(
        projMatr1=cam1_proj,
        projMatr2=cam2_proj,
        projPoints1=np.squeeze(cam1_blobs).T,
        projPoints2=np.squeeze(cam2_blobs).T
    )
    blob3ds=blob3ds[:3]/np.repeat(np.expand_dims(blob3ds[3],axis=0),3,axis=0)
    blob3ds=blob3ds.T.reshape(-1,3,3)
    # d_average=0
    d_s=[]
    for blob3d in blob3ds:
        d1=np.linalg.norm(blob3d[0]-blob3d[1])
        d2=np.linalg.norm(blob3d[1]-blob3d[2])
        # d1/d2 的比例应该与标定杆保持一致
        d=d1+d2
        if state == "init":
            pass
        else:
            print({'state':state,'d1':d1,'d2':d2})
        # 从这个地方发现, 误差实际上是非常大的
        # d_average+=d/len(blob3ds)
        d_s.append(d)
    # 使用示例
    # plot_histogram(np.array(d_s,dtype=np.float32), title="尺度分布数据")
    # find_peak_by_kde(np.array(d_s,dtype=np.float32))
    # d_average=sum(d_s)/len(blob3ds)
    d_average=np.median(np.array(d_s,dtype=np.float32))
    # d_average=find_peak_by_kde(np.array(d_s,dtype=np.float32))
    t_ratio=pole_length/d_average
    return t_ratio

def get_scaled_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic):
    t_ratio=get_scale_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic,"init")
    return R21,t21*t_ratio,t_ratio

def verify_scaled_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic):
    t_ratio=get_scale_intrinsic(pole_length,pole_pairs,R21,t21,cam1_intrinsic,cam2_intrinsic,"verify_init")
    import pdb;pdb.set_trace()

if __name__=="__main__":
    pass