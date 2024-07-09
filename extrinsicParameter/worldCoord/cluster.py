import os
import sys

import numpy as np
import cv2

def sort_cluster(
        init_cluster,
        cost_matrix
    ):
    """
    接受一个cost_matrix,返回一个labels数组
    """
    clumps=[set([point_index]) for point_index in init_cluster]
    row,colume=cost_matrix.shape
    assert row==colume,"cost_matrix is not a n*n matrix"
    n=row # or n=colume
    # 先聚类 没有歧义的点
    for i in range(n):
        distances=[]
        for init_point_index,clump in list(zip(init_cluster,clumps)):
            cost=cost_matrix[i][init_point_index]
            distances.append(cost)
        # 如果有一个<10 其他两个都 >100 则可以加入某一个clump
        if np.sum(np.array(distances)<10)==1 and np.sum(np.array(distances)>100)==2:
            clumps[np.argmin(distances)].add(i)
    no_ambiguity_point_indexs=set()
    for clump in clumps:
        no_ambiguity_point_indexs=no_ambiguity_point_indexs.union(clump)
    # 再聚类 有歧义的点
    for i in range(n):
        if i in no_ambiguity_point_indexs:
            continue
        distances=[]
        for clump in clumps:
            cost=np.mean(np.array([cost_matrix[i][point_index] for point_index in clump]))
            distances.append(cost)
        clumps[np.argmin(distances)].add(i)
    labels=[0 for _ in range(n)]
    for clump_index,clump in enumerate(clumps):
        for point_index in clump:
            labels[point_index]=clump_index
    # import pdb;pdb.set_trace()
    return labels


if __name__=="__main__":
    pass