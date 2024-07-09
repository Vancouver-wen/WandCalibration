import os
import sys

import numpy as np
import cv2
from loguru import logger

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

class IterativeMaximunCalique(object):
    def __init__(self,eps,min_samples) -> None:
        self.threshold=eps
        self.min_samples=min_samples

    def find_maximum_clique(self,graph):
        n=len(graph)
        max_clique=[]
        def expand(current_calique,candidates):
            nonlocal max_clique
            if not candidates:
                if len(current_calique)>len(max_clique):
                    max_clique=current_calique[:]
                return 
            if len(current_calique)+len(candidates)<=len(max_clique):
                return 
            for v in candidates:
                if all(graph[v][u] for u in current_calique):
                    new_candidates=[u for u in candidates if graph[v][u]]
                    expand(current_calique+[v],new_candidates)
        expand([],list(range(n)))
        return max_clique
    
    def test_maximum_clique(self,):
        graph=[
            [0,1,1,0],
            [1,0,1,1],
            [1,1,0,1],
            [0,1,1,0]
        ]
        result=self.find_maximum_clique(graph)
        print(result) # [0,1,2]
        return result
    
    def update_graph(self,graph,clique):
        """
        将已经成为clique的节点全部变成false
        """
        n=len(graph)
        for i in range(n):
            for j in range(n):
                if (i in clique) or (j in clique):
                    graph[i][j]=False
        return graph

    def fit_predict(
            self,
            cost_matrix
        ):
        """
        使用 eps将 cost_matrix转换为 无向图
        对该无向图求解最大团
        删除最大团的节点
        继续求解最大团
        直到最大团的大小不足 self.min_samples
        """
        graph=cost_matrix<self.threshold
        cliques=[]
        while True:
            clique=self.find_maximum_clique(graph)
            if len(clique)<self.min_samples:
                break
            logger.info(f"find a clique containing {len(clique)} point")
            graph=self.update_graph(graph,clique)
            cliques.append(clique)
        labels=[-1 for _ in range(len(graph))]
        for label,clique in enumerate(cliques):
            for index in clique:
                labels[index]=label
        return labels
            

def mcp_cluster(
        cost_matrix,
        eps=1e2,
        min_samples=2
    ):
    label=IterativeMaximunCalique(eps=eps,min_samples=min_samples).fit_predict(cost_matrix=cost_matrix)
    return label

if __name__=="__main__":
    pass