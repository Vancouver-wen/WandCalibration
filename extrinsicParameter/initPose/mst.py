import os
import sys

import numpy as np

def get_wug(cam_num,pole_lists):
    """
    wug -> weighted undirected graph
    """
    wug=np.zeros(shape=(cam_num,cam_num),dtype=np.uint64)
    for pole_list in pole_lists:
        assert cam_num==len(pole_list),"pole format conflict with cam_num"
        mask=[pole is not None for pole in pole_list]
        for i,bool_i in enumerate(mask):
            for j,bool_j in enumerate(mask):
                if i==j or bool_i==False or bool_j==False:
                    continue
                wug[i][j]+=1
    return wug

def get_edge(cam_num,wug):
    edges=[]
    for i in range(cam_num):
        for j in range(1+i,cam_num):
            obj=[wug[i][j],i,j]
            edges.append(obj)
    return np.array(edges)

def is_connected(cam_num,edges):
    connected_nodes=set()
    connected_nodes.add(0)
    for i in range(cam_num):
        for edge in edges:
            if edge[1] in connected_nodes or edge[2] in connected_nodes:
                connected_nodes.add(edge[1])
                connected_nodes.add(edge[2])
    return cam_num==len(connected_nodes)

def get_mst(cam_num,pole_lists):
    """
    mst -> maximum spanning tree
    """
    wug=get_wug(cam_num,pole_lists)
    edges=get_edge(cam_num,wug)
    assert is_connected(cam_num,edges),"the common view of camera is not connected"
    # Kruskal Algorithm
    sorted_edges=edges[np.argsort(-edges[:,0])]
    selected_edge=[]
    selected_node=set()
    for edge in sorted_edges:
        if edge[1] not in selected_node or edge[2] not in selected_node:
            selected_edge.append(edge)
            selected_node.add(edge[1])
            selected_node.add(edge[2])
    selected_edge=np.array(selected_edge)
    return selected_edge



if __name__=="__main__":
    """
    使用kruskel算法计算最大生成树
    """
    pass