import os
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 

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

def visualize_weighted_graph(edge_array,save_path):
    """
    可视化带权无向图
    
    参数:
    edge_array: numpy数组，每一行为[权重, 节点1, 节点2]
    """
    # 创建无向图
    G = nx.Graph()
    # 添加带权重的边
    for weight, node1, node2 in edge_array:
        G.add_edge(int(node1), int(node2), weight=float(weight))
    # 获取权重列表用于颜色映射
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_weight, max_weight = min(weights), max(weights)
    # 创建颜色映射（权重越大颜色越深）
    # 使用灰度色系，权重越大越接近黑色
    if max_weight > min_weight:
        # 归一化权重到0-1范围
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    else:
        # 如果所有权重相同，都设为中等灰色
        normalized_weights = [0.5] * len(weights)
    # 转换为颜色（权重越大颜色越深）
    alpha=0.7
    edge_colors = [(0.0, 1 - w * 0.8, 0.0,alpha) for w in normalized_weights]
    # 设置图形布局
    plt.figure(figsize=(12, 8))
    # 使用spring布局（可以调整k参数控制节点间距）
    pos = nx.circular_layout(G)
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue',edgecolors='green',linewidths=2)
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    # 绘制带权重的边
    edges = nx.draw_networkx_edges(G, pos,width=3,edge_color=edge_colors)
    # 添加权重标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,font_size=10,label_pos=0.5)
    # 创建颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=plt.gca(), orientation='vertical', shrink=0.8)
    cbar.set_label('Edge Weight (darker = heavier)', fontsize=12)
    # 设置标题
    plt.title(f"Weighted Undirected Graph\nTotal nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}", fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'graph.jpg'))
    return G

def get_mst(cam_num,pole_lists,save_path):
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
        if len(selected_node)>0:
            if not((edge[1] in selected_node)^(edge[2] in selected_node)): # 同或操作
                continue
        selected_edge.append(edge)
        selected_node.add(edge[1])
        selected_node.add(edge[2])
    selected_edge=np.array(selected_edge)
    visualize_weighted_graph(selected_edge,save_path)
    return selected_edge



if __name__=="__main__":
    """
    使用kruskel算法计算最大生成树
    """
    pass