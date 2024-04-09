# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 2024

@author: Wu,Jiao

# k-means 聚类算法

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams,font_manager
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    #"font.sans-serif":['SimHei'],   # 显示中文标签
    "axes.unicode_minus":False   # 正常显示负号
    }
rcParams.update(config)
kaiti_font_title = font_manager.FontProperties(family="KaiTi", size=14)

import seaborn as sns
palette = sns.color_palette("hls", 24)
sns.set_palette(palette) 
sns.set_style("whitegrid")

def draw_cluster(dataset,centers,**kwargs):
    dpi = kwargs.get('dpi',100)       # 图片分辨率 默认300
    figsize = kwargs.get('figsize',(6,5))
    fig = plt.figure(figsize=figsize,dpi=dpi) 
    X,y = dataset 
    y = y.astype(np.int32)
    
    # 设置颜色
    #cmap = sns.color_palette("hls",8)
    cmap = sns.color_palette("Paired",24)
    colors = list(map(lambda x: cmap[x], y.tolist()))
    
    plt.scatter(centers[:,0].tolist(),centers[:,1].tolist(),c='orangered',marker='^',linewidths=6)#  'royalblue'
    plt.scatter(X[:,0],X[:,1],c=colors,linewidths=4)
    plt.show()
    
# K-Means聚类算法
def kMeans(X,K):
    # X  样本集
    # k  簇个数
    P,N = X.shape       # 样本数目P, 样本特征维度N
    tags = np.zeros(P)   # 样本标签
    
    # 随机产生K个初始簇中心: (从样本中选取k个样本加入噪声生成类中心)
    random_args = np.random.choice(P,K)
    centers = X[random_args] + np.random.normal(0,0.05,(K,N))   # K*N
    
    TagChanged = True
    Loss = []
    while TagChanged: # 如果没有点发生分配结果改变，则结束
        TagChanged = False
        loss = 0.0        
        # 计算每个样本点到各簇中心的距离, 将其分配到最小距离所属簇类
        for p,x in enumerate(X):
            dist_x = np.linalg.norm((centers-x),axis=1)  # 第p个样本x_p到簇中心的距离
            cidx = np.argmin(dist_x) 
            if cidx != tags[p]:
                TagChanged = True
            tags[p] = cidx
            loss += np.min(dist_x)**2
            
        # 更新聚类中心
        if TagChanged:
            for k in range(K):
                centers[k,:] = np.sum(X[tags==k,:],axis=0)/(tags==k).sum()
            Loss.append(loss)
            draw_cluster((X,tags),centers)
    
    return centers,tags,Loss
