# -*- coding: utf-8 -*-
"""
Created on Sat Feb 3 09:45:54 2024

@author: Jiao

绘图工具
"""

import os
#import torch
import random
import numpy as np
#%matplotlib inline
import matplotlib
from matplotlib import rcParams,font_manager
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib import pyplot as plt # matplotlib 是 Python 的绘图库
from matplotlib_inline import backend_inline

from nnc import activation as act

config = {
"font.family":'serif',
"font.size": 12,
"mathtext.fontset":'stix',
"font.serif": ['SimSun'],
}
rcParams.update(config)
kaiti_font_title = font_manager.FontProperties(family="KaiTi", size=16)
kaiti_font_legend = font_manager.FontProperties(family="KaiTi", size=11)

import seaborn as sns
palette = sns.color_palette("hls", 24)
sns.set_palette(palette) 
sns.set_style("whitegrid")
# sns.set(style='ticks',palette='pastel')

#绘图 线  名称
linestyle_str = [
     ('solid', 'solid'),      # 同 (0, ()) or '-'
     ('dotted', 'dotted'),    # 同 (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # 同 as '--'
     ('dashdot', 'dashdot')]  # 同  '-.'

markers = ["*","o",  "v","p","s","X","P","^","d","<","h",">","8","D","H"]



def draw_iris(fig_dir,fig_name,data,**kwargs):
        
    #导入seaborn自带iris数据集
    # data=sns.load_dataset("iris")  
    #为了方便大家观看，把列名换成中文的
    data.rename(columns={"sepal_length":"萼片长",
                         "sepal_width":"萼片宽",
                         "petal_length":"花瓣长",
                         "petal_width":"花瓣宽",
                         "species":"种类"},inplace=True)
    # kind_dict = {
    #     "setosa":"山鸢尾",
    #     "versicolor":"杂色鸢尾",
    #     "virginica":"维吉尼亚鸢尾"
    # }
    kind_dict = {
        0:"山鸢尾",
        1:"杂色鸢尾",
        2:"维吉尼亚鸢尾"
    }
    data["种类"] = data["种类"].map(kind_dict)
    data.head() #数据集的内容如下
    
    plt.rcParams['figure.figsize'] = (7.0,5.0)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.family'] = 'SimHei'
    
    sns.pairplot(data,hue="种类")
    
    # 保存图片    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    plt.savefig(os.path.join(fig_dir,fig_name) ) # 保存图像到PDF文件中
    plt.show()
    return plt

# 绘制两类数据分布菜点图
def draw_class2(fig_dir,fig_name,data,**kwargs):
    # # 绘图数据
    X,label = data  
    label_set = set(label)
    
    # 设置颜色
    #cmap = sns.color_palette("Paired",20)
    cmap = sns.color_palette("hls",24)
    np.random.shuffle(cmap)
    #colors = list(map(lambda x: cmap[x], label.tolist()))    
    colors = ['yellowgreen','royalblue']
    # 设置标记
    #np.random.shuffle(markers)
    marker = list(map(lambda x: markers[x], label.tolist()))   
    
    dpi = kwargs.get('dpi',100)       # 图片分辨率 默认300
    figsize = kwargs.get('figsize',(6,5))    
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    linestyle = kwargs.get('linestyle','solid')                    # 默认“solid"
    linewidth = kwargs.get('linewidth',2)    
    
    #plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker=marker, color=colors)
    
    for i,ci in enumerate(label_set):
        idx = np.where(label==ci)[0].tolist()
        plt.scatter(x=X[idx,:][:,0].tolist(), y=X[idx,:][:,1].tolist(), marker=markers[i], color=colors[i], edgecolors=colors[i],linewidths=0.5,s=100)
        #plt.scatter(x=X[idx,:][:,0].tolist(), y=X[idx,:][:,1].tolist(), marker=markers[i], color=cmap[i], edgecolors=cmap[i],linewidths=0.5,s=80)
    
    # x-轴标签
    xlabel = kwargs.get('xlabel','$x_1$')
    plt.xlabel(xlabel,fontsize=14)
    # y-轴标签
    ylabel = kwargs.get('ylabel','$x_2$')     
    plt.ylabel(ylabel,fontsize=14)
    
    # x-轴的范围
    xlim = kwargs.get('xlim',None)
    # y-轴的范围
    ylim = kwargs.get('ylim',None)      
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1]) 
    
    
    # 对2-维线性可分问题画分类判决直线
    # 绘 判别边界线
    #xmin = min([xlim[0],xlim[1]])+ 1
    #xmax = max([xlim[0],xlim[1]])- 1
    num = np.linspace(xlim[0], xlim[1], 100)   # 绘图坐标参数
    w = kwargs.get('weights',None)
    b = kwargs.get('bias',0)
    if w is not None:
        func = 'linear'        
        color_i = 'deeppink'  # palette[((i+1)*7) % len(palette)]
        w1 = w[1]
        if w1 == 0:
            w1 += 10**-10        
        k = round(-w[0]/w1,2)
        b1 = round(-b/w1,2)
        afunc = eval(f'act.{act.func_name[func]}({k},{b1})') 
        label=r'$W=[{},{}],b={}$'.format(round(w[0],2),round(w[1],2),round(b,2))
        title = kwargs.get('title','')
        plt.title(title,fontsize=14,fontproperties=kaiti_font_title)
        # 绘 判别边界线
        fvalue = [afunc(i) for i in num]
        plt.plot(
            num,
            fvalue,
            linewidth=linewidth,
            color=color_i,
            linestyle = linestyle,
            label= label
            )            
        plt.legend()
    
    # 保存图片    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir,fig_name) ) # 保存图像到PDF文件中
    plt.show()

# 绘制数据分布图
def draw_classification_data(fig_dir,fig_name,data,**kwargs):
    # 绘图数据
    X,label = data
    
    # 设置颜色
    #cmap = sns.color_palette("Paired",20)
    cmap = sns.color_palette("hls",24)
    np.random.shuffle(cmap)
    colors = list(map(lambda x: cmap[x], label.tolist()))
    
    # 设置标记
    # markers = list(map(lambda x: map_marker[x], label.tolist()))
    
    dpi = kwargs.get('dpi',100)       # 图片分辨率 默认300
    figsize = kwargs.get('figsize',(6,5))    
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    
    plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', color=colors)
    
    # idx_0 = np.where(label==0)[0].tolist()
    # plt.scatter(x=X[idx_0,:][:,0].tolist(), y=X[idx_0,:][:,1].tolist(), marker='o', color=cmap[0],edgecolors=cmap[0],linewidths=0.5)
    # idx_1 = np.where(label==1)[0].tolist()
    # plt.scatter(x=X[idx_1,:][:,0].tolist(), y=X[idx_1,:][:,1].tolist(), marker='*', color=cmap[4],edgecolors=cmap[4],linewidths=0.5)
    # for i,x in enumerate(X):        
    #     plt.scatter(x=x[0].item(), y=x[1].item(), marker=markers[i], c=colors[i])
    
    title = kwargs.get('title','')
    plt.title(title,fontsize=14,fontproperties=kaiti_font_title)
    
    # x-轴标签
    xlabel = kwargs.get('xlabel','$x_1$')
    plt.xlabel(xlabel,fontsize=14)
    # y-轴标签
    ylabel = kwargs.get('ylabel','$x_2$')     
    plt.ylabel(ylabel,fontsize=14)
    
    # x-轴的范围
    xlim = kwargs.get('xlim',None)
    # y-轴的范围
    ylim = kwargs.get('ylim',None)      
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])  
   
    # 保存图片    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir,fig_name) ) # 保存图像到PDF文件中
    plt.show()


# 绘制数据分布图+决策边界
def draw_classification_data_DB(fig_dir,fig_name,data,DB_data,**kwargs):   
    
    X,label = data
    
    # 设置颜色
    cmap = sns.color_palette("Paired",20)
    colors = list(map(lambda x: cmap[x], label.tolist()))
    # 设置标记
    # markers = list(map(lambda x: map_marker[x], label.tolist()))
    
    plt.rcParams['figure.figsize'] = (6.0,5.0)
    plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', color=colors)
    
    # num_classes = kwargs.get('num_classes',2)
    
    # 绘制 决策边界    
    DB_x,DB_y = DB_data
    
    #if torch.is_tensor(DB_y):
        #DB_y = [DB_y]
    if isinstance(DB_y,np.ndarray):
        DB_y = [DB_y]
    
    for i in range(len(DB_y)):
        plt.plot(DB_x.tolist(), DB_y[i].tolist(), color=cmap[i+4])
    
    # x-轴标签
    xlabel = kwargs.get('xlabel','x1')
    plt.xlabel(xlabel)
    # y-轴标签
    ylabel = kwargs.get('ylabel','x2')     
    plt.ylabel(ylabel)
    
    # x-轴的范围
    xlim = kwargs.get('xlim',None)
    # y-轴的范围
    ylim = kwargs.get('ylim',None)      
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])  
   
    # 保存图片    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir,fig_name) ) # 保存图像到PDF文件中
    plt.show()
    return

# 绘制数据分布图+决策边界
def draw_classification_region(data,c_data,**kwargs):
    
    x,y = c_data
    # 设置标记
    # markers = list(map(lambda x: map_marker[x], label.tolist()))
    
    plt.rcParams['figure.figsize'] = (6.0,5.0)
    # 绘制类别区域
    
    plt.scatter(x[:,0].tolist(), x[:,1].tolist(), c=y.tolist(), cmap=plt.cm.Spectral)

    X,y = data
    # 设置颜色
    cmap = sns.color_palette("Paired",20)
    colors = list(map(lambda x: cmap[x], y.tolist()))
    # colors = list(map(lambda x: cmap[x], label.tolist()))
    plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c= colors)

    # x-轴标签
    xlabel = kwargs.get('xlabel','x1')
    plt.xlabel(xlabel,fontsize=14)
    # y-轴标签
    ylabel = kwargs.get('ylabel','x2')     
    plt.ylabel(ylabel,fontsize=14)
    
    # # 保存图片
    # 图片保存路径  
    fig_dir = kwargs.get('fig_dir','') 
    fig_name = kwargs.get('fig_name','fig.pdf') 
    if fig_dir != '':
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir,fig_name)   
        plt.savefig(fig_path) # 保存图像到PDF文件中
    plt.show()
    return

# 绘制二分类问题的分类边界
def draw_decision_boundary(num,dataset,W,func=None,**kwargs): 
    X,Y = dataset
    if func is None:
        func = 'linear'        
    dpi = kwargs.get('dpi',100)       # 图片分辨率 默认300
    figsize = kwargs.get('figsize',(6,4))
    fig = plt.figure(figsize=figsize,dpi=dpi)
    linestyle = kwargs.get('linestyle','solid')                    # 默认“solid"
    linewidth = kwargs.get('linewidth',2)     
    try:
        plt.xlabel(r'$x_1$')  #  ('x label')
        plt.ylabel(r'$x_2$')
        # cnum = random.randint(1,len(palette))-1 
        if (func == 'linear'):
            for i,w in enumerate(W):   
                color_i = palette[((i+1)*7) % len(palette)]
                w2 = w[2]
                if w2 == 0:
                    w2 += 10**-10        
                k = round(-w[1]/w2,2)
                b = round(w[0]/w2,2)
                afunc = eval(f'act.{act.func_name[func]}({k},{b})') 
                label=r'$W=[{},{}],T={}$'.format(round(w[1],2),round(w[2],2),round(w[0],2))
                title = kwargs.get('title','')
                plt.title(title,fontsize=14,fontproperties=kaiti_font_title)
                # 绘训练样本散点图
                for x,y in zip(X,Y):
                    marker = markers[y % len(markers)]
                    color = palette[y*3 % len(palette)]
                    plt.scatter(x[0], x[1], marker=marker,color=color,s=50)

                # 绘 判别边界线
                fvalue = [afunc(i) for i in num]
                plt.plot(
                    num,
                    fvalue,
                    linewidth=linewidth,
                    color=color_i,
                    linestyle = linestyle,
                    label= label
                    )            
                plt.legend()
            
            plt.ylim([-3,3])    
        # plt.legend(lg_label)
        # plt.tight_layout()
        plt.show()
        
        # # 保存图片
        # 图片保存路径  
        fig_dir = kwargs.get('fig_dir','') 
        fig_name = kwargs.get('fig_name','fig.pdf') 
        if fig_dir != '':
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_path = os.path.join(fig_dir,fig_name)   
            plt.savefig(fig_path) # 保存图像到PDF文件中
    except TypeError:
         print('input function expression is wrong or the funciton is not configured')

def draw_scatter_2c(data,**kwargs):
    # 绘 2-类 2-维样本散点图
    X,Y = data   
    center = kwargs.get('center',0)  # 是否画出样本重心
    if center:
        X_0 = np.mean(X[Y<=0],axis=0)  # 第1类重心
        X_1 = np.mean(X[Y>0],axis=0)  # 第2类重心        
    dpi = kwargs.get('dpi',100)       # 图片分辨率 默认300
    figsize = kwargs.get('figsize',(6,4))
    fig = plt.figure(figsize=figsize,dpi=dpi) 
    for x,y in zip(X,Y):
        if y <= 0:
            plt.scatter(x[0], x[1], marker='o',color=palette[5],s=40)
        else:
            plt.scatter(x[0], x[1], marker='*',color=palette[10],s=60)
    if center:
        plt.scatter(X_0[0], X_0[1], marker= '^',color=palette[5],s=70)
        plt.scatter(X_1[0], X_1[1], marker= '^',color=palette[10],s=70)        
    plt.show() 
    

# 绘制分类性能随参数变化图(训练误差-测试误差-验证误差)
def draw_scores(fig_dir,fig_name,**kwargs):
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_path = os.path.join(fig_dir,fig_name) 
       
    # 评价指标名称  默认 Loss(损失函数)   
    metric = kwargs.get('metric','score') 
    # metric = kwargs.get('metric','score') 
    # 评价指标得分
    tdata = kwargs.get('train_scores',([],[]))  
    train_scores,train_steps = tdata
    #print('train_score:',train_scores[:3])
    #print('train_steps:',train_steps[:3])
    test_scores,test_steps  = kwargs.get('test_scores',([],[]))  
    dev_scores,dev_steps  = kwargs.get('dev_scores',([],[]))    
    # # 参数数据    
    # param_data = kwargs.get('param_data',[])    
    # 参数名称
    param = kwargs.get('param_name','')
    # x-轴的范围
    xlim = kwargs.get('xlim',None)
    # y-轴的范围
    ylim = kwargs.get('ylim',None)
    
    # 绘制图片
    if train_scores == [] and test_scores == [] and dev_scores == []:
        return
    
    if train_steps == [] and test_steps == [] and dev_steps == []:
        num = max([len(train_scores),len(test_scores),len(dev_scores)])
        train_steps = [i for i in range(num)]
        test_steps = [i for i in range(num)]
        dev_steps = [i for i in range(num)]
    
    #print('train_score:',train_scores[:3])
    #print('train_steps:',train_steps[:3])
        
    cmap = sns.color_palette("Paired",20)
    dpi = kwargs.get('dpi',100)
    figsize = kwargs.get('figsize',(7,5))
    plt.rcParams['figure.figsize'] = figsize  #  (8.0, 6.0)
    plt.rcParams['figure.dpi'] = dpi  
    
    if train_scores != []:
        plt.plot(train_steps, train_scores, linestyle='-', linewidth=2.0, color=cmap[1], label='Train {}'.format(metric))
    if test_scores != []:
        plt.plot(test_steps, test_scores, linestyle='--', linewidth=2.0,color=cmap[3], label='Test {}'.format(metric))
    if dev_scores != []:
        plt.plot(dev_steps, dev_scores, linestyle='-.',linewidth=2.0, color=cmap[7], label='Dev {}'.format(metric))        
    
    plt.xlabel(param,fontsize='large')
    plt.ylabel(metric,fontsize='large')
    if xlim is not None:
        plt.xticks(range(len(xlim)),labels=xlim)
        # plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])  
    plt.legend(fontsize='medium')    
        
    # 保存图片
    plt.savefig(fig_path) # 保存图像到PDF文件中
    plt.show()
    return     

# 绘制分类性能随参数变化图(训练误差-测试误差-验证误差)
def draw_score(data,**kwargs):
       
    # 评价指标名称  默认 Loss(损失函数)   
    metric = kwargs.get('metric','loss') 
    
    # # 参数数据    
    # param_data = kwargs.get('param_data',[])    
    # 参数名称
    param = kwargs.get('param_name','epoch')
    # x-轴的范围
    xlim = kwargs.get('xlim',None)
    # y-轴的范围
    ylim = kwargs.get('ylim',None)
            
    cmap = sns.color_palette("Paired",20)
    dpi = kwargs.get('dpi',100)
    figsize = kwargs.get('figsize',(6,4))
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.rcParams['figure.figsize'] = figsize  #  (8.0, 6.0)
    plt.rcParams['figure.dpi'] = dpi  
    
    steps = range(1,len(data)+1)
    plt.plot(steps, data, linestyle='-', linewidth=2.5, color=cmap[4], label=metric)
    
    plt.xlabel(param,fontsize='large')
    plt.ylabel(metric,fontsize='large')
    if xlim is not None:
        plt.xticks(range(len(xlim)),labels=xlim)
        # plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])  
    plt.legend(fontsize='x-large')    
        
    # # 保存图片
    # 图片保存路径  
    fig_dir = kwargs.get('fig_dir','') 
    fig_name = kwargs.get('fig_name','fig.pdf') 
    if fig_dir != '':
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir,fig_name)   
        plt.savefig(fig_path) # 保存图像到PDF文件中
    plt.show()
    return   

def draw_circ(dataset,**kwargs):  
    # 绘制归一化（长度为1）的数据在单位圆上的分布
    # 圆半径  # 默认单位圆
    r = 1  #
    # 生成（单位）圆的参数
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    title = kwargs.get('title','')
    plt.rcParams['figure.figsize'] = (4,4.0)
    plt.rcParams['figure.dpi'] = 100
    plt.title(title,fontsize=16,fontproperties=kaiti_font_title)
    
    # 绘制坐标原点(0,0)
    plt.scatter(0,0,color='midnightblue',marker='o', s=14)
    # 绘制单位圆
    plt.plot(x, y, color='midnightblue',linestyle='dotted')
    
    # samples,weights = dataset   # 样本、权值矩阵
    for num in range(len(dataset)):    #
        data = dataset[num]
        if num == 0:   # 绘制样本
            # 箭头参数设置
            slabel = '$X_{}$'
            arrowprops = dict(arrowstyle="->",
                      color='violet',
                      lw=2,
                      ls='-')
            scatter_color = 'royalblue'
            scatter_marker= 'o'
            scatter_s=160
            scatter_alpha=0.6
        elif num == 1:  # 绘制权值向量
            #winner = kwargs.get('winner','')
            # 箭头参数设置
            slabel = '$W_{}$'
            arrowprops = dict(arrowstyle="->",
                      color='gold',
                      lw=2,
                      ls='-')
            scatter_color = 'orange'
            scatter_marker= '*'
            scatter_s=200
            scatter_alpha=0.9
        points_x,points_y = data[:,0],data[:,1]
        # 绘制各角度散点，以及坐标原点到点的箭头连线        
        for i,(xi,yi) in enumerate(zip(points_x,points_y)):            
            plt.annotate('',xy=(xi,yi),xytext=(0,0),arrowprops=arrowprops)
            plt.scatter(xi,yi, color=scatter_color, marker=scatter_marker, s=scatter_s, alpha=scatter_alpha)
            plt.annotate(slabel.format(i+1), (xi,yi))             

    # 设置坐标轴等比例以确保圆形
    plt.axis('equal')
    # 显示图像
    plt.show()

def draw_cl(dataset,**kwargs):  
    # 绘制归一化（长度为1）的数据在单位圆上的分布
    # 圆半径  # 默认单位圆
    r = 1  #
    # 生成（单位）圆的参数
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    title = kwargs.get('title','')
    plt.rcParams['figure.figsize'] = (4.5,4.0)
    plt.rcParams['figure.dpi'] = 100
    plt.title(title,fontsize=16,fontproperties=kaiti_font_title)
    
    # 绘制坐标原点(0,0)
    plt.scatter(0,0,color='midnightblue',marker='o', s=14)
    # 绘制单位圆
    plt.plot(x, y, color='midnightblue',linestyle='dotted')
    
    winner = kwargs.get('winner',None)
    
    # samples,weights = dataset   # 样本、权值矩阵
    for num in range(len(dataset)):    #
        data = dataset[num]
        if num == 0:   # 绘制样本
            # 箭头参数设置
            slabel = '$X_{}$'
            arrowprops = dict(arrowstyle="->",
                      color='violet',
                      lw=2,
                      ls='-')
            scatter_color = 'royalblue'
            scatter_marker= 'o'
            scatter_s=160
            scatter_alpha=0.6
        elif num == 1:  # 绘制权值向量
            #winner = kwargs.get('winner','')
            # 箭头参数设置
            slabel = '$W_{}$'
            arrowprops = dict(arrowstyle="->",
                      color='gold',
                      lw=2,
                      ls='-')
            arrowprops_w = dict(arrowstyle="->",
                      color='deepskyblue',
                      lw=2,
                      ls='-')
            scatter_color = 'orange'
            scatter_marker= '*'
            scatter_s=200
            scatter_alpha=0.9
        points_x,points_y = data[:,0],data[:,1]
        # 绘制各角度散点，以及坐标原点到点的箭头连线        
        for i,(xi,yi) in enumerate(zip(points_x,points_y)):            
            if (num == 1) and (winner is not None):
                if i == winner[0]:   
                    # 获胜节点未更新的权值
                    xi_old,yi_old = winner[1][0],winner[1][1]
                    plt.annotate('',xy=(xi_old,yi_old),xytext=(0,0),arrowprops=arrowprops)
                    plt.scatter(xi_old,yi_old, color=scatter_color, marker=scatter_marker, s=scatter_s, alpha=scatter_alpha)
                    plt.annotate(slabel.format(str(i+1)+'(winner)'), (xi_old,yi_old))
                    # 获胜节点更新后的权值
                    plt.annotate('',xy=(xi,yi),xytext=(0,0),arrowprops=arrowprops)
                    plt.scatter(xi,yi, color='orangered', marker=scatter_marker, s=scatter_s, alpha=scatter_alpha)
                    plt.annotate(slabel.format(i+1), (xi,yi))
                    # 加入从旧权值 到 新权值的箭头连续
                    plt.annotate('',xy=(xi,yi),xytext=(xi_old,yi_old),arrowprops=arrowprops_w)                    
                else:
                    plt.annotate('',xy=(xi,yi),xytext=(0,0),arrowprops=arrowprops)
                    plt.scatter(xi,yi, color=scatter_color, marker=scatter_marker, s=scatter_s, alpha=scatter_alpha)
                    plt.annotate(slabel.format(i+1), (xi,yi))                   
            else:
                plt.annotate('',xy=(xi,yi),xytext=(0,0),arrowprops=arrowprops)
                plt.scatter(xi,yi, color=scatter_color, marker=scatter_marker, s=scatter_s, alpha=scatter_alpha)
                plt.annotate(slabel.format(i+1), (xi,yi))            

    # 设置坐标轴等比例以确保圆形
    plt.axis('equal')
    # 显示图像
    plt.show()
    

def draw_som_maps(maps,winner=None,text_type=None,**kwargs):
    # 标签字体设置
    SimHei_font = font_manager.FontProperties(family="SimHei", size=12)

    classnum = kwargs.get('classnum',2)
    classname = {}
    for i in range(classnum):
        classname[i] = ('','')
    names = kwargs.get('classname',classname)

    M1,M2 = maps.shape
    y = np.linspace(M1,1,M1)
    x = np.linspace(1,M2,M2)
    # 创建一个二维网格
    X, Y = np.meshgrid(x, y)

    # 按类别数目设置颜色
    colors = [(245/255, 240/255, 245/255)]
    colors.extend(sns.color_palette("hls",classnum))
    cmap = ListedColormap(colors)
    bounds = list(range(-1,classnum+1,1))
    norm = BoundaryNorm(bounds, cmap.N)

    # 绘制网格分布图
    plt.pcolormesh(X, Y, maps, cmap=cmap, norm=norm)   # 'viridis')
    # 添加颜色刻度条
    plt.colorbar()
    # 对聚类结果加标签
    for i in range(M1):
        for j in range(M2):
            #plt.text(X[i,j],Y[M1-i-1,M2-j-1],f'${maps[i,j]}$',fontsize=12,color='w')
            if int(maps[i,j])>=0:
                if text_type is None:
                    plt.text(X[i,j],Y[i,j],f'${int(maps[i,j])}$',fontsize=12,color='w',ha='center',va='center')
                else:
                    if text_type in ['number']:
                        text = f'${int(maps[i,j])}$'
                    elif text_type in ['en']:
                        text = names[int(maps[i,j])][0]
                    elif text_type in ['zh']:
                        text = names[int(maps[i,j])][1]
                    map_ij = np.ravel_multi_index([i,j],(M1,M2))
                    if map_ij in winner:
                        plt.text(X[i,j],Y[i,j],text,fontsize=12,color='w',ha='center',va='center',fontproperties=SimHei_font)

    # 设置坐标轴标签
    #plt.xlabel('X')
    #plt.ylabel('Y')
    # 显示图形
    plt.show()


# 箱线图查看异常值分布
def boxplot(fig_dir,fig_name,data,**kwargs):
    
    # 绘制每个属性的箱线图
    data_col = list(data.columns)
    
    # 连续画几个图片
    plt.figure(figsize=(5, 5), dpi=300)
    # 子图调整
    plt.subplots_adjust(wspace=0.6)
    # 每个特征画一个箱线图
    for i, col_name in enumerate(data_col):
        plt.subplot(3, 5, i+1)
        # 画箱线图
        plt.boxplot(data[col_name], 
                    showmeans=True, 
                    meanprops={"markersize":1,"marker":"D","markeredgecolor":'#f19ec2'}, # 均值的属性
                    medianprops={"color":'#e4007f'}, # 中位数线的属性
                    whiskerprops={"color":'#e4007f', "linewidth":0.4, 'linestyle':"--"},
                    flierprops={"markersize":0.4},
                    ) 
        # 图名
        plt.title(col_name, fontdict={"size":5}, pad=2)
        # y方向刻度
        plt.yticks(fontsize=4, rotation=90)
        plt.tick_params(pad=0.5)
        # x方向刻度
        plt.xticks([])
    
    # 保存图片
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_path = os.path.join(fig_dir,fig_name)   
    plt.savefig(fig_path) # 保存图像到PDF文件中
    plt.show()
    
def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(6.5, 4.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    