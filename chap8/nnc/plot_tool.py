# -*- coding: utf-8 -*-
"""
Created on Sat Feb 3 09:45:54 2024

@author: Jiao

绘图工具
"""

import os
import torch
import random
import numpy as np
#%matplotlib inline
import matplotlib
from matplotlib import rcParams,font_manager
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib import pyplot as plt # matplotlib 是 Python 的绘图库
from matplotlib_inline import backend_inline

#from nnc import activation as act

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

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, **kwargs):
    """Plot a list of images.
    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            img = img.numpy()
            ax.imshow(img)
        else:
            # PIL Image
            ax.imshow(img,cmap='viridis')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i],fontsize=11)  

    left = kwargs.get('left',None)
    bottom = kwargs.get('bottom',None)
    right = kwargs.get('right',None)
    top = kwargs.get('top',None)
    wspace = kwargs.get('wspace',None)
    hspace= kwargs.get('hspace',None)
    plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,
                        wspace=wspace,hspace=hspace)

    #
    dpi = kwargs.get('dpi',100)
    #figsize = kwargs.get('figsize',(7,5))
    #plt.rcParams['figure.figsize'] = figsize  #  (8.0, 6.0)
    plt.rcParams['figure.dpi'] = dpi  
    
    fig_dir = kwargs.get('fig_dir',None) 
    fig_name = kwargs.get('fig_name','predict_image.pdf') 
    # 保存图片
    if fig_dir is not None:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir,fig_name)   
        plt.savefig(fig_path) # 保存图像到PDF文件中
        # plt.show()        
    # return axes



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
    test_scores,test_steps = kwargs.get('test_scores',([],[]))  
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
        plt.plot(test_steps, test_scores, linestyle='--', linewidth=2.0,color=cmap[18], label='Test {}'.format(metric))
    if dev_scores != []:
        plt.plot(dev_steps, dev_scores, linestyle='-.',linewidth=2.0, color=cmap[18], label='Dev {}'.format(metric))        
    
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
    