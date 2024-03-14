# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:55:08 2023

@author: Jiao

Toy Data   ---   Regression
Moon Data  ---   Classification
"""

import os
import torch
from torch.utils import data
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from nnc.plot_tool import *

# (1) 数据处理方法
# 数据归一化
def normalization(X,scale='min_max'):
    # X.shape = [N,M]  N-样本数目 M-特征维度
    if X.ndim == 1:
        X = X.reshape(1,-1)
    if scale == 'min_max':
        X_ = (X.T - np.min(X,axis=1))/(np.max(X,axis=1)-np.min(X,axis=1))
    elif scale == 'z_score':
        X_ = (X.T - np.mean(X,axis=1))/np.std(X,axis=1)
    elif scale == 'norm':
        X_ = X.T/np.linalg.norm(X,axis=1)
    elif scale == 'sigmoid':   # Logistic
        X_ = 1.0 / (1 + np.exp(-X.T))
    return X_.T

# 分类变量转 one-hot 向量
def one_hot(label,class_num):
    X = np.zeros((len(label),class_num))
    for i,c in enumerate(label): 
        X[i,c] = 1
    return X

# 设置随机数种子
def seed_random(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    

# (2) 生成数据集
# 数据类
# 构造一个用于线性回归的小规模数据集
def gen_synthetic_data(func,interval,sample_num,noise,add_outlier=False,outlier_ratio=0.001):
    """
    根据给定的函数，生成样本
    输入：
        - func：          函数
        - interval：       x的取值范围
        - sample_num：    样本数目
        - noise：         噪声均方差
        - add_outlier：   是否生成异常值
        - outlier_ratio： 异常值占比
    输出：
        - X: 特征数据，shape=[n_samples,1]
        - y: 标签数据，shape=[n_samples,1]
    # """
    
    # print('sample_num:',sample_num)

    # 均匀采样
    # 使用 torch.rand 在生成 sample_num 个随机数    
    X = torch.rand(sample_num) * (interval[1]-interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # torch.normal生成0均值，noise标准差的数据
    epsilon = torch.normal(0, noise, y.shape)
    y = y + epsilon
    if add_outlier:     # 生成额外的异常点
        outlier_num = int(len(y)*outlier_ratio)
        if outlier_num != 0:
            # 使用torch.randint生成服从均匀分布的、范围在[0, len(y))的随机Tensor
            outlier_idx = torch.randint(len(y),shape = [outlier_num])
            y[outlier_idx] = y[outlier_idx] * 5            
    return X, y

class sin_data():
    
    # 生成 sin(2*pi*x) 数据
    def __init__(self):
        self.interval = (0,1)
        self.noise = 0
        self.test_num = 0
        self.add_outlier = False
        
    def sin(self,x):
        # sin 函数: sin(2*pi*x)
        return torch.sin(2*np.pi*x)

    def gen_data(self,train_num,**kwargs):        
        # 由 sin(2*pi*x) 生成带噪声的数据
        test_num = kwargs.get('test_num',self.test_num)
        interval = kwargs.get('interval',self.interval)      # 数据区间 x\in (0,1)
        noise = kwargs.get('noise',self.noise)           # 数据中加入噪声
        add_outlier = kwargs.get('add_outlier',self.add_outlier)  # 是否加入离群值
        X_train, y_train = gen_synthetic_data(func=self.sin, interval=interval, sample_num=train_num, noise = noise, add_outlier = add_outlier)
        X_test,y_test = None,None
        if test_num > 0:
            X_test, y_test = gen_synthetic_data(func=self.sin, interval=interval, sample_num=test_num, noise = noise, add_outlier = add_outlier)
        self.train_data = (X_train,y_train)
        self.test_data = (X_test,y_test)
        
        # 由函数生成的绘图数据的曲线
        self.X_underlying = torch.linspace(interval[0],interval[1],steps=100) 
        self.y_underlying = self.sin(self.X_underlying)  

    # 绘制散点图
    def draw_data(self,fig_dir,fig_name,**kwargs):
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir,fig_name)   

        # 训练数据
        X_train,y_train = kwargs.get('train_data',(None,None))
        if X_train is not None and y_train is not None:
            if not isinstance(X_train ,torch.Tensor): X_train = torch.Tensor(X_train)
            if not isinstance(y_train ,torch.Tensor): y_train = torch.Tensor(y_train) 

        # 测试数据
        X_test,y_test = kwargs.get('test_data',(None,None))
        if X_test is not None and y_test is not None:
            if not isinstance(X_test ,torch.Tensor): X_test = torch.Tensor(X_test)
            if not isinstance(y_test ,torch.Tensor): y_test = torch.Tensor(y_test) 

        #　由函数生成的绘图数据的曲线
        X_underlying,y_underlying = kwargs.get('underlying_data',(self.X_underlying,self.y_underlying))
        if X_underlying is not None and y_underlying is not None:
            if not isinstance(X_underlying ,torch.Tensor): X_underlying = torch.Tensor(X_underlying)
            if not isinstance(y_underlying ,torch.Tensor): y_underlying = torch.Tensor(y_underlying)    
        
        # 由模型生成的绘图数据的预测曲线
        X_underlying_pred,y_underlying_pred = kwargs.get('pred_underlying_data',(None,None))
        if X_underlying_pred is not None and y_underlying_pred is not None:
            if not isinstance(X_underlying_pred ,torch.Tensor): X_underlying_pred = torch.Tensor(X_underlying_pred)
            if not isinstance(y_underlying_pred ,torch.Tensor): y_underlying_pred = torch.Tensor(y_underlying_pred) 

        # 由正则化模型生成的绘图数据的预测曲线
        X_underlying_pred_reg,y_underlying_pred_reg = kwargs.get('reg_pred_underlying_data',(None,None))
        if X_underlying_pred_reg is not None and y_underlying_pred_reg is not None:
            if not isinstance(X_underlying_pred_reg ,torch.Tensor): X_underlying_pred_reg = torch.Tensor(X_underlying_pred_reg)
            if not isinstance(y_underlying_pred_reg ,torch.Tensor): y_underlying_pred_reg = torch.Tensor(y_underlying_pred_reg) 
        
        label = kwargs.get('label','')
        label_pred = label
        label_pred_reg = label
        degree = kwargs.get('degree',0)
        if degree != 0:
            label_pred = r"$M={}$".format(degree)
        reg_lambda = kwargs.get('reg_lambda',0)
        if reg_lambda != 0:
            label_pred_reg = r"$M={},\ell_2\ reg\ (\lambda={})$".format(degree,reg_lambda)
        
        
        if (X_train,y_train) == (None,None) and (X_test,y_test) == (None,None) and (X_underlying,y_underlying) == (None,None):
            return 

        plt.rcParams['figure.figsize'] = (8.0,6.0)
        if X_train is not None and y_train is not None:
            plt.scatter(X_train, y_train, marker='*', facecolor="none", edgecolor='#e4007f', s=50, label="training data")

        if X_test is not None and y_test is not None:
            plt.scatter(X_test, y_test, facecolor="none", edgecolor='#f19ec2', s=50, label="testing data")

        if X_underlying is not None and y_underlying is not None:        
            plt.plot(X_underlying, y_underlying, c='b', label=r"$\sin(2\pi x)$")

        if X_underlying is not None and y_underlying_pred is not None:  
            plt.plot(X_underlying, y_underlying_pred, c='#e4007f', linestyle="--", label=label_pred)  #label="predicted function")
            # plt.ylim(-2,1.5)
            # plt.annotate("M = {}".format(degree), xy=(0.95,-1.4))      

        if X_underlying is not None and y_underlying_pred_reg is not None:        
            plt.plot(X_underlying, y_underlying_pred_reg, c='#f19ec2', linestyle="-.", label=label_pred_reg)  #label="predicted function")
        plt.ylim(-1.5,1.5)
        plt.legend(fontsize='x-large') # 给图像加图例        
        # 保存图片
        plt.savefig(fig_path) # 保存图像到PDF文件中
        plt.show()
        return

class moons_data():
    
    # 生成带噪声的弯月形状数据
    def __init__(self):
        self.noise = None
        self.shuffle = True
        self.all_data = None
        self.train_data = None
        self.test_data = None
        self.dev_data = None

    def gen_moons_data(self,n_samples,**kwargs):
        """
        生成带噪音的弯月形状数据
        输入：
            - n_samples：数据量大小，数据类型为int
            - shuffle：是否打乱数据，数据类型为bool
            - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
        输出：
            - X：特征数据，shape=[n_samples,2]
            - y：标签数据, shape=[n_samples]
        """
        
        noise = kwargs.get('noise',self.noise)         # 数据中加入噪声,数据类型为None或float，noise为None时表示不增加噪声
        shuffle = kwargs.get('shuffle',self.shuffle)          # 是否打乱数据，数据类型为bool
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out

        # 采集第1类数据，特征为(x,y)
        # 使用'torch.linspace'在0到pi上均匀取n_samples_out个值
        # 使用'torch.cos'计算上述取值的余弦值作为特征1，使用'torch.sin'计算上述取值的正弦值作为特征2
        outer_circ_x = torch.cos(torch.linspace(0, math.pi, n_samples_out))
        outer_circ_y = torch.sin(torch.linspace(0, math.pi, n_samples_out))

        inner_circ_x = 1 - torch.cos(torch.linspace(0, math.pi, n_samples_in))
        inner_circ_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_samples_in))

        #print('outer_circ_x.shape:', outer_circ_x.shape, 'outer_circ_y.shape:', outer_circ_y.shape)
        #print('inner_circ_x.shape:', inner_circ_x.shape, 'inner_circ_y.shape:', inner_circ_y.shape)

        # 使用'torch.concat'将两类数据的特征1和特征2分别延维度0拼接在一起，得到全部特征1和特征2
        # 使用'torch.stack'将两类特征延维度1堆叠在一起
        X = torch.stack(
            [torch.concat([outer_circ_x, inner_circ_x]),
            torch.concat([outer_circ_y, inner_circ_y])],
            axis=1
        )

        #print('after concat shape:', torch.concat([outer_circ_x, inner_circ_x]).shape)
        #print('X shape:', X.shape)

        # 使用'torch. zeros'将第一类数据的标签全部设置为0
        # 使用'torch. ones'将第一类数据的标签全部设置为1
        y = torch.concat(
            [torch.zeros(size=[n_samples_out],dtype=int), torch.ones(size=[n_samples_in],dtype=int)]
        )

        #print('y shape:', y.shape)

        # 如果shuffle为True，将所有数据打乱
        if shuffle:
            # 使用'torch.randperm'生成一个数值在0到X.shape[0]，随机排列的一维Tensor做索引值，用于打乱数据
            idx = torch.randperm(X.shape[0])
            X = X[idx]
            y = y[idx]

        # 如果noise不为None，则给特征值加入噪声
        if noise is not None:
            # 使用'torch.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
            X += torch.normal(mean=0.0, std=noise, size=X.shape)
        
        self.all_data = (X,y)
        # 构建训练集、验证集和测试集
        train_percent = kwargs.get('train_percent',1.0)           # 训练集占比
        test_percent = kwargs.get('test_percent',1-train_percent)    # 测试集占比
        
        if train_percent < 1.0:
            self.train_data,self.test_data,self.dev_data = data_split(X,y,train_percent=train_percent,test_percent=test_percent)
        else:
            self.train_data = (X,y)    
         
    def draw_data(self,fig_path,fig_name,data_type='all'):
        # 对类别数据绘图
        if data_type == 'all':
            draw_classification_data(fig_path, fig_name,self.all_data)
        elif data_type == 'train':
            draw_classification_data(fig_path, fig_name,self.train_data)
        elif data_type == 'test':
            draw_classification_data(fig_path, fig_name,self.test_data)
        elif data_type == 'dev':
            draw_classification_data(fig_path, fig_name,self.dev_data)
        


def gen_multiclass_data(n_samples=100, n_features=2, n_classes=3, shuffle=True, noise=0.1):
    """
    生成带噪音的多类别数据
    输入：
        - n_samples：数据量大小，数据类型为int
        - n_features：特征数量，数据类型为int
        - shuffle：是否打乱数据，数据类型为bool
        - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
    输出：
        - X：特征数据，shape=[n_samples,2]
        - y：标签数据, shape=[n_samples,1]
    """
    # 计算每个类别的样本数量
    n_samples_per_class = [int(n_samples / n_classes) for k in range(n_classes)]
    for i in range(n_samples - sum(n_samples_per_class)):
        n_samples_per_class[i % n_classes] += 1
    # 将特征和标签初始化为0
    X = torch.zeros([n_samples, n_features])
    y = torch.zeros([n_samples],dtype=torch.int64)
    # 随机生成3个簇中心作为类别中心
    centroids = torch.randperm(2 ** n_features)[:n_classes]
    centroids_bin = np.unpackbits(centroids.numpy().astype('uint8')).reshape((-1, 8))[:, -n_features:]
    centroids = torch.tensor(centroids_bin).float()
    # 控制簇中心的分离程度
    centroids = 1.5 * centroids - 1
    # 随机生成特征值
    X[:, :n_features] = torch.randn(size=[n_samples, n_features])

    stop = 0
    # 将每个类的特征值控制在簇中心附近
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_class[k]
        # 指定标签值
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_features]
        # 控制每个类别特征值的分散程度
        A = 2 * torch.rand(size=[n_features, n_features]) - 1
        X_k[...] = torch.matmul(X_k, A)
        X_k += centroid
        X[start:stop, :n_features] = X_k

    # 如果noise不为None，则给特征加入噪声
    if noise > 0.0:
        # 生成noise掩膜，用来指定给那些样本加入噪声
        noise_mask = torch.rand([n_samples]) < noise
        for i in range(len(noise_mask)):
            if noise_mask[i]:
                # 给加噪声的样本随机赋标签值
                y[i] = torch.randint(n_classes, size=[1],dtype=torch.int64)
    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    return X, y


def gen_timeseries_data(func,num_samples,tau,normal_params=(0,0.1)):    
    # T = 1000 # 总共产⽣1000个点
    time = torch.arange(1, num_samples+1, dtype=torch.float32)
    mu,std = normal_params
    x = func(0.01*time) + torch.normal(mu,std,(num_samples,))
    plot(time,[x], 'time','x',xlim=[1,num_samples],figsize=(6,3))
    
    features = torch.zeros((num_samples-tau, tau))
    for i in range(tau):
        features[:,i] = x[i:num_samples-tau+i]
    labels = x[tau:].reshape((-1,1))
    
    return features,labels


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def data_split(X,y,train_percent=0.8,**kwargs): # test_percent=0.2,dev_percent=0):    
    # 划分训练集、测试集、验证集   
    test_percent = kwargs.get('test_percent',1-train_percent)
    dev_percent = kwargs.get('dev_percent',1-train_percent-test_percent)
    
    assert train_percent+test_percent+dev_percent == 1  # 数据集占比和为1 
    
    n = len(X)
    shuffled_indices = torch.randperm(n) # 返回一个数值在0到n-1、随机排列的1-D Tensor
    train_set_size = int(n*train_percent) 
    train_indices = shuffled_indices[:train_set_size]
    if dev_percent == 0:        
        test_indices = shuffled_indices[train_set_size:]
        dev_indices = []
    else:        
        test_set_size = int(n*test_percent)
        test_indices = shuffled_indices[train_set_size:train_set_size+test_set_size]
        dev_indices = shuffled_indices[train_set_size+test_set_size:] 

    # X = X.values
    # y = y.values
    
    y = y.float()
    X_train = X[train_indices]
    y_train = y[train_indices].reshape([-1,1])
    
    X_test = X[test_indices]
    y_test = y[test_indices].reshape([-1,1])
    
    X_dev = X[dev_indices]
    y_dev = y[dev_indices].reshape([-1,1])

    return (X_train,y_train),(X_test,y_test),(X_dev,y_dev) 



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
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
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