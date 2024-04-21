# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:55:08 2023

@author: Wu,Jiao

sin_data      ---   Regression
moon_data     ---   Classification
animal_data    ---   SOM - Cluster
character_data  ---   SOM - Cluster

"""

import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
#sns.set(style='ticks',palette='pastel       
palette = sns.color_palette("hls", 24)
sns.set_palette(palette) 
sns.set_style("whitegrid")
from matplotlib import pyplot as plt
from matplotlib import rcParams,font_manager
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib_inline import backend_inline
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
from nnc.plot_tool import *

# (1) 数据处理方法
# 数据归一化
def normalization(X,scale='min_max'):
    # X.shape = [N,M]  N-样本数目 M-特征维度
    if X.ndim <= 1:
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

# 数据转换
def axis2theta(W): 
    # 2-维单位向量转极坐标（用角度表示平面上的点）
    Theta = []
    for w in W:
        if w[0] == 0:
            if w[1] > 0:
                theta_rad = np.pi/2     
            elif w[1] < 0:
                theta_rad = -np.pi/2
        else:
            if w[1] == 0:
                if w[0] > 0:
                    theta_rad = np.arctan(w[1]/w[0])
                elif w[0] < 0:
                    theta_rad = np.pi
            elif w[1] >0:   
                if w[0] > 0:  # 第一象限角
                    theta_rad = np.arctan(w[1]/w[0])
                elif w[0] < 0: # 第二象限角
                    theta_rad = np.pi - np.arctan(w[1]/w[0])
            else:   
                if w[0] < 0: # 第三象限角
                    theta_rad = -np.pi + np.arctan(w[1]/w[0]) 
                elif w[0] > 0:   # 第四象限角
                    theta_rad = np.arctan(w[1]/w[0]) 
        Theta.append(np.rad2deg(theta_rad))
    return np.array(Theta)

def theta2axis(theta): 
    # 极坐标（用角度表示平面上的点）转 2-维单位向量转
    x = np.cos(np.deg2rad(theta))
    y = np.sin(np.deg2rad(theta))
    D = np.array([x,y]).T
    return D

# 设置随机数种子
def seed_random(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True   

# 距离
#计算欧式距离函数
def edist(self,X1,X2):
    return (np.linalg.norm(X1-X2))

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

    # 均匀采样
    # 使用 np.random.rand 在生成 sample_num 个随机数    
    X = np.random.rand(sample_num) * (interval[1]-interval[0]) + interval[0]
    # 等间隔生成 sample_num 个数据
    #X= np.linspace(interval[0],interval[1],sample_num)
    y = func(X)

    # 生成高斯分布的标签噪声
    # np.random.normal生成0均值，noise标准差的数据
    epsilon = np.random.normal(0, noise, y.shape)
    y_n = y + epsilon
    if add_outlier:     # 生成额外的异常点
        outlier_num = int(len(y_n)*outlier_ratio)
        if outlier_num != 0:
            # 使用torch.randint生成服从均匀分布的、范围在[0, len(y))的随机Tensor
            outlier_idx = np.random.randint(len(y),size = [outlier_num])
            y_n[outlier_idx] = y_n[outlier_idx] * 5    
    return X,y_n,y

class sin_data():
    
    # 生成 sin(2*pi*x) 数据
    def __init__(self):
        self.interval = (0,1)
        self.noise = 0
        self.test_num = 0
        self.add_outlier = False
        
    def sin(self,x):
        # sin 函数: sin(2*pi*x)
        return np.sin(2*np.pi*x)

    def gen_data(self,train_num,**kwargs):        
        # 由 sin(2*pi*x) 生成带噪声的数据
        test_num = kwargs.get('test_num',self.test_num)
        interval = kwargs.get('interval',self.interval)      # 数据区间 x\in (0,1)
        noise = kwargs.get('noise',self.noise)           # 数据中加入噪声
        add_outlier = kwargs.get('add_outlier',self.add_outlier)  # 是否加入离群值
        X_train, y_train,y_train_n = gen_synthetic_data(func=self.sin, interval=interval, sample_num=train_num, noise = noise, add_outlier = add_outlier)
        X_test,y_test,y_text_n = None,None,None
        if test_num > 0:
            X_test, y_test,y_test_n = gen_synthetic_data(func=self.sin, interval=interval, sample_num=test_num, noise = noise, add_outlier = add_outlier)
        self.train_data = (X_train,y_train,y_train_n)
        self.test_data = (X_test,y_test,y_test_n)
        
        # 由函数生成的绘图数据的曲线
        self.X_underlying = np.linspace(interval[0],interval[1],100) 
        self.y_underlying = self.sin(self.X_underlying)  

    # 绘制散点图
    def draw_data(self,fig_dir,fig_name,**kwargs):
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_path = os.path.join(fig_dir,fig_name)   

        # 训练数据
        X_train,y_train = kwargs.get('train_data',(None,None))
        if X_train is not None and y_train is not None:
            if not isinstance(X_train ,np.ndarray): X_train = np.array(X_train)
            if not isinstance(y_train ,np.ndarray): y_train = np.array(y_train) 
        # 训练数据预测值
        X_train_pred,y_train_pred = kwargs.get('train_data_pred',(None,None))
        if X_train_pred is not None and y_train_pred is not None:
            if not isinstance(X_train_pred ,np.ndarray): X_train_pred = np.array(X_train_pred)
            if not isinstance(y_train_pred ,np.ndarray): y_train_pred = np.array(y_train_pred)                 
            idx = np.argsort(X_train_pred)
            X_train_pred = X_train_pred[idx]
            y_train_pred = y_train_pred[idx]
            

        # 测试数据
        X_test,y_test = kwargs.get('test_data',(None,None))
        if X_test is not None and y_test is not None:
            if not isinstance(X_test ,np.ndarray): X_test = np.array(X_test)
            if not isinstance(y_test ,np.ndarray): y_test = np.array(y_test) 
        # 测试数据预测值
        X_test_pred,y_test_pred = kwargs.get('test_data_pred',(None,None))
        if X_test_pred is not None and y_test_pred is not None:
            if not isinstance(X_test_pred ,np.ndarray): X_test_pred = np.array(X_test_pred)
            if not isinstance(y_test_pred ,np.ndarray): y_test_pred = np.array(y_test_pred)                 
            idx = np.argsort(X_test_pred)
            X_test_pred = X_test_pred[idx]
            y_test_pred = y_test_pred[idx]        
                

        #　由函数生成的绘图数据的曲线
        X_underlying,y_underlying = kwargs.get('underlying_data',(self.X_underlying,self.y_underlying))
        if X_underlying is not None and y_underlying is not None:
            if not isinstance(X_underlying ,np.ndarray): X_underlying = np.array(X_underlying)
            if not isinstance(y_underlying ,np.ndarray): y_underlying = np.array(y_underlying)    
        
        # 由模型生成的绘图数据的预测曲线
        X_underlying_pred,y_underlying_pred = kwargs.get('pred_underlying_data',(None,None))
        if X_underlying_pred is not None and y_underlying_pred is not None:
            if not isinstance(X_underlying_pred ,np.ndarray): X_underlying_pred = np.array(X_underlying_pred)
            if not isinstance(y_underlying_pred ,np.ndarray): y_underlying_pred = np.array(y_underlying_pred) 

        # 由正则化模型生成的绘图数据的预测曲线
        X_underlying_pred_reg,y_underlying_pred_reg = kwargs.get('reg_pred_underlying_data',(None,None))
        if X_underlying_pred_reg is not None and y_underlying_pred_reg is not None:
            if not isinstance(X_underlying_pred_reg ,np.ndarray): X_underlying_pred_reg = np.array(X_underlying_pred_reg)
            if not isinstance(y_underlying_pred_reg ,np.ndarray): y_underlying_pred_reg = np.array(y_underlying_pred_reg) 
        
        label = kwargs.get('label','')
        label_pred = label
        label_pred_reg = label
        degree = kwargs.get('degree',0)
        if degree != 0:
            label_pred = r"$M={}$".format(degree)
        reg_lambda = kwargs.get('reg_lambda',0)
        if reg_lambda != 0:
            label_pred_reg = r"$M={},\ell_2\ reg\ (\lambda={})$".format(degree,reg_lambda)
        
        if (X_train is None and y_train is None) and (X_test is None and y_test is None) and (X_underlying is None and y_underlying is None):
            return 
        
        plt.rcParams['figure.figsize'] = (7.0,5.0)
        plt.rcParams['figure.dpi'] = 100
        title = kwargs.get('title',None)
        if title is not None:
            plt.title(title,fontsize=16,fontproperties=kaiti_font_title)
            
        if X_train is not None and y_train is not None:
            plt.scatter(X_train, y_train, marker='*', facecolor="none", edgecolor='#e4007f', s=50, label="training data")
        if X_train_pred is not None and y_train_pred is not None:
            plt.plot(X_train_pred, y_train_pred, c='#e4007f', linestyle="--", label=label_pred) # c='blueviolet'
            #plt.scatter(X_train, y_train, marker='o', facecolor="none", edgecolor='g', s=50, label="training data pred")

        if X_test is not None and y_test is not None:
            plt.scatter(X_test, y_test, facecolor="none", edgecolor= '#f19ec2', s=50, label="testing data")   # '#00a497'
        if X_test_pred is not None and y_test_pred is not None:
            plt.plot(X_test_pred, y_test_pred, c='#e4007f', linestyle="--", label=label_pred)   # c='blueviolet'
        
        #if X_train_pred is None and X_test_pred is None:
        if X_underlying is not None and y_underlying is not None:        
            plt.plot(X_underlying, y_underlying, c='royalblue', label=r"$\sin(2\pi x)$")

        if X_underlying_pred is not None and y_underlying_pred is not None:  
            plt.plot(X_underlying_pred, y_underlying_pred, c='#e4007f', linestyle="--", label=label_pred)  #label="predicted function")
            # plt.ylim(-2,1.5)
            # plt.annotate("M = {}".format(degree), xy=(0.95,-1.4))      

        if X_underlying_pred_reg is not None and y_underlying_pred_reg is not None:        
            plt.plot(X_underlying_pred_reg, y_underlying_pred_reg, c='#f19ec2', linestyle="-.", label=label_pred_reg)  #label="predicted function")
        plt.ylim(-1.5,1.5)
        plt.legend(fontsize='medium') # 给图像加图例        
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
        # 使用'np.linspace'在0到pi上均匀取n_samples_out个值
        # 使用'np.cos'计算上述取值的余弦值作为特征1，使用'np.sin'计算上述取值的正弦值作为特征2
        outer_circ_x = np.cos(np.linspace(0, math.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, math.pi, n_samples_out))

        inner_circ_x = 1 - np.cos(np.linspace(0, math.pi, n_samples_in))
        inner_circ_y = 0.5 - np.sin(np.linspace(0, math.pi, n_samples_in))

        #print('outer_circ_x.shape:', outer_circ_x.shape, 'outer_circ_y.shape:', outer_circ_y.shape)
        #print('inner_circ_x.shape:', inner_circ_x.shape, 'inner_circ_y.shape:', inner_circ_y.shape)

        # 使用'np.concatenate'将两类数据的特征1和特征2分别延维度0拼接在一起，得到全部特征1和特征2
        # 使用'np.stack'将两类特征延维度1堆叠在一起
        X = np.stack(
            [np.concatenate([outer_circ_x, inner_circ_x]),
            np.concatenate([outer_circ_y, inner_circ_y])],
            axis=1
        )

        # 使用'np. zeros'将第一类数据的标签全部设置为0
        # 使用'torch. ones'将第一类数据的标签全部设置为1
        y = np.concatenate(
            [np.zeros(shape=[n_samples_out],dtype=int), np.ones(shape=[n_samples_in],dtype=int)]
        )

        #print('y shape:', y.shape)

        # 如果shuffle为True，将所有数据打乱
        if shuffle:
            # 使用'np.random.permutation'生成一个数值在0到X.shape[0]，随机排列的一维array做索引值，用于打乱数据
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        # 如果noise不为None，则给特征值加入噪声
        if noise is not None:
            # 使用'torch.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
            X += np.random.normal(loc=0.0, scale=noise, size=X.shape)
        
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

class cake_data():
    
    # 生成 cake 数据
    def __init__(self):
        self.interval = (-2,2)
        self.noise = 0
        self.test_rate = 0
        self.add_outlier = False
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        
    def cake(self,X,**kwargs):
        noise = kwargs.get('noise',0) 
        epsilon = 0
        if X.ndim == 1:
            X = X.reshape(1,len(X))
        labels = np.zeros(len(X),dtype=np.int64)
        for i,x in enumerate(X):
            if noise > 0: epsilon = np.random.normal(0, noise, 1) #  np.random.normal生成0均值，noise标准差的数据
            z = np.sum(x**2) + epsilon
            if z <= 1:
                labels[i] = 1
            else:
                labels[i] = -1
        return labels
    
    def _gen_cake_data(self,n_samples,interval,noise):
        # 等间隔生成
        X1 = np.linspace(interval[0],interval[1],n_samples)
        X2 = np.linspace(interval[0],interval[1],n_samples)
        X = []
        y = []
        for x1 in X1:
            for x2 in X2:                
                X.append(np.array([x1,x2]))
                y.append(self.cake(np.array([x1,x2]),noise=noise))
        X = np.array(X)
        y = np.array(y).squeeze()
        return (X,y,n_samples**2)
    
    def _gen_cake_data_random(self,n_samples,interval,noise):
        # 随机生成
        # 使用标准正态分布随机生成x和y坐标
        X1 = np.random.uniform(interval[0],interval[1],n_samples**2)
        X2 = np.random.uniform(interval[0],interval[1],n_samples**2)
        X = np.stack((X1, X2), axis=-1)
        y = self.cake(X,noise=noise)
        return (X,y,n_samples**2)
    
    def gen_data(self,n_samples,**kwargs):
        interval = kwargs.get('interval',self.interval)
        noise = kwargs.get('noise',self.noise)    
        random_flag = kwargs.get('random_flag',0)
        # 生成训练数据集
        if not random_flag:
            self.train_data = self._gen_cake_data(n_samples,interval,noise)
        else:
            self.train_data = self._gen_cake_data_random(n_samples,interval,noise)
        # 生成测试数据集
        test_rate = kwargs.get('test_rate',self.test_rate)    
        if test_rate > 0:
            # 计算测试集数目
            train_num = n_samples**2
            n_samples_test = np.round(np.sqrt(train_num*test_rate)).astype(np.int64)
            if not random_flag:
                self.test_data = self._gen_cake_data(n_samples_test,interval,noise)   
            else:
                self.test_data = self._gen_cake_data_random(n_samples_test,interval,noise)  
    
    def draw_data(self,fig_path,fig_name,data_type='train'):
        # 对类别数据绘图
        if data_type == 'train':
            draw_classification_data(fig_path, fig_name,(self.train_data[0],self.train_data[1]),title='训练样本集')
        elif data_type == 'test':
            draw_classification_data(fig_path, fig_name,(self.test_data[0],self.test_data[1]),title='测试样本集')
        elif data_type == 'dev':
            draw_classification_data(fig_path, fig_name,(self.dev_data[0],self.dev_data[1]),title='验证样本集')  
        

                

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
    X = np.zeros([n_samples, n_features])
    y = np.zeros([n_samples],dtype=np.int64)
    # 随机生成3个簇中心作为类别中心
    centroids = np.random.permutation(2 ** n_features)[:n_classes]
    centroids_bin = np.unpackbits(centroids.astype('uint8')).reshape((-1, 8))[:, -n_features:]
    centroids = np.array(centroids_bin).astype(np.float64)
    # 控制簇中心的分离程度
    centroids = 1.5 * centroids - 1
    # 随机生成特征值
    X[:, :n_features] = np.random.randn(n_samples, n_features)

    stop = 0
    # 将每个类的特征值控制在簇中心附近
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_class[k]
        # 指定标签值
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_features]
        # 控制每个类别特征值的分散程度
        A = 2 * np.random.rand(n_features, n_features) - 1
        X_k[...] = np.matmul(X_k, A)
        X_k += centroid
        X[start:stop, :n_features] = X_k

    # 如果noise不为None，则给特征加入噪声
    if noise > 0.0:
        # 生成noise掩膜，用来指定给那些样本加入噪声
        noise_mask = np.random.rand(n_samples) < noise
        for i in range(len(noise_mask)):
            if noise_mask[i]:
                # 给加噪声的样本随机赋标签值
                y[i] = np.random.randint(n_classes, size=[1],dtype=np.int64)
    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]
    return X, y


def gen_timeseries_data(func,num_samples,tau,normal_params=(0,0.1)):    
    # T = 1000 # 总共产⽣1000个点
    time = np.arange(1, num_samples+1, dtype=np.float32)
    mu,std = normal_params
    x = func(0.01*time) + np.random.normal(mu,std,(num_samples,))
    plot(time,[x], 'time','x',xlim=[1,num_samples],figsize=(6,3))
    
    features = np.zeros((num_samples-tau, tau))
    for i in range(tau):
        features[:,i] = x[i:num_samples-tau+i]
    labels = x[tau:].reshape((-1,1))
    
    return features,labels

class animal_data():
    # 16种动物，13种属性
    def __init__(self,**kwargs):
        # 动物属性
        self.attrs = ['small','medium','big','two-legs','four-legs','hair','hooves','mane','feather','hunt','run','fly','swim']
        # 动物名称
        self.names = {'pigeon':'鸽子','hen':'母鸡','duck':'鸭子','goose':'鹅','owl':'猫头鹰','hawk':'隼','eagle':'鹰','fox':'狐狸','dog':'狗','wolf':'狼','cat':'猫','tiger':'虎','lion':'狮','horse':'马','zebra':'斑马','ox':'牛'}
        self.attrs_value = {
            'pigeon':[1,0,0,1,0,0,0,0,1,0,0,1,0],
            'hen':  [1,0,0,1,0,0,0,0,1,0,0,0,0],
            'duck':  [1,0,0,1,0,0,0,0,1,0,0,0,1],
            'goose': [1,0,0,1,0,0,0,0,1,0,0,1,1],
            'owl':  [1,0,0,1,0,0,0,0,1,1,0,1,0],
            'hawk':  [1,0,0,1,0,0,0,0,1,1,0,1,0],
            'eagle':  [0,1,0,1,0,0,0,0,1,1,0,1,0],
            'fox':   [0,1,0,0,1,1,0,0,0,1,0,0,0],
            'dog':   [0,1,0,0,1,1,0,0,0,0,1,0,0],
            'wolf':  [0,1,0,0,1,1,0,0,0,1,1,0,0],
            'cat':   [1,0,0,0,1,1,0,0,0,1,0,0,0],
            'tiger':  [0,0,1,0,1,1,0,0,0,1,1,0,0],
            'lion':  [0,0,1,0,1,1,0,0,0,1,1,0,0],
            'horse':  [0,0,1,0,1,1,1,1,0,0,1,0,0],
            'zebra':  [0,0,1,0,1,1,1,1,0,0,1,0,0],
            'ox':   [0,0,1,0,1,1,1,0,0,0,0,0,0]
        }
        self.classnum = len(self.names)   # 动物类别数目
        self.classname = {}           # 动物类别-名称（英文和中文）字典
        for i,name in enumerate(self.names):
            self.classname[i] = (name,self.names[name])
            
        # 数据集 16类动物，每个动物包含13个属性特征（共16个数据）
        self.dataset = pd.DataFrame(columns=self.attrs,index=self.names)
        for name in self.names:
            self.dataset.loc[name] = self.attrs_value[name]
            
        # 训练集 每个动物的特征向量上增加16维的one-hot向量,即16种动物的类别向量，
        columns = [f'a{i+1}' for i in range(len(self.names))]
        columns.extend(self.attrs)
        df = pd.DataFrame(columns=columns,index=self.names)
        for i,name in enumerate(self.names):
            aid = len(self.names)*[0]
            aid[i] = 1
            aid.extend(self.attrs_value[name])
            df.loc[name] = aid
        self.train_set = df
        
        #保存数据
        data_dir = kwargs.get('data_path','')
        data_name = kwargs.get('data_name','animals.xlsx') 
        if data_dir != '':
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            data_path = os.path.join(data_dir,data_name)   
            self.train_set.to_excel(data_path,sheet_name='animals',index=False)     

    def draw_som_maps(self,maps,winner=None,text_type=None,**kwargs):
        # 标签字体设置
        SimHei_font = font_manager.FontProperties(family="SimHei", size=12)
        
        classnum = kwargs.get('classnum',self.classnum)
        names = kwargs.get('classname',self.classname)
        
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

class character_data():
    # 字符数据
    def __init__(self,**kwargs):
        # 字符名称/属性值
        self.chars_value = {
            'A': [1,0,0,0,0],
            'B': [2,0,0,0,0],
            'C': [3,0,0,0,0],
            'D': [4,0,0,0,0],
            'E': [5,0,0,0,0],
            'F': [3,1,0,0,0],
            'G': [3,2,0,0,0],
            'H': [3,3,0,0,0],
            'I': [3,4,0,0,0],
            'J': [3,5,0,0,0],
            'K': [3,3,1,0,0],
            'L': [3,3,2,0,0],
            'M': [3,3,3,0,0],
            'N': [3,3,4,0,0],
            'O': [3,3,5,0,0],
            'P': [3,3,6,0,0],
            'Q': [3,3,7,0,0],
            'R': [3,3,8,0,0],
            'S': [3,3,3,1,0],
            'T': [3,3,3,2,0],
            'U': [3,3,3,3,0],
            'V': [3,3,3,4,0],
            'W': [3,3,6,1,0],
            'X': [3,3,6,2,0],
            'Y': [3,3,6,3,0],
            'Z': [3,3,6,4,0],
            '1': [3,3,6,2,1],
            '2': [3,3,6,2,2],
            '3': [3,3,6,2,3],
            '4': [3,3,6,2,4],
            '5': [3,3,6,2,5],
            '6': [3,3,6,2,6]
        }
        self.chars = list(self.chars_value.keys())   # 字符名称      
        self.classnum = len(self.chars)          # 字符类别数目
        self.classname = {}                  # 字符类别-名称（英文和''）字典
        # 数据集 26个英文字母+6个数字，每个字符包含5个属性特征（共32个数据）
        self.dataset = pd.DataFrame(columns=['x1','x2','x3','x4','x5'],index=self.chars)
        for i,char in enumerate(self.chars):
            self.classname[i] = (char,'')
            self.dataset.loc[char] = self.chars_value[char]        
        #保存数据
        data_dir = kwargs.get('data_path','')
        data_name = kwargs.get('data_name','characters.xlsx') 
        if data_dir != '':
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            data_path = os.path.join(data_dir,data_name)   
            self.dataset.to_excel(data_path,sheet_name='characters',index=False)     
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
class iris_data():
    
    # 生成 iris 数据
    def __init__(self):
        self.test_rate = 0.0
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        
    def gen_data(self,**kwargs):
        # 加载数据
        X,y = self.load_iris_data()
        N = len(X)
        # 生成训练测试数据集
        test_rate = kwargs.get('test_rate',self.test_rate)    
        if test_rate == 0:
            self.train_data = (X,y)
        else:
            # 计算测试集数目
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
            self.train_data = (X_train,y_train)
            self.test_data = (X_test,y_test)
            #train_num = np.round(N*(1-test_rate)).astype(np.int64)
            #test_num = N - train_num
            #shuffled_indices = np.random.permutation(N)
            #train_indices = shuffled_indices[:train_num]
            #test_indices = shuffled_indices[train_num:]
            #self.train_data = (X[train_indices],y[train_indices])
            #self.test_data = (X[test_indices],y[test_indices])
        
    def load_iris_data(self,shuffle=True):
        """
        加载鸢尾花数据
        输入：
            - shuffle：是否打乱数据，数据类型为bool
        输出：
            - X：特征数据，shape=[150,4]
            - y：标签数据, shape=[150]
        """   
        iris = load_iris()
        # 从sklearn.datasets读取iris数据,并转化为tensor
        X = np.array(iris.data,dtype=np.float64)
        y = np.array(iris.target,dtype=np.int64)    
        # 数据归一化
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)

        X = (X-X_min)/(X_max-X_min)
        # 如果shuffle为True，随机打乱数据
        if shuffle:
            idx = np.random.permutation(np.arange(X.shape[0]))
            X = X[idx]
            y = y[idx] 
        self.data,self.target = X,y
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        return X, y

    
    def draw_iris(self,fig_dir,fig_name,**kwargs):
        
        #X = np.array(load_iris().data,dtype=np.float64)
        #y = np.array(load_iris().target,dtype=np.int64)  
        X = self.data
        y = self.target
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        dataset = pd.DataFrame(X,columns=feature_names)
        dataset['species'] = y
 
        #把列名换成中文的
        dataset.rename(columns={"sepal_length":"萼片长",
                             "sepal_width":"萼片宽",
                             "petal_length":"花瓣长",
                             "petal_width":"花瓣宽",
                             "species":"种类"},inplace=True)

        kind_dict = {
            0:"山鸢尾",
            1:"杂色鸢尾",
            2:"维吉尼亚鸢尾"
        }
        dataset["种类"] = dataset["种类"].map(kind_dict)
        dataset.head() #数据集的内容如下
        
        plt.rcParams['figure.dpi'] = 200
        plt.rcParams['figure.figsize'] = (7.0,5.0)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams['font.family'] = 'SimHei'

        #sns.pairplot(dataset,hue="种类")
        palette = 'Paired'   #  'hls'
        sns.pairplot(data = dataset, hue = '种类',palette=palette,kind='scatter',diag_kind='kde')

        # 保存图片    
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        plt.savefig(os.path.join(fig_dir,fig_name) ) # 保存图像到PDF文件中
        plt.show()
        
    


#def load_array(data_arrays, batch_size, is_train=True):
#    #"""Construct a PyTorch data iterator.
#    #Defined in :numref:`sec_linear_concise`"""
#    dataset = data.TensorDataset(*data_arrays)
#    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def data_split(X,y,train_percent=0.8,**kwargs): # test_percent=0.2,dev_percent=0):    
    # 划分训练集、测试集、验证集   
    test_percent = kwargs.get('test_percent',1-train_percent)
    dev_percent = kwargs.get('dev_percent',1-train_percent-test_percent)
    
    assert train_percent+test_percent+dev_percent == 1  # 数据集占比和为1 
    
    n = len(X)
    shuffled_indices = np.random.permutation(n) # 返回一个数值在0到n-1、随机排列的1-D Tensor
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
    
    y = y.astype(np.float64)
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