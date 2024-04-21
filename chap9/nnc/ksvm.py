# -*- coding: utf-8 -*-
"""
Created on Feb 3  2024

@author: Wu,Jiao

支持向量机 (SVM）

"""

import numpy as np
from sklearn import metrics
import cvxopt
from nnc.plot_tool import *

# 定义核函数
# Linear kernel  线性核
def linear_kernel(X,X_):
    if X.ndim == 1:
        X = X.reshape(1,len(X))
    if X_.ndim == 1:
        X_ = X_.reshape(1,len(X))
    # P,N = X_.shape    # 样本数目,样本维度
    K =  X @ X_.T
    return K

# Polynomial kernel   多项式核函数
def polynomial_kernel(X,X_,n=2):
    # 多项式核
    if X.ndim == 1:
        X = X.reshape(1,len(X))
    if X_.ndim == 1:
        X_ = X_.reshape(1,len(X))
    # P,N = X_.shape    # 样本数目,样本维度
    K = (X @ X_.T)**n
    return K

# Gaussian kernel
def gauss_kernel(X,X_,sigma=0.5):
    # Gaussian 核
    if X.ndim == 1:
        X = X.reshape(1,len(X))
    if X_.ndim == 1:
        X_ = X_.reshape(1,len(X))
    # P,N = X_.shape    # 样本数目,样本维度
    K = []
    for x in X:
        K.append(np.exp(-0.5*(np.linalg.norm(X_-x,axis=1)/sigma)**2))
    return np.array(K)  

# Laplace kernel
def laplace_kernel(X,X_,sigma=0.5):
    # Laplace 核
    if X.ndim == 1:
        X = X.reshape(1,len(X))
    if X_.ndim == 1:
        X_ = X_.reshape(1,len(X))
    # P,N = X_.shape    # 样本数目,样本维度
    K = []
    for x in X:
        K.append(np.exp(-np.linalg.norm(X_-x,axis=1)/sigma))
    return np.array(K)  

#输入数据 (加入阈值分量)
def addIntercept(X):   # X.shape = (P,M) 
    return np.hstack((X, np.ones((X.shape[0], 1))))#输出（P,M+1)    

# 分类问题评价指标
# 准确率 Accuracy
def accuracy(labels,preds):
    return metrics.accuracy_score(labels,preds)

# 核 SVM
# 求解二次规划（quadprog）
class kernel_svm(object):
    
    def __init__(self,input_size,sample_num,**kwargs):
        super(kernel_svm,self).__init__()
        self.input_size = input_size      # 输入维度
        self.sample_num = sample_num      # 样本数目
        
        # 核函数
        self.kernel = kwargs.get('kernel',[polynomial_kernel,2])
        
        # 软间隔SVM参数C
        self.C = kwargs.get('C',np.inf)
        
        # SVM 参数
        self.param = {}
        self.param['alpha'] = None
        self.param['weights'] = None
        self.param['bias'] = 0
        
        # 支持向量
        self.support_vectors = None   # 支持向量
        self.support_tags = None      # 支持向量标签
        self.support_alphas = None    # 支持向量系数
        self.support_id = None        # 支持向量索引
        self.support_num = None       # 支持向量个数
        # self.init_param()
        
    def kernel_matrix(self,X):
        # 计算核矩阵K
        kfun = self.kernel[0]
        kparam = self.kernel[1]
        if kparam in [None,'']:
            return kfun(X,X)
        else:
            return kfun(X,X,kparam)
    
    def solve_qp(self,K,y):
        # 解二次规划问题 min_x  1/2 x'Qx + c'x  s.t.  Gx<=h, Ax=b
        Q = cvxopt.matrix(np.diag(y) @ K @ np.diag(y))
        c = cvxopt.matrix(-np.ones(self.sample_num))
        if self.C == 0:
            G = cvxopt.matrix(-np.diag(np.ones(self.sample_num)))  # (y.astype(np.float64)))
            h = cvxopt.matrix(np.zeros(self.sample_num))
        else:
            G = cvxopt.matrix(np.vstack((-np.identity(self.sample_num),np.identity(self.sample_num))))
            h = cvxopt.matrix(np.vstack((np.zeros((self.sample_num,1)),np.ones((self.sample_num,1)) * self.C)))
            
        Aeq = cvxopt.matrix(y.reshape(-1,1).T)   # y.astype(np.float64).reshape(-1,1).T)
        beq = cvxopt.matrix([0.0])
        sol = cvxopt.solvers.qp(Q, c, G, h, Aeq, beq)
        self.param['alpha'] = np.array(sol['x']).squeeze()
        return self.param['alpha']
    
    def find_sv(self,X,y,alpha):
        # 查找支持向量
        self.support_id = np.where(alpha>1e-7)[0].tolist()  # 支持向量索引
        self.support_vectors = X[self.support_id,:]         # 支持向量
        self.support_tags = y[self.support_id]               # 支持向量标签（期望输出） 
        self.support_alphas = alpha[self.support_id]        # 支持向量系数
        self.support_num = len(self.support_id)
    
    def cal_weights(self):
        self.param['weights'] = np.sum(np.multiply(np.multiply(self.support_alphas,self.support_tags).reshape(-1,1),self.support_vectors),axis=0)
        return self.param['weights']
    
    def cal_bias(self):
        # 计算核
        if self.kernel[1] in [None,'']:
            K = self.kernel[0](self.support_vectors,self.support_vectors)
        else:
            K = self.kernel[0](self.support_vectors,self.support_vectors,self.kernel[1])
        ys = np.sum(np.multiply(np.multiply(self.support_alphas,self.support_tags),K),axis=1)
        self.param['bias'] = np.sum(self.support_tags - ys)/self.support_num
        return self.param['bias'] 
    
    def predict(self,X):
        # 计算核
        if self.kernel[1] in [None,'']:
            K = self.kernel[0](X,self.support_vectors)
        else:
            K = self.kernel[0](X,self.support_vectors,self.kernel[1])
        y_pred = np.sum(np.multiply(np.multiply(self.support_alphas,self.support_tags),K),axis=1) + self.param['bias']
        return np.sign(y_pred)    

# 定义Runner类
class Runner(object):
    
    def __init__(self,model,metric=accuracy):        
        self.model = model       # 调用模型
        self.metric = metric  # metric     # 评价函数 
        self.train_score = None
        self.train_pred = None
        self.test_score = None
        self.test_pred = None
    
    def train(self,train_set,test_set=None,**kwargs):
        # 数据集(X,y)
        X,y = train_set  
        P = X.shape[0]    # 样本数目
        
        # 计算核矩阵
        X_ = addIntercept(X)  # 加入偏置
        #X_ = X
        K = self.model.kernel_matrix(X_)
        
        # 解二次规划求系数向量alpha
        alpha = self.model.solve_qp(K,y.astype(np.float64))
        
        # 支持向量
        self.model.find_sv(X,y,alpha)
        
        # 计算权值
        if self.model.kernel[0] == linear_kernel:
            weights = self.model.cal_weights()
        
        # 计算偏置
        bias = self.model.cal_bias()
        
        # 评估模型
        self.train_pred,self.train_score = self.evaluate(train_set)
        if test_set is not None:
            self.test_pred,self.test_score = self.evaluate(test_set)
        
    
    def evaluate(self,dataset):
        X, y = dataset
        # 计算模型输出
        y_pred = self.model.predict(X)  
        # 准确率 metric = Accuracy
        return y_pred,self.metric(y,y_pred)
        
    def predict(self,X):
        return self.model.predict(X)        

