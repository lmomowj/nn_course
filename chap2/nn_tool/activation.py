# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:49:21 2024

@author: Wu,Jiao

常用激活函数

使用sympy包定义的激活函数
"""

import numpy as np
import random
import sympy
from sympy import symbols,evalf,diff,stats

func_name = {'unit':'Unit',
             'sgn':'SGN',
             'logistic':'Logistic',
             'tanh':'Tanh',
             'sigmoid_d':'Sigmoid_d',
             'piecewise_s':'Piecewise_s',
             'piecewise_d':'Piecewise_d',
             'hard_logistic':'Hard_Logistic',
             'hard_tanh':'Hard_Tanh',
             'relu':'ReLU',
             'leakyRelu':'LeakyReLU',
             'elu':'ELU',
             'softplus': 'Softplus',
             'swish':'Swish',
             'gelu':'GELU',
             'softmax':'SoftMax'
             }

def cal_value(func,z):
    return float(func.evalf(subs={symbols("z"): z}))

# 单位阶跃函数
class Unit():    
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.Piecewise((0,z<0),(1,z>=0))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 符号函数
class SGN(): 
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.Piecewise((-1,z<0),(0,z==0),(1,z>0))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# Logistic函数
class Logistic(): 
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return 1./(1+sympy.exp(-z))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# Tanh 双曲正切函数
class Tanh():
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return (sympy.exp(z)-sympy.exp(-z))/(sympy.exp(z) + sympy.exp(-z))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 双极性Sigmoid函数(课本中Tanh)
class Sigmoid_d():
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return (1-sympy.exp(-z))/(1+sympy.exp(-z))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 单极性分段线性变换函数
class Piecewise_s():
    # 定义函数
    def __call__(self,c=0):
        z = symbols('z')
        if c == 0: c=0.5
        return sympy.Piecewise((0,z<0),(c*z,(z>=0) & (c*z<1)),(1,(z>=0) & (c*z>=1)))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 双极性分段线性变换函数
class Piecewise_d():
    # 定义函数
    def __call__(self,c=0):
        z = symbols('z')
        if c == 0: c=0.5
        return sympy.Piecewise((-1,z<0),(c*z-1,(z>=0) & (c*z-1<1)),(1,(z>=0) & (c*z-1>=1)))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)
    
# Hard-Logistic 函数 
class Hard_Logistic():
    # 定义函数
    def __call__(self):
        z = symbols('z')
        g = 0.25*z+0.5
        return sympy.Piecewise((0,g<=0),(g,(g>0) & (g<1)),(1,(g>=1)))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# Hard-Tanh 函数
class Hard_Tanh():
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.Piecewise((-1,z<=-1),(z,(z>-1) & (z<1)),(1,(z>=1)))       
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# ReLU 函数
class ReLU():
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.Piecewise((0,z<0),(z,z>=0))      
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 带泄露的ReLU
class LeakyReLU():
    # 定义函数
    def __init__(self,a=0):
        # a 的值 为0.01左右
        if a == 0: 
            self.a = 0.01
        else:
            self.a = a            
    def __call__(self):        
        z = symbols('z')
        return sympy.Piecewise((self.a*z,z<0),(z,z>=0))   
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 指数线性单元 # Exponential Linear Units
class ELU():
    # 定义函数
    def __call__(self,a=0):
        z = symbols('z')
        # a 的值 为0.01左右
        if a == 0: a=0.01
        return sympy.Piecewise((a*(sympy.exp(z)-1),z<0),(z,z>=0))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)
# softolus 函数 
class Softplus():
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.log(1+sympy.exp(z))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# Swish激活函数
# 自门控激活函数, 由谷歌的研究者发布
class Swish():
    # 定义函数
    def __call__(self,b=1):
        z = symbols('z')
        return z/(1+sympy.exp(-b*z))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# GELU函数
class GELU():
    # 定义函数
    def __call__(self,mu=0,sigma=1):
        x = symbols('x')
        y = sympy.exp(-(x-mu)**2/(2*sigma**2))/sympy.sqrt(2*np.pi)
        Y = sympy.integrate(y)
        z = symbols('z')
        return z*(Y.subs(x, z) - Y.subs(x, -float('inf')))
    # 显示函数表达式
    def formula(self):
        return self.__call__()
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# SoftMax函数
class SoftMax():
    # 定义函数 
    def __call__(self,x:np.ndarray):
        # x: 输入x向量
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x) 
    # 函数的梯度 值
    def gradient(self,z):   
        s = self.__call__(z)
        return s * (1 - s)