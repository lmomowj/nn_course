# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:49:21 2024

@author: Jiao

常用激活函数

"""

import numpy as np
import random
import sympy
from sympy import symbols,evalf,diff,stats,lambdify

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
             'softmax':'SoftMax',
             'linear_2d':'linear_2d',
             'linear':'Linear'
             }

def cal_value(func,x):
    z = symbols('z')
    f = lambdify(z, func, 'numpy')
    return f(x)
    #return float(func.evalf(subs={symbols("z"): z}))
    
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
    # 参数初始化
    def __init__(self,a=1):
        self.a = a   
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return (1-sympy.exp(-self.a*z))/(1+sympy.exp(-self.a*z))
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
    # 参数初始化
    def __init__(self,c=0):
        if c == 0: 
            self.c = 0.5
        else:
            self.c = c   
    # 定义函数
    def __call__(self):
        z = symbols('z')        
        return sympy.Piecewise((0,z<0),(self.c*z,(z>=0) & (self.c*z<1)),(1,(z>=0) & (self.c*z>=1)))
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
    # 参数初始化
    def __init__(self,c=0):
        if c == 0: 
            self.c = 0.5
        else:
            self.c = c   
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.Piecewise((-1,z<0),(self.c*z-1,(z>=0) & (self.c*z-1<1)),(1,(z>=0) & (self.c*z-1>=1)))
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
    def __init__(self,a=0):
        # a 的值 为0.01左右
        if a == 0: 
            self.a = 0.01
        else:
            self.a = a   
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return sympy.Piecewise((self.a*(sympy.exp(z)-1),z<0),(z,z>=0))
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
    def __init__(self,b=1):
         self.b = b   
    # 定义函数
    def __call__(self):
        z = symbols('z')
        return z/(1+sympy.exp(-self.b*z))
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
    def __init__(self,mu=0,sigma=1):
        self.mu = mu
        self.sigma = sigma
    # 定义函数
    def __call__(self):
        x = symbols('x')
        y = sympy.exp(-(x-self.mu)**2/(2*self.sigma**2))/sympy.sqrt(2*np.pi)
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

# Linear 函数
class Linear():
    # 定义函数
    def __init__(self,W,b):
        self.W = np.array(W)
        self.b = np.array(b)
    # 定义函数
    def __call__(self,x):
        # x: 输入x向量
        x = np.array(x)
        return np.dot(self.W,x.T)+self.b
    # 函数的梯度 值
    def gradient(self,z): 
        return self.W

# Linear 函数
class linear_2d():
    # 定义函数
    def __init__(self,W,b):
        self.W = W     
        if len(W) == 2:
            self.k = -self.W[0]/self.W[1]
            self.b = -b/self.W[1]
        elif len(W) == 1:
            self.k = W
            self.b = b
    # 定义函数
    def __call__(self):
        # x: 输入x向量
        z = symbols('z')
        return self.k*z+self.b
    # 计算函数值
    def value(self,z):
        return cal_value(self.__call__(),z)
    # 函数的梯度
    def gradient(self,order=1):        
        return diff(self.__call__(), symbols('z'), order)
    # 计算梯度值
    def gradient_value(self,z,order=1):
        return cal_value(self.gradient(order),z)

# 激活函数绘图
import matplotlib
from matplotlib import rcParams,font_manager
from matplotlib import pyplot as plt # matplotlib 是 Python 的绘图库
from matplotlib_inline import backend_inline
config = {
"font.family":'serif',
"font.size": 14,
"mathtext.fontset":'stix',
"font.serif": ['SimSun'],
}
rcParams.update(config)
kaiti_font_title = font_manager.FontProperties(family="KaiTi", size=20)
kaiti_font_legend = font_manager.FontProperties(family="KaiTi", size=14)
import seaborn as sns
palette = sns.color_palette("hls", 16)
# sns.set(style='ticks',palette='pastel')
# sns.set(style='whitegrid',palette='pastel')
sns.set_style("whitegrid")
sns.set_palette(palette) 

#绘图 线  名称
linestyle_str = [
     ('solid', 'solid'),      # 同 (0, ()) or '-'
     ('dotted', 'dotted'),    # 同 (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # 同 as '--'
     ('dashdot', 'dashdot')]  # 同  '-.'   

func_name_p = {'unit':['Unit','Unit','单位阶跃函数'],
         'sgn':['Sgn','Sgn','符号函数'],
         'relu':['ReLU','Rectified Linear Unit','修正线性单元'],
         'leakyRelu':['LeakyReLU','Leaky ReLU','带泄露的ReLU'],
         'elu':['ELU','Exponential Linear Unit','指数线性单元'],
         'softplus': ['Softplus','Softplus','Softplus函数'],
         'swish':['Swish','Swish','Swish函数'],
         'logistic': ['Logistic','Logistic','Logistic函数'],
         'sigmoid':['单极性Sigmoid函数','',''],
         'sigmoid_d':['双极性Sigmoid函数','',''],
         'tanh':['Tanh','Tanh','双曲正切函数'],
         'hard_logistic':['Hard-Logistic','Hard-Logistic','Hard-Logistic函数'],
         'hard_tanh':['Hard-Tanh','Hard-Tanh','Hard-Tanh函数'],
         'piecewise_s':['单极性分段线性变换函数','',''],
         'piecewise_d':['双极性分段线性变换函数','',''],
         'softMax':['SoftMax','SoftMax','SoftMax函数'],
         'normal_pdf':['Normal_PDF','Normal_PDF','正态分布概率密度函数'],
         'normal_cdf':['Normal_CDF','Normal_CDF','正态分布分布函数'],
         'gelu':['GELU','Gaussian Error Linear Unit','高斯误差线性单元'],
         'linear_2d':['linear_2d','Linear','线性函数(2维)'],
         'linear':['Linear','Linear','线性函数'],    
         }


def draw_func(num=None,func=None,**kwargs):
    
    if num is None:
        # 样本数据范围及数目  Sample data
        num = np.linspace(-10, 10, 100)
        
    if func is None:
        func = input( 'Input function expression what you want draw: \n(unit,sgn,logistic,tanh,sigmoid,sigmoid_d,relu,leakyRelu,elu,softplus,swish,piecewise_s,piecewise_d,hard_logistic,hard_tanh,gelu )\n' )
    
    dpi = kwargs.get('dpi',100)       # 图片分辨率 默认300
    figsize = kwargs.get('figsize,',(6,4))
    color = kwargs.get('color',palette[random.randint(1,len(palette))-1])   # 默认随机生成
    linestyle = kwargs.get('linestyle','solid')                    # 默认“solid"
    linewidth = kwargs.get('linewidth',2)                        # 默认 2
    
    # plot 标题
    title_en = '{} Activation Function'.format(func_name_p[func][0])
    title_zh = func_name_p[func][2]
    title_text = title_en     # 标题 默认为英文
    
    # 是否绘制导函数图像
    draw_grad = kwargs.get('draw_grad',None)
    if draw_grad is None:
        grad_order = 0  
    else:
        grad_order = kwargs.get('grad_order',1)
                     
    fig = plt.figure(figsize=figsize,dpi=dpi)
    try:
        plt.xlabel(r'$z$')  #  ('x label')
        plt.ylabel(r'$f(z)$')        
        if (func in ['piecewise_s','piecewise_d']):
            plt.title(func_name_p[func][0],fontsize=16,fontproperties=kaiti_font_title)
            a = input('参数 a = ')
            afunc = eval(f'{func_name[func]}({a})')   
            if (func == 'piecewise_s'):                
                a_c = round(1/float(a),2)
            elif (func == 'piecewise_d'):  
                a_c = round(2/float(a),2)
            label = r'$c={},z_c={}$'.format(a,a_c)
            if draw_grad is None:
                plt.plot(
                    num,
                    [afunc.value(i) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            else:
                plt.plot(
                    num,
                    [afunc.gradient_value(i,grad_order) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            plt.legend()
        elif (func in ['leakyRelu','elu']):
            #if func == 'leakyRelu': color = palette[10]   # blue
            #if func == 'elu': color = palette[5]       # green
            plt.title(title_en,fontsize=16)
            a = input('参数 a = ')
            afunc = eval(f'{func_name[func]}()({a})')
            label=f'$\\alpha={a}$'
            if draw_grad is None:
                plt.plot(
                    num,
                    [afunc.value(i) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            else:
                plt.plot(
                    num,
                    [afunc.gradient_value(i,grad_order) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            plt.legend()
        elif (func in ['swish']):
            plt.title(title_en,fontsize=16)
            b = input('参数 b = ')
            afunc = eval(f'{func_name[func]}({b})')
            label=f'$\\beta={b}$'
            if draw_grad is None:
                plt.plot(
                    num,
                    [afunc.value(i) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            else:
                plt.plot(
                    num,
                    [afunc.gradient_value(i,grad_order) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            plt.legend()
        elif (func in ['normal_pdf','normal_cdf','gelu']):   
            # 输入参数
            mu = input('mu = ')
            sigma = input('sigma = ')
            afunc = eval(f'{func_name[func]}({mu},{sigma})')
            label=f'$\\mu={mu},\\sigma={sigma}$'
            if func in ['normal_pdf','normal_cdf']:
                plt.title(title_zh,fontsize=16,fontproperties=kaiti_font_title)
            else:
                plt.title(title_en,fontsize=16)            
            if draw_grad is None:
                plt.plot(
                    num,
                    [afunc.value(i) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            else:
                plt.plot(
                    num,
                    [afunc.gradient_value(i,grad_order) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            plt.legend()
        elif (func in ['linear_2d','linear']):
            # 输入参数
            w = input('w = ')
            b = input('b = ')
            w = [float(wi.strip()) for wi in w.strip('[] ').split(',')]
            b = float(b)
            if func == 'linear_2d':
                afunc = eval(f'{func_name[func]}({w},{b})')
            if func == 'linear':
                k = -w[0]/w[1]
                b_0 = -b/w[1]
                afunc = eval(f'{func_name[func]}({k},{b_0})')
            label=f'$w={w},b={b}$'
            plt.title('Linear Function',fontsize=16) 
            if draw_grad is None:
                if func == 'linear_2d':
                    fvalue = [afunc.value(i) for i in num]
                elif func == 'linear':
                    fvalue = [afunc(i) for i in num]
                plt.plot(
                    num,
                    fvalue,  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            else:
                if func == 'linear_2d':
                    fgvalue = [afunc.gradient_value(i,grad_order) for i in num]
                elif func == 'linear':
                    fgvalue = [afunc.gradient(i) for i in num]
                plt.plot(
                    num,
                    fgvalue,
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    label= label
                    )
            plt.legend()
        else:
            #if func == 'relu': color = palette[5]     # green  
            #if func == 'logistic': color = palette[10]  # blue 
            afunc = eval(f'{func_name[func]}()')
            if (func in ['sigmoid','sigmoid_d']):
                plt.title(func_name_p[func][0],fontsize=16,fontproperties=kaiti_font_title)
            else:
                plt.title(title_en,fontsize=16)  
            if draw_grad is None:
                plt.plot(
                    num,
                    [afunc.value(i) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    )
            else:
                plt.plot(
                    num,
                    [afunc.gradient_value(i,grad_order) for i in num],  
                    linewidth=linewidth,
                    color=color,
                    linestyle = linestyle,
                    )
            #plt.legend()
        #plt.tight_layout()
        plt.show()
    except TypeError:
        print(
            'Input function expression is wrong or the funciton is not configured'
        )