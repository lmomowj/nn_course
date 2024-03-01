# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:45:54 2024

@author: Wu,Jiao
"""

import os
import random
import torch
#%matplotlib inline
import numpy as np

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

# "text.usetex":True,
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

from nn_tool import activation as act
#from nn_tool.activation import *

func_name = {'unit':['Unit','Unit','单位阶跃函数'],
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
         'gelu':['GELU','Gaussian Error Linear Unit','高斯误差线性单元']
         }


def draw_func(num=None,func=None,**kwargs):
    
    if num is None:
        # 样本数据范围及数目  Sample data
        num = np.linspace(-10, 10, 100)
        
    if func is None:
        func = input( 'Input function expression what you want draw: \n(unit,sgn,logistic,tanh,sigmoid,sigmoid_d,relu,leakyRelu,elu,softplus,swish,piecewise_s,piecewise_d,hard_logistic,hard_tanh,gelu )\n' )
    
    dpi = kwargs.get('dpi',300)       # 图片分辨率 默认300
    color = kwargs.get('color',palette[random.randint(1,len(palette))-1])   # 默认随机生成
    linestyle = kwargs.get('linestyle','solid')                    # 默认“solid"
    linewidth = kwargs.get('linewidth',2)                        # 默认 2
    
    # plot 标题
    title_en = '{} Activation Function'.format(func_name[func][0])
    title_zh = func_name[func][2]
    title_text = title_en     # 标题 默认为英文
    
    # 是否绘制导函数图像
    draw_grad = kwargs.get('draw_grad',None)
    if draw_grad is None:
        grad_order = 0  
    else:
        grad_order = kwargs.get('grad_order',1)
                     
    fig = plt.figure(dpi=dpi)
    try:
        plt.xlabel(r'$z$')  #  ('x label')
        plt.ylabel(r'$f(z)$')        
        if (func in ['piecewise_s','piecewise_d']):
            plt.title(func_name[func][0],fontsize=16,fontproperties=kaiti_font_title)
            a = input('参数 a = ')
            afunc = eval(f'act.{act.func_name[func]}({a})')   
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
            afunc = eval(f'act.{act.func_name[func]}()({a})')
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
            afunc = eval(f'act.{act.func_name[func]}({b})')
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
            afunc = eval(f'act.{act.func_name[func]}({mu},{sigma})')
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
        else:
            #if func == 'relu': color = palette[5]     # green  
            #if func == 'logistic': color = palette[10]  # blue 
            afunc = eval(f'act.{act.func_name[func]}()')
            if (func in ['sigmoid','sigmoid_d']):
                plt.title(func_name[func][0],fontsize=16,fontproperties=kaiti_font_title)
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
        plt.tight_layout()
        plt.show()
    except TypeError:
        print(
            'Input function expression is wrong or the funciton is not configured'
        )
    
def draw_funcs(num=np.linspace(-10, 10, 100),funcs=[]):
        
    if funcs == []:
        funcs = input( 'Input function expression what you want draw: \n(unit,sgn,logistic,tanh,sigmoid,sigmoid_d,relu,leakyRelu,elu,softplus,swish,piecewise_s,piecewise_d,hard_logistic,hard_tanh,gelu )\n' ).split(',')
    
    funcs = [func.strip() for func in funcs]
    fig = plt.figure(dpi=300)
    try:
        plt.xlabel(r'$z$')  #  ('x label')
        plt.ylabel(r'$f(z)$')            
        for func in funcs:            
            cnum = random.randint(1,len(palette))-1      
            print('func:',func)
            if (func in ['unit','sgn']):
                plt.title('阈值型函数',fontsize=16,fontproperties=kaiti_font_title)
                if func == 'unit':cnum = 9   # blue
                if func == 'sgn':cnum =  14   # zise
                plt.plot(
                    num,
                    [eval(f'{func}()').evalf(subs={symbols("x"): i}) for i in num],
                    linewidth=2.5,
                    color=palette[cnum],  
                    label=func_name[func]+' 函数'
                    )
                plt.legend(prop=kaiti_font_legend) # ,loc='best')                
            elif (func in ['s_piecewise','d_piecewise']):
                # plt.title(func_name[func],fontsize=16,fontproperties=kaiti_font)
                a = input('a = ')
                if (func == 's_piecewise'):                
                    ac = round(1/float(a),2)
                    cnum = 13
                elif (func == 'd_piecewise'):  
                    ac = round(2/float(a),2)
                    cnum = 1
                plt.plot(
                    num,
                    [eval(f'{func}({a})').evalf(subs={symbols("x"): i}) for i in num],
                    linewidth=2.5,
                    color=palette[cnum],
                    label=func_name[func]+'$(c={},z_c={})$'.format(a,ac)
                    )
                plt.legend(prop=kaiti_font_legend) # ,loc='best')
            elif (func in ['leakyRelu','elu']):
                # plt.title(func_name[func]+' Activation Function',fontsize=16)
                if func == 'leakyRelu':cnum = 14   # blue
                if func == 'elu': cnum = 4     # green
                a = input('a =')
                plt.plot(
                    num,
                    [eval(f'{func}({a})').evalf(subs={symbols("x"): i}) for i in num],
                    linewidth=2.5,
                    color=palette[cnum],
                    label=func_name[func]+'$(\\alpha={})$'.format(a)
                    )  
                plt.legend()    
            elif (func in ['logistic','sigmoid_d','tanh','hard_logistic','hard_tanh']):
                if func == 'logistic':cnum = 10  #   5   # blue
                if func == 'sigmoid_d':cnum = 14
                if (func == 'tanh'): cnum = 4    #   2     # green
                if (func == 'hard_logistic'): cnum = 13
                if (func == 'hard_tanh'): cnum = 0
                fnames = {'logistic':'Logistic','tanh':'Tanh','sigmoid_d':'S','hard_logistic':'Hard-Logistic','hard_tanh':'Hard-Tanh'}
                if (func in ['hard_logistic','hard_tanh']):
                    plt.title('Hard-Sigmoid型函数',fontsize=16,fontproperties=kaiti_font_title)
                else:
                    plt.title('Sigmoid型函数',fontsize=16,fontproperties=kaiti_font_title)
                    
                plt.plot(
                    num,
                    [eval(f'{func}()').evalf(subs={symbols("x"): i}) for i in num],
                    linewidth=2.5,
                    color=palette[cnum],  
                    label= fnames[func]+' 函数'  #  func_name[func]+' 函数'
                    )
                plt.legend(prop=kaiti_font_legend)   
            elif (func == 'swish'):
                # beta = input('betas = ')
                # betas = [0,0.5,1,100]
                betas = [1]
                # plt.title('Swish 函数',fontsize=16,fontproperties=kaiti_font_title)
                for i,beta in enumerate(betas):
                    # cnum = random.randint(1,len(palette))-1   
                    cnum = 10
                    plt.plot(
                        num,
                        [eval(f'{func}({beta})').evalf(subs={symbols("x"): i}) for i in num],
                        linewidth=2.5,
                        linestyle =  linestyle_str[i%4][1],   # 'dashed',
                        color=palette[cnum],  
                        label= 'Swish $(\\beta={})$'.format(beta)
                        )
                    plt.legend(prop=kaiti_font_legend) 
            elif (func in ['normal_pdf','normal_cdf','gelu']):
                mu = input('mu = ')
                sigma = input('sigma = ')
                cnum = 6
                # plt.title(func_name[func]+' Activation Function',fontsize=16)
                plt.plot(
                    num,
                    [eval(f'{func}({mu},{sigma})').evalf(subs={symbols("z"): i}) for i in num],
                    linewidth=2.5,
                    color=palette[cnum],
                    label=f'GELU $(\\mu={mu},\\sigma={sigma})$'
                    )
                plt.legend()
            else:
                # plt.title(func_name[func]+' Activation Function',fontsize=16)                
                if (func == 'relu'): cnum = 0
                if (func == 'softplus'): cnum = 10
                plt.plot(
                    num,
                    [eval(f'{func}()').evalf(subs={symbols("x"): i}) for i in num],
                    linewidth=2.5,
                    color=palette[cnum],  
                    label=func_name[func]
                    )
                plt.legend()
        plt.tight_layout()
        plt.show()
    except TypeError:
        print(
            'input function expression is wrong or the funciton is not configured'
        )
            
def func_color(num,func='logistic'):   
    
    cnum = random.randint(1,len(palette))-1   
    for cnum in range(len(palette)):
        fig = plt.figure(dpi=300)
        try:
            plt.xlabel(r'$z$')  #  ('x label')
            plt.ylabel(r'$f(z)$')
            plt.title('激活函数 - color-{}'.format(cnum),fontsize=16,fontproperties=kaiti_font_title)
            plt.plot(
                num,
                [eval(f'{func}()').evalf(subs={symbols("x"): i}) for i in num],
                linewidth=2.5,
                color=palette[cnum],  
                label= func_name[func]+' 函数'  #  func_name[func]+' 函数'
                )
            plt.legend(prop=kaiti_font_legend)  
            plt.tight_layout()
            plt.show()
        except TypeError:
            print(
                'input function expression is wrong or the funciton is not configured'
            )
    
        


if __name__ == '__main__':
    
    # x = np.linspace(-13, 5, 100)  # Sample data
    # x = np.linspace(-10, 10, 100)  # Sample data
    x = np.linspace(-6, 6, 100)  # Sample data
    # x = np.linspace(-4, 4, 100)  # Sample data
    # func_color(x)
    
    # draw_func(x)
    draw_funcs(x)



    

