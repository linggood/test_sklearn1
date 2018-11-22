# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:57:05 2018

@author: HP
"""
'''
import pandas as pd
#初始化参数，数据标准化
filename = '内蒙古碳排放指标统计数据.xls'
filename1 = '离差标准化后的数据.xls'
data = pd.read_excel(filename,index_col ="年份（内蒙古）")
data = (data-data.min())/(data.max()-data.min())
data = data.reset_index()
data.to_excel(filename1,index = False)
'''
'''
import pandas as pd
import numpy as np
from sklearn import linear_model

filename = '内蒙古碳排放指标统计数据.xls'
data = pd.read_excel(filename,index_col ="年份（内蒙古）")
data['二氧化碳排放量/万吨'] = np.log(data['二氧化碳排放量/万吨'])
data['经济水平（人均GDP）/万元'] = np.log(data['经济水平（人均GDP）/万元'])
data['产业结构（第二产业占比）/%'] = np.log(data['产业结构（第二产业占比）/%'])
data['人口规模/万人'] = np.log(data['人口规模/万人'])
data['城镇化率/%'] = np.log(data['城镇化率/%'])
data['能源结构(煤炭占比)/%'] = np.log(data['能源结构(煤炭占比)/%'])
data['能源强度（吨标准煤/万元）'] = np.log(data['能源强度（吨标准煤/万元）'])**2
clf = linear_model.Ridge (alpha = .5)
print(data['能源强度（吨标准煤/万元）'])
#print(data.head())
'''
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:59:47 2018

@author: HP
"""
import xlrd
import numpy
from sklearn.linear_model import LinearRegression,Ridge 
from matplotlib.font_manager import  FontProperties
import matplotlib.pyplot as plt
def read_xlrd():
    excel1 = xlrd.open_workbook(r'离差标准化后的数据.xls')
    sheet2_name = excel1.sheet_names()[0]
    sheet2 = excel1.sheet_by_name('Sheet1')
    row_list = []
    #print (sheet2.name,sheet2.nrows,sheet2.ncols)
    for i in range(sheet2.nrows):
        #print(sheet2.row_values(i))
        if i != 0:
            row_list.append(sheet2.row_values(i))
    #print (row_list)
    row_list = numpy.array(row_list,dtype='float')
    #print (row_list)
    '''
    list1 = numpy.log(row_list[:,2:7])
    list2 = numpy.log(row_list[:,7])
    print(list1)
    print(list2)
    X = numpy.insert(list1,0,values = list2,axis=1)
    print(X)  
    '''
    X = row_list[:,2:]
    Y = row_list[:,1]
    Z = row_list[:,0]
    #print(X,Y)
    return X,Y,Z     
def Linear():
    X,Y,Z = read_xlrd()    
    model = Ridge(alpha = 0.000001)#设置线性回归
    model.fit(X, Y)     # 训练模型
    a = model.coef_     #各个参数的权重
    b = model.intercept_#截距b的值 Y=a1*x1+a2*x2+a3*x3+b
# 输出错误差值平方均值 
    print('各个参数的权重：',a)
#    print('截距：',b)
    x_test = X[20:]#测试数据的X数组的值
    y_test = Y[20:]#测试数据的Y的值
    predictions = model.predict(x_test)#选择后11个数据作为测试数据
    for i, prediction in enumerate(predictions):
        print('预测: %s, 结果: %s' % (prediction, y_test[i]))
    c = np.mean((model.predict(x_test) - y_test) ** 2)
    print('错误差值平方均值',c)
    a = model.score(x_test,y_test)
    print('得分：%.2f' %a)#模型得分 
def getChineseFont():
    return FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
def matplotlab1():
    X,Y,Z = read_xlrd()
    plt.subplot(3,1,3)
    plt.ylabel('二氧化碳排放量/万吨',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())  
    plt.plot(Z,Y, color='blue', marker='o', linestyle='solid')
    plt.title('年份与碳排放的关系图', fontproperties=getChineseFont())
    plt.subplot(3,3,1)
    plt.ylabel('经济水平（人均GDP）/万元',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与人均GDP的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,0], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,2)
    plt.ylabel('产业结构（第二产业占比）/%',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与第二产业占比%的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,1], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,3)
    plt.ylabel('人口规模/万人',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与人口规模/万人的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,2], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,4)
    plt.ylabel('城镇化率/%',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与城镇化率/%的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,3], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,5)
    plt.ylabel('能源结构(煤炭占比)/%',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与能源结构(煤炭占比)/%的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,4], color='red', marker='o', linestyle='solid')
    plt.subplot(3,3,6)
    plt.ylabel('能源强度（吨标准煤/万元）',fontproperties=getChineseFont())
    plt.xlabel("年份",fontproperties=getChineseFont())
    plt.title('年份与能源强度（吨标准煤/万元）的关系图', fontproperties=getChineseFont())
    plt.plot(Z,X[:,5], color='red', marker='o', linestyle='solid')
    plt.show()
#read_xlrd()
if __name__=='__main__':
    Linear()
#    matplotlab1()