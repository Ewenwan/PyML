#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
Hadoop 流   多机器分发作业合作
mrMeanReducer.py
'''
# 运行
# linux 平台
# cat inputFile.txt | python mrMeanMapper.py | python mrMeanReducer.py    # 打开文件 管道 管道 
# windows 平台 DOS 窗口下
# python mrMeanMapper.py < inputFile.txt | python mrMeanReducer.py

import sys# 系统输入输出
from numpy import mat, mean, power#矩阵 均值 平方

def read_input(file):
    for line in file:      #  每一行
        yield line.rstrip()#  列表
       
input = read_input(sys.stdin)# 系统输入流 >>> 列表

#按 制表符 分割 每一行 成 列表
mapperOut = [line.split('\t') for line in input]

#accumulate total number of samples, overall sum and overall sum sq
cumVal=0.0
cumSumSq=0.0
cumN=0.0
for instance in mapperOut:
    nj = float(instance[0]) # 数量
    cumN += nj              # 数量和
    cumVal += nj*float(instance[1])  # 均值*数量
    cumSumSq += nj*float(instance[2])# 平方均值 *数量
    
#总的均值
mean = cumVal/cumN#总均值
meanSq = cumSumSq/cumN#总平方均值

#输出 大小 均值 平方均值 
print "%d\t%f\t%f" % (cumN, mean, meanSq)
print >> sys.stderr, "report: still alive"
