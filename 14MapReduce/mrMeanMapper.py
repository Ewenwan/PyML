#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
Hadoop 流   多机器分发作业合作
mrMeanMapper.py
'''
# 运行
# linux 平台
# cat inputFile.txt | python mrMeanMapper.py     # 打开文件 管道 
# windows 平台 DOS 窗口下
# python mrMeanMapper.py < inputFile.txt
import sys # 系统输入输出
from numpy import mat, mean, power #矩阵 均值 平方

def read_input(file):
    for line in file:      #  每一行
        yield line.rstrip()#  列表
        
input = read_input(sys.stdin) # 系统输入流 >>> 列表
input = [float(line) for line in input] # 转换成 浮点型数据
numInputs = len(input) # 数组长度
input = mat(input)     # 转成 矩阵
sqInput = power(input,2)# 矩阵平方

#输出 大小 均值 平方均值 
print "%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput))  
print >> sys.stderr, "report: still alive" #向标准错误输出发送报告 表明还在工作
