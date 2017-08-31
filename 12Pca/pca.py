#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
主成份分析  principal component analysis  PCA
先求解 协方差矩阵
再求解 协方差举矩阵的特征值和特征向量
'''
# 测试程序
# import pca  pca.pca_test()
# import pca  pca.secomTest()
from numpy import *

#载入文件数据   文件名    分隔符
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]# 转换成浮点型 列表
    return mat(datArr)# 转换成 numpy 下的数组mat类型

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0) # 每个样本的平均值 一行
    meanRemoved = dataMat - meanVals # 减去平均值
    covMat = cov(meanRemoved, rowvar=0)# 计算协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))# 计算协方差矩阵的特征值 和特征向量
    eigValInd = argsort(eigVals)              # 特征值从小到大排列
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 前 topNfeat 个较大的特征值对应的 排序下标
    redEigVects = eigVects[:,eigValInd]       # 逆序排列得到前   topNfeat 个特征向量
    lowDDataMat = meanRemoved * redEigVects   # 将原去均值花后的数据 乘以 特征向量 转换到新 坐标空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals#按照逆转换 到原来的 空间数据
    return lowDDataMat, reconMat # 降维后的数据，以及还原后的主成分数据

# 测试
def pca_test(file_name='testSet.txt'):
    import matplotlib
    import matplotlib.pyplot as plot
    datMat = loadDataSet(file_name)
    lowDMat, reconMat  = pca(datMat, 1)# 降维成1维矩阵  前一个主成份
    print "原数据维度: "
    print shape(datMat)
    print "降维后数据维度: "
    print shape(lowDMat)
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datMat[:,0].flatten().A[0], datMat[:,1].flatten().A[0],marker="^",s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker="o",s=50,c='red')
    plot.show()

# 处理缺失值  用均值代替
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ') #半导体数据
    numFeat = shape(datMat)[1]# 样本 的特征 维度
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) # 某特征 非缺失值的均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal        # 缺失值有 上述均值替代
    return datMat

# 实验
def secomTest():
    datMat = replaceNanWithMean()
    meanVals = mean(datMat, axis=0) # 每个样本的平均值 一行
    meanRemoved = datMat - meanVals # 减去平均值
    covMat = cov(meanRemoved, rowvar=0)# 计算协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))# 计算协方差矩阵的特征值 和特征向量
    print eigVals
    lowDMat1, reconMat1  = pca(datMat, 1)# 降维成1维矩阵  前一个主成份
    lowDMat2, reconMat2  = pca(datMat, 2)# 降维成2维矩阵  前两个主成份
    lowDMat3, reconMat3  = pca(datMat, 3)# 降维成3维矩阵  前三个主成份
    lowDMat6, reconMat6  = pca(datMat, 6)# 降维成6维矩阵  前六个主成份
    print "原数据维度: "
    print shape(datMat)
    print "降维后数据维度1: "
    print shape(lowDMat1)
    print "降维后数据维度2: "
    print shape(lowDMat2)
    print "降维后数据维度3: "
    print shape(lowDMat3)
    print "降维后数据维度6: "
    print shape(lowDMat6)



