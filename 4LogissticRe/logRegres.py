#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
逻辑回归
'''
# 测试 import logRegres as lr   lr.multiTest()

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []# 训练样本集  和  对应标签（0/1）
    fr = open('testSet.txt')
    for line in fr.readlines():# 每一行
        lineArr = line.strip().split() # 分解成向量列表 
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 前两个为 输入 x1 x2  , x0 为外加的常数项 不变量
        labelMat.append(int(lineArr[2]))                            # 后一个为 标签
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX)) #  sigmoid 函数输出 把输出变成  0 ~ 1

# 梯度上升算法 优化 权重   w = w + a * det f(x,y)
def gradAscent(dataMatIn, classLabels):     # 数据 和 标签
    dataMatrix = mat(dataMatIn)             # 转换成 矩阵
    labelMat = mat(classLabels).transpose() # 转换成 矩阵 转置 成 列矩阵
    m,n = shape(dataMatrix) # 列数 为 n = 3（每个样本维度）   行数为m 为样本个数 
    alpha = 0.001   # 学习率（权重更新步长）
    maxCycles = 500 # 最大循环次数
    weights = ones((n,1))# 每个样本 的每个 数据都乘以一个权重
    for k in range(maxCycles):              # 循环
        h = sigmoid(dataMatrix*weights)     # 矩阵乘 在z=w0*x0+...+wn*xn  1/(1+exp(-z)) 函数 激活输出 到0~1之间
        error = (labelMat - h)              # 误差 矩阵 减法
        weights = weights + alpha * dataMatrix.transpose()* error #按差值 的方向 调整 权重
        # w = w + alpha * (c-f(x))*x   迭代公式 需要数据情况调整 
    return weights

# 画 出 回归线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []# 1类数据
    xcord2 = []; ycord2 = []# 0类数据
    for i in range(n):
        if int(labelMat[i])== 1:# 1类数据
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])# 0类数据

    fig = plt.figure()#窗口
    ax = fig.add_subplot(111)#图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2] # 拟合曲线
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


# 随机梯度上升法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)# m个数据   每个数据的维度为n
    alpha = 0.01
    weights = ones(n)   # 权重全部初始化为1
    for i in range(m):  # 每个样本 
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i] # w = w + alpha * (c-f(x))*x   迭代公式 需要数据情况调整 
    return weights

# 改进的 随机梯度上升法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)# m个数据   每个数据的维度为n
    weights = ones(n)   # 权重全部初始化为1
    for j in range(numIter):# 最大循环次数
        dataIndex = range(m)# 随机样本序列
        for i in range(m):# 每个样本 
            alpha = 4/(1.0+j+i)+0.0001    # alphe 每次迭代都变化 动态步长，在计算中步长逐渐缩短
            randIndex = int(random.uniform(0,len(dataIndex)))# 随机的一个样本 用于更新 权重
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]# w = w + alpha * (c-f(x))*x   迭代公式 需要数据情况调整 
            del(dataIndex[randIndex])     # 删除所选的随机样本
    return weights

# 根据输出  得 到 分类标签
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))#使用权重 预测
    if prob > 0.5: return 1.0
    else: return 0.0

# 病马存活率数据 处理 训练测试
def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')# 训练和测试数据
    # 训练
    trainingSet = []; trainingLabels = []# 训练集 和 对于的标签

    for line in frTrain.readlines():# 训练数据的每一行
        currLine = line.strip().split('\t')# 按制表符分割 每一行
        lineArr =[]
        for i in range(21):# 每个数据 21维度
            lineArr.append(float(currLine[i]))# 转换成 浮点型数据 添加数据的一个维度数据
        trainingSet.append(lineArr)           # 添加样本到训练数据集
        trainingLabels.append(float(currLine[21]))# 每个样本的最后一列为标签，添加标签
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)# 梯度上升法 更新权重
    
    # 测试
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():#测试数据的每一行
        numTestVec += 1.0#总测试计数
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):#对测试数据进行预测 并和 实际标签比较
            errorCount += 1# 错误次数+1
    errorRate = (float(errorCount)/numTestVec)#计算错误率
    print "本次测试错误率: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print " %d 次测试后，平均错误率为 : %f" % (numTests, errorSum/float(numTests))
        
