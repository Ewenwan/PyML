#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
集成学习  AdaBoost元算法 更新数据集 增大判断错误的样本权重
自举汇聚法（bootstrap aggregating），也称为bagging方法，
是在从原始数据集选择S次后得到S个新数据集的一种技术。
bagging中的数据集相当于是有放回取样，比如要得到一个大小为n的新数据集，
该数据集中的每个样本都是在原始数据集中随机抽样（有放回），
也就是说原始数据集中的样本有部分可能就不会出现在新的数据集中，而有些样本可能会出现不止一次。
bagging算法有很多种：例如随机森林等就自行研究.

boosting是一种与bagging很类似的技术，这两种技术使用的分类器的类型都是一致的，
前者是通过不同的分类器串行训练得到的，每个分类器都根据已训练出的分类器的性能来进行训练。
boosting是通过集中关注被已有分类器错分的哪些数据来获得新的分类器。
由于boosting分类结果是基于所有分类器的加权求和结果的，
因此和bagging不太一样，bagging中的分类器的权重是相等的。
boosting中的分类器权重并不相等，每个权重代表的是其对应分类器在上一轮迭代中的成功度。


AdaBoost的运行过程如下：训练数据中的每个样本，并赋予其一个权重，这些权重构成了向量D。
一开始这些权重都初始化相等值。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，
然后在统一数据上再次训练弱分类器。在分类器的第二次训练当中，将会重新调整每个样本的权重，
其中每一次分对的样本的权重将会降低，而第一次分错的样本的权重就会提高。
为了从所有弱分类器中得到最终的分类结果，Adaboost为每个分类器都分配了一个权重值alpha，
这些alpha值是基于每个弱分类器的错误率errrat进行计算的。


'''

# 测试程序 
# 假数据测试
# import adaboost as ab 
# datMat,classLabels=ab.loadSimpData()   
# classify ,predictLabel= ab.adaBoostTrainDS(datMat,classLabels)
# ab.adaClassify([0,0], classify)
# 病马数据测试
# import adaboost as ab  ab.adaClassifyTest()
from numpy import *

# 假的数据
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],#数据集
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]# 标签
    return datMat,classLabels

# 载入文件数据 得到数据集合和标签
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) # 数据样本 维度
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():# 每一行 为一个 样本数据
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)#数据集合
        labelMat.append(float(curLine[-1]))#标签
    return dataMat,labelMat

#  简单阈值 分类器  对于样本数据集合  的 某一维度
#  弱分类器 按某一 样本维度的大小分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))# 初始化 分类结果矩阵 为1 
    if threshIneq == 'lt':# 比较方法切换标志
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #小于阈值的为 -1 其余为1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0  #大于阈值的为 -1 其余为1
    return retArray

    
# dataArr 原始数据集  classLabels 类标签  D初始 数据 权重
# 遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)# 样本个数m   数据维度n
    numSteps = 10.0        # 迭代 步数
    bestStump = {}         # 最优分类器   字典形式保存 
    bestClasEst = mat(zeros((m,1)))
    minError = inf         # 初始错误率 设置为 无限大

    for i in range(n):#更新所有的数据 每一维度的所有样本数据
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        # 同一维度 的所有样本数据 的 最大最小值

        stepSize = (rangeMax-rangeMin)/numSteps# 步长
        for j in range(-1,int(numSteps)+1): # 在当前的 样本某一维度 以不同的阈值条件 产生不同的分类器
            for inequal in ['lt', 'gt']:    # 分类器的两种条件选项
                threshVal = (rangeMin + float(j) * stepSize)# 变阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#调用分类器得到 分类结果
                errArr = mat(ones((m,1)))  # 先初始化为1
                errArr[predictedVals == labelMat] = 0# 分对的 为0,
                weightedError = D.T*errArr  #带有原来 样本权重的 错误率 之和
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy() # 分类标签
                    bestStump['dim'] = i               # 最优 分类 样本 维度
                    bestStump['thresh'] = threshVal    # 对应 样本维度上  的 阈值
                    bestStump['ineq'] = inequal        # 分类器条件   less then 或者 great  then 
    return bestStump,minError,bestClasEst  # 返回最优分类器 维度，阈值，条件  最小误差   最优分类标签


# 训练模型
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]    # 数据 样本个数m
    D = mat(ones((m,1))/m)   # 初始 数据 权重 init D to all equal
    aggClassEst = mat(zeros((m,1)))#
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)# 训练本次循环得到的 最优分类器
        #print "D:",D.T
  # 更新分类器权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))# 计算 alpha=1/2*((1-error)/error) # max(error,1e-16)分母不会太小
        bestStump['alpha'] = alpha       # 最优分类器加入 分类器权重
        weakClassArr.append(bestStump)   # 以数组形式 保存每次循环得到的最优分类器(字典形式)
  # 计算 样本权重 print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) 
        # 正确的标签 classLabels 分类预测的标签classEst 因为之为1或-1 ,如果分类正确 乘积为1,分类错误乘积为-1 后再 乘以 -1
        # 则 分类正确为 -alpha 分类错误为 alpha
        D = multiply(D,exp(expon))                             # 
        D = D/D.sum()  # 更新样本权重
  # 得到多个分类器的 分类结果
        aggClassEst += alpha*classEst # 分类预测的标签classEst 乘以分类器权重 得到 多个弱分类器的 投票 分类结果
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))# 得到分类错误之和
        errorRate = aggErrors.sum()/m # 计算错误率
        print "total error: ",errorRate
        if errorRate == 0.0: break    # 错误率为0 就结束 
    return weakClassArr,aggClassEst   # 多个弱分类器   分类结果

#分类测试 
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m , n= shape(dataMatrix)
    aggClassEst = mat(zeros((m,1)))# 分类结果
    for i in range(len(classifierArr)):
       # for j in range(len(classifierArr[i])):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
		                            classifierArr[i]['thresh'],\
		                            classifierArr[i]['ineq'])#call stump classify
	aggClassEst += classifierArr[i]['alpha']*classEst#多个弱分类器  加权和

        print sign(aggClassEst)
    return sign(aggClassEst)

#画出ROC函数  predStrengths 预测的标签   classLabels 实际的标签
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) # 浮点数二元组 点位置坐标
    ySum = 0.0      # 计算 AUC
    numPosClas = sum(array(classLabels)==1.0)# 数组过滤 正类数量 len(classLabels)-numPosClas 负类数量
    yStep = 1/float(numPosClas) #真正类率 轴 步进  在0~1 区间上绘制点
    xStep = 1/float(len(classLabels)-numPosClas)# 负正类率 轴 步进
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:#实际为正类的
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0; #实际为负类的
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Postive Rate'); plt.ylabel('True Postive Rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep

# 测试程序
def adaClassifyTest():
    # 训练
    Traindata,Trainlabel = loadDataSet('horseColicTraining2.txt')# 训练数据
    Classify,predictLabel= adaBoostTrainDS(Traindata,Trainlabel ,20)# 得到分类器 维度，阈值，条件  最小误差 
    # 测试
    Testdata,Testlabel = loadDataSet('horseColicTest2.txt')#测试数据
    m = shape(Testdata)[0]    # 数据 样本个数m
    preict = adaClassify(Testdata,Classify)
    Errors = mat(ones((m,1)))   
    errorRate = Errors[sign(preict) != mat(Testlabel).T].sum()/m
    print "错误率: ",errorRate
    plotROC(preict,Testlabel)


