#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
线性回归

@author: Peter
'''

# 测试代码 import regression as lr   lr.lrTest()  lr.lrTest(0.05)
#  import regression as lr   lr.ridgeTestPlot()
#  import regression as lr   lr.stageWiseTestPlot()
from numpy import *


# txt 文件数据提取 以TAB键值分割
def loadDataSet(fileName): 
    numFeat = len(open(fileName).readline().split('\t')) - 1 #样本数据维度 最后一个为标签
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():#每一行 一个样本
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))# 一个样本数据
        dataMat.append(lineArr)# 放入数据集合
        labelMat.append(float(curLine[-1]))#标签
    return dataMat,labelMat# 数据集 和 标签 

# 标准线性回归 
def standRegres(xArr,yArr):
    xMat = mat(xArr)   #自变量  X 
    yMat = mat(yArr).T #标签
    xTx = xMat.T*xMat  #X转置*X
    if linalg.det(xTx) == 0.0:#如果行列式的值为零 则不能计算 逆矩阵
        print "行列式的值为零 不能计算 逆矩阵"
        return
    ws = xTx.I * (xMat.T*yMat)# 计算 回归系数
    return ws

# locally weighted linear regression 
# 局部(某些点附近)加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]# 数据样本个数
    weights = mat(eye((m)))# 数据权重  对角矩阵
    for j in range(m):                      # 
        diffMat = testPoint - xMat[j,:]     #各样本与testPoint的差值 diffMat*diffMat.T距离
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))#高斯核权重
    xTx = xMat.T * (weights * xMat)# 加权后的 #X转置*weights*X
    if linalg.det(xTx) == 0.0:
        print "行列式的值为零 不能计算 逆矩阵"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws #返回 应变量的值  


def lwlrTest(testArr,xArr,yArr,k=1.0):  
    m = shape(testArr)[0]# 样本个数
    yHat = zeros(m)      # 估计值
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)# 估计值
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

# 计算 误差和
def rssError(yArr,yHatArr):  
    return ((yArr-yHatArr)**2).sum()

def lrTest(k=1.0): 
    xArr,yArr = loadDataSet('abalone.txt') # abalone.txt  ex1.txt ex0.txt
    m = shape(xArr)[0]# 样本个数
    yHat = zeros(m)   # 估计值
    for i in range(m):
        yHat[i] = lwlr(xArr[i],xArr,yArr,k)# 估计值
        print "真值: %f, \t估计值: %f" % (yArr[i],yHat[i])
    print '误差和：'+ str(rssError(yArr,yHat))
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)# x1值按升序 排列 方便画图 返回下标序列
    xSort  = xMat[srtInd][:,0,:] # 升序排列后
    import matplotlib.pyplot as plot
    fig = plot.figure()#窗口
    ax  = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd]) # 预测曲线
    #散点图
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plot.show()#显示


# 岭 回归 当样本特征维度大于 样本数量时  X转置*X 不能计算逆矩阵
# 需要加上一个 对角矩阵*系数
# 同样 也可以 当成 偏差
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "行列式的值为零 不能计算 逆矩阵"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

#    先对数据做标准化处理 再用 岭 回归 
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #标签 减去 均值
    #regularize X's
    xMeans = mean(xMat,0)   #均值
    xVar = var(xMat,0)      #方差
    xMat = (xMat - xMeans)/xVar#标准化 减去均值 再除以方差
    numTestPts = 30 #30组不同的 岭 回归测试 系数 可以得到 30组 回归系数 
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))# 在30个不同的 lambda 系数下进行 回归系数求解
        wMat[i,:]=ws.T
    return wMat

# 岭 回归测试 
def ridgeTestPlot():
    xArr,yArr = loadDataSet('abalone.txt') # abalone.txt  ex1.txt ex0.txt
    #m = shape(xArr)[0]# 样本个数
    ridgeWeights = ridgeTest(xArr,yArr)
    import matplotlib.pyplot as plot
    fig = plot.figure()#窗口
    ax  = fig.add_subplot(111)
    ax.plot(ridgeWeights) # 每个特征 回归系数 变化
    plot.show()#显示

# 标准化数据
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


# 前向逐步线性回归  有 遗传算法 淘汰的思想
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     # 标签 减去 均值 消除影响
    xMat = regularize(xMat) # 标准化 样本数据
    m,n=shape(xMat) # m为样本个数  n为样本特征维度
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):# 进化循环100次
        print ws.T
        lowestError = inf; # 误差初始化
        for j in range(n): # 对每个特征
            for sign in [-1,1]: # 增 
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError: # 误差小 更优秀
                    lowestError = rssE
                    wsMax = wsTest # 保留较好的 系数
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#  前向逐步线性回归 测试 
def stageWiseTestPlot():
    xArr,yArr = loadDataSet('abalone.txt') # abalone.txt  ex1.txt ex0.txt
    #m = shape(xArr)[0]# 样本个数
    stageWiseWeights = stageWise(xArr,yArr)
    import matplotlib.pyplot as plot
    fig = plot.figure()#窗口
    ax  = fig.add_subplot(111)
    ax.plot(stageWiseWeights)# 每个特征 回归系数 变化
    plot.show()#显示

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()


# 网络 获取  Google Shopping 的数据
# 从网络返回的 jason字符串数据中抽取价格数据
# 可视化并观察分析数据
# 构建不同的模型，采用逐步线性回归 和 直接线性回归模型
# 使用交叉验证来验证不同的模型，分析哪一个最好
# 生成 估计产品价格的模型 
from time import sleep
import json
import urllib2
#                                    年份
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)# 休眠10秒钟 防止短时间内 有过多的 API调用
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'# API 的KEY  注册 Google 帐号获取
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum) #生成特定URL
    pg = urllib2.urlopen(searchURL) # 打开 URL 等待返回数据
    retDict = json.loads(pg.read()) # jason 解析返回的字符串 成 字典
    for i in range(len(retDict['items'])):# 各个 项
        try:
            currItem = retDict['items'][i]# 当前项
            if currItem['product']['condition'] == 'new':
                newFlag = 1   # 新产品
            else: newFlag = 0 # 老产品
            listOfInv = currItem['product']['inventories'] # 各个产品套件 
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:# 若价格 高于原价格 的 50% 则认为是 完整的 一套产品
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc]) # 保存 产品参数
                    retY.append(sellingPrice) # 售价
        except: print 'problem with item %d' % i
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

#交叉验证   
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)# 数据集 大小                         
    indexList = range(m)
    errorMat = zeros((numVal,30))#10次测试  每次有30组 回归系数 可以得到误差
    for i in range(numVal):#测试次数
        trainX=[]; trainY=[]   # 训练数据集和标签
        testX = []; testY = [] # 测试数据集和标签
        random.shuffle(indexList)# 打乱 样本排序
        for j in range(m):# 从原始数据集中随机 90% 生成训练数据集和标签 以及测试数据集和标签
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]]) #90% 为训练集
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])  #10% 为测试集
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)           
        #岭 回归 得到 权重 有30组 不同的参数 可以得到30组不同的 回归系数
        for k in range(30):# 对于30组 不同的 岭回归 得到的回归系数 进行测试 计算误差 选取最好的
            matTestX = mat(testX)#测试数据集合
            matTrainX=mat(trainX)#训练数据集合
            meanTrain = mean(matTrainX,0)# 训练数据 均值
            varTrain = var(matTrainX,0)  # 训练数据 方差
            matTestX = (matTestX-meanTrain)/varTrain #训练数据 标准化
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)# 标准化后 的 matTestX测试数据 乘以 回归系数后 应该在加上 测试标签的均值
            errorMat[i,k]=rssError(yEst.T.A,array(testY))#计算误差
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#总误差均值
    minMean = float(min(meanErrors))# 误差最小值
    bestWeights = wMat[nonzero(meanErrors==minMean)]# 最好的回归系数
    # 将标准化后的数据还原  用于 可视化   
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX # 还原后 的 回归系数
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat) #还原计算 预测结果
