#-*- coding:utf-8 -*-
#!/usr/bin/python

'''
k Means  K均值聚类
'''
# 测试
# Ｋ均值聚类      import kMeans as KM  KM.kMeansTest()
# 二分Ｋ均值聚类  import kMeans as KM  KM.biKMeansTest()
# 地理位置 二分Ｋ均值聚类 import kMeans as KM  KM.clusterClubs()
from numpy import *

# 导入数据集
def loadDataSet(fileName):      # 
    dataMat = []                # 
    fr = open(fileName)
    for line in fr.readlines(): # 每一行
        curLine = line.strip().split('\t')# 按 Tab键 分割成 列表
        fltLine = map(float,curLine)      # 映射成 浮点型
        dataMat.append(fltLine)           # 放入数据集里
    return dataMat

# 计算欧几里的距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

# 初始构建质心(随机)  数据集  质心个数
def randCent(dataSet, k):
    n = shape(dataSet)[1] # 样本特征维度
    centroids = mat(zeros((k,n))) # 初始化 k个 质心
    for j in range(n):    # 每种样本特征
        minJ = min(dataSet[:,j]) #  每种样本特征最小值 需要转换成 numpy 的mat
        rangeJ = float(max(dataSet[:,j]) - minJ)#每种样本特征的幅值范围
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
        # 在每种样本的最大值和最小值间随机生成K个样本特征值
    return centroids

# 简单k均值聚类算法  
#       数据集  中心数量   距离算法            初始聚类中心算法  
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]             # 样本个数
    clusterAssment = mat(zeros((m,2)))# 样本标记 分配结果 第一列索引 第二列误差
    centroids = createCent(dataSet, k)# 初始聚类中心
    clusterChanged = True# 设置质心是否仍然发送变化
    while clusterChanged:
        clusterChanged = False
        for i in range(m): #对每个样本 计算最近的中心
        # 更新 样本所属关系
            minDist = inf; minIndex = -1 # 距离变量 以及 最近的中心索引
            for j in range(k): # 对每个中心
                distJI = distMeas(centroids[j,:],dataSet[i,:])# 计算距离
                if distJI < minDist:
                    minDist = distJI; minIndex = j# 得到最近的 中心 索引
            if clusterAssment[i,0] != minIndex: clusterChanged = True 
            # 所属索引发生了变化 即质心还在变化，还可以优化
            clusterAssment[i,:] = minIndex,minDist**2 # 保存 所属索引 以及距离平方 用以计算误差平方和 SSE
        # 更新质心
        print centroids # 每次迭代打印质心
        for cent in range(k):# 
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]# 数组过滤 得到各个中心所属的样本
            centroids[cent,:] = mean(ptsInClust, axis=0) # 按列求平均 得到新的中心
    return centroids, clusterAssment# 返回质心 和各个样本分配结果

def kMeansTest(k=5):
    MyDatMat = mat(loadDataSet("testSet.txt"))
    MyCenters, ClustAssing = kMeans(MyDatMat, k)

# bisecting K-means 二分K均值算法 克服局部最优值
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]             # 样本个数
    clusterAssment = mat(zeros((m,2)))# 样本标记 分配结果 第一列索引 第二列误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]# 创建一个初始质心
    centList =[centroid0] # 一个中心的 列表
    for j in range(m):    # 计算初始误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2#每个样本与中心的距离平方
    while (len(centList) < k):# 中心数俩个未达到指定中心数量 继续迭代
        lowestSSE = inf       # 最小的 误差平方和 SSE
        for i in range(len(centList)):# 对于每一个中心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] # 处于当前中心的样本点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 对此中心内的点进行二分类
            # 该样本中心 二分类之后的 误差平方和 SSE
	    sseSplit = sum(splitClustAss[:,1])
            # 其他未划分数据集的误差平方和 SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            # 划分后的误差和没有进行划分的数据集的误差为本次误差
            if (sseSplit + sseNotSplit) < lowestSSE: # 小于上次 的 误差 
                bestCentToSplit = i # 记录应该被划分的中心 的索引
                bestNewCents = centroidMat # 最好的新划分出来的中心
                bestClustAss = splitClustAss.copy()# 新中心 对于的 划分记录 索引(0或1)以及 误差平方
                lowestSSE = sseSplit + sseNotSplit # 更新总的 误差平方和
        # 记录中心 划分 数据
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)  # 现有中心数量
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit# 最应该被划分的中心
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        # 将最应该被划分的中心 替换为 划分后的 两个 中心(一个替换，另一个 append在最后添加)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]# 替换
        centList.append(bestNewCents[1,:].tolist()[0])           # 添加
        # 更新 样本标记 分配结果 替换 被划分中心的记录
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment

def biKMeansTest(k=5):
    MyDatMat = mat(loadDataSet("testSet.txt"))
    MyCenters, ClustAssing = biKmeans(MyDatMat, k)

####位置数据聚类测试#####
# 利用雅虎的服务器将地址转换为 经度和纬度
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # 
    params = {}
    params['flags'] = 'J'        # 设置返回类型为JSON字符串  
    params['appid'] = 'aaa0VN6k' # 注册 帐号后获得 http://developer.yahoo.com
    params['location'] = '%s %s' % (stAddress, city) # 位置信息
    url_params = urllib.urlencode(params)# 将字典转换成可以通过ＵＲＬ进行传递的字符串格式
    yahooApi = apiStem + url_params      # 加入网络地址 
    print yahooApi                       # 打印 ＵＲＬ
    c=urllib.urlopen(yahooApi)           # 打开 ＵＲＬ
    return json.loads(c.read())          # 读取返回的jason字符串   对位置进行了编码 得到经度和纬度 


from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w') # 打开位置信息文件
    for line in open(fileName).readlines():# 每一行
        line = line.strip()
        lineArr = line.split('\t')# 得到列表
        retDict = geoGrab(lineArr[1], lineArr[2])# 第二列为号牌 第三列为城市 进行地址解码
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude']) #经度
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])#纬度
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng)) #再写入到文件
        else: print "error fetching"
        sleep(1)#延迟1s
    fw.close()

# 返回地球表面两点之间的距离  单位英里 输入经纬度(度)  球面余弦定理
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi  in numpy


# 位置聚类测试 画图可视化显示
import matplotlib
import matplotlib.pyplot as plt

def clusterClubs(numClust=5):
    datList = [] # 样本
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])# 保存经纬度
    datMat = mat(datList)# 数据集 numpy的mat类型
    # 进行二分K均值算法聚类
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()# 窗口
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)#轴
    imgP = plt.imread('Portland.png') # 标注在实际的图片上
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):#每一个中心
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]# 属于每个中心的样本点
        markerStyle = scatterMarkers[i % len(scatterMarkers)]# 点的类型 画图
        # 散点图 每个中心的样本点
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    # 散 点图 每个中心
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()# 显示
