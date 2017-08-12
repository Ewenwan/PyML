#-*- coding:utf-8 -*-
#!/usr/bin/python

# 测试代码 约会数据分类 import KNN   KNN.datingClassTest1() 标签为字符串    KNN.datingClassTest2() 标签为整形 
# 测试代码 手写字体分类 import KNN   KNN.handwritingClassTest()  # 子编写 KNN 函数 错误率 0.012685
# 		       import KNN   KNN.sklKNNhandwritingClassTest() 
# 子编写 KNN 函数  错误率 0.012685 速度更快 



import sys
reload(sys)  
sys.setdefaultencoding('utf-8')  # 指定默认中文编码

from numpy import  *   # 科学计算包
import operator        # 运算符模块
from os import listdir # 获得指定目录中的内容（手写字体文件夹下样本txt）  类型命令行 ls
from sklearn.neighbors import KNeighborsClassifier as sklKNN  # KNN 库函数 


import matplotlib                  # 画图可视化操作
import matplotlib.pyplot as plot

# 显示一个 二维图
def myPlot(x, y, labels):
    fig = plot.figure()#创建一个窗口
    ax = fig.add_subplot(111)# 画一个图
    #ax.scatter(x,y)
    ax.scatter(x,y,15.0*array(labels),15.0*array(labels)) # 支持 分类颜色显示
    ax.axis([-2,25,-0.2,2.0])
    plot.xlabel('Percentage of Time Spent Playing Video Games')# 坐标轴名称
    plot.ylabel('Liters of Ice Cream Consumed Per Week')
    plot.show()
    

# 创建假 的数据测试
def createDataSet():
    groop  = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]]) # numpy的array 数组格式
    labels = ['A','A','B','B']# 标签 list
    return groop, labels

# 定义 KNN 分类函数
def knnClassify0(inX, dataSet, labels, k):
    # inX 待分类的点  数据集和标签 DataSet, label  最近领域个数 k
    dataSetSize = dataSet.shape[0]  # 数据集大小（行数）    
    # tile（A,（行维度，列维度）） A沿各个维度重复的次数
    # 点A 重复每一行 到 数据集大小行
    differeMat  = tile(inX, (dataSetSize,1)) - dataSet  # 求 待分类点 与个个数据集点的 差值
    sqDiffMat = differeMat**2                           # 求 平方
    sqDistances = sqDiffMat.sum(axis=1)                 # 求 和（各行求和）
    distances = sqDistances**0.5                        # 开方  得到 点A 与 数据集个点 的欧式距离
    sortedDistIndicies = distances.argsort()            # 返回  递增排序后 的 原位置序列（不是值）    
    # 取得最近的 k个点 统计 标签类出现的频率
    classCount={}  # 字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#从小到大 对应距离 数据点 的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # 对于类标签 字典单词 的 值 + 1        
    # 对 类标签 频率（字典的 第二列（operator.itemgetter(1)）） 排序 从大到小排序 reverse=True
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # 返回 最近的 对应的标签


# 真实数据的处理   输入TXT文本文件  返回 数据集和标签(已转化成数字) 列表 list
def file2matrix(filename):
    fr = open(filename)                  # 打开文件             
    numberOfLines = len(fr.readlines())  # 得到文件所有的行数
    returnMat = zeros((numberOfLines,3))
 # 创建一个用于存储返回数据的矩阵  数据集  每个数据的大小根据实际情况！！ 即是 3 列数应根据 数据维度确定
    classLabelVector = []                # 对应标签
    fr = open(filename)
    index = 0
    for line in fr.readlines():          # 每一行
        line = line.strip()              # 默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        listFromLine = line.split('\t')  # 按 制表符(\t) 分割字符串 成 元素列表
        returnMat[index,:] = listFromLine[0:3]         # 前三个为 数据集数据
        classLabelVector.append(int(listFromLine[-1])) # 最后一个 为 标签  整形
        index += 1
    return returnMat,classLabelVector


# 真实数据的处理   输入TXT文本文件  返回 数据集和标签(为字符串) 列表 list
def file2matrix2(filename):
    fr = open(filename)                  # 打开文件             
    numberOfLines = len(fr.readlines())  # 得到文件所有的行数
    returnMat = zeros((numberOfLines,3))
 # 创建一个用于存储返回数据的矩阵  数据集  每个数据的大小根据实际情况！！ 即是 3 列数应根据 数据维度确定
    classLabelVector = []                # 对应标签
    fr = open(filename)
    index = 0
    for line in fr.readlines():          # 每一行
        line = line.strip()              # 默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        listFromLine = line.split('\t')  # 按 制表符(\t) 分割字符串 成 元素列表
        returnMat[index,:] = listFromLine[0:3]         # 前三个为 数据集数据
        classLabelVector.append(str(listFromLine[-1])) # 最后一个 为 标签  字符串型
        '''
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        '''
        index += 1
    return returnMat,classLabelVector



def showdatas(datingDataMat, datingLabels):
    from matplotlib.font_manager import FontProperties
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plot
    #设置汉字格式
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) #window 下
    # font = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/ukai.ttc") # linux 下 字体
    #font = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/font_ch_wq/font_ch_wq.ttf") # linux 下 字体
    '''
    linux 需要配置修改 matplotlib的资源配置文件   import matplotlib   matplotlib.matplotlib_fname() 可查找位置  /etc/matplotlibrc
 
# 查看 系统中文字体
~$ python
from matplotlib.font_manager import FontManager
import subprocess
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print mat_fonts
output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
print '*' * 10, '系统可用的中文字体', '*' * 10
print output
zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
available = mat_fonts & zh_fonts
print '*' * 10, '可用的字体', '*' * 10
for f in available:
    print f

>>> print output
AR PL UMing CN
AR PL UKai TW MBE
AR PL UKai HK
Droid Sans Fallback
AR PL UKai CN
AR PL UKai TW
文泉驿等宽正黑,文泉驛等寬正黑,WenQuanYi Zen Hei Mono
AR PL UMing HK
AR PL UMing TW
AR PL UMing TW MBE

查看安装位置
fc-match -v "AR PL UKai CN"           >>>  file: "/usr/share/fonts/truetype/arphic/ukai.ttc"(w)
fc-match -v "WenQuanYi Zen Hei Mono"  >>>  file: "/usr/share/fonts/font_ch_wq/font_ch_wq.ttf"(w)

    先安装 中文字体 http://font.chinaz.com/130130474870.htm
    sudo mv  XXX.ttf   /usr/share/fonts
    cd /usr/share/fonts
    sudo mkdir XXX
    sudo mv XXX.ttf XXX/
    cd /usr/share/fonts/XXX
    #生成字体索引信息. 会显示字体的font-family
    sudo mkfontscale
    sudo mkfontdir
    #更新字体缓存：
    fc-cache

    vim /etc/matplotlibrc
    # font.family  取消注释
    并且在font.serif 取消注释 后 支持字体加上一个中文字体 WenQuanYi Zen Hei Mono
    font.sans-serif  取消注释 后 支持字体加上一个中文字体 WenQuanYi Zen Hei Mono
    修改 Ture 为False 并去掉注释 axes.unicode_minus : False
    为matplotlib增加中文字体
    重要 复制一份 ttf字体 到 /usr/share/matplotlib/mpl-data/fonts/ttf  下
    sudo cp /usr/share/fonts/XXX/XXX.ttf  /usr/share/matplotlib/mpl-data/fonts/ttf

    !!!!! 进入 $HOME/.cache/matplotlib  隐藏文件 Ctrl+H 可显示
    找到文件夹下的fontList.cache文件     删除
    遇到中文字体就会自动调用 上面添加的中文字体显示 
    '''
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plot.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black') #黑色
        if i == 2:
            LabelsColors.append('orange')#橙色
        if i == 3:
            LabelsColors.append('red')   #红色 
# 第一行第一列的图    
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为s=15,透明度为alpha=0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text  = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')#标题
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')# 子图 x轴坐标
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占')  # 子图 y轴坐标
    plot.setp(axs0_title_text, size=9, weight='bold', color='red')    #标题 大小 颜色
    plot.setp(axs0_xlabel_text, size=7, weight='bold', color='black') #坐标轴 
    plot.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
# 第一行第二列的图    
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')
    plot.setp(axs1_title_text, size=9, weight='bold', color='red') 
    plot.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
    plot.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
# 第二行第一列的图 
    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
    plot.setp(axs2_title_text, size=9, weight='bold', color='red') 
    plot.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
    plot.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plot.show()



# 数据集 各个类型数据归一化  平等化 影响权值
def dataAutoNorm(dataSet):
    minVals = dataSet.min(0) # 最小值 每一列的 每一种属性 的最小值 
    maxVals = dataSet.max(0) # 最大值
    ranges = maxVals - minVals # 数据范围
    normDataSet = zeros(shape(dataSet)) # 初始化输出 数组
    m = dataSet.shape[0]                # 行维度  样本总数
    normDataSet = dataSet - tile(minVals, (m,1))    # 扩展 minVals 成 样本总数行m行 1列（属性值个数） 
    normDataSet = normDataSet/tile(ranges, (m,1))   # 矩阵除法 每种属性值 归一化  numpy库 为（linalg.solve(matA,matB)）
    return normDataSet, ranges, minVals             # 返回 归一化后的数组 和 个属性范围以及最小值

# 约会数据 KNN分类 测试
# 标签为 字符串型
def datingClassTest1(test_ret=0.1):
    hoRatio = test_ret              # 测试的样本比例 剩下的作为 训练集
    datingDataMat,datingLabels = file2matrix2('datingTestSet.txt')                #载入数据集
    normMat, ranges, minVals = dataAutoNorm(datingDataMat)
    m = normMat.shape[0]            # 总样本数量
    numTestVecs = int(m*hoRatio)    # 总测试样本数
    errorCount = 0.0                # 错误次数记录
    for i in range(numTestVecs):    # 对每个测试样本
        # KNN 分类                        测试样本       剩下的作为数据集               数据集对应的标签  最近 的三个
        classifierResult = knnClassify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "分类结果: %s,\t真实标签: %s" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0    
    print "总错误次数: %d" %  errorCount
    print "测试总数:   %d" %  numTestVecs
    print "总错误率:   %f" % (errorCount/float(numTestVecs))

# 标签为 整形 int 
def datingClassTest2(test_ret=0.1):
    hoRatio = test_ret              # 测试的样本比例 剩下的作为 训练集
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')                #载入数据集
    normMat, ranges, minVals = dataAutoNorm(datingDataMat)
    m = normMat.shape[0]            # 总样本数量
    numTestVecs = int(m*hoRatio)    # 总测试样本数
    errorCount = 0.0                # 错误次数记录
    for i in range(numTestVecs):    # 对每个测试样本
        # KNN 分类                        测试样本       剩下的作为数据集               数据集对应的标签  最近 的三个
        classifierResult = knnClassify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "分类结果: %d, 真实标签: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0    
    print "总错误次数: %d" %  errorCount
    print "测试总数:   %d" %  numTestVecs
    print "总错误率:   %f" % (errorCount/float(numTestVecs))
    showdatas(datingDataMat, datingLabels)

# 根据用户输入的 样本的属性值 判断用户所倾向的类型(有点问题？？)
def classifyPerson():
    resultList = ['讨厌','一般化','非常喜欢']
    percent = float(raw_input("打游戏所花时间比例： "))
    mile    = float(raw_input("每年飞行的里程数量： "))
    ice     = float(raw_input("每周消费的冰淇淋量： "))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')                #载入数据集
    normMat, ranges, minVals   = dataAutoNorm(datingDataMat)
    # 新测试样本 归一化
    print  ranges, minVals 
    testSampArry     = array([mile, percent, ice])   # 用户输入的 测试样例
    testSampArryNorm = (testSampArry-minVals)/ranges # 样例归一化
    print  testSampArry ,testSampArryNorm
    # 分类
    classifierResult = knnClassify0(testSampArryNorm,normMat,datingLabels,3)
    print  classifierResult
    print "他是不是你的菜： ", resultList[classifierResult-1]
    


# 手写字体 图像 32*32 像素转化成  1*1024 的向量  
def img2vector(filename):
    returnVect = zeros((1,1024)) # 创建空的 返回向量
    fr = open(filename)          # 打开文件
    for i in range(32):          # 对每一行
        lineStr = fr.readline()  # 每一行元素
        for j in range(32):      # 每一行的每个值
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


# 手写字体的 KNN识别 每个数字图片被转换成 32*32 的 0 1 矩阵
def handwritingClassTest(k=3):
    # 得到训练数据集
    hwLabels = []                                # 识别的标签
    trainingFileList = listdir('trainingDigits') # 加载手写字体训练数据集 (所有txt文件列表)
    m = len(trainingFileList)                    # 总训练样本数
    trainingMat = zeros((m,1024))                # 训练数据集
    for i in range(m):
        fileNameStr = trainingFileList[i]        # 每个训练数据样本文件  0_0.txt  0_1.txt  0_2.txt 
        fileStr = fileNameStr.split('.')[0]      # 以.分割 第一个[0]为文件名   第二个[1]为类型名 txt文件
        classNumStr = int(fileStr.split('_')[0]) # 以_分割，第一个[0]为该数据表示的数字 标签
        hwLabels.append(classNumStr)                                      # 训练样本标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  # 训练样本数据
        
    # 得到测试数据集   
    testFileList = listdir('testDigits')         # 测试数据集
    errorCount = 0.0                             # 错误次数计数
    mTest = len(testFileList)                    # 总测试 数据样本个数
    for i in range(mTest):
        fileNameStr = testFileList[i]            # 每个测试样本文件
        fileStr = fileNameStr.split('.')[0]      # 得到文件名
        classNumStr = int(fileStr.split('_')[0]) # 得到对应的真实标签
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)                # 测试样本数据
        classifierResult = knnClassify0(vectorUnderTest, trainingMat, hwLabels, k) # 分类
        print "KNN分类标签: %d, 真实标签: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\n总的错误次数: %d" % errorCount
    print "\n总的错误比例: %f" % (errorCount/float(mTest))

def sklKNNhandwritingClassTest(k=3):
   # 得到训练数据集
    hwLabels = []                                # 识别的标签
    trainingFileList = listdir('trainingDigits') # 加载手写字体训练数据集 (所有txt文件列表)
    m = len(trainingFileList)                    # 总训练样本数
    trainingMat = zeros((m,1024))                # 训练数据集
    for i in range(m):
        fileNameStr = trainingFileList[i]        # 每个训练数据样本文件  0_0.txt  0_1.txt  0_2.txt 
        fileStr = fileNameStr.split('.')[0]      # 以.分割 第一个[0]为文件名   第二个[1]为类型名 txt文件
        classNumStr = int(fileStr.split('_')[0]) # 以_分割，第一个[0]为该数据表示的数字 标签
        hwLabels.append(classNumStr)                                      # 训练样本标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  # 训练样本数据
    #库函数构建kNN分类器
    neigh = sklKNN(k , algorithm = 'auto')
    #拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)     
    # 得到测试数据集   
    testFileList = listdir('testDigits')         # 测试数据集
    errorCount = 0.0                             # 错误次数计数
    mTest = len(testFileList)                    # 总测试 数据样本个数
    for i in range(mTest):
        fileNameStr = testFileList[i]            # 每个测试样本文件
        fileStr = fileNameStr.split('.')[0]      # 得到文件名
        classNumStr = int(fileStr.split('_')[0]) # 得到对应的真实标签
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)                # 测试样本数据
        classifierResult = neigh.predict(vectorUnderTest)# 分类
        print "KNN分类标签: %d, \t真实标签: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\n总的错误次数: %d" % errorCount
    print "\n总的错误比例: %f" % (errorCount/float(mTest))



