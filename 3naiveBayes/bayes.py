#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
朴素贝叶斯
'''
# 测试 侮辱性语句 import bayes as bs     bs.testingNB()
# 测试 垃圾邮件   import bayes as bs     bs.spamTest()

# 博客数据分析 测试  import feedparser   import bayes as bs
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.ress')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.ress')
# bs.getTopWords(ny,sf)

from numpy import *


# import feedparser # rss数据分析器

dist_lab =  {0: '非侮辱', 1: '侮辱'}
dist_mail = {0: '非垃圾邮件', 1: '垃圾邮件'}
# 自建立简单数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 为侮辱性言论 , 0 非侮辱性言论
    return postingList,classVec

# 创建 词汇表 (生成 字典 单词表)            
def createVocabList(dataSet):
    vocabSet = set([])                      # 创建空集合（不重复）
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 两个集合的并集 set（返回不重复的词表）
    return list(vocabSet)

# 词集模型 词向量 ，出现很多次 也为1
# 用字典中的单词 线性表示 输入的词汇 流
# 词表转为向量 classVec 类别标签向量  词汇表 vocabList,    输入文档 inputSet 
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) # 类别标签向量 与  词表(单词字典)   等长
    for word in inputSet:          # 遍历文档中的单词
        if word in vocabList:      # 如果出现了 词汇表中的单词
            returnVec[vocabList.index(word)] = 1      # 相应单词计数为1, 标记为1 出现很多次 也为1，表示使用过词典里的该单词
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 训练 朴素贝叶斯分类器
# trainMatrix 训练文档矩阵（已用字典线性表示）   trainCategory 训练文档类别标签向量 1 为侮辱性言论 , 0 非侮辱性言论
# p(类别/词向量) = p(词向量/类别)* p(类别) / p(词向量)
# p(ci/w)       = p(w/ci)       * p(ci)   / p(w)
# 而   p(w/ci) = p(w1/ci)*p(w2/ci)*。。。*p(wn/ci) 
# 而   p(w1/ci)  为对应 文章中 所有同类别文档 w1单词出现的 概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) # 文档个数
    numWords = len(trainMatrix[0])  # 文档总字典单词数

    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 属于侮辱性文档的概率 p(ci=1)
    # p(word/ci)  各类别（侮辱和非侮辱）中出现 各个单词的总次数 初始化为1
    p0Num = ones(numWords); p1Num = ones(numWords)      # 避免出现乘以0的情况
    p0Denom = 2.0; p1Denom = 2.0                        # change to 2.0

    for i in range(numTrainDocs):   # 每个文档
        if trainCategory[i] == 1:   # 对应类别  如果是侮辱类
            p1Num += trainMatrix[i]          # 所有文档中 侮辱类类别 各个单词出现的次数 直接用词向量+
            p1Denom += sum(trainMatrix[i])   # 侮辱类文档单词总数 用以计算 侮辱类文档中每个单词出现的概率
        else:
            p0Num += trainMatrix[i]          # p(x/c0) 所有文档中 非侮辱类 各个单词出现的次数
            p0Denom += sum(trainMatrix[i])   # 非侮辱类文档单词总数
    p1Vect = log(p1Num/p1Denom)              # change to log()   p（w/c1） 取 对数避免下溢出
    p0Vect = log(p0Num/p0Denom)              # change to log()   p（w/c0）
    return p0Vect,p1Vect,pAbusive # 返回[p(w1/c0),...,p(wn/c0)] [p(w1/c1),...,p(wn/c1)] p(c1)
'''
上面的代码中输入的是特征向量组成的矩阵，和一个由标签组成的向量，
其中pAbusive是类别概率P(ci)，因为只有两类，计算一类后，另外一类可以直接用1-p得出。
接下来初始化计算p(wi|c1)和p(wi|c0)的分子和分母，
这里惟一让人好奇的就是为什么分母p0Denom和p1Denom都初始化为2？
这是因为在实际应用中，我们计算出了（公式三）右半部分的概率后，也就是p(wi|ci)后，
注意wi表示消息中的一个字，接下来就是判断整条消息属于某个类别的概率，
就要计算p(w0|1)p(w1|1)p(w2|1)的形式，这样如果某个wi为0，这样整个概率都为0，
或者都很小连乘后会更小，甚至round off 0。
这样就会影响判断，因此把他们转到对数空间中来做运算，对数在机器学习里经常用到，
在保持单调的情况下避免因数值运算带来的歧义问题，而且对数可以把乘法转到加法运算，加速了运算。
因此上面的代码中把所有的出现次数初始化为1，然后把分母初始为2，接着都是累加，
在对数空间中从0还是1开始累加，最后比较大小不会受影响的。
'''

# 测试 朴树贝叶斯分类器
def test_trainNB0():
    pList,cVec = loadDataSet()          # 文档句子  和 对于标签
    vocList    = createVocabList(pList) # 生成字典
    trainMat   = []                     # 用字典线性表示的 文档
    for pDoc in pList:                  # 字典   待测试文档
        trainMat.append(setOfWords2Vec(vocList, pDoc))
    #        线性表示后的矩阵（词向量）  与类别标签
    p0_not_Vect,p1_yes_Vect,pAbusive=trainNB0(trainMat,cVec )# 得到 参数向量
    return p0_not_Vect,p1_yes_Vect,pAbusive


# 根据模型计算词向量属于每一类的概率，最大的为分类概率
# vec2Classify 要分类的词向量
# p(ci/w新)= p(w/ci)* p(ci)/ p(w) 正比  p(w/ci)* p(ci) 
# 正比log(p(w/ci)* p(ci) ) = log(p(w/ci)) + log(p(ci)) = 
# sum(log(p(w1/ci))+...+log(p(wn/ci))) + log(p(ci))

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)       # 对数 + 即为乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1 # 属于1类 侮辱类
    else: 
        return 0 # 属于0类

# 词袋模型   统计单词出现的次数
#  字典单词 线性 表示 句子 成 词向量
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # 统计出现的次数
    return returnVec


# 测试分类器
def testingNB():
    listOPosts,listClasses = loadDataSet()    # 句子和标签
    myVocabList = createVocabList(listOPosts) # 从所有句子中得到 所有不相同词汇表  字典
    trainMat=[]
    for postinDoc in listOPosts:# 每个句子
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))# 用字典线性表示成 词向量
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) # 得到 朴树贝叶斯分类器参数
    #测试句子包含的单词1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))    # 测试句子的 词向量
    print testEntry,'分类为: ',dist_lab[classifyNB(thisDoc,p0V,p1V,pAb)]# 打印分类结果
    #测试句子包含的单词2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'分类为: ',dist_lab[classifyNB(thisDoc,p0V,p1V,pAb)]# 打印分类结果

# 垃圾邮件过滤  英文句子 解析成 单词向量
def textParse(bigString):     # 输入为长长的字符串, # 输出为单词列表向量
    import re
    listOfTokens = re.split(r'\W*', bigString)# 按除去单词、数字外的任意字符分割字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] # 排除 单词个数少于2 的小词 变小些
   
# 垃圾邮件测试 
def spamTest():
    docList=[]; classList = []; fullText =[]# 文本词向量  标签  所有的词向量
    for i in range(1,26):# 总共有25个测试文本
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)  # 垃圾文档向量
        fullText.extend(wordList)
        classList.append(1)       # 标签为1
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)  # 非垃圾文档向量
        fullText.extend(wordList)
        classList.append(0)       # 标签为0

    vocabList = createVocabList(docList)          # 创建 单词字典
    trainingSet = range(50); testSet=[] 
          
    # 50个训练样本   从训练样本随机抽取 10个 测试样本，并从训练样本中删除
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])    # 测试样本
        del(trainingSet[randIndex])               # 从训练样本中删除 测试样本 

    # 训练
    trainMat=[]; trainClasses = []# 训练样本词向量 (词袋模型生成)
    for docIndex in trainingSet: 
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))#训练样本 词袋模型生成词向量
        trainClasses.append(classList[docIndex])                       #训练样本 标签
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))      #得到模型参数

    # 测试
    errorCount = 0
    for docIndex in testSet:      # 对每一个测试样本分类
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex]) #测试样本 词袋模型生成词向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "分类错误",docList[docIndex]
    print '错误率为: ',float(errorCount)/len(testSet)
    #return vocabList,fullText


# 计算最常出现的词汇 的频率 以及 词汇 前30个
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

# 
def localWords(feed1,feed0):
    import feedparser # ＲＳＳ阅读器  下载 wget https://pypi.python.org/packages/source/f/feedparser/feedparser-5.1.3.tar.gz#md5=f2253de78085a1d5738f626fcc1d8f71   sudo python setup.py install

    docList=[]; classList = []; fullText =[] # 文本词向量  标签  所有的词向量

    minLen = min(len(feed1['entries']),len(feed0['entries'])) # 访问所有的条目  好像关键字没有！！！
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])  # 对ＲＳＳ源数据 分解
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)       # NY is class 1 类别1

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)       # 类别2

    vocabList = createVocabList(docList)            # 字典
    top30Words = calcMostFreq(vocabList,fullText)   # 字典单词中 在 所有词汇中出现次数最多的30个词汇

    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0]) # 去除出现次数最高的词汇

    # 创建测试数据
    trainingSet = range(2*minLen); testSet=[]                #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
	print  randIndex, len(trainingSet)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) 

    # 创建训练数据 词向量
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet: 
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))# 词向量
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))      # 得到朴树贝叶斯分类器参数

    errorCount = 0
    for docIndex in testSet:        # 对每一个测试样本分类
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])    # 测试样本 词袋模型生成词向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#识别错误的样本
            errorCount += 1
    print '错误率: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V


# 
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)

    topNY=[]; topSF=[]

    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
