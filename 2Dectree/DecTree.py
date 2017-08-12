# -*- coding:utf-8 -*-
#!/usr/bin/python

# 测试 import DecTree as DT   DT.test_dt()
# 测试 import DecTree as DT   DT.test_skl_dt()

from math import log   # 对数
import operator        # 操作符

import copy            # 列表复制，不改变原来的列表


# 画树
import plot_deci_tree as pt

## 自定义数据集 来进行测试
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers'] # 属性值列表
    #change to discrete values
    return dataSet, labels


## 计算给定数据集的熵（混乱度） sum（概率 * （- log2(概率）））
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 数据集 样本总数
    labelCounts = {}          # 标签集合 字典 标签 对应 出现的次数
    # 计算 类标签 各自出现的次数
    for featVec in dataSet:   # 每个样本
        currentLabel = featVec[-1]     # 每个样本的最后一列为标签
        # 若当前标签值不存在，则扩展字典，加入当前关键字
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1 # 对应标签 计数 + 1
    # 计算信息熵
    shannonEnt = 0.0   
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 计算每类出现的概率
        shannonEnt -= prob * log(prob,2)           # 计算信息熵
    return shannonEnt
  
  
##  按照给定特征划分数据集 （取 划分属性值列 的剩余 部分）
def splitDataSet(dataSet, axis, value):
    # dataSet 数据集   axis划分的属性（哪一列）   对应属性（axis）值value 的那一行要去掉
    retDataSet = []         # 划分好的数据集
    for featVec in dataSet: # 每一行 即一个 样本
        if featVec[axis] == value:                  #  选取 符合 对应属性值 的列
            reducedFeatVec = featVec[:axis]         # 对应属性值 之前的其他属性值
            reducedFeatVec.extend(featVec[axis+1:]) # 加入 对应属性值 之前后的其他属性值  成一个 列表
            retDataSet.append(reducedFeatVec)       # 将其他部分 加入新 的列表里
    return retDataSet
 


## 选取最好的 划分属性值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       # 总特征维度， 最后一列是标签
    baseEntropy = calcShannonEnt(dataSet)   # 计算原来数据集的 信息熵
    bestInfoGain = 0.0; bestFeature = -1    # 信息增益 和 最优划分属性初始化
    
    for i in range(numFeatures):            # 对于所有的特征（每一列特征对应一个 属性，即对已每一个属性）
        featList = [example[i] for example in dataSet] # 列表推导 所有 对应 特征属性
        uniqueVals = set(featList)          # 从列表创建集合 得到每（列）个属性值的集合 用于划分集合
        newEntropy = 0.0
        for value in uniqueVals:            # 对于该属性  的每个 属性值
            subDataSet = splitDataSet(dataSet, i, value)    # 选取对应属性对应属性值 的新集合
            prob = len(subDataSet)/float(len(dataSet))      # 计算 该属性下该属性值的样本所占总样本数 的比例
            newEntropy += prob * calcShannonEnt(subDataSet) # 比例 * 对于子集合的信息熵，求和得到总信息熵  
        infoGain = baseEntropy - newEntropy                 # 原始集合信息熵 - 新划分子集信息熵和  得到信息增益
        if (infoGain > bestInfoGain):       # 信息熵 比划分前 减小了吗？ 减小的话 （信息增益增大）
            bestInfoGain = infoGain         # 更新最优 信息熵
            bestFeature = i                 # 记录当前最优 的划分属性
    return bestFeature                      # 返回全局最有的划分属性


# 统计样本集合 类出现的次数 返回出现最多的 分类名称
def majorityCnt(classList):
    classCount={}
    for vote in classList: # 每一个样本
        if vote not in classCount.keys(): classCount[vote] = 0 # 增加类标签到字典中
        classCount[vote] += 1                                  # 统计次数
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)# 按类出现次数 从大到小排序
    return sortedClassCount[0][0]                              # 返回出现次数最多的


# 输入数据集 和 属性标签 生成 决策树
def createTree(dataSet,labels):
    #copy_labels = labels
    classList = [example[-1] for example in dataSet]    # 每个样本的分类标签
    # 终止条件1 所有类标签完全相同
    if classList.count(classList[0]) == len(classList): 
        return classList[0]           # 返回该类标签（分类属性）
    # 终止条件2 遍历完所有特征
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList) # 返回出现概率最大的类标签
    # 选择最好的划分属性（划分特征）
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 对于特征值（属性值）的特征（属性）
    bestFeatLabel = labels[bestFeat]
    # 初始化树
    myTree = {bestFeatLabel:{}} # 树的形状 分类属性：子树
    del(labels[bestFeat]) # 有问题 改变了 原来属性序列
    # 根据最优的划分属性 的 值列表 创建子树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:          # 最优的划分属性 的 值列表
        subLabels = labels[:]         # 每个子树的 子属性标签
        # 递归调用 生成决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            


# 使用训练好的决策树做识别分类   
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]    # 起始分类属性  节点
    secondDict = inputTree[firstStr]  # 子树
    featIndex = featLabels.index(firstStr)  # 属性标签索引
   # key = testVec[featIndex]               # 测试属性值向量 起始分类属性 对应 的属性
   # valueOfFeat = secondDict[key]          # 对应的子树 或 叶子节点
   # if isinstance(valueOfFeat, dict):      # 子树还是 树
   #     classLabel = classify(valueOfFeat, featLabels, testVec) #递归调用
   # else: classLabel = valueOfFeat         # 将叶子节点的标签赋予 类标签 输出
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':      # 子树还是 树
                classLabel = classify(secondDict[key], featLabels, testVec) #递归调用
            else: classLabel = secondDict[key]         # 将叶子节点的标签赋予 类标签 输出
            
    return classLabel

# 使用 pickle 对象 存储决策数
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

# 使用 pickle 对象 载入决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# 使用佩戴眼镜数据测试

def test_dt():
    print '载入数据 lenses.txt ...'
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_lab  = ['age','prescript','astigmatic','teatRate']
    print '创建 lenses  决策数...'
    lenses_tree = createTree(lenses,lenses_lab)
    print lenses_tree
    print '保存 lenses  决策数...'
    storeTree(lenses_tree,'lenses_tree.txt')
    print '可视化 lenses  决策数...'
    pt.createPlot(lenses_tree)
    
# 使用佩戴眼镜数据测试   sklearn.tree.DecisionTreeClassifier() 数据需要是numpy的float32类型
# 如果是字符串类型 需要 使用pandas生成pandas数据后 再 序列化 完成数据编码
# 如果样本数量少但是样本特征非常多，在拟合决策树模型前，推荐先做维度规约，比如主成分分析（PCA），
# 特征选择（Losso）或者独立成分分析（ICA）。这样特征的维度会大大减小。再来拟合决策树模型效果会好。

def test_skl_dt():
    from sklearn import tree # sklearn 的决策数  更新 sudo pip install scikit-learn --upgrade
    import pandas as pd      # 使用pandas生成pandas数据 序列化
    from sklearn.preprocessing import LabelEncoder # 数据编码
    import pydotplus # 决策数可视化
    import graphviz  # 决策数可视化
    from sklearn.externals.six import StringIO
    print '载入数据 lenses.txt ...'
    fr = open('lenses.txt')#打开文件
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]#读取文件成列表
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1]) #分类标签  硬材质 软材质  不需要佩戴  
    lenses_label= ['age','prescript','astigmatic','teatRate']#特征：年龄 症状 是否散光 眼泪数量 
    lenses_list = []  # 保存 lenses数据的临时列表
    lenses_dict = {}  # 保存 lenses数据的临时字典 用于生成 pandas数据
    for each_label in lenses_label:   #提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lenses_label.index(each_label)])# 每个特征标签的值
        lenses_dict[each_label] = lenses_list #每个特征标签对应的 所有样本的 值
        lenses_list = []
    print(lenses_dict) #打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict) # 生成pandas.DataFrame
    print(lenses_pd)                      # 打印pandas.DataFrame
    le = LabelEncoder()                   # 创建LabelEncoder()对象，用于序列化            
    for col in lenses_pd.columns:         # 为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)                      # 打印编码信息

    clf = tree.DecisionTreeClassifier(max_depth = 4)       # 创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)# 使用数据，构建决策树
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file = dot_data, # 绘制决策树
                        feature_names = lenses_pd.keys(),
                        class_names = clf.classes_,       
                                                   # 如果提示没有 class_names 的话，请更新scikit-learn
                        filled=True, rounded=True,
                        special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf") #保存绘制好的决策树，以PDF的形式存储。
    
    
