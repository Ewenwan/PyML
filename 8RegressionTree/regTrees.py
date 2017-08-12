#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
回归树   连续值回归预测 的 回归树 
'''
# 测试代码
# import regTrees as RT  RT.RtTreeTest()  RT.RtTreeTest('ex0.txt') RT.RtTreeTest('ex2.txt')
# import regTrees as RT  RT.RtTreeTest('ex2.txt',ops=(10000,4))
# import regTrees as RT  RT.pruneTest()
# 模型树 测试
# import regTrees as RT  RT.modeTreeTest(ops=(1,10)
# 模型回归树和普通回归树 效果比较 计算相关系数  
# import regTrees as RT  RT.MRTvsSRT()
from numpy import *


# Tab 键值分隔的数据 提取成 列表数据集 成浮点型数据
def loadDataSet(fileName):      #   
    dataMat = []                # 目标数据集 列表
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #转换成浮点型数据
        dataMat.append(fltLine)
    return dataMat

# 按特征值 的数据集二元切分    特征(列)    对应的值
# 某一列的值大于value值的一行样本全部放在一个矩阵里，其余放在另一个矩阵里
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]  # 数组过滤
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0] # 
    return mat0,mat1

# 常量叶子节点
def regLeaf(dataSet):# 最后一列为标签  为数的叶子节点
    return mean(dataSet[:,-1])# 目标变量的均值
# 方差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]# 目标变量的平方误差 * 样本个数（行数）的得到总方差

# 选择最优的 分裂属性和对应的大小
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0] # 允许的误差下降值
    tolN = ops[1] # 切分的最少样本数量
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: # 特征剩余数量为1 则返回
        return None, leafType(dataSet)             #### 返回 1 #### 
    m,n = shape(dataSet) # 当前数据集大小 形状
    S = errType(dataSet) # 当前数据集误差  均方误差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):# 遍历 可分裂特征
        for splitVal in set(dataSet[:,featIndex]):# 遍历对应 特性的 属性值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)# 进行二元分割
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue #样本数量 小于设定值，则不切分
            newS = errType(mat0) + errType(mat1)# 二元分割后的 均方差
            if newS < bestS: # 弱比分裂前小 则保留这个分类
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS: # 弱分裂后 比 分裂前样本方差 减小的不多 也不进行切分
        return None, leafType(dataSet)             #### 返回 2 #### 
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #样本数量 小于设定值，则不切分
        return None, leafType(dataSet)             #### 返回 3 #### 
    return bestIndex,bestValue # 返回最佳的 分裂属性 和 对应的值

# 创建回归树 numpy数组数据集 叶子函数    误差函数   用户设置参数（最小样本数量 以及最小误差下降间隔）
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
 # 找到最佳的待切分特征和对应 的值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#
 # 停止条件 该节点不能再分，该节点为叶子节点
    if feat == None: return val 
    retTree = {}
    retTree['spInd'] = feat #特征
    retTree['spVal'] = val  #值
 # 执行二元切分  
    lSet, rSet = binSplitDataSet(dataSet, feat, val)# 二元切分  左树 右树
 # 创建左树
    retTree['left'] = createTree(lSet, leafType, errType, ops)   #  左树  最终返回子叶子节点 的属性值
 # 创建右树
    retTree['right'] = createTree(rSet, leafType, errType, ops)  #  右树
    return retTree 

# 未进行后剪枝的回归树测试  
def RtTreeTest(filename='ex00.txt',ops=(1,4)):
    MyDat = loadDataSet(filename) # ex00.txt y = w*x 两维   ex0.txt y = w*x+b 三维
    MyMat = mat(MyDat)
    print createTree(MyMat,ops=ops)
# 判断是不是树 (按字典形式存储)
def isTree(obj):
    return (type(obj).__name__=='dict')

# 返回树的平均值  塌陷处理
def getMean(tree):
    if isTree(tree['right']): 
	tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): 
	tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0  # 两个叶子节点的 平均值

# 后剪枝   待剪枝的树   剪枝所需的测试数据
def prune(tree, testData):
    if shape(testData)[0] == 0: 
	return getMean(tree) #没有测试数据 返回
    if (isTree(tree['right']) or isTree(tree['left'])): # 如果回归树的左右两边是树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])#对测试数据 进行切分
    if isTree(tree['left']): 
	tree['left'] = prune(tree['left'], lSet)   # 对左树进行剪枝
    if isTree(tree['right']): 
	tree['right'] =  prune(tree['right'], rSet)# 对右树进行剪枝
    if not isTree(tree['left']) and not isTree(tree['right']):#两边都是叶子
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])#对测试数据 进行切分
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2)) # 对两边叶子合并前计算 误差 
        treeMean = (tree['left']+tree['right'])/2.0  # 合并后的 叶子 均值
        errorMerge = sum(power(testData[:,-1] - treeMean,2))# 合并后 的误差
        if errorMerge < errorNoMerge: # 合并后的误差小于合并前的误差
            print "merging"           # 说明合并后的树 误差更小
            return treeMean           # 返回两个叶子 的均值 作为 合并后的叶子节点
        else: return tree
    else: return tree
    
def pruneTest():
    MyDat  = loadDataSet('ex2.txt')  
    MyMat  = mat(MyDat)
    MyTree = createTree(MyMat,ops=(0,1))    # 为了得到  最大的树  误差设置为0  个数设置为1 即不进行预剪枝
    MyDatTest  = loadDataSet('ex2test.txt')
    MyMatTest  = mat(MyDatTest)
    print prune(MyTree,MyMatTest)


######叶子节点为线性模型的模型树#########
# 线性模型
def linearSolve(dataSet):    
    m,n = shape(dataSet) # 数据集大小
    X = mat(ones((m,n))) # 自变量
    Y = mat(ones((m,1))) # 目标变量  
    X[:,1:n] = dataSet[:,0:n-1]# 样本数据集合
    Y = dataSet[:,-1]          # 标签
    # 线性模型 求解
    xTx = X.T*X                
    if linalg.det(xTx) == 0.0:
        raise NameError('行列式值为零,不能计算逆矩阵，可适当增加ops的第二个值')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

# 模型叶子节点
def modelLeaf(dataSet): 
    ws,X,Y = linearSolve(dataSet)
    return ws

# 计算模型误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

# 模型树测试
def modeTreeTest(filename='ex2.txt',ops=(1,4)):
    MyDat = loadDataSet(filename) # 
    MyMat = mat(MyDat)
    print createTree(MyMat,leafType=modelLeaf, errType=modelErr,ops=ops)#带入线性模型 和相应 的误差计算函数


# 模型效果计较
# 线性叶子节点 预测计算函数 直接返回 树叶子节点 值
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))# 增加一列
    X[:,1:n+1]=inDat
    return float(X*model) # 返回 值乘以 线性回归系数

# 树预测函数
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): 
	return modelEval(tree, inData) # 返回 叶子节点 预测值
    if inData[tree['spInd']] > tree['spVal']:      # 左树
        if isTree(tree['left']): 
	    return treeForeCast(tree['left'], inData, modelEval)# 还是树 则递归调用
        else: 
	    return modelEval(tree['left'], inData) # 计算叶子节点的值 并返回
    else:
        if isTree(tree['right']):                  # 右树
	    return treeForeCast(tree['right'], inData, modelEval)
        else: 
	    return modelEval(tree['right'], inData)# 计算叶子节点的值 并返回

# 得到预测值        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))#预测标签
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

# 常量回归树和线性模型回归树的预测结果比较
def MRTvsSRT():
    TestMat  = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    TrainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
# 普通回归树 预测结果
    # 得到普通回归树树
    StaTree = createTree(TrainMat, ops=(1,20))
    # 得到预测结果
    StaYHat = createForeCast(StaTree, TestMat[:,0], regTreeEval)# 第一列为 自变量
    # 预测结果和真实标签的相关系数
    StaCorr = corrcoef(StaYHat, TestMat[:,1], rowvar=0)[0,1] # NumPy 库函数 
# 模型回归树 预测结果
    # 得到模型回归树
    ModeTree = createTree(TrainMat,leafType=modelLeaf, errType=modelErr, ops=(1,20))
    # 得到预测结果
    ModeYHat = createForeCast(ModeTree, TestMat[:,0], modelTreeEval)  
    # 预测结果和真实标签的相关系数
    ModeCorr = corrcoef(ModeYHat, TestMat[:,1], rowvar=0)[0,1] # NumPy 库函数   
    print "普通回归树 预测结果的相关系数R2: %f" %(StaCorr)                                              
    print "模型回归树 预测结果的相关系数R2: %f" %(ModeCorr)
    if ModeCorr>StaCorr:
	print "模型回归树效果优于普通回归树"
    else:
	print "回归回归树效果优于模型普通树"
       

