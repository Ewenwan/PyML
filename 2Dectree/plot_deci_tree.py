# -*- coding:utf-8 -*-
#!/usr/bin/python
'''
绘制决策树生成的树类型字典
'''
import matplotlib.pyplot as plt


# 文本框类型 和 箭头类型
decisionNode = dict(boxstyle="sawtooth", fc="0.8") # 决策节点 文本框类型 花边矩形
leafNode = dict(boxstyle="round4", fc="0.8")       # 叶子节点 文本框类型 倒角矩形
arrow_args = dict(arrowstyle="<-")                 # 箭头类型

# 得到 叶子节点总树，用以确定 图的横轴长度
def getNumLeafs(myTree):
    # myTree = {bestFeatLabel:{}} # 树的形状 分类属性：子树  {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    numLeafs = 0
    firstStr = myTree.keys()[0]   # 开始 节点 分类属性
    secondDict = myTree[firstStr] # 对应后面的子节点（子字典）
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':   # 子节点 是字典 （孩子节点） 循环调用
            numLeafs += getNumLeafs(secondDict[key]) #
        else:   numLeafs +=1                         # 子节点的 如果不是 字典（孩子节点），就是叶子节点
    return numLeafs

# 得到 树的深度(层数)，用以确定 图的纵轴长度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]   # 开始 节点 分类属性
    secondDict = myTree[firstStr] # 对应后面的子节点（子字典）
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':         # 子节点 是字典 （孩子节点） 循环调用
            thisDepth = 1 + getTreeDepth(secondDict[key])  # 
        else:   thisDepth = 1                              # 深度为1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# 绘制节点 带箭头的注释   框内文本  起点  终点   框类型
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


# 在 父子节点中间连线的线上 添加文本信息（属性分类值）
#                起点     终点      文本信息
def plotMidText(cntrPt, parentPt, txtString):
    # 计算中点位置（文本放置的位置）
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]                            
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30) # 偏转角度 初始位置为90 度


# 画树
def plotTree(myTree, parentPt, nodeTxt):# if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)      # 树的宽度（图的横轴坐标，叶子节点的数量）
    depth = getTreeDepth(myTree)        # 树的高度（图的纵坐标，  树的层数）
    firstStr = myTree.keys()[0]         # 父节点，开始节点信息（分裂属性）
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)# 起点
    plotMidText(cntrPt, parentPt, nodeTxt)               # 线上注释
    plotNode(firstStr, cntrPt, parentPt, decisionNode)   # 画分类属性节点 和 箭头
    secondDict = myTree[firstStr]                        # 后面的子树，子字典
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':           # 子字典内还有字典
            plotTree(secondDict[key],cntrPt,str(key))        # 递归调用画子树
        else:                                                # 画叶子节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict


# 画树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white') # 图1 背景白色
    fig.clf()                              # 清空显示
    axprops = dict(xticks=[], yticks=[])   # 标尺
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    # no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False)              # ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))  # 叶子节点总数（以便确定图的横轴长度）
    plotTree.totalD = float(getTreeDepth(inTree)) # 树的深度（层树）（以便确定 纵轴长度）
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
    
# 带箭头的含有文本框的  图
def plot_tree_demo():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# 事先存储一个 树信息
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

#createPlot(thisTree)