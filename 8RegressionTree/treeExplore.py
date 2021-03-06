#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
模型回归树 与 普通回归树  的ＧＵＩ编程
'''
from numpy import *

from Tkinter import *
import regTrees

import matplotlib
matplotlib.use('TkAgg')#后端使用 TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg# 后端
from matplotlib.figure import Figure # 画布

def reDraw(tolS,tolN):
    reDraw.f.clf()        # 清除图像显示
    reDraw.a = reDraw.f.add_subplot(111)# 显示一个图
    if chkBtnVar.get():# 复选框选中的话 为模型树
        if tolN < 2: tolN = 2 # 最小数量限制
	# 生成 模型树
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,\
                                   regTrees.modelErr, (tolS,tolN))
	# 用模型树 进行预测
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, \
                                       regTrees.modelTreeEval)
    else:              # 不选的话 默认为 普通回归树
	# 生成 普通回归树
        myTree=regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
	# 用回归树 进行预测
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    # 散点图
    reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5) #大小为5
    # 画出预测曲线
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #线宽为2
    # 显示画布
    reDraw.canvas.show()

# 从输入文本框内得到 参数   
def getInputs():
    try: tolN = int(tolNentry.get())# 得到数量文本框的值
    except: # 如果输入了 非整数 会进入异常
        tolN = 10    # 异常 默认为 10
        print "enter Integer for tolN"# 提示输入整数
        tolNentry.delete(0, END)      # 删除非正常输入值
        tolNentry.insert(0,'10')      # 显示默认值
    try: tolS = float(tolSentry.get())# 得到 最小均方误差文本框的值
    except:          # 非浮点型数据
        tolS = 1.0   # 异常状态默认1.0
        print "enter Float for tolS"  # 提示 输入浮点型数据
        tolSentry.delete(0, END)      # 删除非正常输入值
        tolSentry.insert(0,'1.0')     # 显示默认值
    return tolN,tolS

# 花树按钮对应 的 回调函数
def drawNewTree():
    tolN,tolS = getInputs()#get values from Entry boxes
    reDraw(tolS,tolN)
    
root=Tk()# TK类型的根部件
# 画图按钮对应的 显示画布 大小  分辨率
reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)# columnspan=3跨三列
 
Label(root, text="tolN").grid(row=1, column=0) # 第一行 第0列位置显示标签 树样本个数
tolNentry = Entry(root)        # 输入文本框
tolNentry.grid(row=1, column=1)# 第一行 地一列 一个输入文本框
tolNentry.insert(0,'10')       # 最开始 输入文本框默认插入 10
Label(root, text="tolS").grid(row=2, column=0)# 第二行 第0列位置显示标签 最小均方误差
tolSentry = Entry(root)        # 输入文本框
tolSentry.grid(row=2, column=1)# 第二行 地一列 一个输入文本框
tolSentry.insert(0,'1.0')      # 最开始 输入文本框默认插入 1.0

# 普通的按钮 连接到画树的程序 位置第二行  第二列  rowspan=3 横夸三行 
Button(root, text="ReDraw", command=drawNewTree).grid(row=2, column=2, rowspan=2)
# 复选按钮 按钮前带 勾选框
chkBtnVar = IntVar() # 勾选框的值
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)# 带有勾选框的值变量
chkBtn.grid(row=3, column=0, columnspan=2)#第三行 地0列 横跨2列

# 退出按钮
Button(root, text="quit", command=root.quit()).grid(row=1, column=2)

# 全局变量
# 训练数据
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
# 测试数据
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
# 默认参数
reDraw(1.0, 10)
# 启动事件循环              
root.mainloop()
