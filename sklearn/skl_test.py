#!usr/bin/env python  
#-*- coding: utf-8 -*-  
# http://blog.csdn.net/zouxy09/article/details/48903179
     
import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np  
import cPickle as pickle  
      
reload(sys)  
sys.setdefaultencoding('utf8')    # 指定默认中文编码
      
# 朴素贝叶斯分类器 Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
      
      
# 最近邻分类器 KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
      
      
# 逻辑回归分类器 Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
      
      
# 随机森林 分类器 Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
      
      
# 决策树 分类器 Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
      
# 梯度提升决策树分类器
# 集成学习 三个臭屁将顶一个诸葛亮 
# http://www.jianshu.com/p/005a4e6ac775  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
      
      
# 支持向量机分类器 VM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
      
# 交叉验证 支持向量机 SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print para, val  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model
 
# 读取 手写字体 数据集  
def read_data(data_file):  
    import gzip  
    f = gzip.open(data_file, "rb")  
    train, val, test = pickle.load(f)  
    f.close()  
    train_x = train[0]  
    train_y = train[1]  
    test_x = test[0]  
    test_y = test[1]  
    return train_x, train_y, test_x, test_y  
          
if __name__ == '__main__':  
    data_file = "mnist.pkl.gz"  
    thresh = 0.5  
    model_save_file = None  
    model_save = {}  
          
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']  
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
        }  
          
    print '读取 训练数据以及测试数据...'  
    train_x, train_y, test_x, test_y = read_data(data_file)  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2)  
    print '******************** 数据信息 *********************'  
    print '#训练数据: %d, #测试数据: %d, 维度: %d' % (num_train, num_test, num_feat)  
          
    for classifier in test_classifiers:  
        print '*****************分类器: %s ********************' % classifier  
        start_time = time.time()  
        model = classifiers[classifier](train_x, train_y)  
        print '训练时间: %fs!' % (time.time() - start_time)  
        predict = model.predict(test_x)  
        if model_save_file != None:  
            model_save[classifier] = model  
        if is_binary_class:  
            precision = metrics.precision_score(test_y, predict)  
            recall = metrics.recall_score(test_y, predict) 
            #  P = TP/(TP+FP) ;  精确度（Precision） 反映了被分类器判定的正例中真正的正例样本的比重
	    #  召回率(Recall)，也称为 True Positive Rate:
	    #  R = TP/(TP+FN) = 1 - FN/T;  反映了被正确判定的正例占总的正例的比重 
            print '精确度: %.2f%%, 召回率: %.2f%%' % (100 * precision, 100 * recall)  
        accuracy = metrics.accuracy_score(test_y, predict) 
	#  准确率（Accuracy） A = (TP + TN)/(P+N) = (TP + TN)/(TP + FN + FP + TN);    
	#  反映了分类器统对整个样本的判定能力——能将正的判定为正，负的判定为负
        print '准确率: %.2f%%' % (100 * accuracy)   
      
    if model_save_file != None:  
        pickle.dump(model_save, open(model_save_file, 'wb'))  
