
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import mglearn
from mglearn.datasets import *
#from mglearn.plot_helpers import cm2, cm3
from mglearn.plot_helpers import discrete_scatter
from mglearn.plot_2d_separator import plot_2d_separator
from mglearn.plot_2d_separator import plot_2d_scores
#from mglearn.plot_2d_separator import plot_2d_classification


cm_cycle = ListedColormap(['#0000aa', '#ff5050', '#50ff50', '#9040a0', '#fff000'])
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
cm2 = ListedColormap(['#0000aa', '#ff2020'])
cm = cm2


cdict = {'red': [(0.0, 0.0, cm2(0)[0]),
                 (1.0, cm2(1)[0], 1.0)],

         'green': [(0.0, 0.0, cm2(0)[1]),
                   (1.0, cm2(1)[1], 1.0)],

         'blue': [(0.0, 0.0, cm2(0)[2]),
                  (1.0, cm2(1)[2], 1.0)]}

ReBl = LinearSegmentedColormap("ReBl", cdict)



## 5.3　评估指标与评分

##5.3.2　二分类指标
digits = load_digits()
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
print(type(X_test),X_test.shape,np.bincount(X_test.astype(int).reshape(-1,)))


#DummyClassifier 使用简单规则进行预测的分类器，
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)  #strategy='most_frequent'  预测值是出现频率最高的类别
pred_most_frequent = dummy_majority.predict(X_test)  #y_train 为 False 的值最多，固 pred_most_frequent 全为False
print("Unique predicted labels: {}".format(np.unique(pred_most_frequent)))
print("Test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))


tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("Test score: {:.2f}".format(tree.score(X_test, y_test)))

dummy = DummyClassifier(strategy='stratified').fit(X_train, y_train)  #strategy='stratified' : 根据训练集中的频率分布给出随机预测
pred_dummy = dummy.predict(X_test)
print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))

#confusion_matrix 计算混淆矩阵
confusion = confusion_matrix(y_test, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))


print("Most frequent class:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nDummy model:")
print(confusion_matrix(y_test, pred_dummy))
print("\nDecision tree:")
print(confusion_matrix(y_test, pred_tree))
print("\nLogistic Regression")
print(confusion_matrix(y_test, pred_logreg))


#f1_score 1/2*(准确率*召回率)/(准确率+召回率)
print("f1 score most frequent: {:.2f}".format(f1_score(y_test, pred_most_frequent)))
print("f1 score dummy: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("f1 score tree: {:.2f}".format(f1_score(y_test, pred_tree)))
print("f1 score logistic regression: {:.2f}".format(
f1_score(y_test, pred_logreg)))

#classification_report 同时返回准确率、召回率和 f1_score
print(classification_report(y_test, pred_most_frequent,target_names=["not nine", "nine"]))
print(classification_report(y_test, pred_dummy,target_names=["not nine", "nine"]))
print(classification_report(y_test, pred_logreg,target_names=["not nine", "nine"]))


#4. 考虑不确定性
X, y = make_blobs(n_samples=(400, 50), cluster_std=[7.0, 2],
                  random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)

#决策函数的热图与改变决策阈值的影响
mglearn.plots.plot_decision_threshold()

print(classification_report(y_test, svc.predict(X_test)))  #svc.predict(X_test) 同 (svc.decision_function(X_test) > 0).astype(int)

#减小阈值,将更多的点划为类别 1
y_pred_lower_threshold = svc.decision_function(X_test) > -0.8  #decision_function 计算样本点到分割超平面的函数距离（函数距离:将几何距离进行了归一化）
print(classification_report(y_test, y_pred_lower_threshold))


#5. 准确率-召回率曲线

#precision_recall_curve 函数返回一个列表，包含按顺序排序的所有可能阈值（在决策函数中出现的所有值）对应的准确率和召回率
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))


#SVC（gamma=0.05）的准确率 - 召回率曲线
plt.figure()
# 使用更多数据点来得到更加平滑的曲线
X, y = make_blobs(n_samples=(4000, 500), cluster_std=[7.0, 2],random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
# 找到最接近于0的阈值
close_zero = np.argmin(np.abs(thresholds))  #np.argmin 返回最小值在数组中所在的位置
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,label="threshold zero", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")


#不同的分类器可能在曲线上不同的位置（即在不同的工作点）表现很好。我们来比较一下在同一数据集上训练的 SVM 与随机森林。
plt.figure()

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

# 随机森林里面有predict_proba，但没有decision_function
# RandomForestClassifier has predict_proba, but not decision_function
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:, 1])  #rf.predict_proba(X_test)[:, 1]) 返回属于类别 1 的概率

plt.plot(precision, recall, label="svc")

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero svc", fillstyle="none", c='k', mew=2)
print(precision[close_zero], recall[close_zero])
plt.plot(precision_rf, recall_rf, label="rf")
# 找到最接近于0.5的阈值(二分类问题的 predict_proba 的默认阈值是 0.5)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k',
         markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)
print(precision_rf[close_default_rf], recall_rf[close_default_rf])
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")

#默认阈值和上面最接近0或0.5时的阈值(是否有区别？)
print("f1_score of svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))
print("f1_score of random forest: {:.3f}".format(f1_score(y_test, rf.predict(X_test))))
print(classification_report(y_test, svc.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test)))


#平均准确率,即曲线下的面积(感觉没啥用)
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("Average precision of random forest: {:.3f}".format(ap_rf))
print("Average precision of svc: {:.3f}".format(ap_svc))


#6. 受试者工作特征（ROC）与AUC，对于不平衡类别的分类问题，使用 AUC 进行模型选择通常比使用精度更有意义

#SVM 的 ROC 曲线
plt.figure()
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# 找到最接近于0的阈值
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)


# SVM 和随机森林的 ROC 曲线对比
plt.figure()
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label="ROC Curve SVC")
plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")

plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,label="threshold zero SVC", fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf], tpr[close_default_rf], '^', markersize=10,label="threshold 0.5 RF", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)


#计算 ROC 曲线下的面积
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("AUC for Random Forest: {:.3f}".format(rf_auc))
print("AUC for SVC: {:.3f}".format(svc_auc))



#对比不同 gamma 值的 SVM 的 ROC 曲线
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

plt.figure()

for gamma in [1, 0.05, 0.01]:
    svc = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test , svc.decision_function(X_test))
    print("gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}".format(
          gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc="best")


##5.3.3　多分类指标

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred)))

plt.figure()
scores_image = mglearn.tools.heatmap(
    confusion_matrix(y_test, pred), xlabel='Predicted label',
    ylabel='True label', xticklabels=digits.target_names,
    yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()

print(classification_report(y_test, pred))

print("Micro average f1 score: {:.3f}".format(
    f1_score(y_test, pred, average="micro")))
print("Macro average f1 score: {:.3f}".format(
    f1_score(y_test, pred, average="macro")))


##5.3.4　回归指标(一般直接用 R^2 评估回归模型)


##5.3.5　在模型选择中使用评估指标
'''
对于分类问题， scoring 参数最重要的取值包括： accuracy （默认值）、 roc_auc （ROC 曲线下方的面积）、 average_precision （准确率 - 召回率曲线下方的面积）、 
f1 、 f1_macro 、f1_micro 和 f1_weighted （这四个是二分类的 f 1 - 分数以及各种加权变体）。
对于回归问题，最常用的取值包括： r2 （R 2 分数）、 mean_squared_error （均方误差）和 mean_absolute_error （平均绝对误差）。
也可以查看 metrics.scorer 模块中定义的 SCORER 字典
from sklearn.metrics.scorer import SCORERS
print("Available scorers:\n{}".format(sorted(SCORERS.keys())))
'''
# 分类问题的默认评分是精度
print("Default scoring: {}".format(cross_val_score(SVC(), digits.data, digits.target == 9, cv=5)))  #5折交叉验证
# 指定"scoring="accuracy"不会改变结果
explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9,scoring="accuracy", cv=5)
print("Explicit accuracy scoring: {}".format(explicit_accuracy))
roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9,scoring="roc_auc", cv=5)
print("AUC scoring: {}".format(roc_auc))


res = cross_validate(SVC(), digits.data, digits.target == 9,
                     scoring=["accuracy", "roc_auc", "recall_macro"],
                     return_train_score=True, cv=5)
#display(pd.DataFrame(res))


#改变 GridSearchCV 中用于选择最佳参数的指标
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target == 9, random_state=0)

# we provide a somewhat bad grid to illustrate the point:
# 我们给出了不太好的网格来说明：
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
# using the default scoring of accuracy:
# 使用默认的精度：
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("Grid-Search with accuracy")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))


# using AUC scoring instead:
# 使用AUC评分来代替：
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)
print("\nGrid-Search with AUC")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (AUC): {:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

plt.show()