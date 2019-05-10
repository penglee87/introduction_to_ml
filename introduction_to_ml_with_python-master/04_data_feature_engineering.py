import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

from scipy import signal
from sklearn.datasets import load_boston
from sklearn.datasets import make_blobs
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer  #sklearn 0.20版本以后
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

import mglearn
from mglearn.datasets import *
from mglearn.plot_helpers import discrete_scatter
from mglearn.plot_helpers import cm2, cm3




## 4.1　分类变量
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
print(demo_df)

#get_dummies 自动编码字符串特征，不会改变整数特征
print(pd.get_dummies(demo_df))
print(pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature']))  #指定对哪些列进行编码，整数型也可以指定

#demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)  #将Integer Feature列改为字符串类型


# 设置 sparse=False 返回一个 numpy array, 而不是一个 sparse matrix(索引加值表示的稀疏矩阵)
ohe = OneHotEncoder(sparse=False)
print(ohe.fit_transform(demo_df))

#默认sparse=True
ohe = OneHotEncoder()
print(demo_df)
ohe.fit(demo_df)
print(ohe.transform(demo_df))  #按索引加值的形式显示矩阵
print(ohe.transform(demo_df).toarray())  #转化为矩阵，也可设置sparse=False得到
print(ohe.fit_transform(demo_df))




## 4.2　分箱、离散化、线性模型与树
#原始数据决策对与线性模型对比
X, y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_leaf=3).fit(X, y)
plt.plot(line, reg.predict(line), label="decision tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")


#kb = KBinsDiscretizer(n_bins=10, strategy='uniform')
#kb.fit(X)
#print("bin edges: \n", kb.bin_edges_)
#X_binned = kb.transform(X)
#X_binned
#X_binned.toarray()[:10]

#kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='onehot-dense')
#kb.fit(X)
#X_binned = kb.transform(X)

# 使用OneHotEncoder进行变换后，决策对与线性模型对比
figure = plt.figure()

bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)  #记录每个数据点所属的箱子
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

encoder = OneHotEncoder(sparse=False)
# encoder.fit找到which_bin中的唯一值
encoder.fit(which_bin)
# transform创建one-hot编码
X_binned = encoder.transform(which_bin)
print(X_binned[:5])


line_binned = encoder.transform(np.digitize(line, bins=bins))
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")

## 4.3　交互特征与多项式特征
#添加原始数据的交互特征（interaction feature），使用分箱特征和单一全局斜率的线性回归
figure = plt.figure()
X_combined = np.hstack([X, X_binned])
line_combined = np.hstack([line, line_binned])
reg = LinearRegression().fit(X_combined, y)
plt.plot(line, reg.predict(line_combined), label='linear regression combined')

#决策树效果与只用特征X的效果相同
#reg = DecisionTreeRegressor(min_samples_split=3).fit(X_combined, y)
#plt.plot(line, reg.predict(line_combined), label='decision tree binned')
#reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
#plt.plot(line, reg.predict(line), label='decision tree')
for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')
#plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:, 0], y, 'o', c='k')




#添加交互特征或乘积特征，每个箱子具有不同斜率的线性回归
figure = plt.figure()
X_product = np.hstack([X_binned, X * X_binned])
reg = LinearRegression().fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')
for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")



# 使 用 原 始 特 征 的 多 项 式（polynomial）。对于给定特征 x ，我们可以考虑 x ** 2 、 x ** 3 、 x ** 4
figure = plt.figure()

# 包含直到x ** 10的多项式:
# 默认的"include_bias=True"添加恒等于1的常数特征
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


#将多项式特征与线性回归模型一起使用，可以得到经典的多项式回归（polynomial regression）模型
reg = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")


#在原始数据上学到的核 SVM 模型（能够学到一个与多项式回归的复杂度类似的预测）
figure = plt.figure()
for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")



## 4.4　单变量非线性变换
figure = plt.figure()
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)
X = rnd.poisson(10 * np.exp(X_org))  #poisson 从泊松分布中抽取样本(返回一组整数值，其分布符合泊松分布)
y = np.dot(X_org, w)  #np.dot 矩阵相乘

bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='grey')
plt.ylabel("Number of appearances")
plt.xlabel("Value")

#原始数据岭回归模型
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test score: {:.3f}".format(score))

#LOG变换后数据岭回归模型
figure = plt.figure()
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel("Number of appearances")
plt.xlabel("Value")

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("Test score: {:.3f}".format(score))


## 4.5　自动化特征选择
# 4.5.1　单变量统计
figure = plt.figure()
from sklearn.feature_selection import SelectPercentile
cancer = load_breast_cancer()
# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# 向数据中添加噪声特征
# 前30个特征来自数据集，后50个是噪声
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(
X_w_noise, cancer.target, random_state=0, test_size=.5)
# 使用f_classif（默认值）和SelectPercentile来选择50%的特征
select = SelectPercentile(percentile=50)  #特征选择算法,选取50%的特征数
select.fit(X_train, y_train)
# 对训练集进行变换
X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

mask = select.get_support()  #get_support 查看哪些特征被选中
print(mask)
# 将选取的特征可视化——黑色为True，白色为False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

#比较 Logistic 回归在所有特征上的性能与仅使用所选特征的性能
X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test)))

# 4.5.2　基于模型的特征选择
figure = plt.figure()
from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),threshold="median")  #使用中位数作为阈值

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Test score: {:.3f}".format(score))

# 4.5.3　迭代特征选择
figure = plt.figure()
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),n_features_to_select=40)
select.fit(X_train, y_train)
# 将选中的特征可视化：
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())

X_train_rfe= select.transform(X_train)
X_test_rfe= select.transform(X_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Test score: {:.3f}".format(score))
#利用在 RFE 内使用的模型(随机森林)来进行预测
print("Test score: {:.3f}".format(select.score(X_test, y_test)))

plt.show()