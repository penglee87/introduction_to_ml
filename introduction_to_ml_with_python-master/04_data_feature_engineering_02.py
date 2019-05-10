
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

## 4.6　利用专家知识
def load_citibike():
    data_mine = pd.read_csv(os.path.join(DATA_PATH, "citibike.csv"))
    data_mine['one'] = 1
    data_mine['starttime'] = pd.to_datetime(data_mine.starttime)
    data_starttime = data_mine.set_index("starttime")
    #t1 = data_starttime.resample("3h").sum()
    #print('t1',t1)
    data_resampled = data_starttime.resample("3h").sum().fillna(0)  #resample：在给定的时间单位内重取样
    print(data_starttime.resample("3h").sum().shape,data_resampled.shape)
    return data_resampled.one


citibike = load_citibike()
print("Citi Bike data:\n{}".format(citibike.head()))



plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
#plt.xticks(xticks.astype("int"), xticks.strftime("%a %m-%d"), rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Rentals")


# 提取目标值（租车数量）
y = citibike.values
# 利用"%s"将时间转换为POSIX时间
#X = citibike.index.strftime("%s").astype("int").reshape(-1, 1)
X = citibike.index.astype("int64").values.reshape(-1, 1) // 10**9

n_train = 184

# 对给定特征集上的回归进行评估和作图的函数
# function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    # split the given features into a training and a test set
    X_train, X_test = features[:n_train], features[n_train:]
    # also split the target array
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")



# 随机森林仅使用 POSIX 时间做出的预测
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)

# 随机森林仅使用每天的时刻做出的预测
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

# 随机森林使用一周的星期几和每天的时刻两个特征做出的预测
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),citibike.index.hour.values.reshape(-1, 1)])
eval_on_features(X_hour_week, y, regressor)

# 线性回归使用一周的星期几和每天的时刻两个特征做出的预测
eval_on_features(X_hour_week, y, LinearRegression())  #结果较差，因为线性回归会将星期几和小时当连续变量


#将整数转换为分类变量（用 OneHotEncoder 进行变换）
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()  #转换后含15列特征(7个不同天+8个不同小时数)，有警告，原因不懂
eval_on_features(X_hour_week_onehot, y, LinearRegression())
eval_on_features(X_hour_week_onehot, y, Ridge())

# PolynomialFeatures 多项式生成函数
# degree：默认为2，多项式次数(就同几元几次方程中的次数一样)
# interaction_only：是否包含单个自变量**n(n>1)特征数据标识，默认为False，为True则表示去除与自己相乘的情况
# include_bias：是否包含偏差标识，默认为True，为False则表示不包含偏差项
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True,include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)  #转换后含120列特征(15*14/2+15  15个特征两两组合+原始15个特征)
print('X_hour_week_onehot_poly',X_hour_week_onehot_poly.shape)

lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)
print(lr.coef_)  #lr.coef_ 斜率,即每个特征的系数

#将线性模型学到的系数作图，而这对于随机森林来说是不可能的
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features = day + hour

features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]  #仅保留系数不为零的特征
coef_nonzero = lr.coef_[lr.coef_ != 0]  #仅保留系数不为零的特征的系数

#将线性模型学到的系数可视化
plt.figure(figsize=(15, 2))
#plt.plot(lr.coef_, 'o')  #所有系数
#plt.xticks(np.arange(len(features_poly)), features_poly, rotation=90)
plt.plot(coef_nonzero, 'o')  #仅保留非零系数
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature name")
plt.ylabel("Feature magnitude")

plt.show()



