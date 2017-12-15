# coding:utf-8

# 导入pandas并且更名为pd。
import pandas as pd


def get_titantic():
    titanic = pd.read_csv('../Datasets/titanic.txt')
    # 分离数据特征与预测目标。
    y = titanic['survived']
    X = titanic.drop(['row.names', 'name', 'survived'], axis=1)
    # 对对缺失数据进行填充。
    X['age'].fillna(X['age'].mean(), inplace=True)
    X.fillna('UNKNOWN', inplace=True)
    return X, y


# 分割数据，依然采样25%用于测试。
from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


# 类别型特征向量化。
from sklearn.feature_extraction import DictVectorizer


def vectorize(X_train, X_test):
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))
    # 输出处理后特征向量的维度。
    print len(vec.feature_names_)
    return X_train, X_test


# 从sklearn导入特征筛选器。
from sklearn import feature_selection


def selection_percentile(X_train, X_test, y_train, percentile):
    # 筛选前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能。
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile)
    X_train_fs = fs.fit_transform(X_train, y_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs


# 使用决策树模型依靠所有特征进行预测，并作性能评估。
from sklearn.tree import DecisionTreeClassifier


def get_DT():
    dt = DecisionTreeClassifier(criterion='entropy')
    return dt


def classifiaction_DT(X_train, X_test, y_train, y_test):
    dt = get_DT()
    dt.fit(X_train, y_train)
    dt.score(X_test, y_test)


def classifiaction_DT_percent(X_train, X_test, y_train, y_test, percentile):
    dt = get_DT()
    X_train_fs, X_test_fs = selection_percentile(X_train, X_test, y_train, percentile)
    dt.fit(X_train_fs, y_train)
    dt.score(X_test_fs, y_test)


import pylab as pl


def pl_plot(percentiles, results):
    pl.plot(percentiles, results)
    pl.xlabel('percentiles of features')
    pl.ylabel('accuracy')
    pl.show()


# 通过交叉验证（下一节将详细介绍）的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化。
from sklearn.cross_validation import cross_val_score
import numpy as np


def get_percentiles_results(X_train, y_train):
    percentiles = range(1, 100, 2)
    results = []
    dt = get_DT()
    for i in percentiles:
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
        X_train_fs = fs.fit_transform(X_train, y_train)
        scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
        results = np.append(results, scores.mean())
    print results
    opt = np.where(results == results.max())[0][0]  # 这一句跟源代码有出入，查看文档np.where返回的是 ndarray or tuple of ndarrays类型数据
    print 'Optimal number of features %d' % percentiles[opt]
    return percentiles, results


def main():
    print("===========start===========")
    X, y = get_titantic()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(X, y)
    X_train, X_test = vectorize(X_train, X_test)
    dt = get_DT()
    classifiaction_DT(X_train, X_test, y_train, y_test)
    print("===========running===========")
    classifiaction_DT_percent(X_train, X_test, y_train, y_test, 20)

    print("===========running===========")
    classifiaction_DT_percent(X_train, X_test, y_train, y_test, 7)

    print("===========running===========")
    percentiles, results = get_percentiles_results(X_train, y_train)
    pl_plot(percentiles, results)

    print("===========end===========")

if __name__ == '__main__':
    main()
