# coding=utf-8

import pandas as pd
import numpy as np


def get_column_names():
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses', 'Class']
    return column_names


def get_data():
    column_names = get_column_names()
    data = pd.read_csv("../Datasets/breast-cancer-wisconsin.csv", names=column_names)
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')
    print(data.shape)
    return data


from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(data):
    column_names = get_column_names()
    X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                        test_size=0.25, random_state=33)  # 样品编号不考虑0 ， 1~9 ， 10 是分类结果
    print(y_train.value_counts())
    print(y_test.value_counts())
    return X_train, X_test, y_train, y_test


from sklearn.preprocessing import StandardScaler


# 必须先用fit_transform(partData)，之后再transform(restData)
# 如果直接transform(partData)，程序会报错
# 如果fit_transfrom(partData)后，使用fit_transform(restData)而不用transform(restData)，
# 虽然也能归一化，但是两个结果不是在同一个“标准”下的

# 标准化数据，保证每个维度的特征数据方差为1，均值为0。
def normalized(X_train, X_test):
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    return X_train, X_test


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 逻辑斯蒂回归 学习样本，输出 准确性， 精确率， 召回率，F1指标 四大性能评测指标
def classfication_Lgs(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_y_predict = lr.predict(X_test)
    print 'Accuracy of LR Classifier:', lr.score(X_test, y_test)
    print classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant'])


from sklearn.linear_model import SGDClassifier


# 随机梯度参数估计 学习样本，输出 准确性， 精确率， 召回率，F1指标 四大性能评测指标
def classfication_SGD(X_train, y_train, X_test, y_test):
    sgdc = SGDClassifier()
    sgdc.fit(X_train, y_train)
    sgdc_y_predict = sgdc.predict(X_test)
    print 'Accuarcy of SGD Classifier:', sgdc.score(X_test, y_test)
    print classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant'])


def main():
    data = get_data()
    print("===========start===========")
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(data)

    print("===========running===========")
    X_train, X_test = normalized(X_train, X_test)

    print("===========running===========")
    classfication_Lgs(X_train, y_train, X_test, y_test)

    print("===========running===========")
    classfication_SGD(X_train, y_train, X_test, y_test)

    print("===========end===========")


if __name__ == '__main__':
    main()
