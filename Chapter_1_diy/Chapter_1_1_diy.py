# coding=utf-8
import pandas as pd

df_train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')

df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

import matplotlib.pyplot as plt


def plt_scatter(plt, df_test_negative, df_test_positive):
    plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()


# 绘制 测试样本 点
def figure1(df_test_negative, df_test_positive):
    plt.figure(1)
    plt_scatter(plt, df_test_negative, df_test_positive)


import numpy as np


# 绘制 随机曲线，对测试样本分类
def figure2(df_test_negative, df_test_positive):
    intercept = np.random.random([1])
    coef = np.random.random([2])

    lx = np.arange(0, 12)
    ly = (-intercept - lx * coef[0]) / coef[1]
    # ly 为 随机直线
    plt.figure(2)
    plt.plot(lx, ly, c='yellow')
    plt_scatter(plt, df_test_negative, df_test_positive)


from sklearn.linear_model import LogisticRegression


# 使用 逻辑斯蒂 回归分类器 分类， 学习10条 训练样本
def figure3(df_train, df_test):
    lr = LogisticRegression()
    lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])  # 前面是 x 向量，Type 是 y 向量 进行训练
    print 'Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']],
                                                              df_test['Type'])
    # 训练后直线的 截距
    intercept = lr.intercept_
    # 训练后直线的 系数
    coef = lr.coef_[0, :]

    lx = np.arange(0, 12)
    # 训练后 的 曲线
    ly = (-intercept - lx * coef[0]) / coef[1]

    plt.figure(3)
    plt.plot(lx, ly, c='green')
    plt_scatter(plt, df_test_negative, df_test_positive)


# 使用 逻辑斯蒂 回归分类器 分类， 学习 全部 训练样本
def figure4(df_train, df_test):
    lr = LogisticRegression()

    lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])  # 前面是 x 向量，Type 是 y 向量 进行训练
    print 'Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']],
                                                               df_test['Type'])

    intercept = lr.intercept_
    coef = lr.coef_[0, :]
    lx = np.arange(0, 12)
    ly = (-intercept - lx * coef[0]) / coef[1]

    plt.figure(4)
    plt.plot(lx, ly, c='blue')
    plt_scatter(plt, df_test_negative, df_test_positive)


def main():
    # 读取测试样本， 构建测试机中 正负分类 样本
    df_test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')
    df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
    df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

    print("===========start===========")
    figure1(df_test_negative, df_test_positive)

    print("===========running===========")
    # figure2 中 为 随机线， 不固定
    figure2(df_test_negative, df_test_positive)

    print("===========running===========")
    # 读取 训练样本
    df_train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')

    figure3(df_train, df_test)

    print("===========running===========")
    figure4(df_train, df_test)

    print("===========end===========")


if __name__ == '__main__':
    main()

    # backup running
    # print("===========running===========")
