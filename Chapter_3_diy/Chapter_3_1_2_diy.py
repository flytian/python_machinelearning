# coding:utf-8

def get_train():
    # 输入训练样本的特征以及目标值，分别存储在变量X_train与y_train之中。
    X_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    return X_train, y_train


# 导入numpy并且重命名为np。
import numpy as np


def get_tezt_xx():
    # 在x轴上从0至25均匀采样100个数据点。
    xx = np.linspace(0, 26, 100)
    xx = xx.reshape(xx.shape[0], 1)
    return xx


# 从sklearn.linear_model中导入LinearRegression。
from sklearn.linear_model import LinearRegression

def get_LR():
    return LinearRegression()
def classification_LR(X_train, y_train, xx):
    # 使用默认配置初始化线性回归模型。
    regressor = get_LR()
    # 直接以披萨的直径作为特征训练模型。
    regressor.fit(X_train, y_train)
    yy = regressor.predict(xx)
    # 输出线性回归模型在训练样本上的R-squared值。
    print 'The R-squared value of Linear Regressor performing on the training data is', regressor.score(X_train,y_train)
    return regressor, yy

# 对回归预测到的直线进行作图。
import matplotlib.pyplot as plt

def figure1(X_train, y_train, xx, yy):
    plt.scatter(X_train, y_train)
    plt1, = plt.plot(xx, yy, label="Degree=1")
    plt.axis([0, 25, 0, 25])
    plt.xlabel('Diameter of Pizza')
    plt.ylabel('Price of Pizza')
    plt.legend(handles=[plt1])
    plt.show()

# 从sklearn.preproessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures

def get_poly(degree):
   return PolynomialFeatures(degree)

def polynominal_2(X_train, xx):
    # 使用PolynominalFeatures(degree=2)映射出2次多项式特征，存储在变量X_train_poly2中。
    poly2 = get_poly(2)
    X_train_poly2 = poly2.fit_transform(X_train)
    # 从新映射绘图用x轴采样数据。
    xx_poly2 = poly2.transform(xx)
    return  X_train_poly2, xx_poly2

# 和 上 一个 函数， 仅参数 不一样， 可以和并成一个，进行传参 degree
def polynominal_4(X_train, xx):
    # 初始化4次多项式特征生成器。
    poly4 = PolynomialFeatures(degree=4) # get_poly(4)
    X_train_poly4 = poly4.fit_transform(X_train)
    # 从新映射绘图用x轴采样数据。
    xx_poly4 = poly4.transform(xx)
    return  X_train_poly4, xx_poly4

def classification_LR_2(X_train_poly2, y_train, xx_poly2):
    # 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然是线性模型。
    regressor_poly2 = LinearRegression() # get_LR()
    # 对2次多项式回归模型进行训练。
    regressor_poly2.fit(X_train_poly2, y_train)
    # 使用2次多项式回归模型对应x轴采样数据进行回归预测。
    yy_poly2 = regressor_poly2.predict(xx_poly2)
    # 输出2次多项式回归模型在训练样本上的R-squared值。
    print 'The R-squared value of Polynominal Regressor (Degree=2) performing on the training data is', regressor_poly2.score(
        X_train_poly2, y_train)
    return regressor_poly2,yy_poly2


def classification_LR_4(X_train_poly4, y_train, xx_poly4):
    # 使用默认配置初始化4次多项式回归器。
    regressor_poly4 = LinearRegression()
    # 对4次多项式回归模型进行训练。
    regressor_poly4.fit(X_train_poly4, y_train)
    # 使用2次多项式回归模型对应x轴采样数据进行回归预测。
    yy_poly4 = regressor_poly4.predict(xx_poly4)
    print 'The R-squared value of Polynominal Regressor (Degree=4) performing on the training data is', regressor_poly4.score(
        X_train_poly4, y_train)
    return regressor_poly4, yy_poly4

def figure2(X_train, y_train, xx, yy, yy_poly2):
    # 分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图。
    plt.scatter(X_train, y_train)
    plt1, = plt.plot(xx, yy, label='Degree=1')
    plt2, = plt.plot(xx, yy_poly2, label='Degree=2')

    plt.axis([0, 25, 0, 25])
    plt.xlabel('Diameter of Pizza')
    plt.ylabel('Price of Pizza')
    plt.legend(handles=[plt1, plt2])
    plt.show()

def figure3(X_train, y_train, xx, yy,yy_poly2, yy_poly4):
    # 分别对训练数据点、线性回归直线、2次多项式以及4次多项式回归曲线进行作图。
    plt.scatter(X_train, y_train)
    plt1, = plt.plot(xx, yy, label='Degree=1')
    plt2, = plt.plot(xx, yy_poly2, label='Degree=2')

    plt4, = plt.plot(xx, yy_poly4, label='Degree=4')
    plt.axis([0, 25, 0, 25])
    plt.xlabel('Diameter of Pizza')
    plt.ylabel('Price of Pizza')
    plt.legend(handles=[plt1, plt2, plt4])
    plt.show()

def get_tezt_x_y():
    # 准备测试数据。
    X_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]
    return X_test, y_test

def regressor_score_tezt(regressor):
    print("---regressor---")
    X_test, y_test = get_tezt_x_y()
    regressor.score(X_test, y_test)

def regressor_poly2_score_tezt(X_train, regressor_poly2):
    print("---regressor_poly2---")
    X_test, y_test = get_tezt_x_y()
    poly2 = get_poly(2)
    X_train_poly2 = poly2.fit_transform(X_train)
    # 使用测试数据对2次多项式回归模型的性能进行评估。
    X_test_poly2 = poly2.transform(X_test)
    regressor_poly2.score(X_test_poly2, y_test)
    return X_test_poly2, y_test

def regressor_poly4_score_tezt(X_train, regressor_poly4):
    print("---regressor_poly4---")
    X_test, y_test = get_tezt_x_y()
    poly4 = get_poly(4)
    X_train_poly4 = poly4.fit_transform(X_train)
    # 使用测试数据对4次多项式回归模型的性能进行评估。
    X_test_poly4 = poly4.transform(X_test)
    regressor_poly4.score(X_test_poly4, y_test)
    # 回顾普通4次多项式回归模型过拟合之后的性能。
    print regressor_poly4.score(X_test_poly4, y_test)
    # 回顾普通4次多项式回归模型的参数列表。
    print regressor_poly4.coef_
    # 输出普通4次多项式回归模型的参数列表。
    print regressor_poly4.coef_
    # 输出上述这些参数的平方和，验证参数之间的巨大差异。
    print np.sum(regressor_poly4.coef_ ** 2)
    return X_test_poly4, y_test

# 从sklearn.linear_model中导入Lasso。
from sklearn.linear_model import Lasso

def classification_Lasso(X_train_poly4, y_train, X_test_poly4, y_test):
    # 从使用默认配置初始化Lasso。
    lasso_poly4 = Lasso()
    # 从使用Lasso对4次多项式特征进行拟合。
    lasso_poly4.fit(X_train_poly4, y_train)
    # 对Lasso模型在测试样本上的回归性能进行评估。
    print lasso_poly4.score(X_test_poly4, y_test)
    # 输出Lasso模型的参数列表。
    print lasso_poly4.coef_

# 从sklearn.linear_model导入Ridge。
from sklearn.linear_model import Ridge

def classification_Ridge(X_train_poly4, y_train, X_test_poly4, y_test):
    # 使用默认配置初始化Riedge。
    ridge_poly4 = Ridge()
    # 使用Ridge模型对4次多项式特征进行拟合。
    ridge_poly4.fit(X_train_poly4, y_train)
    # 输出Ridge模型在测试样本上的回归性能。
    print ridge_poly4.score(X_test_poly4, y_test)
    # 输出Ridge模型的参数列表，观察参数差异。
    print ridge_poly4.coef_
    # 计算Ridge模型拟合后参数的平方和。
    print np.sum(ridge_poly4.coef_ ** 2)

def main():
    print("===========start===========")
    X_train, y_train = get_train()
    xx = get_tezt_xx()
    regressor, yy = classification_LR(X_train, y_train, xx)
    print("===========running===========")
    figure1(X_train, y_train, xx, yy)

    print("===========running===========")
    X_train_poly2, xx_poly2 = polynominal_2(X_train, xx)
    regressor_poly2, yy_poly2 = classification_LR_2(X_train_poly2, y_train, xx_poly2)

    print("===========running===========")
    figure2(X_train, y_train, xx, yy, yy_poly2)


    print("===========running===========")
    X_train_poly4, xx_poly4 = polynominal_4(X_train, xx)
    regressor_poly4, yy_poly4 = classification_LR_4(X_train_poly4, y_train, xx_poly4)

    print("===========running===========")
    figure3(X_train, y_train, xx, yy, yy_poly2, yy_poly4)

    print("===========running===========")
    regressor_score_tezt(regressor)

    print("===========running===========")
    regressor_poly2_score_tezt(X_train, regressor_poly2)

    print("===========running===========")
    X_test_poly4, y_test = regressor_poly4_score_tezt(X_train, regressor_poly4)

    print("===========running===========")
    classification_Lasso(X_train_poly4, y_train, X_test_poly4, y_test)

    print("===========running===========")
    classification_Ridge(X_train_poly4, y_train, X_test_poly4, y_test)


    print("===========end===========")

if __name__ == '__main__':
    main()
