# coding:utf-8

# 从sklearn.datasets导入波士顿房价数据读取器。
from sklearn.datasets import load_boston
# 导入numpy并重命名为np。
import numpy as np


def get_boston():
    boston = load_boston()
    # 输出数据描述。
    print boston.DESCR
    X = boston.data
    y = boston.target
    # 分析回归目标值的差异。
    print "The max target value is", np.max(boston.target)
    print "The min target value is", np.min(boston.target)
    print "The average target value is", np.mean(boston.target)
    return X, y, boston


# 从sklearn.cross_validation导入数据分割器。
from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)
    return X_train, X_test, y_train, y_test


# 从sklearn.preprocessing导入数据标准化模块。
from sklearn.preprocessing import StandardScaler


def normalized(X_train, X_test, y_train, y_test):
    # 分别初始化对特征和目标值的标准化器。
    ss_X = StandardScaler()
    # 分别对训练和测试数据的特征以及目标值进行标准化处理。
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)

    ss_y = StandardScaler()

    # http://blog.csdn.net/llx1026/article/details/77940880
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))
    return X_train, X_test, y_train, y_test, ss_y


# 从sklearn.linear_model导入LinearRegression。
from sklearn.linear_model import LinearRegression
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估。
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def classfication_LR(X_train, y_train, X_test, y_test, ss_y):
    # 使用默认配置初始化线性回归器LinearRegression。
    lr = LinearRegression()
    # 使用训练数据进行参数估计。
    # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    # https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    lr.fit(X_train, y_train.ravel())
    # 对测试数据进行回归预测。
    lr_y_predict = lr.predict(X_test)
    # 使用LinearRegression模型自带的评估模块，并输出评估结果。
    print 'The value of default measurement of LinearRegression is', lr.score(X_test, y_test)
    # 使用r2_score模块，并输出评估结果。
    print 'The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict)
    # 使用mean_squared_error模块，并输出评估结果。
    print 'The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                              ss_y.inverse_transform(lr_y_predict))
    # 使用mean_absolute_error模块，并输出评估结果。
    print 'The mean absoluate error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                                 ss_y.inverse_transform(lr_y_predict))


# 从sklearn.linear_model导入SGDRegressor。
from sklearn.linear_model import SGDRegressor


def classfication_SGDR(X_train, y_train, X_test, y_test, ss_y):
    # 使用默认配置初始化线性回归器SGDRegressor。
    sgdr = SGDRegressor()
    # 使用训练数据进行参数估计。
    # https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    sgdr.fit(X_train, y_train.ravel())
    # 对测试数据进行回归预测。
    sgdr_y_predict = sgdr.predict(X_test)
    # 使用SGDRegressor模型自带的评估模块，并输出评估结果。
    print 'The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test)

    # 使用r2_score模块，并输出评估结果。
    print 'The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict)

    # 使用mean_squared_error模块，并输出评估结果。
    print 'The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                          ss_y.inverse_transform(sgdr_y_predict))

    # 使用mean_absolute_error模块，并输出评估结果。
    print 'The mean absoluate error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                             ss_y.inverse_transform(sgdr_y_predict))


# 从sklearn.svm中导入支持向量机（回归）模型。
from sklearn.svm import SVR


# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
# ='linear'  线性核函数配置
# ='poly'    多项式核函数配置
# ='rbf'     径向基核函数配置
def classfication_SVR(X_train, y_train, X_test, y_test, ss_y, kernel):
    svr = SVR(kernel)
    # https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    svr.fit(X_train, y_train.ravel())
    svr_y_predict = svr.predict(X_test)
    # 使用R-squared、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估。
    print 'R-squared value of ' + kernel + ' SVR is', svr.score(X_test, y_test)
    print 'The mean squared error of ' + kernel + '  SVR is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                                 ss_y.inverse_transform(svr_y_predict))
    print 'The mean absoluate error of ' + kernel + ' SVR is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                                   ss_y.inverse_transform(
                                                                                       svr_y_predict))


# 从sklearn.neighbors导入KNeighborRegressor（K近邻回归器）。
from sklearn.neighbors import KNeighborsRegressor


# 使得预测的方式为平均回归：weights='uniform'。
# 使得预测的方式为根据距离加权回归：weights='distance'。
def classfication_KNR(X_train, y_train, X_test, y_test, ss_y, weights):
    # 初始化K近邻回归器，并且调整配置
    knr = KNeighborsRegressor(weights)
    knr.fit(X_train, y_train)
    knr_y_predict = knr.predict(X_test)  # predict方法有问题
    # 使用R-squared、MSE以及MAE三种指标对根据距离加权回归配置的K近邻模型在测试集上进行性能评估。
    print 'R-squared value of ' + weights + '-weighted KNeighorRegression:', knr.score(X_test, y_test)
    print 'The mean squared error of ' + weights + '-weighted KNeighorRegression:', mean_squared_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(knr_y_predict))
    print 'The mean absoluate error of uniform-weighted KNeighorRegression', mean_absolute_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(knr_y_predict))  # 从sklearn.tree中导入DecisionTreeRegressor。


from sklearn.tree import DecisionTreeRegressor


def classfication_DTR(X_train, y_train, X_test, y_test, ss_y):
    # 使用默认配置初始化DecisionTreeRegressor。
    dtr = DecisionTreeRegressor()
    # 用波士顿房价的训练数据构建回归树。
    dtr.fit(X_train, y_train)
    # 使用默认配置的单一回归树对测试数据进行预测，并将预测值存储在变量dtr_y_predict中。
    dtr_y_predict = dtr.predict(X_test)
    # 使用R-squared、MSE以及MAE指标对默认配置的回归树在测试集上进行性能评估。
    print 'R-squared value of DecisionTreeRegressor: ', dtr.score(X_test, y_test)
    print 'The mean squared error of DecisionTreeRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                                  ss_y.inverse_transform(dtr_y_predict))
    print 'The mean absoluate error of DecisionTreeRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                                     ss_y.inverse_transform(
                                                                                         dtr_y_predict))


# 从sklearn.ensemble中导入RandomForestRegressor、ExtraTreesGressor以及GradientBoostingRegressor。
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


def classfication_RFR(X_train, y_train, X_test, y_test, ss_y):
    # 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量rfr_y_predict中。
    rfr = RandomForestRegressor()
    #   https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    #  DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    rfr.fit(X_train, y_train.ravel())
    rfr_y_predict = rfr.predict(X_test)
    # 使用R-squared、MSE以及MAE指标对默认配置的随机回归森林在测试集上进行性能评估。
    print 'R-squared value of RandomForestRegressor: ', rfr.score(X_test, y_test)
    print 'The mean squared error of RandomForestRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                                  ss_y.inverse_transform(rfr_y_predict))
    print 'The mean absoluate error of RandomForestRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                                     ss_y.inverse_transform(
                                                                                         rfr_y_predict))


def classfication_ETR(X_train, y_train, X_test, y_test, ss_y, boston):
    # 使用ExtraTreesRegressor训练模型，并对测试数据做出预测，结果存储在变量etr_y_predict中。
    etr = ExtraTreesRegressor()
    # https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    etr.fit(X_train, y_train.ravel())
    etr_y_predict = etr.predict(X_test)
    # 使用R-squared、MSE以及MAE指标对默认配置的极端回归森林在测试集上进行性能评估。
    print 'R-squared value of ExtraTreesRegessor: ', etr.score(X_test, y_test)
    print 'The mean squared error of  ExtraTreesRegessor: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                                ss_y.inverse_transform(etr_y_predict))
    print 'The mean absoluate error of ExtraTreesRegessor: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                                  ss_y.inverse_transform(etr_y_predict))
    # 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度。
    print np.sort(zip(etr.feature_importances_, boston.feature_names), axis=0)


def classfication_GBR(X_train, y_train, X_test, y_test, ss_y):
    # 使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中。
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train.ravel())
    gbr_y_predict = gbr.predict(X_test)
    # 使用R-squared、MSE以及MAE指标对默认配置的梯度提升回归树在测试集上进行性能评估。
    print 'R-squared value of GradientBoostingRegressor: ', gbr.score(X_test, y_test)
    print 'The mean squared error of GradientBoostingRegressor: ', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                                      ss_y.inverse_transform(
                                                                                          gbr_y_predict))
    print 'The mean absoluate error of GradientBoostingRegressor: ', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                                         ss_y.inverse_transform(
                                                                                             gbr_y_predict))


def main():
    print("===========start===========")
    X, y, boston = get_boston()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(X, y)
    # print(len(X_test)) # 测试集  总个数

    X_train, X_test, y_train, y_test, ss_y = normalized(X_train, X_test, y_train, y_test)
    classfication_LR(X_train, y_train, X_test, y_test, ss_y)

    print("===========running===========")
    classfication_SVR(X_train, y_train, X_test, y_test, ss_y, 'linear')

    print("===========running===========")
    classfication_SVR(X_train, y_train, X_test, y_test, ss_y, 'poly')

    print("===========running===========")
    classfication_SVR(X_train, y_train, X_test, y_test, ss_y, 'rbf')

    print("===========running===========")
    # classfication_KNR(X_train, y_train, X_test, y_test, ss_y, 'uniform')

    print("===========running===========")
    # classfication_KNR(X_train, y_train, X_test, y_test, ss_y, 'distance')

    print("===========running===========")
    classfication_DTR(X_train, y_train, X_test, y_test, ss_y)

    print("===========running===========")
    classfication_RFR(X_train, y_train, X_test, y_test, ss_y)

    print("===========running===========")
    classfication_ETR(X_train, y_train, X_test, y_test, ss_y, boston)

    print("===========running===========")
    classfication_GBR(X_train, y_train, X_test, y_test, ss_y)

    print("===========end===========")


if __name__ == '__main__':
    main()
