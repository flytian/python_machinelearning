# coding=utf-8

from sklearn.datasets import load_iris


# 使用加载器读取数据并且存入变量iris。
def get_iris():
    iris = load_iris()
    print(iris.data.shape)
    # 查看数据说明。对于一名机器学习的实践者来讲，这是一个好习惯。
    print(iris.DESCR)
    return iris


from sklearn.cross_validation import train_test_split


# 从使用train_test_split，利用随机种子random_state采样25%的数据作为测试集。
def get_train_tezt_X_Y(iris):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


from sklearn.preprocessing import StandardScaler


def normalized(X_train, X_test):
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    return X_train, X_test


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# 使用K近邻分类器对测试数据进行类别预测
def classfication_KN(X_train, y_train, X_test, y_test, iris):
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    print 'The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test)
    print classification_report(y_test, y_predict, target_names=iris.target_names)


def main():
    print("===========start===========")
    iris = get_iris()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(iris)
    X_train, X_test = normalized(X_train, X_test)
    classfication_KN(X_train, y_train, X_test, y_test, iris)
    print("===========end===========")


if __name__ == '__main__':
    main()
