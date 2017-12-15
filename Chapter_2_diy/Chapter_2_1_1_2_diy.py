# coding=utf-8

from sklearn.datasets import load_digits


# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中。
def get_digits():
    digits = load_digits()
    print(digits.data.shape)
    return digits


from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(digits):
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test


from sklearn.preprocessing import StandardScaler


# 标准化数据，保证每个维度的特征数据方差为1，均值为0。
def normalized(X_train, X_test):
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    return X_train, X_test


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def classfication_SVC(X_train, y_train, X_test, y_test, digits):
    lsvc = LinearSVC()
    lsvc.fit(X_train, y_train)
    y_predict = lsvc.predict(X_test)
    print 'The Accuracy of Linear SVC is', lsvc.score(X_test, y_test)
    print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))


def main():
    print("===========start===========")
    digits = get_digits()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(digits)
    X_train, X_test = normalized(X_train, X_test)
    classfication_SVC(X_train, y_train, X_test, y_test, digits)
    print("===========end===========")


if __name__ == '__main__':
    main()
