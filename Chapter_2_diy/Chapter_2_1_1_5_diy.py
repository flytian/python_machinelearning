# coding=utf-8
#  titanic.csv 中 sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征； 处理之后运行正常
import pandas as pd


def get_titantic():
    titanic = pd.read_csv('../Datasets/titanic.csv')
    titanic.head()
    # 使用info()，查看数据的统计特性
    titanic.info()
    print("===========titanic_info===========")
    # 十分重要的一环，特征的选择,sex, age, pclass这些都很有可能是决定幸免与否的关键因素
    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    # 对当前选择的特征进行探查。
    X.info()
    print("===========X_info===========")
    # 借由上面的输出，我们设计如下几个数据处理的任务：
    # 1) age这个数据列，只有633个，需要补完。
    # 2) sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替。

    # 首先我们补充age里的数据，使用 平均数或者中位数 都是对模型偏离  造成最小影响  的策略。
    X['age'].fillna(X['age'].mean(), inplace=True)
    # 对补完的数据重新探查。
    X.info()
    # 由此得知，age特征得到了补完。
    return X, y


# 数据分割。
from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def classfication_DT(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier()

    dtc.fit(X_train, y_train)  # 数据有问题
    y_predict = dtc.predict(X_test)
    print dtc.score(X_test, y_test)
    print classification_report(y_predict, y_test, target_names=['died', 'survived'])


def main():
    print("===========start===========")
    X, y = get_titantic()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(X, y)
    classfication_DT(X_train, y_train, X_test, y_test)
    print("===========end===========")


if __name__ == '__main__':
    main()
