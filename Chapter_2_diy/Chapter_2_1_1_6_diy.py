# coding=utf-8
#  titanic.csv 中 sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征； 处理之后运行正常
import pandas as pd


def get_titantic():
    titanic = pd.read_csv('../Datasets/titanic.csv')

    # 十分重要的一环，特征的选择,sex, age, pclass这些都很有可能是决定幸免与否的关键因素
    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']

    # 对于缺失的年龄信息，我们使用全体乘客的平均年龄代替，这样可以在保证顺利训练模型的同时，尽可能不影响预测任务。
    X['age'].fillna(X['age'].mean(), inplace=True)
    return X, y


# 数据分割。
from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


# 对类别型特征进行转化，成为特征向量。
from sklearn.feature_extraction import DictVectorizer


def vectorizer(X_train, X_test):
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))
    return X_train, X_test


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def classfication_DT(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)  # 数据有问题
    y_predict = dtc.predict(X_test)
    print 'The accuracy of decision tree is', dtc.score(X_test, y_test)
    print classification_report(y_predict, y_test, target_names=['died', 'survived'])


# 使用随机森林分类器进行集成模型的训练以及预测分析。
from sklearn.ensemble import RandomForestClassifier


def classfication_RF(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_y_pred = rfc.predict(X_test)
    print 'The accuracy of decision tree is', rfc.score(X_test, y_test)
    print classification_report(rfc_y_pred, y_test, target_names=['died', 'survived'])


# 使用梯度提升决策树进行集成模型的训练以及预测分析。
from sklearn.ensemble import GradientBoostingClassifier


def classfication_GB(X_train, y_train, X_test, y_test):
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    gbc_y_pred = gbc.predict(X_test)
    print 'The accuracy of decision tree is', gbc.score(X_test, y_test)
    print classification_report(gbc_y_pred, y_test, target_names=['died', 'survived'])


def main():
    print("===========start===========")
    X, y = get_titantic()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(X, y)
    # print(len(X_test)) # 测试集  总个数 329
    classfication_DT(X_train, y_train, X_test, y_test)

    print("===========running===========")
    classfication_RF(X_train, y_train, X_test, y_test)
    print("===========running===========")

    classfication_GB(X_train, y_train, X_test, y_test)
    print("===========end===========")


if __name__ == '__main__':
    main()
