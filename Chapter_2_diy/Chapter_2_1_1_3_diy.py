# coding=utf-8

from sklearn.datasets import fetch_20newsgroups


def get_news():
    news = fetch_20newsgroups(subset='all')
    print len(news.data)
    print news.data[0]
    return news


from sklearn.cross_validation import train_test_split


def get_train_tezt_X_Y(news):
    X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer


def vectorizer(X_train, X_test):
    vec = CountVectorizer()
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)
    return X_train, X_test


# 从sklearn.naive_bayes里导入朴素贝叶斯模型。
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def classfication_Lgs(X_train, y_train, X_test, y_test, news):
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_predict = mnb.predict(X_test)

    print 'The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test)
    print classification_report(y_test, y_predict, target_names=news.target_names)


def main():
    print("===========start===========")
    news = get_news()
    X_train, X_test, y_train, y_test = get_train_tezt_X_Y(news)
    X_train, X_test = vectorizer(X_train, X_test)
    classfication_Lgs(X_train, y_train, X_test, y_test, news)
    print("===========end===========")


if __name__ == '__main__':
    main()
