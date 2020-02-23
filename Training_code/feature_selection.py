import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics


bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/ft_1.csv")
X = bankdata.drop('class', axis=1)
y = bankdata['class']



scores = []
cv = KFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    # print(train_index)
    # print(test_index)
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                           oob_score=False, random_state=None, verbose=0,
                           warm_start=False)
    feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
    print(feature_imp[413])

    feature_imp.nlargest(5).plot(kind='barh')
    plt.show()
    break
