import numpy as np
import pandas as pd
from matplotlib.pyplot import hist
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt


# bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/ft_1.csv")
# X = bankdata.drop('class', axis=1)
# y = bankdata['class']
#

bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean-std-var/contrast.csv")
bankdata1 = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean-std-var/mel.csv")
bankdata1 = bankdata1.drop('class', axis=1)
bankdata2 = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean-std-var/mfcc.csv")
bankdata2 = bankdata2.drop('class', axis=1)
bankdata3 = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean-std-var/tempogram.csv")
bankdata3 = bankdata3.drop('class', axis=1)
bankdata=pd.concat([bankdata, bankdata1], axis=1)
bankdata=pd.concat([bankdata, bankdata2], axis=1)
bankdata=pd.concat([bankdata, bankdata3], axis=1)
X = bankdata.drop('class', axis=1)
y = bankdata['class']
from sklearn.feature_selection import SelectFromModel

# scores1 = []
# scores2 = []
scores = []
cv = KFold(n_splits=10, random_state=None, shuffle=True)
# sel = SelectFromModel(
#     PermutationImportance(RandomForestClassifier(), cv=5),
#     threshold=0.001,
# # ).fit(X, y)
# for train_index, test_index in cv.split(X):
#     X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
#
#     clf = RandomForestClassifier(n_estimators=100, random_state=0)
#     clf.fit(X_train, y_train)
#
#
#     #
#     # sfm = SelectFromModel(clf, threshold=0.001)
#     # sfm.fit(X_train, y_train)
#     #
#
#     # importances = clf.feature_importances_
#     # indices = np.argsort(importances)
#
#     # plt.title('Feature Importances')
#     # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#     # # plt.yticks(range(len(indices)), [features[i] for i in indices])
#     # plt.xlabel('Relative Importance')
#     # plt.show()
#     #
#     # X_important_train = sel.transform(X_train)
#     # X_important_test = sel.transform(X_test)
#
#     # print(X_important_train.shape)
#     print(X_train.shape)
#
#     clf_important = RandomForestClassifier(n_estimators=100, random_state=0)
#     # clf_important.fit(X_important_train, y_train)
#     # X_important_train = sfm.transform(X_train)
#     # X_important_test = sfm.transform(X_test)
#     # X_important_train = sel.transform(X_train)
#     # X_important_test = sel.transform(X_test)
#
#
#     accuracy = clf.score(X_test, y_test)
#     print('RandomForest :', accuracy)
#     # accuracy1 = clf_important.score(X_important_test, y_test)
#     # print('RandomForestFt :', accuracy1)
#
#     # svclassifier = svm.SVC(kernel='linear',verbose=0)
#     # svclassifier.fit(X_important_train, y_train)
#     # accuracy2 = svclassifier.score(X_important_test, y_test)
#     y_pred = clf.predict(X_test)
#     # cm=confusion_matrix(y_test, y_pred)
#     skplt.metrics.plot_confusion_matrix(
#         y_test,
#         y_pred,
#         figsize=(12, 12))
#     plt.show()
#     # print('SVMFt :', accuracy2)
#     scores.append(accuracy)
#     # scores1.append(accuracy1)
#     # scores2.append(accuracy2)
# print(np.mean(scores))
# print(np.mean(scores1))
# print(np.mean(scores2))


for train_index, test_index in cv.split(X):
    # print(train_index)
    # print(test_index)
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]


    test = np.array(y_test)
    train = np.array(y_train)
    r = test.size
    for i in range(0, r):
        if (test[i] == 2):
            test[i] = 1;
    r = train.size
    for i in range(0, r):
        if (train[i] == 2):
            train[i] = 1;

    #             SVM
    # svclassifier = svm.SVC(kernel='linear')
    # svclassifier.fit(X_train, train)
    # y_pred = svclassifier.predict(X_test)
    #   Random Forest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100,max_depth=500, random_state=0)
    clf.fit(X_train, train)
    y_pred = clf.predict(X_test)

    accuracy= metrics.accuracy_score(test, y_pred)
    print(confusion_matrix(test, y_pred))
    print("Accuracy(normal$extra-Murmur):", accuracy)
    pos = []
    r = y_pred.size
    for i in range(0, r):
        if (y_pred[i] == 1):
            pos = np.append(pos, i)
    x_test = X_test.iloc[pos, :]

    bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/features_ME.csv")
    XX = bankdata.drop('class', axis=1)
    yy = bankdata['class']
    yy=np.array(yy)
    XX=np.array(XX)
    MEpos=[]
    print(XX.shape)
    for p in test_index:
        if(p>199):
            MEpos=np.append(MEpos,p-199)
    # print(MEpos)
    XX=np.delete(XX,MEpos,0)
    yy=np.delete(yy,MEpos,0)

    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(XX, yy)
    y_pred1 = svclassifier.predict(x_test)
    actual_predict = y_pred
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    # clf.fit(XX, yy)
    # y_pred1 = clf.predict(x_test)
    # actual_predict = y_pred

    s = actual_predict.size
    j = 0
    for i in range(0, s):
        if (actual_predict[i] == 1):
            actual_predict[i] = y_pred1[j]
            j = j + 1



    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracy=metrics.accuracy_score(y_test, actual_predict)
    print("Accuracy:",accuracy)

    svclassifier.fit(X_train, y_train)
    scores.append(accuracy)
# sum=0
print("Mean Accuracy : ")
print(np.mean(scores))

'''X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25)

test=np.array(y_test)
train=np.array(y_train)
r=test.size
for i in range(0,r):
    if(test[i]==2):
        test[i]=1;
r=train.size
for i in range(0,r):
    if(train[i]==2):
        train[i]=1;


from sklearn import svm

from sklearn import metrics
svclassifier = svm.SVC( kernel='linear')
# svclassifier = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)

svclassifier.fit(X_train, train)

y_pred = svclassifier.predict(X_test)

print("Accuracy(normal$extra-Murmur):",metrics.accuracy_score(test, y_pred))

pos=[]
r=y_pred.size
for i in range(0,r):
    if(y_pred[i]==1):
        pos=np.append(pos,i)

print("my info :")

x_test=X_test.iloc[pos,:]

bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/train_ME_2.csv")
X = bankdata.drop('class', axis=1)
y =bankdata['class']



svclassifier = svm.SVC( kernel='linear')
svclassifier.fit(X, y)
y_pred1 = svclassifier.predict(x_test)


actual_predict=y_pred
s=actual_predict.size
j=0
for i in range(0,s):
    if(actual_predict[i]==1):
        actual_predict[i]=y_pred1[j]
        j=j+1



from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, actual_predict))


#
# scores = []
# cv = KFold(n_splits=5, random_state=None, shuffle=False)
# for train_index, test_index in cv.split(X):
#     print("Train Index: ", train_index)
#     print("Test Index: ", test_index, "\n")
#     X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
#     svclassifier.fit(X_train, y_train)
#     scores.append(svclassifier.score(X_test, y_test))
# sum=0
# print("accuracy : ")
# print(np.mean(scores))'''
