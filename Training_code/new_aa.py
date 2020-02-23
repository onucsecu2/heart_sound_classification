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
import pickle

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

flag=0
for i in bankdata['class']:
    if(i!=0):
        break
    flag+=1

from sklearn.feature_selection import SelectFromModel


scores = []
cv = KFold(n_splits=10, random_state=False, shuffle=True)
f1scoreSVM=[]
accuracySVM=[]
precisionSVM=[]
recallSVM=[]
f1scoreRF=[]
accuracyRF=[]
precisionRF=[]
recallRF=[]
count=0
def SVM_for_first(X_important_train, train, X_important_test):
    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(X_important_train, train)
    y_pred = svclassifier.predict(X_important_test)
    return y_pred


def RF_for_first(X_important_train, train, X_important_test):
    clf_important = RandomForestClassifier(n_estimators=100, random_state=0)
    clf_important.fit(X_important_train, train)
    y_pred=clf_important.predict(X_important_test)
    return y_pred
def intermediate_XX_YY(y_pred, X_test,X, y, test_index):
    pos = []
    r = y_pred.size
    for i in range(0, r):
        if (y_pred[i] == 1):
            pos = np.append(pos, i)
    x_test = X_test.iloc[pos, :]
    XX=X.iloc[flag:]
    yy=y.iloc[flag:]

    yy=np.array(y)
    XX=np.array(X)
    MEpos=[]
    MEposy=[]
    i=0
    for p in test_index:
        if(y_pred[i]>0):
            MEpos=np.append(MEpos,p-200)
            MEposy=np.append(MEposy,p)
        i=i+1

    XX=np.delete(XX, MEpos, 0)
    yytest=y[MEposy]
    yytest=np.array(yytest)
    yy=np.delete(yy,MEpos,0)
    clf2 = RandomForestClassifier(n_estimators=100,max_depth=500, random_state=0)
    clf2.fit(XX, yy)
    #selection of features
    sfm2 = SelectFromModel(clf2, max_features=100)
    sfm2.fit(XX, yy)

    X_important_train = sfm2.transform(XX)
    X_important_test = sfm2.transform(x_test)
    return XX,yy,yytest,x_test,X_important_train,X_important_test

def RF_for_second(X_important_train, yy,X_important_test,y_pred):
    clf_important2 = RandomForestClassifier(n_estimators=100, random_state=0)
    clf_important2.fit(X_important_train, yy)
    y_pred1=clf_important2.predict(X_important_test)
    actual_predict = y_pred
    s = actual_predict.size
    j = 0
    for i in range(0, s):
        if (actual_predict[i] == 1):
            actual_predict[i] = y_pred1[j]
            j = j + 1
    return actual_predict
def SVM_for_second(X_important_train, yy,X_important_test,y_pred):
    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(X_important_train, yy)
    y_pred2 = svclassifier.predict(X_important_test)

    actual_predict = y_pred

    s = actual_predict.size
    j = 0
    for i in range(0, s):
        if (actual_predict[i] == 1):
            actual_predict[i] = y_pred2[j]
            j = j + 1
    return actual_predict
def save_confusion_matrix(y_test,actual_predict,str):
    import scikitplot as skplt
    skplt.metrics.plot_confusion_matrix(
        y_test,
        actual_predict,
        figsize=(12, 12))
    # plt.show()
    plt.savefig(str)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    count=count+1
    print(count)
    # X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
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


    #   Random Forest
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100,max_depth=500, random_state=0)
    clf.fit(X_train, train)


    #selection of features
    sfm = SelectFromModel(clf, max_features=100)
    sfm.fit(X_train, train)

    # print(y_train.shape)
    X_important_train = sfm.transform(X_train)
    X_important_test = sfm.transform(X_test)

    # #         Random Forest

    y_pred_RF1 = RF_for_first(X_important_train, train, X_important_test)
    # clf_important = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf_important.fit(X_important_train, train)
    # y_pred=clf_important.predict(X_important_test)

    # #             SVM
    y_pred_SVM1=SVM_for_first(X_important_train,train,X_important_test)
    # svclassifier = svm.SVC(kernel='linear')
    # svclassifier.fit(X_important_train, train)
    # y_pred = svclassifier.predict(X_important_test)

    # pos = []
    # r = y_pred.size
    # for i in range(0, r):
    #     if (y_pred[i] == 1):
    #         pos = np.append(pos, i)
    # x_test = X_test.iloc[pos, :]
    # XX=X.iloc[flag:]
    # yy=y.iloc[flag:]
    #
    # yy=np.array(y)
    # XX=np.array(X)
    # MEpos=[]
    # MEposy=[]
    # i=0
    # for p in test_index:
    #     if(y_pred[i]>0):
    #         MEpos=np.append(MEpos,p-200)
    #         MEposy=np.append(MEposy,p)
    #     i=i+1
    #
    # XX=np.delete(XX, MEpos, 0)
    # yytest=y[MEposy]
    # yytest=np.array(yytest)
    # yy=np.delete(yy,MEpos,0)
    print("RF-RF")
    XX,yy,yytest,x_test,X_important_train,X_important_test=intermediate_XX_YY(y_pred_RF1, X_test,X, y, test_index)
    actual_predict=RF_for_second(X_important_train,yy,X_important_test,y_pred_SVM1)
    print(classification_report(y_test, actual_predict))
    save_confusion_matrix(y_test,actual_predict,"RF-RF")
    print("RF-SVM")
    XX,yy,yytest,x_test,X_important_train,X_important_test=intermediate_XX_YY(y_pred_RF1, X_test,X, y, test_index)
    actual_predict=SVM_for_second(X_important_train,yy,X_important_test,y_pred_SVM1)
    print(classification_report(y_test, actual_predict))
    save_confusion_matrix(y_test,actual_predict,"RF-SVM")
    print("SVM-RF")
    XX,yy,yytest,x_test,X_important_train,X_important_test=intermediate_XX_YY(y_pred_SVM1, X_test,X, y, test_index)
    actual_predict=RF_for_second(X_important_train,yy,X_important_test,y_pred_SVM1)
    print(classification_report(y_test, actual_predict))
    save_confusion_matrix(y_test,actual_predict,"SVM-RF")
    print("SVM-SVM")
    XX,yy,yytest,x_test,X_important_train,X_important_test=intermediate_XX_YY(y_pred_SVM1, X_test,X, y, test_index)
    actual_predict=SVM_for_second(X_important_train,yy,X_important_test,y_pred_SVM1)
    print(classification_report(y_test, actual_predict))
    save_confusion_matrix(y_test,actual_predict,"SVM-SVM")
    # clf2 = RandomForestClassifier(n_estimators=100,max_depth=500, random_state=0)
    # clf2.fit(XX, yy)
    # #selection of features
    # sfm2 = SelectFromModel(clf2, max_features=100)
    # sfm2.fit(XX, yy)
    #
    # X_important_train = sfm2.transform(XX)
    # X_important_test = sfm2.transform(x_test)
    # #             Random Forest again
    # clf_important2 = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf_important2.fit(X_important_train, yy)
    # y_pred1=clf_important2.predict(X_important_test)
    #
    # actual_predict = y_pred
    # s = actual_predict.size
    # j = 0
    # for i in range(0, s):
    #     if (actual_predict[i] == 1):
    #         actual_predict[i] = y_pred1[j]
    #         j = j + 1
#           SVM Again
#     actual_predict=SVM_for_second(X_important_train,yy,X_important_test,y_pred)
#     svclassifier = svm.SVC(kernel='linear')
#     svclassifier.fit(X_important_train, yy)
#     y_pred2 = svclassifier.predict(X_important_test)
#
#     actual_predict = y_pred
#
#     s = actual_predict.size
#     j = 0
#     for i in range(0, s):
#         if (actual_predict[i] == 1):
#             actual_predict[i] = y_pred2[j]
#             j = j + 1
#     print(classification_report(y_test, actual_predict))
#     accuracySVM.append(metrics.accuracy_score(y_test,actual_predict))

    break
print("its over :)")
# print(np.mean(accuracyRF),np.mean(accuracySVM))
