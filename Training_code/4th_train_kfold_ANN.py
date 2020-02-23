import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn import metrics


bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/features.csv")
X = bankdata.drop('class', axis=1)
y = bankdata['class']



scores = []
cv = KFold(n_splits=5, random_state=None, shuffle=True)
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
    from sklearn.preprocessing import LabelEncoder

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(train)
    encoded_Y = encoder.transform(train)
    # convert integers to dummy variables (i.e. one hot encoded)
    etrain = to_categorical(encoded_Y)
    encoded_Y = encoder.transform(test)
    # convert integers to dummy variables (i.e. one hot encoded)
    etest = to_categorical(encoded_Y)


    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(425, input_dim=425, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    from keras.callbacks import EarlyStopping

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    # Fit the model
    es = EarlyStopping(monitor='val_loss', min_delta=0.2, mode='min', verbose=1, patience=3)
    model.fit(X_train, etrain, epochs=100, batch_size=None, callbacks=[es], verbose=2)

    y_pred=model.predict_classes(X_test)
    accuracy= metrics.accuracy_score(test, y_pred)
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
    print(MEpos)
    XX=np.delete(XX,MEpos,0)
    yy=np.delete(yy,MEpos,0)

    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(XX, yy)
    y_pred1 = svclassifier.predict(x_test)
    actual_predict = y_pred
    s = actual_predict.size
    j = 0
    for i in range(0, s):
        if (actual_predict[i] == 1):
            actual_predict[i] = y_pred1[j]
            j = j + 1

    from sklearn.metrics import classification_report, confusion_matrix

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracy=metrics.accuracy_score(y_test, actual_predict)
    print("Accuracy:",accuracy)

    svclassifier.fit(X_train, y_train)
    scores.append(accuracy)
# sum=0
print("TotalAccuracy : ")
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
