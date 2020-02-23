import numpy as np
import pandas as pd
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/features.csv")
X = bankdata.drop('class', axis=1)
y = bankdata['class']





from sklearn.preprocessing import LabelEncoder
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X,dummy_y, test_size=0.1)




# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(activation= 'logistic',early_stopping =True,max_iter=1000,verbose=100,learning_rate_init =0.0001)
# # clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
# #                      solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from keras.models import Sequential
from keras import layers
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#
# nX_train=np.array(X_train)
# nX_test=np.array(X_test)
#
# nX_train = nX_train.reshape((nX_train.shape[0], nX_train.shape[1],1))
# nX_test = nX_test.reshape((nX_test.shape[0], nX_test.shape[1],1))
#
# print(nX_test.shape)
#
# n_timesteps, n_features, n_outputs = nX_train.shape[1], nX_train.shape[2], y_train.shape[1]
# print('thats : ',n_timesteps,n_features,n_outputs)
# model = Sequential()
# model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(n_outputs, activation='softmax'))
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# model.fit(nX_train, y_train, epochs=150,batch_size=50,verbose=1)
# scores = model.evaluate(nX_train, y_train)
# #
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model = Sequential()
model.add(Dense(425,input_dim=425,activation='relu'))
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
model.add(Dense(3, activation='softmax'))
from keras.callbacks import EarlyStopping
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# Fit the model
es = EarlyStopping(monitor='val_loss',min_delta=0.2, mode='min', verbose=1,patience=3)
model.fit(X_train, y_train, epochs=100,batch_size=None,callbacks=[es],verbose=2)
pred=model.predict_classes(X_test)
print(pred)
scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25)

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

bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/train/features_ME.csv")
X = bankdata.drop('class', axis=1)
y =bankdata['class']



svclassifier = svm.SVC( kernel='linear')
svclassifier.fit(X, y)
y_pred1 = svclassifier.predict(x_test)

# model = Sequential()
# model.add(Dense(20,input_dim=20,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dropout(0.1))
# model.add(Dense(5, activation='sigmoid'))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid'))
#
# from keras.callbacks import EarlyStopping
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# # Fit the model
# es = EarlyStopping(monitor='val_loss',min_delta=0.2, mode='min', verbose=1,patience=3)
# model.fit(X, y, epochs=150,batch_size=None,callbacks=[es],verbose=2)
# scores = model.evaluate(X_test, y_test)
#
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# y_pred1 = model.predict(x_test)


# clf = MLPClassifier(activation= 'logistic',early_stopping =True,max_iter=1000,verbose=True,learning_rate_init =0.0001)
# # clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
# #                      solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
# clf.fit(X, y)
# y_pred1 = clf.predict(x_test)

# print(pos)
actual_predict=y_pred
# print(actual_predict)
s=actual_predict.size
j=0
for i in range(0,s):
    if(actual_predict[i]==1):
        actual_predict[i]=y_pred1[j]
        j=j+1



# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, actual_predict))


# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB().fit(X_train, y_train)
# gnb_predictions = gnb.predict(X_test)
#
# # accuracy on X_test
# accuracy = gnb.score(X_test, y_test)
# print ('BAyesian : ',accuracy)
#


# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
#
# # accuracy on X_test
# accuracy = knn.score(X_test, y_test)
# print ('KNN :',accuracy)
#


# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print ('RandomForest :',accuracy)