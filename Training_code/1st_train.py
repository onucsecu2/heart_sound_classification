import numpy as np
import pandas as pd
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
bankdata = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean/ftempogram_mean.csv")
X = bankdata.drop('class', axis=1)
y = bankdata['class']


from sklearn.preprocessing import LabelEncoder
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)


X_train, X_test, y_train, y_test = train_test_split(X,dummy_y, test_size=0.2)

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(activation= 'logistic',early_stopping =True,max_iter=100,verbose=100,learning_rate_init =0.001)
# # clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
# #                      solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from keras.models import Sequential
from keras import layers
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#
nX_train=np.array(X_train)
nX_test=np.array(X_test)

nX_train = nX_train.reshape((nX_train.shape[0], nX_train.shape[1],1))
nX_test = nX_test.reshape((nX_test.shape[0], nX_test.shape[1],1))

print(nX_test.shape)

n_timesteps, n_features, n_outputs = nX_train.shape[1], nX_train.shape[2], y_train.shape[1]
print('thats : ',n_timesteps,n_features,n_outputs)
model = Sequential()
model.add(Conv1D(filters=20, kernel_size=2, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=10, kernel_size=2, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(nX_train, y_train, epochs=120,batch_size=50,verbose=1)
scores = model.evaluate(nX_train, y_train)
#
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


'''
model = Sequential()
model.add(Dense(425+128,input_dim=425+128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))
from keras.callbacks import EarlyStopping
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',min_delta=0.2, mode='min', verbose=1,patience=3)
model.fit(X_train, y_train, epochs=60,batch_size=None,callbacks=[es],verbose=2)
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))'''

data=[]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

from sklearn import svm
from sklearn import metrics
svclassifier = svm.SVC( kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)

# Bayesian
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB().fit(X_train, y_train)
# gnb_predictions = gnb.predict(X_test)
# accuracy = gnb.score(X_test, y_test)
# print ('BAyesian : ',accuracy)

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
# accuracy = knn.score(X_test, y_test)
# print ('KNN :',accuracy)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=None,verbose=1)
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print ('RandomForest :',accuracy)
# y_pred=clf.predict(X_test)
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# report = classification_report(y_test, y_pred, output_dict=True)
#


df = pd.DataFrame(report).transpose()

print(df)

export_csv = df.to_csv('report_SVM3.csv', index = True, header=True) #Don't forget to add '.csv' at the end of the path
