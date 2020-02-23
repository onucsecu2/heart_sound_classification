import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
bankdata = pd.read_csv("/home/onu/Desktop/machineLearning-master/Data/data.csv")
#X = bankdata.drop('buys_computer', axis=1)
X = bankdata
#y = bankdata['buys_computer']

#X['age']= pd.get_dummies(X['age'])
le = LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])
X['buys_computer']=le.fit_transform(X['buys_computer'])
from keras.models import Sequential
from keras.layers import Dense,Dropout
import keras
x=X.to_numpy()

score=[]
cv = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index)
    print("Test Index: ", test_index, "\n")

    X_train, X_test, y_train, y_test = x[train_index,0:4], x[test_index,4], x[train_index,0:4], x[test_index,4]
    print(X_test)
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='sigmoid'))
    #model.add(Dropout(0.6))
    model.add(Dense(5, activation='sigmoid'))
    #model.add(Dropout(0.6))
    model.add(Dense(5, activation='sigmoid'))
    #model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
    model.fit(X_train, y_train, epochs=50,batch_size=32)
    scores = model.evaluate(X_train, y_train)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    score.append(scores[1])
sum=0
print("accuracy : ")
print(np.mean(scores)*100)