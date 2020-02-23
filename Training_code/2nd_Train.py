import numpy as np
import pandas as pd
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split



# bankdata = pd.read_csv("all.csv")
# X = bankdata.drop('class', axis=1)
# y = bankdata['class']
loaded = np.arange(384*21*600).reshape(600,384,21)
bankdata = pd.read_csv("dataset_ac_fourier_tempogram.csv")
X = bankdata.drop('class', axis=1)
y = bankdata['class']
X=np.array(X)
r,c=X.shape
for i in range (0,r):
    for j in range (0,c):
        loaded[i][j][0]=X[i][j]

bankdata = pd.read_csv("dataset_mfcc.csv")
Y = bankdata.drop('class', axis=1)
Y=np.array(Y)
r,c=Y.shape
for i in range (0,r):
    for j in range (0,c):
        loaded[i][0][j+1]=Y[i][j]



from sklearn.preprocessing import LabelEncoder
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)
X_train, X_test, y_train, y_test = train_test_split(loaded,dummy_y, test_size=0.1)

print(X_train.shape)


from keras.models import Sequential
from keras import layers
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
print('thats : ',n_timesteps,n_features,n_outputs)
model = Sequential()
model.add(Conv1D(filters=384, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(404, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15,batch_size=50,verbose=1)
scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))