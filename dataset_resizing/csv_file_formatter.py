import numpy as np
import pandas as pd
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

loaded = np.arange(384*21*600).reshape(600,384,21)
bankdata = pd.read_csv("dataset_ac_fourier_tempogram.csv")
X = bankdata.drop('class', axis=1)
y = bankdata['class']
X=np.array(X)
r,c=X.shape
print(r,c)
for i in range (0,r):
    for j in range (0,c):
        loaded[i][j][0]=X[i][j]

bankdata = pd.read_csv("dataset_mfcc.csv")
Y = bankdata.drop('class', axis=1)
y = bankdata['class']
Y=np.array(Y)
r,c=Y.shape
print(r,c)
for i in range (0,r):
    for j in range (0,c):
        loaded[i][0][j+1]=Y[i][j]

# df = pd.DataFrame(data)
#
# export_csv = df.to_csv ('1.csv', index = False, header=False) #Don't forget to add '.csv' at the end of the path
