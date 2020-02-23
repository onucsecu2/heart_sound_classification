import numpy as np
import pandas as pd

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
# bankdata=np.append(bankdata,bankdata1,axis=1)
# bankdata=np.append(bankdata,bankdata2,axis=1)
# bankdata=np.append(bankdata,bankdata3,axis=1)
# X = bankdata.drop('class', axis=1)
# y = bankdata['class']
print(bankdata.shape)