# import the pyplot and wavfile modules
import glob
import pandas as pd
import numpy as np

data=[]

cntrast = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean-std-var/contrast.csv")
mel_mean = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean/mel_mean.csv")
mel_std = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/std/mel_std.csv")
mel_var = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/var/mel_var.csv")
mfcc_std = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean/mfcc_std.csv")
mfcc_mean = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/std/mfcc_mean.csv")
mfcc_var = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/var/mfcc_var.csv")
actemp_mean = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/mean/actempogram_mean.csv")
actemp_std = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/std/actempogram_std.csv")
actemp_var = pd.read_csv("/home/onu/PycharmProjects/Heart/RAW_ft/var/actempogram_var.csv")

X1 = cntrast
X2 = mel_mean.drop('class', axis=1)
X3 = mel_std.drop('class', axis=1)
X4 = mel_var.drop('class', axis=1)
X5 = mfcc_mean.drop('class', axis=1)
X6 = mfcc_std.drop('class', axis=1)
X7 = mfcc_var.drop('class', axis=1)
X8 = actemp_mean.drop('class', axis=1)

X1=np.append(X1,X2,axis=1)
X1=np.append(X1,X3,axis=1)
X1=np.append(X1,X4,axis=1)
X1=np.append(X1,X5,axis=1)
X1=np.append(X1,X6,axis=1)
X1=np.append(X1,X7,axis=1)
X1=np.append(X1,X8,axis=1)

print(X1.shape)
# print(data.shape)
df = pd.DataFrame(X1)
export_csv = df.to_csv ('all.csv', index = False, header=True) #Don't forget to add '.csv' at the end of the path
