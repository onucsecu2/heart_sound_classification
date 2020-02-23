# import the pyplot and wavfile modules
import glob
import wave
import heartpy as hp
import noisereduce as nr
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.io import wavfile
import librosa
'''
# Read the wav file (mono)
dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_normal/Training B Normal/*.wav")]

df = pd.DataFrame.from_dict(dataset)
df['x'] = df['path'].apply(lambda x: wavfile.read(x)[1])
normal = df[df['label'] == 'normal' ].sample(1)
plot.figure(1, figsize=(10,5))
plot.title('normal')
plot.plot(normal['x'].values[0], c='m')

# Plot the signal read from wav file
'''

sampling_rate,signalData = wavfile.read('/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_extrasystole/Btraining_extrastole/163_1307104470471_C.wav')
spf = wave.open('/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_normal/Training B Normal/103_1305031931979_D1.wav','r')
print(sampling_rate)

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
#Time=np.linspace(0, len(signal)/spf.getframerate(), num=len(signal))
#print(Time)
#plot.plot(Time,signal)
plot.subplot(212)
plot.title('Normal')
plot.plot(signalData)
plot.show()


#
# print(type(signalData))
#
# print(len(signalData))
#
# # X=np.array([[1,0]])
# max=31032
# cnt=0
# for x in signalData:
#     if(x>max):
#         cnt=cnt+1
#     # new=np.array([[x,1]])
#     # X=np.append(X, new,axis=0)
#     #print(new)
#     #print(new.shape)
# # print(X.shape)
# print(cnt)
# #X = X.reshape((X.shape[0], 1))
# X=X.reshape(-1,1)
# # print(X)
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# plot.scatter(X[:,0],X[:,0], c=kmeans.labels_, cmap='rainbow')
# print(kmeans.cluster_centers_)

# plot.xlabel('Sample')
#
# plot.ylabel('Amplitude')
# samplingFrequency, signalData = wavfile.read('/home/onu/Desktop/Thesis Work/Datasets/Pascal/mur1.wav')

# Plot the signal read from wav file


#plot.subplot(212)
#
# plot.specgram(signalData, Fs=samplingFrequency)
#
# plot.xlabel('Time')
#
# plot.ylabel('Frequency')
#
plot.show()