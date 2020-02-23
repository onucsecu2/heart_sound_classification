# import the pyplot and wavfile modules
import glob
import wave
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from scipy import signal as sc


dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_normal/Training B Normal/*.wav")]

df = pd.DataFrame.from_dict(dataset)
# print(df['path'])
X=[]
p=0
for x in df['path']:
    p=p+1
    spf = wave.open(x,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    max_amp=0
    g=0
    for i in signal:
        spf = wave.open(x, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')

        frames=spf.getnframes()
        print(frames)
        sos = sc.cheby2(10, 40, 17, btype='highpass', fs=4000, output='sos')
        filtered = sc.sosfilt(sos, signal)
        aplot=plot.plot(signal,c='b')
        print(filtered)
        # for i in range (0,frames):
        #     if(filtered[i]>0):
        #         filtered[i]=filtered[i]-800
        #     else:
        #         filtered[i] = filtered[i] + 800
        #sos = sc.cheby2(10, 40, 17, btype='lowpass', fs=1000, output='sos')
        filtered = sc.sosfilt(sos, signal)
        vplot=plot.plot(filtered,c='c')
        print(filtered)

        break
    if p==2:
        break



plot.show()