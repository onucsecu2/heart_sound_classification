import glob
import wave
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np

dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_normal/Training B Normal/*.wav")]

df = pd.DataFrame.from_dict(dataset)
X=[]
for x in df['path']:
    spf = wave.open(x,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    Time=np.linspace(0, len(signal)/spf.getframerate(), num=len(signal))
    X.append(Time[-1])
print(X)
plot.title('Time Length of Normal')
plot.plot(X)
plot.show()