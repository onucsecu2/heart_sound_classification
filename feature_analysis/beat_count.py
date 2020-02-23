# import the pyplot and wavfile modules
import glob
import wave
import heartpy as hp
import noisereduce as nr
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from scipy import signal as sc
import numpy.fft as fft
def get_min_max_amp(signal):
    max_amp=0
    min_amp=0
    for i in signal:
        if(i>max_amp):
            max_amp=i
        if(i<min_amp):
            min_amp=i
    return max_amp,min_amp
def beat_cout(signal,threshold,max_amp):
    threshold_amp=max_amp-max_amp*threshold
    beat=0
    g=0
    for i in signal:
        g=g+1
        if(g<5):
            continue
        else:
            g=0
        if(threshold_amp<=i):
            beat=beat+1
    return beat
def local_maxima_plot(signal):
    peaks,_=sc.find_peaks(signal,distance=1000)
    peaksex,_=sc.find_peaks(signal,distance=2500)
    plot.plot(signal)
    plot.plot(peaks, signal[peaks], "x")
    plot.plot(peaksex, signal[peaksex], "v")

    # print(peaks.__len__(),beat,x)
    plot.show()

# def frinch_search_beat(signal,frames):
#     first=0
#     second=0
#     third=0
#     peaks=[]
#     for i in range(0,frames-4000):
#         for j in range(i,i+4000):
#

dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Atraining_murmur/Atraining_murmur/*.wav")]

df = pd.DataFrame.from_dict(dataset)
# print(df['path'])
X=[]
k=0
for x in df['path']:
    k=k+1
    spf = wave.open(x,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    frames=spf.getnframes()

    max_amp,min_amp=get_min_max_amp(signal)

    for i in range(0,frames):
        signal[i]=(min_amp-signal[i]/max_amp-min_amp)*1000

    beat=beat_cout(signal,0.25,max_amp)
    # frinch_search_beat(signal,frames)
    sos = sc.cheby2(10, 40, 17, btype='highpass', fs=4000, output='sos')
    filtered = sc.sosfilt(sos, signal)
    local_maxima_plot(signal)
    # spectrum=fft.fft(signal)
    # freq = fft.fftfreq(len(spectrum))
    #
    # threshold = 0.5 * max(abs(spectrum))
    # mask = abs(spectrum) > threshold
    # peaks = freq[mask]
    # plot.plot(freq, abs(spectrum))
    # plot.show()
    # plot.plot(signal)




