# import the pyplot and wavfile modules
import glob
import wave

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
    plot.show()

dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_normal/Training B Normal/*.wav")]

df = pd.DataFrame.from_dict(dataset)
X=[]
k=0
min_frame=99999999999
max_frame=0
for x in df['path']:
    k=k+1
    spf = wave.open(x,'r')
    spfM = wave.open('/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_murmur/Btraining_murmur/185_1308073325396_B.wav', 'r')
    tsignal = spf.readframes(-1)
    tsignalM = spfM.readframes(-1)
    tsignal = np.fromstring(tsignal, 'Int16')
    tsignalM = np.fromstring(tsignalM, 'Int16')
    signal=tsignal[:5000]
    signalM=tsignalM[:5000]
    frames=spf.getnframes()
    if(frames<5000):
        continue
    if(frames>max_frame):
        max_frame=frames
    if(min_frame>frames):
        min_frame=frames

    max_amp,min_amp=get_min_max_amp(signal)
    max_amp_m,min_amp_m=get_min_max_amp(signalM)

    for i in range(0,5000):
         signal[i]=(min_amp-signal[i]/max_amp-min_amp)*10000
         signalM[i]=(min_amp_m-signalM[i]/max_amp_m-min_amp_m)*10000

    # beat=beat_cout(signal,0.25,max_amp)
    sos = sc.cheby2(10, 40, 17, btype='lowpass', fs=65, output='sos')
    filtered = sc.sosfilt(sos, signal)
    plot.plot(signal,c='c')
    plot.plot(filtered)
    # spectrum=fft.fft(signal)
    # spectrumM=fft.fft(signalM)
    # freq = fft.fftfreq(len(spectrum))
    # freqM = fft.fftfreq(len(spectrumM))
    #
    # threshold = 0.3 * max(abs(spectrum))
    # thresholdM = 0.3 * max(abs(spectrumM))
    # mask = abs(spectrum) < threshold
    # maskM = abs(spectrumM) < thresholdM
    # peaks = freq[mask]
    # peaksM = freq[maskM]
    # plot.plot(freqM, spectrumM, c='g')
    # plot.plot(freq, spectrum)


    plot.show()

    # print(spf.getnframes(),spf.getframerate(),spf.getnframes()/spf.getframerate())
    # local_maxima_plot(signal)

print(min_frame,max_frame)



