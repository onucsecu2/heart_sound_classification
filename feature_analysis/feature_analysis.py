# import the pyplot and wavfile modules
import glob
import wave
import wavefile as wavefile
from scipy.io import wavfile
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from scipy import signal as sc
from scipy import signal
import librosa
import pywt
import heartpy as hp


def get_min_max_amp(signal):
    max_amp=0
    min_amp=0
    for i in signal:
        if(i>max_amp):
            max_amp=i
        if(i<min_amp):
            min_amp=i
    return max_amp,min_amp

def lowpassfilter(signal, thresh, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_murmur/Btraining_murmur/*.wav")]

df = pd.DataFrame.from_dict(dataset)
for x in df['path']:
    y, sr = librosa.load(x)
    S = np.abs(librosa.stft(y))
    #Normalize the amplitude
    max_amp, min_amp = get_min_max_amp(y)
    for i in range(0,len(y)):
        y[i]=(min_amp-y[i]/max_amp-min_amp)

    y = librosa.resample(y, sr, 4000)
    rec = lowpassfilter(y, 0.1)
    # rms = librosa.feature.rms(y=rec)
    # st=librosa.feature.stack_memory(rec)
    # S = np.abs(librosa.stft(rec))
    # contrast = librosa.feature.spectral_contrast(S=S)
    # cent = librosa.feature.spectral_centroid(y=rec, sr=4000)
    # rolloff = librosa.feature.spectral_rolloff(y=rec, sr=4000)
    # flatness = librosa.feature.spectral_flatness(y=rec)
    # plot.plot(flatness)
    # plot.show()
    # print(flatness)
    # print(flatness.shape)

    # np.diff(peaks)
    # enhanced = hp.enhance_peaks(rec, iterations=1)
    # plot.subplot(2, 1, 1)
    #
    plot.plot(y)
    plot.plot(rec)
    # plot.subplot(2, 1, 2)
    # plot.plot(enhanced)
    plot.show()

    # plot.plot(y)
    # plot.plot(rec)
    # plot.show()
    # (cA1, cD1) = pywt.dwt(rec, 'db4', 'smooth')
    # f, t, Sxx = signal.spectrogram(rec, fs=sr)
    # plot.pcolormesh(t, f, Sxx)


#
# df = pd.DataFrame(tmp)
# export_csv = df.to_csv ('dataset_melspec.csv', index = True, header=True) #Don't forget to add '.csv' at the end of the path

