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


def get_min_max_amp(signal):
    max_amp=0
    min_amp=0
    for i in signal:
        if(i>max_amp):
            max_amp=i
        if(i<min_amp):
            min_amp=i
    return max_amp,min_amp

dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_normal/Training B Normal/*.wav")]

df = pd.DataFrame.from_dict(dataset)
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


data = np.arange(21).reshape(1,21)
k=0


def plot_stftcontrast(rec,mfcc,S):
    import librosa.display
    plot.plot(rec)
    plot.figure()
    plot.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S,ref = np.max),y_axis = 'log')
    plot.colorbar(format='%+2.0f dB')
    plot.title('Power spectrogram')
    plot.subplot(2, 1, 2)
    librosa.display.specshow(mfcc, x_axis='time')
    plot.colorbar()
    plot.ylabel('Frequency bands')
    plot.title('Spectral contrast')
    plot.tight_layout()
    plot.show()
def plot_tempogram(rec,mfcc,oenv,sr):
    import librosa.display
    plot.plot(rec)
    plot.figure(figsize=(8, 8))
    plot.subplot(4, 1, 1)
    plot.plot(oenv, label='Onset strength')
    plot.xticks([])
    plot.legend(frameon=True)
    plot.axis('tight')
    plot.subplot(4, 1, 2)
    # We'll truncate the display to a narrower range of tempi
    librosa.display.specshow(mfcc, sr=sr, hop_length=512,x_axis = 'time', y_axis = 'tempo')
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,hop_length = 512)[0]
    plot.axhline(tempo, color='w', linestyle='--', alpha=1,label='Estimated tempo={:g}'.format(tempo))
    plot.legend(frameon=True, framealpha=0.75)
    plot.show()
def plot_ftempogram(rec,S,sr,oenv):
    import librosa.display
    plot.plot(rec)
    plot.figure(figsize=(8, 8))
    plot.subplot(3, 1, 1)
    plot.plot(oenv, label='Onset strength')
    plot.xticks([])
    plot.legend(frameon=True)
    plot.axis('tight')
    plot.subplot(3, 1, 2)
    librosa.display.specshow(S, sr=sr, hop_length=512, x_axis = 'time', y_axis = 'fourier_tempo', cmap = 'magma')
    plot.title('Fourier tempogram')
    plot.subplot(3, 1, 3)
    ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length = 512, norm = None)
    librosa.display.specshow(ac_tempogram, sr=sr, hop_length=512, x_axis = 'time', y_axis = 'tempo', cmap = 'magma')
    plot.title('Autocorrelation tempogram')
    plot.tight_layout()
    plot.show()
def plot_mfcc(rec,mfcc):
    import librosa.display
    plot.plot(rec)
    plot.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plot.colorbar()
    plot.title('MFCC-normal')
    plot.tight_layout()
    plot.show()
def plot_melspectogram(rec,mfcc):
    import librosa.display
    plot.plot(rec)
    plot.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(mfcc, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',y_axis = 'mel', sr = 16000,fmax = 8000)
    plot.colorbar(format='%+2.0f dB')
    plot.title('Mel-frequency spectrogram')
    plot.tight_layout()
    plot.show()

# for x in df['path']:
#     k=k+1
#     y, sr = librosa.load(x)
#     S = np.abs(librosa.stft(y))
#     #Normalize the amplitude
#     max_amp, min_amp = get_min_max_amp(y)
#     for i in range(0,len(y)):
#         y[i]=(min_amp-y[i]/max_amp-min_amp)
#     # y = librosa.resample(y, sr, 16000)
#     rec = lowpassfilter(y, 0.1)
#     oenv = librosa.onset.onset_strength(y=rec, sr=sr, hop_length=512)
#     S = np.abs(librosa.stft(rec))
#     # mfcc = librosa.feature.spectral_contrast(S=S)
#     # mfcc = librosa.feature.tempogram(onset_envelope=oenv, sr=16000,hop_length = 512, norm = None)
#     # mfcc = librosa.feature.fourier_tempogram(y=rec, sr=16000)
#     # mfcc = librosa.feature.chroma_stft(y=rec, sr=16000)
#     # mfcc = librosa.feature.mfcc(y=rec, sr=16000)
#     # mfcc = librosa.feature.zero_crossing_rate(y=rec)
#     # mfcc = librosa.feature.melspectrogram(y=rec, sr=sr, hop_length=256)
#
#     tmp=[]
#     tmp=np.append(tmp,'0')
#     r,c=mfcc.shape
#     #plotting function
#
#     #statistical analysis
#     count=0
#     for i in range(0,r):
#         mean=np.mean(mfcc[i])
#         std=np.std(mfcc[i])
#         var=np.var(mfcc[i])
#         # tmp=np.append(tmp,mean)
#         # tmp=np.append(tmp,std)
#         tmp=np.append(tmp,var)
#     print(x)
#     tmp=tmp.reshape(1,1*r+1)
#     data=np.append(data,tmp,axis=0)
#
#
# dataset = [{'path': path, 'label':'murmur'}
#                for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_murmur/Btraining_murmur/*.wav")]

for x in df['path']:
    y, sr = librosa.load(x)
    S = np.abs(librosa.stft(y))
    # Normalize the amplitude
    max_amp, min_amp = get_min_max_amp(y)
    for i in range(0, len(y)):
        y[i] = (min_amp - y[i] / max_amp - min_amp)
    rec = lowpassfilter(y, 0.4)
    oenv = librosa.onset.onset_strength(y=rec, sr=sr, hop_length=512)
    S = np.abs(librosa.stft(rec))
    # mfcc = librosa.feature.spectral_contrast(S=S)

    mfcc = librosa.feature.tempogram(onset_envelope=oenv, sr=16000,hop_length = 512, norm = None)

    # mfcc = librosa.feature.fourier_tempogram(y=rec, sr=16000)

    # mfcc = librosa.feature.chroma_stft(y=rec, sr=16000)
    # mfcc = librosa.feature.mfcc(y=rec, sr=16000)

    # mfcc = librosa.feature.zero_crossing_rate(y=rec)
    # mfcc = librosa.feature.melspectrogram(y=rec, sr=sr, hop_length=512)  #actually its mel
    # mfcc=mfcc.T
    tmp=[]
    tmp=np.append(tmp,'1')
    r,c=mfcc.shape
    for i in range(0,r):
        mean=np.mean(mfcc[i])
        std=np.std(mfcc[i])
        var=np.var(mfcc[i])
        # tmp=np.append(tmp,mean)
        # tmp=np.append(tmp,std)
        tmp=np.append(tmp,var)

    tmp=tmp.reshape(1,1*r+1)
    data=np.append(data,tmp,axis=0)
    print(x)
print(data.shape)
dataset = [{'path': path, 'label':'extrasys'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_extrasystole/Btraining_extrastole/*.wav")]

# for x in df['path']:
#     y, sr = librosa.load(x)
#     S = np.abs(librosa.stft(y))
#     # Normalize the amplitude
#     max_amp, min_amp = get_min_max_amp(y)
#     for i in range(0, len(y)):
#         y[i] = (min_amp - y[i] / max_amp - min_amp)
#     rec = lowpassfilter(y, 0.1)
#
#     oenv = librosa.onset.onset_strength(y=rec, sr=sr, hop_length=512)
#     S = np.abs(librosa.stft(rec))
#     mfcc = librosa.feature.spectral_contrast(S=S)
#     mfcc = librosa.feature.tempogram(onset_envelope=oenv, sr=16000,hop_length = 512, norm = None)
#     mfcc = librosa.feature.fourier_tempogram(y=rec, sr=16000)
#     # mfcc = librosa.feature.chroma_stft(y=rec, sr=16000)
#     mfcc = librosa.feature.mfcc(y=rec, sr=16000)
#     # mfcc = librosa.feature.zero_crossing_rate(y=rec)
#     mfcc = librosa.feature.melspectrogram(y=rec, sr=sr, hop_length=512)  #actually its mel
#     # mfcc=mfcc.T
#     tmp=[]
#     tmp=np.append(tmp,'2')
#     r,c=mfcc.shape
#     for i in range(0,r):
#         mean=np.mean(mfcc[i])
#         std=np.std(mfcc[i])
#         var=np.var(mfcc[i])
#         # tmp=np.append(tmp,mean)
#         # tmp=np.append(tmp,std)
#         tmp=np.append(tmp,var)
#
#     tmp=tmp.reshape(1,1*r+1)
#     data=np.append(data,tmp,axis=0)
#     print(x)
#
# df = pd.DataFrame(data)
#
# export_csv = df.to_csv ('ftempogram_var.csv', index = False, header=False) #Don't forget to add '.csv' at the end of the path'''
