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
def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
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

class_vector=[]
chroma_stft_mean=[]  # chroma stft
chroma_stft_std=[]
chroma_stft_var=[]
chroma_cq_mean=[]
chroma_cq_std=[]
chroma_cq_var=[]
chroma_cens_mean=[]
chroma_cens_std=[]
chroma_cens_var=[]
mel_mean=[]
mel_std=[]
mel_var=[]
rmse_mean=[]
rmse_std=[]
rmse_var=[]
f_tempo_mean=[]
f_tempo_std=[]
f_tempo_var=[]
temp_mean=[]
temp_std=[]
temp_var=[]
spec_bw_mean=[]
spec_bw_std=[]
spec_bw_var=[]
zcr_mean=[]
zcr_std=[]
zcr_var=[]
harm_mean=[]
harm_std=[]
harm_var=[]
for x in df['path']:
    y, sr = librosa.load(x)
    S = np.abs(librosa.stft(y))
    #Normalize the amplitude
    max_amp, min_amp = get_min_max_amp(y)
    for i in range(0,len(y)):
        y[i]=(min_amp-y[i]/max_amp-min_amp)


    rec = lowpassfilter(y, 0.4)
    # rec = lowpassfilter(rec, 0.9)
    # plot.plot(y)
    # plot.plot(rec)
    (cA1, cD1) = pywt.dwt(rec, 'db2', 'smooth')

    f, t, Sxx = signal.spectrogram(rec, fs=sr)
    plot.pcolormesh(t, f, Sxx)

    harmonic = librosa.effects.harmonic(rec)
    melspectrogram = librosa.feature.melspectrogram(y=rec, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=rec, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=rec, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=rec, sr=sr)
    fourier_tempogram=librosa.feature.fourier_tempogram(y=rec,sr=sr)
    tempogram=librosa.feature.tempogram(y=rec,sr=sr)
    spec_bw=librosa.feature.spectral_bandwidth(y=rec,sr=sr)
    rms=librosa.feature.rms(S=rec)
    zcr= librosa.feature.zero_crossing_rate(rec)

    class_vector=np.append(class_vector, '0')  # sname
    chroma_stft_mean=np.append(chroma_stft_mean, np.mean(chroma_stft))  # chroma stft
    chroma_stft_std=np.append(chroma_stft_std, np.std(chroma_stft))
    chroma_stft_var=np.append(chroma_stft_var, np.var(chroma_stft))
    chroma_cq_mean=np.append(chroma_cq_mean, np.mean(chroma_cq))  # chroma cq
    chroma_cq_std=np.append(chroma_cq_std, np.std(chroma_cq))
    chroma_cq_var=np.append(chroma_cq_var, np.var(chroma_cq))
    chroma_cens_mean=np.append(chroma_cens_mean, np.mean(chroma_cens))  # chroma cens
    chroma_cens_std=np.append(chroma_cens_std, np.std(chroma_cens))
    chroma_cens_var=np.append(chroma_cens_var, np.var(chroma_cens))
    mel_mean=np.append(mel_mean, np.mean(melspectrogram))  # melspectrogram
    mel_std=np.append(mel_std, np.std(melspectrogram))
    mel_var=np.append(mel_var, np.var(melspectrogram))
    spec_bw_mean = np.append(spec_bw_mean, np.mean(spec_bw))  # spectral bandwidth
    spec_bw_std=np.append(spec_bw_std, np.std(spec_bw))
    spec_bw_var=np.append(spec_bw_var, np.var(spec_bw))
    zcr_mean=np.append(zcr_mean, np.mean(zcr))  # zero crossing rate
    zcr_std=np.append(zcr_std, np.std(zcr))
    zcr_var=np.append(zcr_var, np.var(zcr))
    f_tempo_mean=np.append(f_tempo_mean,np.mean(fourier_tempogram))
    f_tempo_std=np.append(f_tempo_std,np.std(fourier_tempogram))
    f_tempo_var=np.append(f_tempo_var,np.var(fourier_tempogram))
    temp_mean=np.append(temp_mean,np.mean(tempogram))
    temp_std=np.append(temp_std,np.std(tempogram))
    temp_var=np.append(temp_var,np.var(tempogram))
    harm_mean=np.append(harm_mean, np.mean(harmonic))  # harmonic
    harm_std=np.append(harm_std, np.std(harmonic))
    harm_var=np.append(harm_var, np.var(harmonic))
    print(x)
dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_murmur/Btraining_murmur/*.wav")]

for x in df['path']:
    y, sr = librosa.load(x)
    S = np.abs(librosa.stft(y))
    # Normalize the amplitude
    max_amp, min_amp = get_min_max_amp(y)
    for i in range(0, len(y)):
        y[i] = (min_amp - y[i] / max_amp - min_amp)

    rec = lowpassfilter(y, 0.4)
    # rec = lowpassfilter(rec, 0.9)
    # plot.plot(y)
    # plot.plot(rec)
    (cA1, cD1) = pywt.dwt(rec, 'db2', 'smooth')

    f, t, Sxx = signal.spectrogram(rec, fs=sr)
    plot.pcolormesh(t, f, Sxx)

    harmonic = librosa.effects.harmonic(rec)
    melspectrogram = librosa.feature.melspectrogram(y=rec, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=rec, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=rec, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=rec, sr=sr)
    fourier_tempogram = librosa.feature.fourier_tempogram(y=rec, sr=sr)
    tempogram = librosa.feature.tempogram(y=rec, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=rec, sr=sr)
    rms = librosa.feature.rms(S=rec)
    zcr = librosa.feature.zero_crossing_rate(rec)

    class_vector = np.append(class_vector, '1')  # sname
    chroma_stft_mean = np.append(chroma_stft_mean, np.mean(chroma_stft))  # chroma stft
    chroma_stft_std = np.append(chroma_stft_std, np.std(chroma_stft))
    chroma_stft_var = np.append(chroma_stft_var, np.var(chroma_stft))
    chroma_cq_mean = np.append(chroma_cq_mean, np.mean(chroma_cq))  # chroma cq
    chroma_cq_std = np.append(chroma_cq_std, np.std(chroma_cq))
    chroma_cq_var = np.append(chroma_cq_var, np.var(chroma_cq))
    chroma_cens_mean = np.append(chroma_cens_mean, np.mean(chroma_cens))  # chroma cens
    chroma_cens_std = np.append(chroma_cens_std, np.std(chroma_cens))
    chroma_cens_var = np.append(chroma_cens_var, np.var(chroma_cens))
    mel_mean = np.append(mel_mean, np.mean(melspectrogram))  # melspectrogram
    mel_std = np.append(mel_std, np.std(melspectrogram))
    mel_var = np.append(mel_var, np.var(melspectrogram))
    spec_bw_mean = np.append(spec_bw_mean, np.mean(spec_bw))  # spectral bandwidth
    spec_bw_std = np.append(spec_bw_std, np.std(spec_bw))
    spec_bw_var = np.append(spec_bw_var, np.var(spec_bw))
    zcr_mean = np.append(zcr_mean, np.mean(zcr))  # zero crossing rate
    zcr_std = np.append(zcr_std, np.std(zcr))
    zcr_var = np.append(zcr_var, np.var(zcr))
    f_tempo_mean = np.append(f_tempo_mean, np.mean(fourier_tempogram))
    f_tempo_std = np.append(f_tempo_std, np.std(fourier_tempogram))
    f_tempo_var = np.append(f_tempo_var, np.var(fourier_tempogram))
    temp_mean = np.append(temp_mean, np.mean(tempogram))
    temp_std = np.append(temp_std, np.std(tempogram))
    temp_var = np.append(temp_var, np.var(tempogram))
    harm_mean=np.append(harm_mean, np.mean(harmonic))  # harmonic
    harm_std=np.append(harm_std, np.std(harmonic))
    harm_var=np.append(harm_var, np.var(harmonic))
    print(x)

Datas = {'class': class_vector,
         'chroma_stft_mean':  chroma_stft_mean,
         'chroma_stft_std': chroma_stft_std,
         'chroma_stft_var': chroma_stft_var,
         'chroma_cq_mean':chroma_cq_mean,
         'chroma_cq_std':chroma_cq_std,
         'chroma_cq_var':chroma_cq_var,
         'chroma_cens_mean':chroma_cens_mean,
         'chroma_cens_std':chroma_cens_std,
         'chroma_cens_var':chroma_cens_var,
         'mel_mean':mel_mean,
         'mel_std':mel_std,
         'mel_var':mel_var,
         'spec_bw_mean':spec_bw_mean,
         'spec_bw_std':spec_bw_std,
         'spec_bw_var':spec_bw_var,
         'zcr_mean':zcr_mean,
         'zcr_std':zcr_std,
         'zcr_var':zcr_var,
         'harm_mean':harm_mean,
         'harm_std':harm_std,
         'harm_var':harm_var,
         'f_tempo_mean':f_tempo_mean,
         'f_tempo_std':f_tempo_std,
         'f_tempo_var':f_tempo_var,
         'temp_mean':temp_mean,
         'temp_std':temp_std,
         'temp_var':temp_var
        }

df = pd.DataFrame(Datas, columns= ['name','chroma_stft_mean', 'chroma_stft_std','chroma_stft_var','chroma_cq_mean','chroma_cq_std','chroma_cq_var','chroma_cens_mean','chroma_cens_std','chroma_cens_var','mel_mean','mel_std','mel_var','spec_bw_mean','spec_bw_std','spec_bw_var','zcr_mean','zcr_std','zcr_var','harm_mean','harm_std','harm_var','f_tempo_mean','f_tempo_std','f_tempo_var','temp_mean','temp_std','temp_var'])

export_csv = df.to_csv ('dataset_27.csv', index = True, header=True) #Don't forget to add '.csv' at the end of the path
