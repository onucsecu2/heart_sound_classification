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
# def local_maxima_plot(signal):
#     peaks,_=sc.find_peaks(signal,distance=1000)
#     peaksex,_=sc.find_peaks(signal,distance=2500)
#     plot.plot(signal)
#     plot.plot(peaks, signal[peaks], "x")
#     plot.plot(peaksex, signal[peaksex], "v")
#     plot.show()

dataset = [{'path': path, 'label':'normal'}
               for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_murmur/Btraining_murmur/*.wav")]


df = pd.DataFrame.from_dict(dataset)
X=[]
k=0
id=1

# name_vector=[]
# tempo_vector=[]
# total_beats=[]
# average_beats=[]
# chroma_stft_mean=[]  # chroma stft
# chroma_stft_std=[]
# chroma_stft_var=[]
# chroma_cq_mean=[]
# chroma_cq_std=[]
# chroma_cq_var=[]
# chroma_cens_mean=[]
# chroma_cens_std=[]
# chroma_cens_var=[]
# mel_mean=[]
# mel_std=[]
# mel_var=[]
# mfcc_mean=[]
# mfcc_std=[]
# mfcc_var=[]
# mfcc_delta_mean=[]
# mfcc_delta_std=[]
# mfcc_delta_var=[]
# # rmse_mean=[]
# # rmse_std=[]
# # rmse_var=[]
# cent_mean=[]
# cent_std=[]
# cent_var=[]
# spec_bw_mean=[]
# spec_bw_std=[]
# spec_bw_var=[]
# contrast_mean=[]
# contrast_std=[]
# contrast_var=[]
# rolloff_mean=[]
# rolloff_std=[]
# rolloff_var=[]
# poly_mean=[]
# poly_std=[]
# poly_var=[]
# tonnetz_mean=[]
# tonnetz_std=[]
# tonnetz_var=[]
# zcr_mean=[]
# zcr_std=[]
# zcr_var=[]
# harm_mean=[]
# harm_std=[]
# harm_var=[]
# perc_mean=[]
# perc_std=[]
# perc_var=[]
# frame_mean=[]
# frame_std=[]
# frame_var=[]
# feature_set = []
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
for x in df['path']:
    y, sr = librosa.load(x)
    S = np.abs(librosa.stft(y))
    #Normalize the amplitude
    max_amp, min_amp = get_min_max_amp(y)
    for i in range(0,len(y)):
        y[i]=(min_amp-y[i]/max_amp-min_amp)

    # (cA1, cD1) = pywt.dwt(y, 'db2', 'smooth')

    #butterworth bandpass otota valo denoise korte parce na...
    # b, a = butter_bandpass(50, 150,  sr, order=4)
    # bp_sig = signal.lfilter(b, a, y)
    # plot.plot(bp_sig,c='c')
    rec = lowpassfilter(y, 0.4)
    # rec = lowpassfilter(rec, 0.9)
    plot.plot(y)
    plot.plot(rec)


    plot.show()

    # print(coeffs2)



    # #applying butterworth Lowpass filter
    # fc = 150  # Cut-off frequency of the filter
    # w = fc / (sr / 2)  # Normalize the frequency
    # b, a = signal.butter(5, w, 'low')
    # output = signal.filtfilt(b, a, y)
    # plot.plot(y)
    # plot.plot(output, label='filtered',c='c')
    #applying IIRNotch
    # q=30.0  #Quality factor
    # b1, a1 = signal.iirnotch(111, q, sr)
    # output1 = signal.filtfilt(b1, a1, y)
    # plot.plot(output1, label='filtered', c='r')
    # finding peak
    # peaks, _ = signal.find_peaks(output,height=0.40)
    # plot.plot(peaks, y[peaks], "x")
    #
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    # chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    # chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    # melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # # rmse = librosa.feature.rmse(y=y)
    # cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # poly_features = librosa.feature.poly_features(S=S, sr=sr)
    # tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    # zcr = librosa.feature.zero_crossing_rate(y)
    # harmonic = librosa.effects.harmonic(y)
    # percussive = librosa.effects.percussive(y)
    #
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # mfcc_delta = librosa.feature.delta(mfcc)
    #
    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
    #
    # # Transforming Features
    # name_vector=np.append(name_vector, 'normal')  # sname
    # chroma_stft_mean=np.append(chroma_stft_mean, np.mean(chroma_stft))  # chroma stft
    # chroma_stft_std=np.append(chroma_stft_std, np.std(chroma_stft))
    # chroma_stft_var=np.append(chroma_stft_var, np.var(chroma_stft))
    # chroma_cq_mean=np.append(chroma_cq_mean, np.mean(chroma_cq))  # chroma cq
    # chroma_cq_std=np.append(chroma_cq_std, np.std(chroma_cq))
    # chroma_cq_var=np.append(chroma_cq_var, np.var(chroma_cq))
    # chroma_cens_mean=np.append(chroma_cens_mean, np.mean(chroma_cens))  # chroma cens
    # chroma_cens_std=np.append(chroma_cens_std, np.std(chroma_cens))
    # chroma_cens_var=np.append(chroma_cens_var, np.var(chroma_cens))
    # mel_mean=np.append(mel_mean, np.mean(melspectrogram))  # melspectrogram
    # mel_std=np.append(mel_std, np.std(melspectrogram))
    # mel_var=np.append(mel_var, np.var(melspectrogram))
    # mfcc_mean=np.append(mfcc_mean, np.mean(mfcc))  # mfcc
    # mfcc_std=np.append(mfcc_std, np.std(mfcc))
    # mfcc_var=np.append(mfcc_var, np.var(mfcc))
    # mfcc_delta_mean=np.append(mfcc_delta, np.mean(mfcc_delta))  # mfcc delta
    # mfcc_delta_std=np.append(mfcc_delta_std, np.std(mfcc_delta))
    # mfcc_delta_var=np.append(mfcc_delta_var, np.var(mfcc_delta))
    # # rmse_mean=np.append(id, np.mean(rmse))  # rmse
    # # rmse_std=np.append(id, np.std(rmse))
    # # rmse_var=np.append(id, np.var(rmse))
    # cent_mean=np.append(cent_mean, np.mean(cent))  # cent
    # cent_std=np.append(cent_std, np.std(cent))
    # cent_var=np.append(cent_var, np.var(cent))
    # spec_bw_mean=np.append(spec_bw_mean, np.mean(spec_bw))  # spectral bandwidth
    # spec_bw_std=np.append(spec_bw_std, np.std(spec_bw))
    # spec_bw_var=np.append(spec_bw_var, np.var(spec_bw))
    # contrast_mean=np.append(contrast_mean, np.mean(contrast))  # contrast
    # contrast_std=np.append(contrast_std, np.std(contrast))
    # contrast_var=np.append( contrast_var, np.var(contrast))
    # rolloff_mean=np.append(rolloff_mean, np.mean(rolloff))  # rolloff
    # rolloff_std=np.append(rolloff_std, np.std(rolloff))
    # rolloff_var=np.append(rolloff_var, np.var(rolloff))
    # poly_mean=np.append(poly_mean, np.mean(poly_features))  # poly features
    # poly_std=np.append(poly_std, np.std(poly_features))
    # poly_var=np.append(poly_var, np.var(poly_features))
    # tonnetz_mean=np.append( tonnetz_mean, np.mean(tonnetz))  # tonnetz
    # tonnetz_std=np.append(tonnetz_std, np.std(tonnetz))
    # tonnetz_var=np.append(tonnetz_var, np.var(tonnetz))
    # zcr_mean=np.append(zcr_mean, np.mean(zcr))  # zero crossing rate
    # zcr_std=np.append(zcr_std, np.std(zcr))
    # zcr_var=np.append(zcr_var, np.var(zcr))
    # harm_mean=np.append(harm_mean, np.mean(harmonic))  # harmonic
    # harm_std=np.append(harm_std, np.std(harmonic))
    # harm_var=np.append(harm_var, np.var(harmonic))
    # perc_mean=np.append(perc_mean, np.mean(percussive))  # percussive
    # perc_std=np.append(perc_std, np.std(percussive))
    # perc_var=np.append(perc_var, np.var(percussive))
    # frame_mean=np.append(frame_mean, np.mean(frames_to_time))  # frames
    # frame_std=np.append(frame_std, np.std(frames_to_time))
    # frame_var=np.append(frame_var, np.var(frames_to_time))
    print (x)


# dataset = [{'path': path, 'label':'normal'}
#                for path in glob.glob("/home/onu/Desktop/Thesis Work/Datasets/Pascal/Btraining_murmur/Btraining_murmur/*.wav")]
# for x in df['path']:
#     y, sr = librosa.load(x)
#     S = np.abs(librosa.stft(y))
#     tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
#     chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
#     melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#     # rmse = librosa.feature.rmse(y=y)
#     cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     poly_features = librosa.feature.poly_features(S=S, sr=sr)
#     tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     harmonic = librosa.effects.harmonic(y)
#     percussive = librosa.effects.percussive(y)
#
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     mfcc_delta = librosa.feature.delta(mfcc)
#
#     onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
#     frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
#
#     # Transforming Features
#     name_vector=np.append(name_vector, 'murmur')  # sname
#     chroma_stft_mean=np.append(chroma_stft_mean, np.mean(chroma_stft))  # chroma stft
#     chroma_stft_std=np.append(chroma_stft_std, np.std(chroma_stft))
#     chroma_stft_var=np.append(chroma_stft_var, np.var(chroma_stft))
#     chroma_cq_mean=np.append(chroma_cq_mean, np.mean(chroma_cq))  # chroma cq
#     chroma_cq_std=np.append(chroma_cq_std, np.std(chroma_cq))
#     chroma_cq_var=np.append(chroma_cq_var, np.var(chroma_cq))
#     chroma_cens_mean=np.append(chroma_cens_mean, np.mean(chroma_cens))  # chroma cens
#     chroma_cens_std=np.append(chroma_cens_std, np.std(chroma_cens))
#     chroma_cens_var=np.append(chroma_cens_var, np.var(chroma_cens))
#     mel_mean=np.append(mel_mean, np.mean(melspectrogram))  # melspectrogram
#     mel_std=np.append(mel_std, np.std(melspectrogram))
#     mel_var=np.append(mel_var, np.var(melspectrogram))
#     mfcc_mean=np.append(mfcc_mean, np.mean(mfcc))  # mfcc
#     mfcc_std=np.append(mfcc_std, np.std(mfcc))
#     mfcc_var=np.append(mfcc_var, np.var(mfcc))
#     mfcc_delta_mean=np.append(mfcc_delta, np.mean(mfcc_delta))  # mfcc delta
#     mfcc_delta_std=np.append(mfcc_delta_std, np.std(mfcc_delta))
#     mfcc_delta_var=np.append(mfcc_delta_var, np.var(mfcc_delta))
#     # rmse_mean=np.append(id, np.mean(rmse))  # rmse
#     # rmse_std=np.append(id, np.std(rmse))
#     # rmse_var=np.append(id, np.var(rmse))
#     cent_mean=np.append(cent_mean, np.mean(cent))  # cent
#     cent_std=np.append(cent_std, np.std(cent))
#     cent_var=np.append(cent_var, np.var(cent))
#     spec_bw_mean=np.append(spec_bw_mean, np.mean(spec_bw))  # spectral bandwidth
#     spec_bw_std=np.append(spec_bw_std, np.std(spec_bw))
#     spec_bw_var=np.append(spec_bw_var, np.var(spec_bw))
#     contrast_mean=np.append(contrast_mean, np.mean(contrast))  # contrast
#     contrast_std=np.append(contrast_std, np.std(contrast))
#     contrast_var=np.append( contrast_var, np.var(contrast))
#     rolloff_mean=np.append(rolloff_mean, np.mean(rolloff))  # rolloff
#     rolloff_std=np.append(rolloff_std, np.std(rolloff))
#     rolloff_var=np.append(rolloff_var, np.var(rolloff))
#     poly_mean=np.append(poly_mean, np.mean(poly_features))  # poly features
#     poly_std=np.append(poly_std, np.std(poly_features))
#     poly_var=np.append(poly_var, np.var(poly_features))
#     tonnetz_mean=np.append( tonnetz_mean, np.mean(tonnetz))  # tonnetz
#     tonnetz_std=np.append(tonnetz_std, np.std(tonnetz))
#     tonnetz_var=np.append(tonnetz_var, np.var(tonnetz))
#     zcr_mean=np.append(zcr_mean, np.mean(zcr))  # zero crossing rate
#     zcr_std=np.append(zcr_std, np.std(zcr))
#     zcr_var=np.append(zcr_var, np.var(zcr))
#     harm_mean=np.append(harm_mean, np.mean(harmonic))  # harmonic
#     harm_std=np.append(harm_std, np.std(harmonic))
#     harm_var=np.append(harm_var, np.var(harmonic))
#     perc_mean=np.append(perc_mean, np.mean(percussive))  # percussive
#     perc_std=np.append(perc_std, np.std(percussive))
#     perc_var=np.append(perc_var, np.var(percussive))
#     frame_mean=np.append(frame_mean, np.mean(frames_to_time))  # frames
#     frame_std=np.append(frame_std, np.std(frames_to_time))
#     frame_var=np.append(frame_var, np.var(frames_to_time))
#     print (x)
#
#

#
# Datas = {'name': name_vector,
#          'chroma_stft_mean':  chroma_stft_mean,
#          'chroma_stft_std': chroma_stft_std,
#          'chroma_stft_var': chroma_stft_var,
#          'chroma_cq_mean':chroma_cq_mean,
#          'chroma_cq_std':chroma_cq_std,
#          'chroma_cq_var':chroma_cq_var,
#          'chroma_cens_mean':chroma_cens_mean,
#          'chroma_cens_std':chroma_cens_std,
#          'chroma_cens_var':chroma_cens_var,
#          'mel_mean':mel_mean,
#          'mel_std':mel_std,
#          'mel_var':mel_var,
#          'mfcc_mean':mfcc_mean,
#          'mfcc_std':mfcc_std,
#          'mfcc_var':mfcc_var,
#          # 'mfcc_delta_mean':mfcc_delta_mean,
#          # 'mfcc_delta_std':mfcc_delta_std,
#          # 'mfcc_delta_var':mfcc_delta_var,
#          'cent_mean':cent_mean,
#          'cent_std':cent_std,
#          'cent_var':cent_var,
#          'spec_bw_mean':spec_bw_mean,
#          'spec_bw_std':spec_bw_std,
#          'spec_bw_var':spec_bw_var,
#          'contrast_mean':contrast_mean,
#          'contrast_std':contrast_std,
#          'contrast_var':contrast_var,
#          'rolloff_mean':rolloff_mean,
#          'rolloff_std':rolloff_std,
#          'rolloff_var':rolloff_var,
#          'poly_mean':poly_mean,
#          'poly_std':poly_std,
#          'poly_var':poly_var,
#          'tonnetz_mean':tonnetz_mean,
#          'tonnetz_std':tonnetz_std,
#          'tonnetz_var':tonnetz_var,
#          'zcr_mean':zcr_mean,
#          'zcr_std':zcr_std,
#          'zcr_var':zcr_var,
#          'harm_mean':harm_mean,
#          'harm_std':harm_std,
#          'harm_var':harm_var,
#          'perc_mean':perc_mean,
#          'perc_std':perc_std,
#          'perc_var':perc_var,
#          'frame_mean':frame_mean,
#          'frame_std':frame_std,
#          'frame_var':frame_var
#         }
#
# df = pd.DataFrame(Datas, columns= ['name','chroma_stft_mean', 'chroma_stft_std','chroma_stft_var','chroma_cq_mean','chroma_cq_std','chroma_cq_var','chroma_cens_mean','chroma_cens_std','chroma_cens_var','mel_mean','mel_std','mel_var','mfcc_mean','mfcc_std','mfcc_var','cent_mean','cent_std','cent_var','spec_bw_mean','spec_bw_std','spec_bw_var','contrast_mean','contrast_std','contrast_var','rolloff_mean','rolloff_std','rolloff_var','poly_mean','poly_std','poly_var','tonnetz_mean','tonnetz_std','tonnetz_var','zcr_mean','zcr_std','zcr_var','harm_mean','harm_std','harm_var','perc_mean','perc_std','perc_var','frame_mean','frame_std','frame_var'])
#
# export_csv = df.to_csv ('export_dataframe.csv', index = True, header=True) #Don't forget to add '.csv' at the end of the path
#


# ,'mfcc_delta_mean','mfcc_delta_std','mfcc_delta_var'
