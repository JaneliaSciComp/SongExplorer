#!/usr/bin/python3

# threshold an audio recording in both the time and frequency spaces

# detect.py <full-path-to-wavfile> <wav-tic-rate> <wav-nchannels> <time-sigma> <time-smooth-ms> <frequency-n-ms> <frequency-nw> <frequency-p> <frequency-smooth-ms>

# e.g.
# deepsong detect.py `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 10000 1 4 6.4 25.6 4 0.1 25.6

import os
import numpy as np
import scipy.io.wavfile as spiowav
import skimage
from skimage.morphology import closing, opening
import nitime.utils as utils
import nitime.algorithms as tsa
from sys import argv
import csv
from scipy import stats

_, filename, audio_tic_rate, audio_nchannels, time_sigma, time_smooth_ms, frequency_n_ms, frequency_nw, frequency_p, frequency_smooth_ms = argv
print('filename: '+filename)
print('audio_tic_rate: '+audio_tic_rate)
print('audio_nchannels: '+audio_nchannels)
print('time_sigma: '+time_sigma)
print('time_smooth_ms: '+time_smooth_ms)
print('frequency_n_ms: '+frequency_n_ms)
print('frequency_nw: '+frequency_nw)
print('frequency_p: '+frequency_p)
print('frequency_smooth_ms: '+frequency_smooth_ms)

audio_tic_rate = int(audio_tic_rate)
audio_nchannels = int(audio_nchannels)
time_sigma = int(time_sigma)
time_smooth = int(float(time_smooth_ms)/1000*audio_tic_rate)
frequency_n = int(float(frequency_n_ms)/1000*audio_tic_rate)
frequency_nw = int(frequency_nw)
frequency_p = float(frequency_p)
frequency_smooth = int(float(frequency_smooth_ms)/1000*audio_tic_rate) // (frequency_n//2)

fs, song = spiowav.read(filename)
assert fs==audio_tic_rate
if audio_nchannels==1:
  song = np.expand_dims(song, axis=1)
nsamples = np.shape(song)[0]
nchannels = np.shape(song)[1]
assert nchannels == audio_nchannels


selem = np.ones((time_smooth), dtype=np.uint8)

timestamps_time = []
for ichannel in range(nchannels):
  song_median = np.median(song[:,ichannel])
  song_mad = stats.median_absolute_deviation(song[:,ichannel])
  song_thresholded = np.abs(song[:,ichannel]-song_median) > time_sigma*song_mad
  song_morphed = closing(song_thresholded, selem)
  song_2D = np.array([song_morphed, song_morphed])
  song_labelled = skimage.measure.label(song_2D)
  song_props = skimage.measure.regionprops(song_labelled)
  ichannel_str = str(ichannel) if nchannels>1 else ''
  for iprop in range(len(song_props)):
    timestamps_time.append((song_props[iprop]['bbox'][1],
                            song_props[iprop]['bbox'][3],
                            ichannel_str))


N = frequency_n
NW = frequency_nw
fft_pow = int( np.ceil(np.log2(N) + 2) )
NFFT = 2**fft_pow
p = 1/NFFT*frequency_p

timestamps_freq = []
for ichannel in range(nchannels):
  song_reshaped1 = np.reshape(song[:nsamples//N*N,ichannel],(-1,N))
  f = utils.detect_lines(song_reshaped1, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p)
  timestamps_f1 = [2*i+0 for (i,ii) in enumerate(f) if ii!=()]

  song_reshaped2 = np.reshape(song[N//2:N//2+(nsamples-N//2)//N*N,ichannel],(-1,N))
  f = utils.detect_lines(song_reshaped2, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p)
  timestamps_f2 = [2*i+1 for (i,ii) in enumerate(f) if ii!=()]

  song_thresholded = np.zeros((len(song_reshaped1)+len(song_reshaped2)), dtype=np.uint8)
  song_thresholded[np.concatenate((timestamps_f1,timestamps_f2))] = 1
  selem = np.ones((frequency_smooth), dtype=np.uint8)
  song_morphed = closing(opening(song_thresholded, selem), selem)
  song_2D = skimage.measure.label(np.array([song_morphed, song_morphed]))
  song_labelled = skimage.measure.label(song_2D)
  song_props = skimage.measure.regionprops(song_labelled)

  ichannel_str = str(ichannel) if nchannels>1 else ''
  for iprop in range(len(song_props)):
    timestamps_freq.append((song_props[iprop]['bbox'][1]*N//2-N//4,
                            song_props[iprop]['bbox'][3]*N//2+N//4,
                            ichannel_str))


basename = os.path.basename(filename)
with open(os.path.splitext(filename)[0]+'-detected.csv', 'w') as fid:
  csvwriter = csv.writer(fid)
  for i in timestamps_time:
    csvwriter.writerow([basename,i[0],i[1],'detected','time'+i[2]])
  for i in timestamps_freq:
    csvwriter.writerow([basename,i[0],i[1],'detected','frequency'+i[2]])
