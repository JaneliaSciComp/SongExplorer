#!/usr/bin/python3

# threshold an audio recording in both the time and frequency spaces

# detect.py <full-path-to-wavfile> <sample-rate> <time-sigma> <time-smooth-ms> <frequency-n-ms> <frequency-nw> <frequency-p> <frequency-smooth-ms>

# e.g.
# deepsong detect.py `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 10000 4 6.4 25.6 4 0.1 25.6

import os
import numpy as np
import wave
import skimage
from skimage.morphology import closing, opening
import nitime.utils as utils
import nitime.algorithms as tsa
from sys import argv
import csv
from scipy import stats

_, filename, tic_rate, time_sigma, time_smooth_ms, frequency_n_ms, frequency_nw, frequency_p, frequency_smooth_ms = argv
print('filename: '+filename)
print('tic_rate: '+tic_rate)
print('time_sigma: '+time_sigma)
print('time_smooth_ms: '+time_smooth_ms)
print('frequency_n_ms: '+frequency_n_ms)
print('frequency_nw: '+frequency_nw)
print('frequency_p: '+frequency_p)
print('frequency_smooth_ms: '+frequency_smooth_ms)

tic_rate = int(tic_rate)
time_sigma = int(time_sigma)
time_smooth = int(float(time_smooth_ms)/1000*tic_rate)
frequency_n = int(float(frequency_n_ms)/1000*tic_rate)
frequency_nw = int(frequency_nw)
frequency_p = float(frequency_p)
frequency_smooth = int(float(frequency_smooth_ms)/1000*tic_rate) // (frequency_n//2)

fid=wave.open(filename)
song=fid.readframes(fid.getnframes())
song = np.frombuffer(song, dtype=np.int16)
fs = fid.getframerate()
assert fs==tic_rate
fid.close()


### in time, using a simple threshold

song_median = np.median(song)
song_mad = stats.median_absolute_deviation(song)
song_thresholded = np.abs(song-song_median) > time_sigma*song_mad
selem = np.ones((time_smooth), dtype=np.uint8)
song_morphed = closing(song_thresholded, selem)
song_2D = skimage.measure.label(np.array([song_morphed, song_morphed]))
song_labelled = skimage.measure.label(song_2D)
song_props = skimage.measure.regionprops(song_labelled)
timestamps_t = []
for iprop in range(len(song_props)):
  timestamps_t.append((song_props[iprop]['bbox'][1], song_props[iprop]['bbox'][3]))


### in frequency, using multi-taper F-test

N = frequency_n
NW = frequency_nw
fft_pow = int( np.ceil(np.log2(N) + 2) )
NFFT = 2**fft_pow
p = 1/NFFT*frequency_p

song_reshaped1 = np.reshape(song[:len(song)//N*N],(-1,N))
f = utils.detect_lines(song_reshaped1, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p)
timestamps_f1 = [2*i+0 for (i,ii) in enumerate(f) if ii!=()]

song_reshaped2 = np.reshape(song[N//2:N//2+(len(song)-N//2)//N*N],(-1,N))
f = utils.detect_lines(song_reshaped2, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p)
timestamps_f2 = [2*i+1 for (i,ii) in enumerate(f) if ii!=()]

song_thresholded = np.zeros((len(song_reshaped1)+len(song_reshaped2)), dtype=np.uint8)
song_thresholded[np.concatenate((timestamps_f1,timestamps_f2))] = 1
selem = np.ones((frequency_smooth), dtype=np.uint8)
song_morphed = closing(opening(song_thresholded, selem), selem)
song_2D = skimage.measure.label(np.array([song_morphed, song_morphed]))
song_labelled = skimage.measure.label(song_2D)
song_props = skimage.measure.regionprops(song_labelled)

timestamps_f = []
for iprop in range(len(song_props)):
  timestamps_f.append((song_props[iprop]['bbox'][1]*N//2-N//4,
                       song_props[iprop]['bbox'][3]*N//2+N//4))


### save

basename = os.path.basename(filename)
with open(os.path.splitext(filename)[0]+'-detected.csv', 'w') as fid:
  csvwriter = csv.writer(fid)
  for i in timestamps_t:
    csvwriter.writerow([basename,i[0],i[1],'detected','time'])
  for i in timestamps_f:
    csvwriter.writerow([basename,i[0],i[1],'detected','frequency'])


### debug

#import matplotlib.pyplot as plt
#plt.ion()
#idx=int(1108.06*tic_rate)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(song[idx-200:idx+200])
#ax.plot(song[timestamps[0]-100:timestamps[0]+100])
