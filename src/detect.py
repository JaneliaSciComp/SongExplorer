#!/usr/bin/python3

# threshold an audio recording in both the time and frequency spaces

# detect.py <full-path-to-wavfile> <audio-tic-rate> <audio-nchannels> <time-sigma-signal> <time-sigma-noise> <time-smooth-ms> <frequency-n-ms> <frequency-nw> <frequency-p-signal> <frequency-p-noise> <frequency-smooth-ms>

# e.g.
# detect.py `pwd`/groundtruth-data/round2/20161207T102314_ch1_p1.wav 2500 1 4 2 6.4 25.6 4 0.1 1.0 25.6

import os
import numpy as np
import scipy.io.wavfile as spiowav
import skimage
from skimage.morphology import closing, opening
import nitime.utils as utils
import nitime.algorithms as tsa
import sys
import csv
from scipy import stats
from itertools import cycle

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

_, filename, audio_tic_rate, audio_nchannels, time_sigma_signal, time_sigma_noise, time_smooth_ms, frequency_n_ms, frequency_nw, frequency_p_signal, frequency_p_noise, frequency_smooth_ms = sys.argv
print('filename: '+filename)
print('audio_tic_rate: '+audio_tic_rate)
print('audio_nchannels: '+audio_nchannels)
print('time_sigma_signal: '+time_sigma_signal)
print('time_sigma_noise: '+time_sigma_noise)
print('time_smooth_ms: '+time_smooth_ms)
print('frequency_n_ms: '+frequency_n_ms)
print('frequency_nw: '+frequency_nw)
print('frequency_p_signal: '+frequency_p_signal)
print('frequency_p_noise: '+frequency_p_noise)
print('frequency_smooth_ms: '+frequency_smooth_ms)

audio_tic_rate = int(audio_tic_rate)
audio_nchannels = int(audio_nchannels)
time_sigma_signal = int(time_sigma_signal)
time_sigma_noise = int(time_sigma_noise)
time_smooth = round(float(time_smooth_ms)/1000*audio_tic_rate)
frequency_n = round(float(frequency_n_ms)/1000*audio_tic_rate)
frequency_nw = int(frequency_nw)
frequency_p_signal = float(frequency_p_signal)
frequency_p_noise = float(frequency_p_noise)
frequency_smooth = round(float(frequency_smooth_ms)/1000*audio_tic_rate) // (frequency_n//2)

fs, song = spiowav.read(filename)
if fs!=audio_tic_rate:
  print("ERROR: sampling rate of WAV file (="+str(fs)+
        ") is not the same as specified in the config file (="+str(audio_tic_rate)+")")
  exit()

if not (frequency_n & (frequency_n-1) == 0) or frequency_n == 0:
  print("ERROR: 'freq N (msec)' should be a power of two when converted to tics")
  exit()

if np.ndim(song)==1:
  song = np.expand_dims(song, axis=1)
nsounds = np.shape(song)[0]
nchannels = np.shape(song)[1]
if nchannels != audio_nchannels:
  print("ERROR: number of channels in WAV file (="+str(nchannels)+
        ") is not the same as specified in the config file (="+str(audio_nchannels)+")")
  exit()

def bool2stamp(song_morphed, scale):
  intervals_time = []
  song_2D = np.array([song_morphed, song_morphed])
  song_labelled = skimage.measure.label(song_2D)
  song_props = skimage.measure.regionprops(song_labelled)
  ichannel_str = str(ichannel) if nchannels>1 else ''
  for iprop in range(len(song_props)):
    intervals_time.append(('',
                            *scale(song_props[iprop]['bbox'][1],
                                   song_props[iprop]['bbox'][3]),
                            ichannel_str))
  return intervals_time


selem = np.ones((time_smooth), dtype=np.uint8)

for ichannel in range(nchannels):
  song_median = np.median(song[:,ichannel])
  song_mad = stats.median_absolute_deviation(song[:,ichannel])

  song_thresholded = np.abs(song[:,ichannel]-song_median) > time_sigma_signal*song_mad
  song_morphed = closing(song_thresholded, selem)
  intervals_time_signal = bool2stamp(song_morphed, lambda x,y: (x,y))

  song_thresholded = np.abs(song[:,ichannel]-song_median) > time_sigma_noise*song_mad
  song_morphed = closing(song_thresholded, selem)
  intervals_time_noise = bool2stamp(song_morphed, lambda x,y: (x,y))


N = frequency_n
NW = frequency_nw
fft_pow = int( np.ceil(np.log2(N) + 2) )
NFFT = 2**fft_pow
p_signal = 1/NFFT*frequency_p_signal
p_noise = 1/NFFT*frequency_p_noise

selem = np.ones((frequency_smooth), dtype=np.uint8)

chunk_size_tics = 1024*1024

intervals_freq_signal = []
intervals_freq_noise = []
for ichannel in range(nchannels):
  ioffset=0
  while ioffset < nsounds:
    ilast = min(nsounds, ioffset+chunk_size_tics)//N*N
    song_reshaped1 = np.reshape(song[ioffset : ilast, ichannel], (-1,N))
    f = utils.detect_lines(song_reshaped1, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_signal)
    intervals_f1 = [2*i+0 for (i,ii) in enumerate(f) if ii!=()]

    ilast -= N//2
    song_reshaped2 = np.reshape(song[ioffset+N//2 : ilast, ichannel], (-1,N))
    f = utils.detect_lines(song_reshaped2, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_signal)
    intervals_f2 = [2*i+1 for (i,ii) in enumerate(f) if ii!=()]

    song_thresholded = np.zeros((len(song_reshaped1)+len(song_reshaped2)), dtype=np.uint8)
    song_thresholded[np.concatenate((intervals_f1,intervals_f2)).astype(np.int)] = 1
    song_morphed = closing(opening(song_thresholded, selem), selem)
    intervals_freq_signal += bool2stamp(song_morphed,
                                         lambda x,y: (ioffset+x*N//2-N//4, ioffset+y*N//2+N//4))

    ilast = min(nsounds, ioffset+chunk_size_tics)//N*N
    song_reshaped1 = np.reshape(song[ioffset : ilast, ichannel], (-1,N))
    f = utils.detect_lines(song_reshaped1, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_noise)
    intervals_f1 = [2*i+0 for (i,ii) in enumerate(f) if ii!=()]

    ilast -= N//2
    song_reshaped2 = np.reshape(song[ioffset+N//2 : ilast, ichannel], (-1,N))
    f = utils.detect_lines(song_reshaped2, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_noise)
    intervals_f2 = [2*i+1 for (i,ii) in enumerate(f) if ii!=()]

    song_thresholded = np.zeros((len(song_reshaped1)+len(song_reshaped2)), dtype=np.uint8)
    song_thresholded[np.concatenate((intervals_f1,intervals_f2)).astype(np.int)] = 1
    song_morphed = closing(opening(song_thresholded, selem), selem)
    intervals_freq_noise += bool2stamp(song_morphed,
                                        lambda x,y: (ioffset+x*N//2-N//4, ioffset+y*N//2+N//4))

    ioffset += chunk_size_tics


start_times_neither, stop_times_neither, ifeature = combine_events(
      intervals_time_noise, intervals_freq_noise,
      lambda x,y: np.logical_and(np.logical_not(x), np.logical_not(y)))


basename = os.path.basename(filename)
with open(os.path.splitext(filename)[0]+'-detected.csv', 'w') as fid:
  csvwriter = csv.writer(fid)
  for i in intervals_time_signal:
    csvwriter.writerow([basename,i[1],i[2],'detected','time'+i[3]])
  for i in intervals_freq_signal:
    csvwriter.writerow([basename,i[1],i[2],'detected','frequency'+i[3]])
  csvwriter.writerows(zip(cycle([basename]), \
                          start_times_neither[:ifeature], stop_times_neither[:ifeature], \
                          cycle(['detected']), cycle(['neither'])))
