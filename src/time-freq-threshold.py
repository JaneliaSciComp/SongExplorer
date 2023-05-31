#!/usr/bin/env python3

# threshold an audio recording in both the time and frequency spaces

# e.g.
# $SONGEXPLORER_BIN time-freq-threshold.py \
#     --filename=`pwd`/groundtruth-data/round2/20161207T102314_ch1_p1.wav \
#     --parameters={"time_sigma":"9,4", "time_smooth_ms":"6.4", "frequency_n_ms":"25.6", "frequency_nw":"4", "frequency_p":"0.1,1.0", "frequency_smooth_ms":"25.6", "time_sigma_robust":"median"} \
#     --audio_tic_rate=2500 \
#     --audio_nchannels=1 \
#     --audio_read_plugin=load-wav \
#     --audio_read_plugin_kwargs={}
 
import argparse
import os
import numpy as np
import importlib
import skimage
from skimage.morphology import closing, opening
import nitime.utils as utils
import nitime.algorithms as tsa
import sys
import csv
from scipy import stats
from itertools import cycle
from datetime import datetime
import socket
import json
import shutil

repodir = os.path.dirname(os.path.dirname(os.path.realpath(shutil.which("songexplorer"))))

sys.path.append(os.path.join(repodir, "src"))
from lib import *

def _frequency_n_callback(M,V,C):
    C.time.sleep(0.5)
    V.detect_parameters['frequency_n_ms'].css_classes = []
    V.detect_parameters['frequency_smooth_ms'].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def frequency_n_callback(n,M,V,C):
    changed, frequency_n_ms2 = M.next_pow2_ms(float(V.detect_parameters['frequency_n_ms'].value))
    if changed:
        V.detect_parameters['frequency_n_ms'].css_classes = ['changed']
        V.detect_parameters['frequency_n_ms'].value = str(frequency_n_ms2)
    if float(V.detect_parameters['frequency_smooth_ms'].value) < float(V.detect_parameters['frequency_n_ms'].value):
        V.detect_parameters['frequency_smooth_ms'].css_classes = ['changed']
        V.detect_parameters['frequency_smooth_ms'].value = V.detect_parameters['frequency_n_ms'].value
    if V.bokeh_document:
        V.bokeh_document.add_next_tick_callback(lambda: _frequency_n_callback(M,V,C))
    else:
        _frequency_n_callback(M,V,C)

# a list of lists specifying the detect-specific hyperparameters in the GUI
detect_parameters = [
    ["time_sigma",          "time σ",        '',        '9,4',     1, [], None,                 True],
    ["time_smooth_ms",      "time smooth",   '',        '6.4',     1, [], None,                 True],
    ["frequency_n_ms",      "freq N (msec)", '',        '25.6',    1, [], frequency_n_callback, True],
    ["frequency_nw",        "freq NW",       '',        '4',       1, [], None,                 True],
    ["frequency_p",         "freq ρ",        '',        '0.1,1.0', 1, [], None,                 True],
    ["frequency_smooth_ms", "freq smooth",   '',        '25.6',    1, [], None,                 True],
    ["time_sigma_robust",   "robust",        ['median',
                                              'mean'],  'median',  1, [], None,                 True],
    ]

# a function which returns a vector of strings used to annotate the detected events
def detect_labels(audio_nchannels):
    labels=[]
    for i in range(audio_nchannels):
        i_str = str(i) if audio_nchannels>1 else ''
        labels.append("time"+i_str+",frequency"+i_str+",neither"+i_str)
    return labels

FLAGS = None

def main():
  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    print('%s = %s' % (key, flags[key]))

    sys.path.append(os.path.dirname(FLAGS.audio_read_plugin))
    audio_read_module = importlib.import_module(os.path.basename(FLAGS.audio_read_plugin))
    def audio_read(wav_path, start_tic=None, stop_tic=None):
        return audio_read_module.audio_read(wav_path, start_tic, stop_tic,
                                            **FLAGS.audio_read_plugin_kwargs)

    time_sigma_signal, time_sigma_noise = [int(x) for x in FLAGS.parameters['time_sigma'].split(',')]
    
    time_smooth = round(float(FLAGS.parameters['time_smooth_ms'])/1000*FLAGS.audio_tic_rate)
    frequency_n = round(float(FLAGS.parameters['frequency_n_ms'])/1000*FLAGS.audio_tic_rate)
    frequency_nw = int(FLAGS.parameters['frequency_nw'])
    frequency_p_signal, frequency_p_noise = [float(x) for x in FLAGS.parameters['frequency_p'].split(',')]
    frequency_smooth = round(float(FLAGS.parameters['frequency_smooth_ms'])/1000*FLAGS.audio_tic_rate) // (frequency_n//2)
    time_sigma_robust = FLAGS.parameters['time_sigma_robust']

    fs, song = audio_read(FLAGS.filename)
    if fs!=FLAGS.audio_tic_rate:
      raise Exception("ERROR: sampling rate of WAV file (="+str(fs)+
            ") is not the same as specified in the config file (="+str(FLAGS.audio_tic_rate)+")")

    if not (frequency_n & (frequency_n-1) == 0) or frequency_n == 0:
      next_higher = np.power(2, np.ceil(np.log2(frequency_n))).astype(int)
      next_lower = np.power(2, np.floor(np.log2(frequency_n))).astype(int)
      sigdigs = np.ceil(np.log10(next_higher)).astype(int)+1
      next_higher_ms = np.around(next_higher/FLAGS.audio_tic_rate*1000, decimals=sigdigs)
      next_lower_ms = np.around(next_lower/FLAGS.audio_tic_rate*1000, decimals=sigdigs)
      raise Exception("ERROR: 'freq N (msec)' should be a power of two when converted to tics.  "+FLAGS.parameters['frequency_n_ms']+" ms is "+str(frequency_n)+" tics for Fs="+str(FLAGS.audio_tic_rate)+".  try "+str(next_lower_ms)+" ms (="+str(next_lower)+") or "+str(next_higher_ms)+"ms (="+str(next_higher)+") instead.")

    if np.ndim(song)==1:
      song = np.expand_dims(song, axis=1)
    nsounds = np.shape(song)[0]
    nchannels = np.shape(song)[1]
    if nchannels != FLAGS.audio_nchannels:
      raise Exception("ERROR: number of channels in WAV file (="+str(nchannels)+
            ") is not the same as specified in the config file (="+str(FLAGS.audio_nchannels)+")")

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
      if time_sigma_robust=='median':
        song_median = np.median(song[:,ichannel])
        song_mad = stats.median_abs_deviation(song[:,ichannel])
      else:
        song_median = np.mean(song[:,ichannel])
        song_mad = np.std(song[:,ichannel])

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
        song_thresholded[np.concatenate((intervals_f1,intervals_f2)).astype(int)] = 1
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
        song_thresholded[np.concatenate((intervals_f1,intervals_f2)).astype(int)] = 1
        song_morphed = closing(opening(song_thresholded, selem), selem)
        intervals_freq_noise += bool2stamp(song_morphed,
                                            lambda x,y: (ioffset+x*N//2-N//4, ioffset+y*N//2+N//4))

        ioffset += chunk_size_tics


    start_times_neither, stop_times_neither, ifeature = combine_events(
          intervals_time_noise, intervals_freq_noise,
          lambda x,y: np.logical_and(np.logical_not(x), np.logical_not(y)))


    basename = os.path.basename(FLAGS.filename)
    with open(os.path.splitext(FLAGS.filename)[0]+'-detected.csv', 'w') as fid:
      csvwriter = csv.writer(fid)
      for i in intervals_time_signal:
        csvwriter.writerow([basename,i[1],i[2],'detected','time'+i[3]])
      for i in intervals_freq_signal:
        csvwriter.writerow([basename,i[1],i[2],'detected','frequency'+i[3]])
      csvwriter.writerows(zip(cycle([basename]), \
                              start_times_neither[:ifeature], stop_times_neither[:ifeature], \
                              cycle(['detected']), cycle(['neither'])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        type=str)
    parser.add_argument(
        '--parameters',
        type=json.loads)
    parser.add_argument(
        '--audio_tic_rate',
        type=int)
    parser.add_argument(
        '--audio_nchannels',
        type=int)
    parser.add_argument(
        '--audio_read_plugin',
        type=str)
    parser.add_argument(
        '--audio_read_plugin_kwargs',
        type=json.loads)
  
    FLAGS, unparsed = parser.parse_known_args()
  
    print(str(datetime.now())+": start time")
    with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
        print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
    print("hostname = "+socket.gethostname())
  
    try:
        main()
  
    except Exception as e:
      print(e)
  
    finally:
      if hasattr(os, 'sync'):
        os.sync()
      print(str(datetime.now())+": finish time")
