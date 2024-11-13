#!/usr/bin/env python

# threshold an audio recording in both the time and frequency spaces.
# specifically, in the time domain, subtract the median, take the absolute value,
# threshold by the median absolute deviation times the first number in `time σ`,
# and morphologically close gaps shorter than `time smooth` milliseconds.
# separately, use multi-taper harmonic analysis ([thomson, 1982;
# IEEE](https://ieeexplore.ieee.org/document/1456701)) in the frequency domain to
# create a spectrogram using a window of length `freq N` milliseconds (`freq N` /
# 1000 * `audio_tic_rate` should be a power of two) and twice `freq NW` slepian
# tapers, include sounds only within `freq range`, multiply the default threshold
# of the F-test by the first number in `freq ρ`, and open islands and close gaps
# shorter than `freq smooth` milliseconds.  sound events are considered to be
# periods of time which pass either of these two criteria.  quiescent intervals
# are similarly defined as those which pass neither the time nor the frequency
# domain criteria using the second number in `time σ` and `freq ρ` text boxes.

# e.g. time-freq-threshold.py \
#     --filename=`pwd`/groundtruth-data/round2/20161207T102314_ch1_p1.wav \
#     --parameters={"time_sigma":"9,4", "time_smooth":"6.4", "frequency_n":"25.6", "frequency_nw":"4", "frequency_p":"0.1,1.0", "frequency_range":"0-", "frequency_smooth":"25.6", "time_sigma_robust":"median"} \
#     --time_units=ms \
#     --freq_units=Hz \
#     --time_scale=0.001 \
#     --freq_scale=1 \
#     --audio_tic_rate=2500 \
#     --audio_nchannels=1 \
#     --audio_read_plugin=load-wav \
#     --audio_read_plugin_kwargs={}
 
import argparse
import os
import numpy as np
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

from lib import combine_events, load_audio_read_plugin

def _frequency_n_callback(M,V,C):
    C.time.sleep(0.5)
    V.detect_parameters['frequency_n'].css_classes = []
    V.detect_parameters['frequency_smooth'].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def frequency_n_callback(n,M,V,C):
    changed, frequency_n_sec2 = M.next_pow2_sec(float(V.detect_parameters['frequency_n'].value) * M.time_scale)
    if changed:
        V.detect_parameters['frequency_n'].css_classes = ['changed']
        V.detect_parameters['frequency_n'].value = str(frequency_n_sec2 / M.time_scale)
    if float(V.detect_parameters['frequency_smooth'].value) < float(V.detect_parameters['frequency_n'].value):
        V.detect_parameters['frequency_smooth'].css_classes = ['changed']
        V.detect_parameters['frequency_smooth'].value = V.detect_parameters['frequency_n'].value
    if V.bokeh_document:
        V.bokeh_document.add_next_tick_callback(lambda: _frequency_n_callback(M,V,C))
    else:
        _frequency_n_callback(M,V,C)

def detect_parameters(time_units, freq_units, time_scale, freq_scale):
    return [
        ["time_sigma",        "time σ",                       '',        '9,4',                  1, [], None,                 True],
        ["time_smooth",       "time smooth ("+time_units+")", '',        str(0.0064/time_scale), 1, [], None,                 True],
        ["frequency_n",       "freq N ("+time_units+")",      '',        str(0.0256/time_scale), 1, [], frequency_n_callback, True],
        ["frequency_nw",      "freq NW",                      '',        '4',                    1, [], None,                 True],
        ["frequency_p",       "freq ρ",                       '',        '0.1,1.0',              1, [], None,                 True],
        ["frequency_range",   "freq range ("+freq_units+")",  '',        '0-',                   1, [], None,                 True],
        ["frequency_smooth",  "freq smooth ("+time_units+")", '',        str(0.0256/time_scale), 1, [], None,                 True],
        ["time_sigma_robust", "robust",                       ['median',
                                                               'mean'],  'median',               1, [], None,                 True],
    ]

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

    audio_tic_rate = FLAGS.audio_tic_rate
    time_scale = FLAGS.time_scale
    time_units = FLAGS.time_units

    load_audio_read_plugin(FLAGS.audio_read_plugin, FLAGS.audio_read_plugin_kwargs)
    from lib import audio_read, trim_ext

    time_sigma_signal, time_sigma_noise = [int(x) for x in FLAGS.parameters['time_sigma'].split(',')]
    
    time_smooth_tic = round(float(FLAGS.parameters['time_smooth']) * time_scale * audio_tic_rate)
    frequency_n_tic = round(float(FLAGS.parameters['frequency_n']) * time_scale * audio_tic_rate)
    frequency_nw = int(FLAGS.parameters['frequency_nw'])
    frequency_p_signal, frequency_p_noise = [float(x) for x in FLAGS.parameters['frequency_p'].split(',')]
    frequency_range_lo, frequency_range_hi = FLAGS.parameters['frequency_range'].split('-')
    frequency_range_lo = 0.0 if frequency_range_lo=='' else \
                         float(frequency_range_lo) / audio_tic_rate
    frequency_range_hi = 0.5 if frequency_range_hi=='' else \
                         float(frequency_range_hi) / audio_tic_rate
    frequency_smooth_tic = round(float(FLAGS.parameters['frequency_smooth'])*time_scale*audio_tic_rate) // (frequency_n_tic//2)
    time_sigma_robust = FLAGS.parameters['time_sigma_robust']

    fs, _, song = audio_read(FLAGS.filename)
    if fs!=audio_tic_rate:
      raise Exception("ERROR: sampling rate of WAV file (="+str(fs)+
            ") is not the same as specified in the config file (="+str(audio_tic_rate)+")")

    if not (frequency_n_tic & (frequency_n_tic-1) == 0) or frequency_n_tic == 0:
      next_higher_tic = np.power(2, np.ceil(np.log2(frequency_n_tic))).astype(int)
      next_lower_tic = np.power(2, np.floor(np.log2(frequency_n_tic))).astype(int)
      sigdigs = np.ceil(np.log10(next_higher_tic)).astype(int)+1
      next_higher_sec = np.around(next_higher_tic/audio_tic_rate, decimals=sigdigs)
      next_lower_sec = np.around(next_lower_tic/audio_tic_rate, decimals=sigdigs)
      raise Exception("ERROR: 'freq N ("+time_units+")' should be a power of two when converted to tics.  "+FLAGS.parameters['frequency_n']+" "+time_units+" is "+str(frequency_n_tic)+" tics for Fs="+str(audio_tic_rate)+".  try "+str(next_lower_sec/time_scale)+" "+time_units+" (="+str(next_lower_tic)+") or "+str(next_higher_sec/time_scale)+" "+time_units+" (="+str(next_higher_tic)+") instead.")

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


    selem = np.ones((time_smooth_tic), dtype=np.uint8)

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


    N = frequency_n_tic
    NW = frequency_nw
    fft_pow = int( np.ceil(np.log2(N) + 2) )
    NFFT = 2**fft_pow
    p_signal = 1/NFFT*frequency_p_signal
    p_noise = 1/NFFT*frequency_p_noise

    selem = np.ones((frequency_smooth_tic), dtype=np.uint8)

    chunk_size_tics = 1024*1024

    intervals_freq_signal = []
    intervals_freq_noise = []
    for ichannel in range(nchannels):
      ioffset=0
      while ioffset < nsounds:
        ilast = min(nsounds, ioffset+chunk_size_tics)//N*N
        song_reshaped1 = np.reshape(song[ioffset : ilast, ichannel], (-1,N))
        f = utils.detect_lines(song_reshaped1, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_signal)
        intervals_f1 = [2*i+0 for (i,ii) in enumerate(f) if ii!=() and
                any([frequency_range_lo <= f <= frequency_range_hi for f in ii[0]])]

        ilast -= N//2
        song_reshaped2 = np.reshape(song[ioffset+N//2 : ilast, ichannel], (-1,N))
        f = utils.detect_lines(song_reshaped2, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_signal)
        intervals_f2 = [2*i+1 for (i,ii) in enumerate(f) if ii!=() and
                any([frequency_range_lo <= f <= frequency_range_hi for f in ii[0]])]

        song_thresholded = np.zeros((len(song_reshaped1)+len(song_reshaped2)), dtype=np.uint8)
        song_thresholded[np.concatenate((intervals_f1,intervals_f2)).astype(int)] = 1
        song_morphed = closing(opening(song_thresholded, selem), selem)
        intervals_freq_signal += bool2stamp(song_morphed,
                                             lambda x,y: (ioffset+x*N//2-N//4, ioffset+y*N//2+N//4))

        ilast = min(nsounds, ioffset+chunk_size_tics)//N*N
        song_reshaped1 = np.reshape(song[ioffset : ilast, ichannel], (-1,N))
        f = utils.detect_lines(song_reshaped1, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_noise)
        intervals_f1 = [2*i+0 for (i,ii) in enumerate(f) if ii!=() and
                any([frequency_range_lo <= f <= frequency_range_hi for f in ii[0]])]

        ilast -= N//2
        song_reshaped2 = np.reshape(song[ioffset+N//2 : ilast, ichannel], (-1,N))
        f = utils.detect_lines(song_reshaped2, (NW, 2*NW), low_bias=True, NFFT=NFFT, p=p_noise)
        intervals_f2 = [2*i+1 for (i,ii) in enumerate(f) if ii!=() and
                any([frequency_range_lo <= f <= frequency_range_hi for f in ii[0]])]

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
    with open(trim_ext(FLAGS.filename)+'-detected.csv', 'w') as fid:
      csvwriter = csv.writer(fid, lineterminator='\n')
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
        '--time_units',
        type=str,
        default="ms",
        help='Units of time',)
    parser.add_argument(
        '--freq_units',
        type=str,
        default="Hz",
        help='Units of frequency',)
    parser.add_argument(
        '--time_scale',
        type=float,
        default="ms",
        help='This many seconds are in time_units',)
    parser.add_argument(
        '--freq_scale',
        type=float,
        default="Hz",
        help='This many frequencies are in freq_units',)
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
    print('time-freq-threshold.py version = 0.1')
    print("hostname = "+socket.gethostname())
  
    try:
        main()
  
    except Exception as e:
      print(e)
  
    finally:
      if hasattr(os, 'sync'):
        os.sync()
      print(str(datetime.now())+": finish time")
