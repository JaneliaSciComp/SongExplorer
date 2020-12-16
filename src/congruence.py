#!/usr/bin/python3

# generate figures of false positives and negatives

# congruence.py <path-to-groundtruth> <wavfiles-with-dense-annotations-and-predictions> <nprobabilities> <audio-tic-rate> <congruence-parallelize>

# congruence.py /groups/stern/sternlab/behavior/arthurb/groundtruth/kyriacou2017 PS_20130625111709_ch3.wav,PS_20130625111709_ch7.wav 20 2500 1

import sys
import os
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ion()
import csv
import pandas as pd
from itertools import cycle
from interval import interval, inf
from functools import reduce
from natsort import natsorted
from matplotlib_venn import venn2, venn3
from scipy import interpolate

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

_,basepath,wavfiles,nprobabilities,audio_tic_rate,congruence_parallelize = sys.argv
print('basepath: '+basepath)
print('wavfiles: '+wavfiles)
print('nprobabilities: '+nprobabilities)
print('audio_tic_rate: '+audio_tic_rate)
print('congruence_parallelize: '+congruence_parallelize)
wavfiles=set([os.path.basename(x) for x in wavfiles.split(',')])
nprobabilities=int(nprobabilities)
audio_tic_rate=int(audio_tic_rate)
congruence_parallelize=int(congruence_parallelize)

wavdirs = {}
for subdir in filter(lambda x: os.path.isdir(os.path.join(basepath,x)), \
                     os.listdir(basepath)):
  commonsubfiles = wavfiles & set(os.listdir(os.path.join(basepath, subdir)))
  if len(commonsubfiles) > 0:
    wavdirs[subdir] = commonsubfiles

labels=None
for wavdir in wavdirs:
  for wavfile in wavdirs[wavdir]:
    wavfile_noext = os.path.splitext(wavfile)[0]

    if not labels:
      with open(os.path.join(basepath,wavdir,wavfile_noext+'-ethogram.log'), 'r') as fid:
        for line in fid:
          if line.startswith('logdir: '):
            m=re.search('logdir: (.*)',line)
            logdir = m.group(1)
          elif line.startswith('model: '):
            m=re.search('model: (.*)',line)
            model = m.group(1)
          elif line.startswith('thresholds_file: '):
            m=re.search('thresholds_file: (.*)',line)
            thresholds_file = m.group(1)
            m=re.search('ckpt-([0-9]+)',thresholds_file)
            ckpt = m.group(1)
            break
      with open(os.path.join(logdir,model,'vgg_labels.txt'), 'r') as fid:
        labels = fid.read().splitlines()
      precision_recalls_sparse, thresholds_sparse = read_thresholds(logdir, model,
                                                                    thresholds_file)

    sample_rate_probabilities, half_stride_sec, probability_matrix = \
          read_probabilities(os.path.join(basepath, wavdir, wavfile_noext), labels)
    for threshold in np.linspace(0, 1, num=nprobabilities+2)[1:-1]:
      features, start_tics, stop_tics = discretize_probabilites(probability_matrix,
                                                                threshold,
                                                                labels,
                                                                sample_rate_probabilities,
                                                                half_stride_sec,
                                                                audio_tic_rate)
      filename = os.path.join(basepath, wavdir,
                              wavfile_noext+'-predicted-'+str(threshold)+'th.csv')
      isort = np.argsort(start_tics)
      with open(filename,'w') as fid:
        csvwriter = csv.writer(fid)
        csvwriter.writerows(zip(cycle([wavfile]), \
                                start_tics[isort], stop_tics[isort], \
                                cycle(['predicted']),
                                features[isort]))

humans = set()
precision_recalls = set()
timestamps = {}
annotator_keys = set()

for wavdir in wavdirs:
  for csvfile in filter(lambda x: ("-annotated-" in x or "-predicted-" in x) and
                                  x.endswith('.csv'), \
                        os.listdir(os.path.join(basepath,wavdir))):
    filepath = os.path.join(basepath, wavdir, csvfile)
    if os.path.getsize(filepath) == 0:
      continue
    df = pd.read_csv(filepath, header=None, index_col=False)
    if len(set(df[0]) & wavdirs[wavdir]) == 0 or \
       len(set(df[3]) & set(['annotated','predicted'])) == 0:
      continue
    print(os.path.join(wavdir, csvfile))
    m = re.search('(-annotated|-predicted)-(.*).csv', csvfile)
    annotator = csvfile[m.span(0)[0]:]
    csvbase = os.path.join(wavdir, csvfile[:m.span(0)[0]])
    if '-annotated-' in csvfile:
      humans.add(m.groups()[1])
    if '-predicted-' in csvfile:
      precision_recalls.add(m.groups()[1])
    for word in set(df[4]):
      if word not in timestamps:
        timestamps[word] = {}
      if csvbase not in timestamps[word]:
        timestamps[word][csvbase] = {}
      timestamps[word][csvbase][annotator] = df.loc[df[4]==word, 1:2]
      timestamps[word][csvbase][annotator].sort_values(by=[1],inplace=True)
      annotator_keys.add(annotator)

print('humans = '+str(humans))
print('precision_recalls = '+str(precision_recalls))
print('words = '+str(timestamps.keys()))

for word in timestamps.keys():
  for csvbase in timestamps[word].keys():
    for annotator in annotator_keys - timestamps[word][csvbase].keys():
      timestamps[word][csvbase][annotator] = pd.DataFrame({1:[], 2:[]})

def get_csvbases(word):
  return natsorted(list(filter(lambda x: x, \
                               [f if any(['-annotated' in a \
                                          for a in timestamps[word][f].keys()]) and \
                                     any(['-predicted-'+pr in a \
                                          for a in timestamps[word][f].keys()]) \
                                  else None for f in timestamps[word].keys()])))

def delete_interval(intervals, idx):
  mask = interval()
  if idx < len(intervals)-1:
    mask |= interval([intervals[idx+1].inf,inf])
  if idx > 0:
    mask |= interval([-inf,intervals[idx-1].sup])
  intervals &= mask
  return intervals

def _interval_diff(A, B):
  ret_val = interval()
  if len(A)==0:
    return ret_val
  if A[0].inf < B[0].inf:
    ret_val |= interval([A[0].inf, min(A[0].sup, B[0].inf-1)])
  if A[0].sup > B[0].sup:
    ret_val |= interval([max(A[0].inf, B[0].sup+1), A[0].sup])
  return ret_val

# set diff limited to len(B)==1
def interval_diff(A,B):
  ret_val = interval()
  for thisinterval in A.components:
    ret_val |= _interval_diff(thisinterval, B)
  return ret_val

def doit(intervals):

  #for the intervals everyone agrees upon (i.e. "everyone"), choose one of the
  #sets at random.  iterate through each interval therein, testing whether it
  #overlaps with any of the intervals in all of the other sets.  if it does,
  #delete the matching intervals in the other sets, and add to the "everyone"
  #set just intersection of the matching intervals.

  intervals_copy = intervals.copy()
  key0 = next(iter(intervals_copy.keys()))
  everyone = interval()
  for interval0 in intervals_copy[key0].components:
    ivalues = {}
    for keyN in set(intervals_copy.keys()) - set([key0]):
      for i,intervalN in enumerate(intervals_copy[keyN].components):
        if len(interval0 & intervalN)>0:
          ivalues[keyN] = i
          break
      if keyN not in ivalues:
        break
    if len(ivalues)==len(intervals_copy)-1:
      for keyN in ivalues.keys():
        interval0 &= interval(intervals_copy[keyN][ivalues[keyN]])
        intervals_copy[keyN] = delete_interval(intervals_copy[keyN], ivalues[keyN])
      everyone |= interval0

  #to calculate the intervals which only one set contains (e.g. "only
  #songexplorer"), iteratively test if each interval therein overlaps
  #with any of the other sets.  if it does, delete the matching intervals
  #in the other sets; otherwise add this interval to the "only word" set.
  #for tics, delete from the interval the points in each matching interval
  #and add what remains to the "only tic" set.

  onlyone_tic = {}
  onlyone_word = {}
  for key0 in intervals.keys():
    intervals_copy = intervals.copy()
    onlyone_tic[key0] = interval()
    onlyone_word[key0] = interval()
    for interval0 in intervals_copy[key0].components:
      ivalues = {}
      for keyN in set(intervals_copy.keys()) - set([key0]):
        for i,intervalN in enumerate(intervals_copy[keyN].components):
          if len(interval0 & intervalN)>0:
            ivalues[keyN] = i
            break
      if len(ivalues)==0:
        onlyone_word[key0] |= interval0
      for keyN in ivalues.keys():
        interval0 = interval_diff(interval0, interval(intervals_copy[keyN][ivalues[keyN]]))
        intervals_copy[keyN] = delete_interval(intervals_copy[keyN], ivalues[keyN])
      onlyone_tic[key0] |= interval0

  #to calculate the intervals which only one set does not contain (e.g. "not
  #david"), choose one of the other sets at random.  iteratively test whether
  #its intervals overlap with an interval in the rest of the other sets
  #but not with the set of interest.  for those intervals which meet this
  #criteria, delete the matching intervals in the rest of the other sets,
  #and add this interval to the "not" set.  for tics, add to the "not tic"
  #set the intersection of all the matching intervals.

  notone_tic = {}
  notone_word = {}
  for key0 in intervals.keys():
    intervals_copy = intervals.copy()
    notone_tic[key0] = interval()
    notone_word[key0] = interval()
    key1 = next(iter(set(intervals_copy.keys()) - set([key0])))
    for interval1 in intervals_copy[key1].components:
      ivalues = {}
      for keyN in set(intervals_copy.keys()) - set([key1]):
        for i,intervalN in enumerate(intervals_copy[keyN].components):
          if len(interval1 & intervalN)>0:
            ivalues[keyN] = i
            break
      if len(ivalues)==len(intervals_copy)-2 and key0 not in ivalues.keys():
        notone_word[key0] |= interval1
        for keyN in ivalues.keys():
          interval1 &= interval(intervals_copy[keyN][ivalues[keyN]])
          intervals_copy[keyN] = delete_interval(intervals_copy[keyN], ivalues[keyN])
        notone_tic[key0] |= interval1
      
  return everyone, onlyone_tic, notone_tic, onlyone_word, notone_word

if congruence_parallelize!=1:
  from multiprocessing import Pool
  nprocs = os.cpu_count() if congruence_parallelize==-1 else congruence_parallelize
  pool = Pool(nprocs)

everyone = {}
onlyone_tic = {}
notone_tic = {}
onlyone_word = {}
notone_word = {}

for pr in precision_recalls:
  print('P/R = '+pr)
  everyone[pr] = {}
  onlyone_tic[pr] = {}
  notone_tic[pr] = {}
  onlyone_word[pr] = {}
  notone_word[pr] = {}

  for word in timestamps.keys():
    print('word = '+word)
    csvbases = get_csvbases(word)
    if len(csvbases)==0:
      continue
    everyone[pr][word] = {}
    onlyone_tic[pr][word] = {}
    notone_tic[pr][word] = {}
    onlyone_word[pr][word] = {}
    notone_word[pr][word] = {}

    timestamps_curated = {}
    for csvbase in csvbases:
      print('csv = '+csvbase)
      predicted_key = '-predicted-'+pr+'.csv'
      if predicted_key not in timestamps[word][csvbase]:
        continue
      intervals = {}
      intervals[pr] = interval(*[[x[1],x[2]] for _,x in \
                                    timestamps[word][csvbase][predicted_key].iterrows()])
      annotated_left = np.inf
      annotated_right = 0
      for human in humans:
        annotator_key = '-annotated-'+human+'.csv'
        annotated_left = min(annotated_left, annotated_left, \
                             *timestamps[word][csvbase][annotator_key][1])
        annotated_right = max(annotated_right, annotated_right, \
                              *timestamps[word][csvbase][annotator_key][2])
        intervals[human] = interval(*[[x[1],x[2]] for _,x in \
                                          timestamps[word][csvbase][annotator_key].iterrows()])
      intervals[pr] &= interval([annotated_left, annotated_right])
      if congruence_parallelize!=0:
        everyone[pr][word][csvbase] = pool.apply_async(doit, (intervals,))
      else:
        everyone[pr][word][csvbase], \
            onlyone_tic[pr][word][csvbase], notone_tic[pr][word][csvbase], \
            onlyone_word[pr][word][csvbase], notone_word[pr][word][csvbase] = \
            doit(intervals)

def plot_file(fig, fig_venn, only_data, not_data):
  ax = fig.add_subplot(nrows,ncols,iplot)
  if len(sorted_hm)<4:
    ax_venn = fig_venn.add_subplot(nrows,ncols,iplot)
    ax_venn.set_title(os.path.basename(csvbase), fontsize=8)
  xdata=['everyone', *['only '+(x if x!=pr else 'songexplorer') for x in sorted_hm]]
  ydata=only_data
  if len(sorted_hm)>2:
    xdata.extend(['not '+(x if x!=pr else 'songexplorer') for x in sorted_hm])
    ydata.extend(not_data)
  if len(sorted_hm)==2:
    idx = [1,2,0]  # Ab, aB, AB
    venn2(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'songexplorer' for x in sorted_hm],
          ax=ax_venn)
  elif len(sorted_hm)==3:
    idx = [1,2,6,3,5,4,0]  # Abc, aBc, ABc, abC, AbC, aBC, ABC
    venn3(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'songexplorer' for x in sorted_hm],
          ax=ax_venn)
  ax.bar(xdata, ydata, color='k')
  ax.set_title(os.path.basename(csvbase), fontsize=8)
  ax.set_xticklabels(xdata, rotation=40, ha='right')

def plot_sumfiles(fig, fig_venn, only_data, not_data):
  ax = fig.add_subplot(nrows,ncols,1)
  if len(sorted_hm)<4:
    ax_venn = fig_venn.add_subplot(nrows,ncols,1)
    ax_venn.set_title('all files', fontsize=8)
  xdata=['everyone', *['only '+(x if x!=pr else 'songexplorer') for x in sorted_hm]]
  ydata=only_data
  if len(sorted_hm)>2:
    xdata.extend(['not '+(x if x!=pr else 'songexplorer') for x in sorted_hm])
    ydata.extend(not_data)
  if len(sorted_hm)==2:
    idx = [1,2,0]
    venn2(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'songexplorer' for x in sorted_hm],
          ax=ax_venn)
  elif len(sorted_hm)==3:
    idx = [1,2,6,3,5,4,0]
    venn3(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'songexplorer' for x in sorted_hm],
          ax=ax_venn)
  ax.bar(xdata, ydata, color='k')
  ax.set_xticklabels(xdata, rotation=40, ha='right')
  ax.set_title('all files', fontsize=8)

for pr in precision_recalls:
  print('P/R = '+pr)
  for word in timestamps.keys():
    print('word = '+word)
    csvbases = get_csvbases(word)
    if len(csvbases)==0:
      continue

    if pr.endswith('pr'):
      all_files_flag = len(csvbases)>1
      nrows = np.floor(np.sqrt(all_files_flag+len(csvbases))).astype(np.int)
      ncols = np.ceil((all_files_flag+len(csvbases))/nrows).astype(np.int)
      fig_tic = plt.figure(figsize=(2*ncols,2*nrows))
      fig_word = plt.figure(figsize=(2*ncols,2*nrows))
      if len(humans)<3:
        fig_tic_venn = plt.figure(figsize=(2*ncols,2*nrows))
        fig_word_venn = plt.figure(figsize=(2*ncols,2*nrows))
      else:
        fig_tic_venn = None
        fig_word_venn = None
      iplot=nrows*ncols

    for csvbase in reversed(csvbases):
      print('csv = '+csvbase)
      predicted_key = '-predicted-'+pr+'.csv'
      if predicted_key not in timestamps[word][csvbase]:
        continue
      if congruence_parallelize!=0:
        everyone[pr][word][csvbase], \
            onlyone_tic[pr][word][csvbase], notone_tic[pr][word][csvbase], \
            onlyone_word[pr][word][csvbase], notone_word[pr][word][csvbase] = \
            everyone[pr][word][csvbase].get()

      if not pr.endswith('pr'):
        continue

      sorted_hm = natsorted(onlyone_tic[pr][word][csvbase].keys())

      plot_file(fig_tic, fig_tic_venn,
                [sum([x[1]-x[0]+1 for x in everyone[pr][word][csvbase]]), \
                 *[sum([y[1]-y[0]+1 for y in onlyone_tic[pr][word][csvbase][x]]) \
                   for x in sorted_hm]],
                [sum([y[1]-y[0]+1 for y in notone_tic[pr][word][csvbase][x]]) \
                 for x in sorted_hm] if len(sorted_hm)>2 else None)
      plot_file(fig_word, fig_word_venn,
                [len(everyone[pr][word][csvbase]), \
                 *[len(onlyone_word[pr][word][csvbase][x]) for x in sorted_hm]],
                [len(notone_word[pr][word][csvbase][x]) for x in sorted_hm]
                    if len(sorted_hm)>2 else None)
      iplot-=1

    if not pr.endswith('pr'):
      continue

    if all_files_flag:
      csvbase0 = list(onlyone_tic[pr][word].keys())[0]
      plot_sumfiles(fig_tic, fig_tic_venn,
                    [sum([x[1]-x[0]+1 for f in everyone[pr][word].values() for x in f]),
                     *[sum([sum([y[1]-y[0]+1 for y in f[hm]])
                            for f in onlyone_tic[pr][word].values()])
                       for hm in sorted_hm]],
                    [sum([sum([y[1]-y[0]+1 for y in f[hm]]) \
                           for f in notone_tic[pr][word].values()])
                     for hm in onlyone_tic[pr][word][csvbase0].keys()]
                        if len(sorted_hm)>2 else None)
      plot_sumfiles(fig_word, fig_word_venn,
                    [sum([len(f) for f in everyone[pr][word].values()]),
                     *[sum([len(f[hm]) for f in onlyone_word[pr][word].values()])
                       for hm in sorted_hm]],
                    [sum([len(f[hm]) for f in notone_word[pr][word].values()])
                     for hm in sorted_hm]
                        if len(sorted_hm)>2 else None)

    fig_tic.tight_layout()
    plt.figure(fig_tic.number)
    plt.savefig(os.path.join(basepath, 'congruence.tic.'+word+'.'+pr+'.pdf'))
    plt.close()
    fig_word.tight_layout()
    plt.figure(fig_word.number)
    plt.savefig(os.path.join(basepath, 'congruence.word.'+word+'.'+pr+'.pdf'))
    plt.close()
    if len(sorted_hm)<4:
      fig_tic_venn.tight_layout()
      plt.figure(fig_tic_venn.number)
      plt.savefig(os.path.join(basepath, 'congruence.tic.'+word+'.'+pr+'-venn.pdf'))
      plt.close()
      fig_word_venn.tight_layout()
      plt.figure(fig_word_venn.number)
      plt.savefig(os.path.join(basepath, 'congruence.word.'+word+'.'+pr+'-venn.pdf'))
      plt.close()

def to_csv(intervals, csvbase, whichset):
  with open(os.path.join(basepath,csvbase+'-disjoint-'+whichset+'.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for iword,word in enumerate(timestamps.keys()):
      for i in intervals[iword]:
        csvwriter.writerow([os.path.basename(csvbase)+'.wav',
                            int(i[0]), int(i[1]), whichset, word])

for pr in filter(lambda x: x.endswith("pr"), precision_recalls):
  for csvbase in csvbases:
    to_csv([everyone[pr][word][csvbase] for word in timestamps.keys()],
            csvbase, 'everyone')
    word0 = next(iter(natsorted(onlyone_tic[pr].keys())))
    sorted_hm = natsorted(onlyone_tic[pr][word][csvbase].keys())
    for hm in sorted_hm:
      to_csv([onlyone_tic[pr][word][csvbase][hm] for word in timestamps.keys()],
              csvbase, 'tic-only'+hm)
      if len(sorted_hm)>2:
        to_csv([notone_tic[pr][word][csvbase][hm] for word in timestamps.keys()],
               csvbase, 'tic-not'+hm)
    for hm in sorted_hm:
      to_csv([onlyone_word[pr][word][csvbase][hm] for word in timestamps.keys()],
             csvbase, 'word-only'+hm)
      if len(sorted_hm)>2:
        to_csv([notone_word[pr][word][csvbase][hm] for word in timestamps.keys()],
               csvbase, 'word-not'+hm)

roc_table_tic = {}
roc_table_word = {}
for pr in precision_recalls:
  for word in sorted(onlyone_tic[pr].keys()):
    if word not in roc_table_tic:
      roc_table_tic[word] = {}
      roc_table_word[word] = {}
    if pr not in roc_table_tic[word]:
      roc_table_tic[word][pr] = {}
      roc_table_word[word][pr] = {}
    csvbase0 = next(iter(onlyone_tic[pr][word].keys()))
    sorted_hm = natsorted(onlyone_tic[pr][word][csvbase0].keys())
    for hm in sorted_hm:
      key = 'only '+hm if hm!=pr else 'only songexplorer'
      roc_table_tic[word][pr][key] = int(sum([sum([y[1]-y[0]+1 for y in f[hm]]) \
                                              for f in onlyone_tic[pr][word].values()]))
      roc_table_word[word][pr][key] = sum([len(f[hm]) for f in onlyone_word[pr][word].values()])
      if len(sorted_hm)>2:
        key = 'not '+hm if hm!=pr else 'not songexplorer'
        roc_table_tic[word][pr][key] = int(sum([sum([y[1]-y[0]+1 for y in f[hm]]) \
                                                for f in notone_tic[pr][word].values()]))
        roc_table_word[word][pr][key] = sum([len(f[hm])
                                             for f in notone_word[pr][word].values()])
    roc_table_tic[word][pr]['everyone'] = int(sum([x[1]-x[0]+1
                                                   for f in everyone[pr][word].values()
                                                   for x in f]))
    roc_table_word[word][pr]['everyone'] = sum([len(f) for f in everyone[pr][word].values()])

if congruence_parallelize!=0:
  pool.close()

def plot_versus_thresholds(roc_table, kind):
  thresholds_touse = {}
  desired_prs = None
  for word in roc_table:
    thresholds = realsorted([x for x in roc_table[word].keys() if x.endswith('th')])
    xdata = [float(x[:-2]) for x in thresholds]
    desired_prs = natsorted([float(x[:-2]) for x in roc_table[word].keys()
                             if x.endswith('pr')])

    fig = plt.figure(figsize=(2*6.4, 4.8))
    ax1 = fig.add_subplot(1,2,1)
    for not_only_every in sorted(roc_table[word][thresholds[0]].keys()):
      ydata = [roc_table[word][x][not_only_every] for x in thresholds]
      ax1.plot(xdata, ydata, '.-' if len(xdata)<10 else '-', label=not_only_every)
      if not_only_every=='everyone':
        TP = ydata
      elif not_only_every=='only songexplorer':
        FP = ydata
      else:
        if len(humans)==1:
          FN = ydata
        elif not_only_every=='not songexplorer':
          FN = ydata
    for (ipr,pr) in enumerate(precision_recalls_sparse):
      th = float([x[1:] for x in thresholds_sparse if x[0]==word][0][ipr])
      if 0<=th<=1 and not np.isnan(th):
        ax1.axvline(x=th, label='sparse P/R = '+pr,
                   color=next(ax1._get_lines.prop_cycler)['color'])
    ax1.set_ylabel('# Annotations')
    ax1.set_xlabel('Threshold')
    ax1.set_xlim(0,1)
    ax2 = fig.add_subplot(1,2,2)
    P = [tp/(tp+fp) if tp+fp>0 else np.nan for (tp,fp) in zip(TP,FP)]
    R = [tp/(tp+fn) if tp+fn>0 else np.nan for (tp,fn) in zip(TP,FN)]
    ax2.plot(R, P, '.-' if len(xdata)<10 else '-')
    F1 = [2*p*r/(p+r) if p+r>0 else np.nan for (p,r) in zip(P,R)]
    ax3 = ax1.twinx()
    ax3.plot(xdata, F1, '.-' if len(xdata)<10 else '-', color='k', label='F1')
    ax3.set_ylabel('F1 = 2PR/(P+R)', color='k')
    ax3.legend(loc=(1.15, 0.0))
    f = interpolate.interp1d([p/r if r!=0 else np.nan for (p,r) in zip(P,R)],
                             xdata, fill_value="extrapolate")
    thresholds_touse[word] = f(desired_prs)
    fP = interpolate.interp1d(xdata, P, fill_value="extrapolate")
    fR = interpolate.interp1d(xdata, R, fill_value="extrapolate")
    for (ipr,pr) in enumerate(precision_recalls_sparse):
      th = float([x[1:] for x in thresholds_sparse if x[0]==word][0][ipr])
      if 0<=th<=1 and not np.isnan(th):
        ax2.plot(fR(th), fP(th), '.', label='sparse P/R = '+pr)
    for (ith,th) in enumerate(thresholds_touse[word]):
      ax2.plot(fR(th), fP(th), '.', label='dense P/R = '+str(desired_prs[ith]))
      if 0<=th<=1 and not np.isnan(th):
        ax1.axvline(x=th, label='dense P/R = '+str(desired_prs[ith]),
                   color=next(ax1._get_lines.prop_cycler)['color'])
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_xlabel('Recall = TP/(TP+FN)')
    ax2.set_ylabel('Precision = TP/(TP+FP)')
    ax2.legend(loc=(1.05, 0.0))
    ax1.legend(loc=(1.2, 0.1))
    fig.tight_layout()
    plt.savefig(os.path.join(basepath,'congruence.'+kind+'.'+word+'.pdf'))
    plt.close()

    with open(os.path.join(basepath,'congruence.'+kind+'.'+word+'.csv'), 'w') as fid:
      csvwriter = csv.writer(fid)
      rows = roc_table[word].keys()
      cols = roc_table[word][next(iter(rows))].keys()
      csvwriter.writerow(['']+list(cols))
      for row in realsorted(rows):
        csvwriter.writerow([row]+[roc_table[word][row][x] for x in cols])

  return thresholds_touse, desired_prs

plot_versus_thresholds(roc_table_tic, kind='tic')
thresholds_touse, desired_prs = plot_versus_thresholds(roc_table_word, kind='word')
 
save_thresholds(logdir, model, ckpt, thresholds_touse, desired_prs,
                list(thresholds_touse.keys()), True)
