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
from itertools import chain, cycle
from interval import interval
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
congruence_parallelize=bool(int(congruence_parallelize))

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

class MyInterval(interval):
    def complement(self) -> 'MyInterval':
        mychain = chain(self, [[sys.maxsize, None]])
        out = []
        prev = [None, 0]
        for this in mychain:
            if prev[1] != this[0]:
                out.append([prev[1], this[0]])
            prev = this
        return self.__class__(*out)

def doit(intervals):
  everyone = reduce(lambda x,y: x&y, intervals.values())
  onlyone_tic = {}
  notone_tic = {}
  onlyone_word = {}
  notone_word = {}
  for hm in intervals.keys():
    onlyone_tic[hm] = intervals[hm]
    for x in intervals.keys()-set([hm]):
      onlyone_tic[hm] &= MyInterval(*intervals[x]).complement()
    notone_tic[hm] = interval()
    for x in intervals.keys()-set([hm]):
      notone_tic[hm] |= intervals[x]
    notone_tic[hm] &= MyInterval(*intervals[hm]).complement()
    someoneelse = reduce(lambda x,y: x|y, set(intervals.values())-set([intervals[hm]]))
    onlyone_word[hm] = interval()
    for x in intervals[hm].components:
      intersection = x & someoneelse
      if len(intersection)==0:
        onlyone_word[hm] |= x
    everyoneelse = reduce(lambda x,y: x&y, set(intervals.values())-set([intervals[hm]]))
    notone_word[hm] = interval()
    for x in everyoneelse.components:
      intersection = x & intervals[hm]
      if len(intersection)==0:
        notone_word[hm] |= x
  return everyone, onlyone_tic, notone_tic, onlyone_word, notone_word

if congruence_parallelize:
  from multiprocessing import Pool
  pool = Pool()

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
      if congruence_parallelize:
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
  xdata=['everyone', *['only '+(x if x!=pr else 'deepsong') for x in sorted_hm]]
  ydata=only_data
  if len(sorted_hm)>2:
    xdata.extend(['not '+(x if x!=pr else 'deepsong') for x in sorted_hm])
    ydata.extend(not_data)
  if len(sorted_hm)==2:
    idx = [1,2,0]  # Ab, aB, AB
    venn2(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'deepsong' for x in sorted_hm],
          ax=ax_venn)
  elif len(sorted_hm)==3:
    idx = [1,2,6,3,5,4,0]  # Abc, aBc, ABc, abC, AbC, aBC, ABC
    venn3(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'deepsong' for x in sorted_hm],
          ax=ax_venn)
  ax_venn.set_title(os.path.basename(csvbase), fontsize=8)
  ax.bar(xdata, ydata, color='k')
  ax.set_title(os.path.basename(csvbase), fontsize=8)
  ax.set_xticklabels(xdata, rotation=40, ha='right')

def plot_sumfiles(fig, fig_venn, only_data, not_data):
  ax = fig.add_subplot(nrows,ncols,1)
  if len(sorted_hm)<4:
    ax_venn = fig_venn.add_subplot(nrows,ncols,1)
  xdata=['everyone', *['only '+(x if x!=pr else 'deepsong') for x in sorted_hm]]
  ydata=only_data
  if len(sorted_hm)>2:
    xdata.extend(['not '+(x if x!=pr else 'deepsong') for x in sorted_hm])
    ydata.extend(not_data)
  if len(sorted_hm)==2:
    idx = [1,2,0]
    venn2(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'deepsong' for x in sorted_hm],
          ax=ax_venn)
  elif len(sorted_hm)==3:
    idx = [1,2,6,3,5,4,0]
    venn3(subsets=[ydata[x] for x in idx],
          set_labels=[x if x!=pr else 'deepsong' for x in sorted_hm],
          ax=ax_venn)
  ax_venn.set_title('all files', fontsize=8)
  ax.bar(xdata, ydata, color='k')
  ax.set_xticklabels(xdata, rotation=40, ha='right')
  ax.set_title('all files', fontsize=8)

def to_csv(intervals, csvbase, whichset, word):
  with open(os.path.join(basepath,csvbase+'-disjoint-'+whichset+'.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for i in intervals:
      csvwriter.writerow([os.path.basename(csvbase)+'.wav',
                          int(i[0]), int(i[1]), whichset, word])

if not os.path.isdir(os.path.join(basepath,'congruence.bar-venn')):
  os.mkdir(os.path.join(basepath,'congruence.bar-venn'))

roc_table_tic = {}
roc_table_word = {}
for pr in precision_recalls:
  print('P/R = '+pr)
  for word in timestamps.keys():
    print('word = '+word)
    csvbases = get_csvbases(word)
    if len(csvbases)==0:
      continue

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
      if congruence_parallelize:
        everyone[pr][word][csvbase], \
            onlyone_tic[pr][word][csvbase], notone_tic[pr][word][csvbase], \
            onlyone_word[pr][word][csvbase], notone_word[pr][word][csvbase] = \
            everyone[pr][word][csvbase].get()

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

      to_csv(everyone[pr][word][csvbase], csvbase, 'everyone', word)
      for hm in sorted_hm:
        to_csv(onlyone_tic[pr][word][csvbase][hm], csvbase, 'tic-only'+hm, word)
        if len(sorted_hm)>2:
          to_csv(notone_tic[pr][word][csvbase][hm], csvbase, 'tic-not'+hm, word)
      for hm in sorted_hm:
        to_csv(onlyone_word[pr][word][csvbase][hm], csvbase, 'word-only'+hm, word)
        if len(sorted_hm)>2:
          to_csv(notone_word[pr][word][csvbase][hm], csvbase, 'word-not'+hm, word)

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
    plt.savefig(os.path.join(basepath, 'congruence.bar-venn',
                             'congruence.tic.'+word+'.'+pr+'.pdf'))
    plt.close()
    fig_word.tight_layout()
    plt.figure(fig_word.number)
    plt.savefig(os.path.join(basepath, 'congruence.bar-venn',
                             'congruence.word.'+word+'.'+pr+'.pdf'))
    plt.close()
    if len(sorted_hm)<4:
      fig_tic_venn.tight_layout()
      plt.figure(fig_tic_venn.number)
      plt.savefig(os.path.join(basepath, 'congruence.bar-venn',
                               'congruence.tic.'+word+'.'+pr+'-venn.pdf'))
      plt.close()
      fig_word_venn.tight_layout()
      plt.figure(fig_word_venn.number)
      plt.savefig(os.path.join(basepath, 'congruence.bar-venn',
                               'congruence.word.'+word+'.'+pr+'-venn.pdf'))
      plt.close()

  for word in sorted(onlyone_tic[pr].keys()):
    if word not in roc_table_tic:
      roc_table_tic[word] = {}
      roc_table_word[word] = {}
    if pr not in roc_table_tic[word]:
      roc_table_tic[word][pr] = {}
      roc_table_word[word][pr] = {}
    for hm in sorted_hm:
      key = 'only '+hm if hm!=pr else 'only deepsong'
      roc_table_tic[word][pr][key] = int(sum([sum([y[1]-y[0]+1 for y in f[hm]]) \
                                              for f in onlyone_tic[pr][word].values()]))
      roc_table_word[word][pr][key] = sum([len(f[hm]) for f in onlyone_word[pr][word].values()])
      if len(sorted_hm)>2:
        key = 'not '+hm if hm!=pr else 'not deepsong'
        roc_table_tic[word][pr][key] = int(sum([sum([y[1]-y[0]+1 for y in f[hm]]) \
                                                for f in notone_tic[pr][word].values()]))
        roc_table_word[word][pr][key] = sum([len(f[hm])
                                             for f in notone_word[pr][word].values()])
    roc_table_tic[word][pr]['everyone'] = int(sum([x[1]-x[0]+1
                                                   for f in everyone[pr][word].values()
                                                   for x in f]))
    roc_table_word[word][pr]['everyone'] = sum([len(f) for f in everyone[pr][word].values()])

if congruence_parallelize:
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
      elif not_only_every=='only deepsong':
        FP = ydata
      else:
        if len(humans)==1:
          FN = ydata
        elif not_only_every=='not deepsong':
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
    f = interpolate.interp1d([p/r for (p,r) in zip(P,R)], xdata, fill_value="extrapolate")
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
