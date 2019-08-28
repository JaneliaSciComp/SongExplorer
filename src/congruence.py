#!/usr/bin/python3

# generate figures of false positives and negatives

# congruence.py <path-to-groundtruth> <wavfiles-with-dense-annotations-and-predictions> <congruence-parallelize>

# congruence.py /groups/stern/sternlab/behavior/arthurb/groundtruth/kyriacou2017 PS_20130625111709_ch3.wav,PS_20130625111709_ch7.wav 1

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
from itertools import chain
from interval import interval
from functools import reduce
from natsort import natsorted
from matplotlib_venn import venn2, venn3

_,basepath,wavfiles,congruence_parallelize = sys.argv
print('basepath: '+basepath)
print('wavfiles: '+wavfiles)
print('congruence_parallelize: '+congruence_parallelize)
wavfiles=set([os.path.basename(x) for x in wavfiles.split(',')])
congruence_parallelize=bool(int(congruence_parallelize))

wavdirs = {}
for subdir in filter(lambda x: os.path.isdir(os.path.join(basepath,x)), \
                     os.listdir(basepath)):
  commonsubfiles = wavfiles & set(os.listdir(os.path.join(basepath, subdir)))
  if len(commonsubfiles) > 0:
    wavdirs[subdir] = commonsubfiles

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
      precision_recalls.add(m.groups()[1][:-2])
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
                                     any(['-predicted-'+pr+'pr' in a \
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
      predicted_key = '-predicted-'+pr+'pr.csv'
      if predicted_key not in timestamps[word][csvbase]:
        continue
      intervals = {}
      intervals[pr+"pr"] = interval(*[[x[1],x[2]] for _,x in \
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
      intervals[pr+"pr"] &= interval([annotated_left, annotated_right])
      if congruence_parallelize:
        everyone[pr][word][csvbase] = pool.apply_async(doit, (intervals,))
      else:
        everyone[pr][word][csvbase], \
            onlyone_tic[pr][word][csvbase], notone_tic[pr][word][csvbase], \
            onlyone_word[pr][word][csvbase], notone_word[pr][word][csvbase] = \
            doit(intervals)

def to_csv(intervals, csvbase, whichset, word):
  with open(os.path.join(basepath,csvbase+'-disjoint-'+whichset+'.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for i in intervals:
      csvwriter.writerow([os.path.basename(csvbase)+'.wav',
                          int(i[0]), int(i[1]), whichset, word])

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

    iplot=nrows*ncols
    for csvbase in reversed(csvbases):
      print('csv = '+csvbase)
      predicted_key = '-predicted-'+pr+'pr.csv'
      if predicted_key not in timestamps[word][csvbase]:
        continue
      if congruence_parallelize:
        everyone[pr][word][csvbase], \
            onlyone_tic[pr][word][csvbase], notone_tic[pr][word][csvbase], \
            onlyone_word[pr][word][csvbase], notone_word[pr][word][csvbase] = \
            everyone[pr][word][csvbase].get()

      sorted_hm = natsorted(onlyone_tic[pr][word][csvbase].keys())

      ax = fig_tic.add_subplot(nrows,ncols,iplot)
      if len(sorted_hm)<4:
        ax_venn = fig_tic_venn.add_subplot(nrows,ncols,iplot)
      xdata=['everyone', *['only '+x for x in sorted_hm]]
      ydata=[sum([x[1]-x[0] for x in everyone[pr][word][csvbase]]), \
             *[sum([y[1]-y[0] for y in onlyone_tic[pr][word][csvbase][x]]) \
               for x in sorted_hm]] 
      if len(sorted_hm)>2:
        xdata.extend(['not '+x for x in sorted_hm])
        ydata.extend([sum([y[1]-y[0] for y in notone_tic[pr][word][csvbase][x]]) \
                      for x in sorted_hm])
      if len(sorted_hm)==2:
        idx = [1,2,0]  # Ab, aB, AB
        venn2(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      elif len(sorted_hm)==3:
        idx = [1,2,6,3,5,4,0]  # Abc, aBc, ABc, abC, AbC, aBC, ABC
        venn3(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      ax_venn.set_title(os.path.basename(csvbase), fontsize=8)
      ax.bar(xdata, ydata, color='k')
      ax.set_title(os.path.basename(csvbase), fontsize=8)
      ax.set_xticklabels(xdata, rotation=40, ha='right')

      ax = fig_word.add_subplot(nrows,ncols,iplot)
      if len(sorted_hm)<4:
        ax_venn = fig_word_venn.add_subplot(nrows,ncols,iplot)
      xdata=['everyone', *['only '+x for x in sorted_hm]]
      ydata=[len(everyone[pr][word][csvbase]), \
             *[len(onlyone_word[pr][word][csvbase][x]) for x in sorted_hm]] 
      if len(sorted_hm)>2:
        xdata.extend(['not '+x for x in sorted_hm])
        ydata.extend([len(notone_word[pr][word][csvbase][x]) for x in sorted_hm])
      if len(sorted_hm)==2:
        idx = [1,2,0]
        venn2(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      elif len(sorted_hm)==3:
        idx = [1,2,6,3,5,4,0]
        venn3(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      ax_venn.set_title(os.path.basename(csvbase), fontsize=8)
      ax.bar(xdata, ydata, color='k')
      ax.set_title(os.path.basename(csvbase), fontsize=8)
      ax.set_xticklabels(xdata, rotation=40, ha='right')
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
      ax = fig_tic.add_subplot(nrows,ncols,1)
      if len(sorted_hm)<4:
        ax_venn = fig_tic_venn.add_subplot(nrows,ncols,1)
      xdata=['everyone', *['only '+x for x in sorted_hm]]
      ydata=[sum([x[1]-x[0] for f in everyone[pr][word].values() for x in f])]
      csvbase0 = list(onlyone_tic[pr][word].keys())[0]
      for hm in sorted_hm:
        ydata.append(sum([sum([y[1]-y[0] for y in f[hm]]) \
                          for f in onlyone_tic[pr][word].values()]))
      if len(sorted_hm)>2:
        xdata.extend(['not '+x for x in sorted_hm])
        for hm in onlyone_tic[pr][word][csvbase0].keys():
          ydata.append(sum([sum([y[1]-y[0] for y in f[hm]]) \
                            for f in notone_tic[pr][word].values()]))
      if len(sorted_hm)==2:
        idx = [1,2,0]
        venn2(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      elif len(sorted_hm)==3:
        idx = [1,2,6,3,5,4,0]
        venn3(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      ax_venn.set_title('all files', fontsize=8)
      ax.bar(xdata, ydata, color='k')
      ax.set_xticklabels(xdata, rotation=40, ha='right')
      ax.set_title('all files', fontsize=8)

      ax = fig_word.add_subplot(nrows,ncols,1)
      if len(sorted_hm)<4:
        ax_venn = fig_word_venn.add_subplot(nrows,ncols,1)
      xdata=['everyone', *['only '+x for x in sorted_hm]]
      ydata=[sum([len(f) for f in everyone[pr][word].values()])]
      csvbase0 = list(onlyone_word[pr][word].keys())[0]
      for hm in sorted_hm:
        ydata.append(sum([len(f[hm]) for f in onlyone_word[pr][word].values()]))
      if len(sorted_hm)>2:
        xdata.extend(['not '+x for x in sorted_hm])
        for hm in sorted_hm:
          ydata.append(sum([len(f[hm]) for f in notone_word[pr][word].values()]))
      if len(sorted_hm)==2:
        idx = [1,2,0]
        venn2(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      elif len(sorted_hm)==3:
        idx = [1,2,6,3,5,4,0]
        venn3(subsets=[ydata[x] for x in idx], set_labels=sorted_hm, ax=ax_venn)
      ax_venn.set_title('all files', fontsize=8)
      ax.bar(xdata, ydata, color='k')
      ax.set_xticklabels(xdata, rotation=40, ha='right')
      ax.set_title('all files', fontsize=8)

    fig_tic.tight_layout()
    plt.figure(fig_tic.number)
    plt.savefig(os.path.join(basepath,'congruence-tic.'+pr+'pr.'+word+'.pdf'))
    plt.close()
    fig_word.tight_layout()
    plt.figure(fig_word.number)
    plt.savefig(os.path.join(basepath,'congruence-word.'+pr+'pr.'+word+'.pdf'))
    plt.close()
    if len(sorted_hm)<4:
      fig_tic_venn.tight_layout()
      plt.figure(fig_tic_venn.number)
      plt.savefig(os.path.join(basepath,'congruence-tic.'+pr+'pr.'+word+'-venn.pdf'))
      plt.close()
      fig_word_venn.tight_layout()
      plt.figure(fig_word_venn.number)
      plt.savefig(os.path.join(basepath,'congruence-word.'+pr+'pr.'+word+'-venn.pdf'))
      plt.close()

  congruence_table_tic = []
  congruence_table_word = []
  for word in sorted(onlyone_tic[pr].keys()):
    csvbase0 = list(onlyone_tic[pr][word].keys())[0]
    if congruence_table_tic==[]:
      column = []
      for hm in sorted_hm:
        column.append('only '+hm)
        if len(sorted_hm)>2:
          column.append('not '+hm)
      column.append('everyone')
      congruence_table_tic.append(column)
      congruence_table_word.append(column)
    column_tic = []
    column_word = []
    for hm in sorted_hm:
      column_tic.append(int(sum([sum([y[1]-y[0] for y in f[hm]]) \
                                 for f in onlyone_tic[pr][word].values()])))
      column_word.append(sum([len(f[hm]) for f in onlyone_word[pr][word].values()]))
      if len(sorted_hm)>2:
        column_tic.append(int(sum([sum([y[1]-y[0] for y in f[hm]]) \
                                   for f in notone_tic[pr][word].values()])))
        column_word.append(sum([len(f[hm]) for f in notone_word[pr][word].values()]))
    column_tic.append(int(sum([x[1]-x[0] for f in everyone[pr][word].values() for x in f])))
    column_word.append(sum([len(f) for f in everyone[pr][word].values()]))
    congruence_table_tic.append(column_tic)
    congruence_table_word.append(column_word)

  congruence_table_tic = list(np.transpose(congruence_table_tic))
  congruence_table_tic.insert(0,['word', *sorted(onlyone_tic[pr].keys())])
  with open(os.path.join(basepath,'congruence-tic.'+pr+'pr.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for x in congruence_table_tic:
      csvwriter.writerow(x)
  congruence_table_word = list(np.transpose(congruence_table_word))
  congruence_table_word.insert(0,['word', *sorted(onlyone_word[pr].keys())])
  with open(os.path.join(basepath,'congruence-word.'+pr+'pr.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for x in congruence_table_word:
      csvwriter.writerow(x)

if congruence_parallelize:
  pool.close()
