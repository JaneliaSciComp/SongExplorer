#!/usr/bin/python3

# generate Venn diagrams of false positives and negatives

# dense.py <folder-with-dense-annotations-and-predictions>

# dense.py /groups/stern/sternlab/behavior/arthurb/congruence-kyriacou2017

import sys
import os
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ion()
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import wave
import csv
import pandas as pd
from scipy import stats
from itertools import chain
from interval import interval
from matplotlib_venn import venn2, venn3
from functools import reduce

_,basepath = sys.argv
print('basepath: '+basepath)

pr_files = filter(lambda x: '-predicted-' in x and x.endswith('.csv'), \
                  os.listdir(basepath))
precision_recalls = list(set([re.search('-predicted-(.*)pr.csv', x).groups(1)[0] \
                              for x in pr_files]))
annotated_files = filter(lambda x: '-annotated-' in x and x.endswith('.csv'), \
                         os.listdir(basepath))
annotators = list(set([re.search('-annotated-(.*).csv', x).groups(1)[0] \
                              for x in annotated_files]))

timestamps = {}

for regexs in [*[['-annotated', '-'+a, '.csv'] for a in annotators], \
              *[['-predicted-'+pr+'pr.csv'] for pr in precision_recalls]]:
  annotator = ''.join(regexs)
  for csvfile in filter(lambda f: all([r in f for r in regexs]), os.listdir(basepath)):
    print(csvfile)
    df = pd.read_csv(os.path.join(basepath,csvfile), header=None)
    m = re.search('(-annotated|-predicted)', csvfile)
    csvbase = csvfile[:m.span(0)[0]]
    for word in set(df[4]):
      if word not in timestamps:
        timestamps[word] = {}
      if csvbase not in timestamps[word]:
        timestamps[word][csvbase] = {}
      timestamps[word][csvbase][annotator] = df.loc[df[4]==word, 1:2]
      timestamps[word][csvbase][annotator].sort_values(by=[1],inplace=True)

def to_csv(timestamps,csvbase,whichset):
  with open(os.path.join(basepath,csvbase+'-disjoint-'+whichset+'.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for i in timestamps:
      csvwriter.writerow([csvbase+'.wav',i,i,'pulse'])

onlythis = {}
notthis = {}
everyone = {}

for precision_recall in precision_recalls:
  print('P/R = '+precision_recall)
  allthis = {}
  onlythis[precision_recall] = {}
  notthis[precision_recall] = {}
  everyone[precision_recall] = {}
  #
  for word in timestamps.keys():
    print('word = '+word)
    csvbases = sorted(list(filter(lambda x: x, \
                           [f if any(['-annotated' in a \
                                      for a in timestamps[word][f].keys()]) and \
                                 any(['-predicted-'+precision_recall+'pr' in a \
                                      for a in timestamps[word][f].keys()]) \
                              else None for f in timestamps[word].keys()])))
    if len(csvbases)==0:
      continue
    csvbases = ['PS_20130625111709_ch3']
    #
    allthis[word] = {}
    onlythis[precision_recall][word] = {}
    notthis[precision_recall][word] = {}
    everyone[precision_recall][word] = 0
    #
    all_files_flag = len(csvbases)>1
    nrows = np.floor(np.sqrt(all_files_flag+len(csvbases))).astype(np.int)
    ncols = np.ceil((all_files_flag+len(csvbases))/nrows).astype(np.int)
    fig = plt.figure(figsize=(2*ncols,2*nrows))
    #
    timestamps_curated = {}
    iplot=all_files_flag+1
    for csvbase in csvbases:
      print('csv = '+csvbase)
      ax = fig.add_subplot(nrows,ncols,iplot)
      predicted_key = '-predicted-'+precision_recall+'pr.csv'
      if predicted_key not in timestamps[word][csvbase]:
        continue
      predicted = interval(*[[x[1],x[2]] for _,x in \
                             timestamps[word][csvbase][predicted_key].iterrows()])
      timestamps_curated[csvbase] = {}
      timestamps_curated[csvbase]['DS'] = set()
      annotated_left = np.inf
      annotated_right = 0
      for annotator in annotators:
        annotator_key = '-annotated-'+annotator+'.csv'
        annotated_left = min(annotated_left, *timestamps[word][csvbase][annotator_key][1])
        annotated_right = max(annotated_right, *timestamps[word][csvbase][annotator_key][2])
        for _,x in timestamps[word][csvbase][annotator_key].iterrows():  ### slow!
          annotated = interval([x[1],x[2]])
          if annotated in predicted:
            intersection = annotated & predicted
            timestamps_curated[csvbase]['DS'].add(int(intersection[0].inf))
        annotator_initials = ''.join([x[0].upper() for x in annotator.split(' ')])
        timestamps_curated[csvbase][annotator_initials] = set([x[1] for _,x in \
                timestamps[word][csvbase][annotator_key].iterrows()])
      deepsong = interval(*list(timestamps_curated[csvbase]['DS']))
      predicted_clipped = predicted & interval([annotated_left, annotated_right])
      for p in predicted_clipped.components:
        if p & deepsong == interval():
          timestamps_curated[csvbase]['DS'].add(int(p[0].inf))
      try:
        ncircles = len(timestamps_curated[csvbase])
        if ncircles==2:
          venn2(timestamps_curated[csvbase].values(), \
                set_labels=timestamps_curated[csvbase].keys())
        elif ncircles==3:
          venn3(timestamps_curated[csvbase].values(), \
                set_labels=timestamps_curated[csvbase].keys())
        else:
          print('more than two human annotators not yet supported')
      except:
        None
      ax.set_title(csvbase, fontsize=8)
      iplot+=1
      for hm in timestamps_curated[csvbase].keys():
        if hm not in allthis[word]:
          allthis[word][hm] = []
        allthis[word][hm].extend([csvbase+str(x) for x in timestamps_curated[csvbase][hm]])
        U = reduce(lambda x,y: x|y, [timestamps_curated[csvbase][x] \
                   for x in timestamps_curated[csvbase].keys() - set([hm])])
        onlythisk = timestamps_curated[csvbase][hm] - U
        to_csv(onlythisk, csvbase, 'only'+hm)
        if hm not in onlythis[precision_recall][word]:
          onlythis[precision_recall][word][hm] = 0
        onlythis[precision_recall][word][hm] += len(onlythisk)
        I = reduce(lambda x,y: x&y, [timestamps_curated[csvbase][x] \
                   for x in timestamps_curated[csvbase].keys() - set([hm])])
        notthisk = I - timestamps_curated[csvbase][hm]
        to_csv(notthisk, csvbase, 'not'+hm)
        if hm not in notthis[precision_recall][word]:
          notthis[precision_recall][word][hm] = 0
        notthis[precision_recall][word][hm] += len(notthisk)
      I = reduce(lambda x,y: x&y, [timestamps_curated[csvbase][x] \
                 for x in timestamps_curated[csvbase].keys()])
      to_csv(I, csvbase, 'everyone')
      everyone[precision_recall][word] += len(I)
    #
    if all_files_flag:
      ax = fig.add_subplot(nrows,ncols,1)
      ncircles = len(allthis[word])
      if ncircles==2:
        venn2([set(x) for x in allthis[word].values()], set_labels=allthis[word].keys())
      elif ncircles==3:
        venn3([set(x) for x in allthis[word].values()], set_labels=allthis[word].keys())
      else:
        print('more than two human annotators not yet supported')
      ax.set_title('all files', fontsize=8)
    #
    fig.tight_layout()
    plt.savefig(os.path.join(basepath,'congruence.'+precision_recall+'.'+word+'.pdf'))
  #
  congruence_table = []
  for w in onlythis[precision_recall].keys():
    if congruence_table==[]:
      column = []
      for hm in onlythis[precision_recall][w].keys():
        column.append('only '+hm)
        column.append('not '+hm)
      column.append('everyone')
      congruence_table.append(column)
    column = []
    for hm in onlythis[precision_recall][w].keys():
      column.append(onlythis[precision_recall][w][hm])
      column.append( notthis[precision_recall][w][hm])
    column.append(everyone[precision_recall][w])
    congruence_table.append(column)
  #
  congruence_table = list(np.transpose(congruence_table))
  congruence_table.insert(0,['word', *onlythis[precision_recall].keys()])
  with open(os.path.join(basepath,'congruence.'+precision_recall+'.csv'), 'w') as fid:
    csvwriter = csv.writer(fid)
    for x in congruence_table:
      csvwriter.writerow(x)

if len(precision_recalls)>1:
  key1 = list(onlythis.keys())[0]
  fig = plt.figure()
  words = list(onlythis[key1].keys())
  for iplot,word in enumerate(words):
    ax = fig.add_subplot(1,len(words),iplot+1)
    xdata = [float(x) for x in everyone.keys()]
    ydata = [x[word] for x in everyone.values()]
    xdata, ydata = zip(*sorted(zip(xdata, ydata)))
    line, = ax.plot(xdata, ydata, '.-')
    line.set_label('everyone')
    #
    ydata_sum = []
    for annotator in list(onlythis[key1][word].keys()):
      xdata = [float(x) for x in onlythis.keys()]
      ydata = [x[word][annotator] for x in onlythis.values()]
      xdata, ydata = zip(*sorted(zip(xdata, ydata)))
      ydata_sum.append(ydata)
      line, = ax.plot(xdata, ydata, '.-')
      line.set_label('only '+annotator)
    #
    line, = ax.plot(xdata, np.sum(np.array(ydata_sum),0), '.-')
    line.set_label('sum only')
    #
    ax.set_ylabel('# annotations')
    ax.set_xlabel('precision / recall')
    ax.legend(bbox_to_anchor=(1.1, 1.05))
  #
  fig.tight_layout()
  plt.savefig(os.path.join(basepath,'congruence-vs-threshold.pdf'))
