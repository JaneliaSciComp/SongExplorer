#!/usr/bin/python3

# plot accuracy across hyperparameter values
 
# compare.py <logdirs-prefix>

# e.g.
# compare.py `pwd`/withhold

import sys
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ion()
import matplotlib.cm as cm
from natsort import natsorted
from functools import reduce
from datetime import datetime
import socket

repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(repodir, "src"))
from lib import *
from jitter import *

print(str(datetime.now())+": start time")
with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
  print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
print("hostname = "+socket.gethostname())

try:

  _,logdirs_prefix = sys.argv
  print('logdirs_prefix: '+logdirs_prefix)

  basename, dirname = os.path.split(logdirs_prefix)

  same_time=False
  outlier_criteria=50

  train_time={}
  validation_accuracy={}
  validation_time={}
  validation_step={}
  labels_touse={}
  label_counts={}
  nparameters_total={}
  nparameters_finallayer={}
  batch_size={}
  nlayers={}
  hyperparameters={}

  logdirs = list(filter(lambda x: x.startswith(dirname+'-') and \
                        os.path.isdir(os.path.join(basename,x)), os.listdir(basename)))

  for logdir in logdirs:
    print(logdir)
    hyperparameters[logdir] = set(logdir.split('-')[1].split('_'))
    _, _, train_time[logdir], _, \
            _, _, validation_accuracy[logdir], validation_time[logdir], validation_step[logdir], \
            _, _, _, \
            labels_touse[logdir], label_counts[logdir], \
            nparameters_total[logdir], nparameters_finallayer[logdir], \
            batch_size[logdir], nlayers[logdir] = \
            read_logs(os.path.join(basename,logdir))
    if len(set([tuple(x) for x in labels_touse[logdir].values()]))>1:
      print('WARNING: not all labels_touse are the same')
    #assert len(set(label_counts[logdir].values()))==1
    if len(set(nparameters_total[logdir].values()))>1:
      print('WARNING: not all nparameters_total are the same')
    if len(set(nparameters_finallayer[logdir].values()))>1:
      print('WARNING: not all nparameters_finallayer are the same')
    if len(set(batch_size[logdir].values()))>1:
      print('WARNING: not all batch_size are the same')
    if len(set(nlayers[logdir].values()))>1:
      print('WARNING: not all nlayers are the same')
    if len(validation_accuracy)>0:
      if set([tuple(x) for x in labels_touse[logdirs[0]].values()])!=set([tuple(x) for x in labels_touse[logdir].values()]):
        print('WARNING: not all labels_touse are the same')
      #assert set(label_counts[logdirs[0]].values())==set(label_counts[logdir].values())
      if set(nparameters_total[logdirs[0]].values())!=set(nparameters_total[logdir].values()):
        print('WARNING: not all nparameters_total are the same')
      if set(nparameters_finallayer[logdirs[0]].values())!=set(nparameters_finallayer[logdir].values()):
        print('WARNING: not all nparameters_finallayer are the same')
      if set(batch_size[logdirs[0]].values())!=set(batch_size[logdir].values()):
        print('WARNING: not all batch_size are the same')
      if set(nlayers[logdirs[0]].values())!=set(nlayers[logdir].values()):
        print('WARNING: not all nlayers are the same')

  commonparameters = reduce(lambda x,y: x&y, hyperparameters.values())
  differentparameters = {x:','.join(natsorted(list(hyperparameters[x]-commonparameters))) \
                         for x in natsorted(logdirs)}

  fig = plt.figure(figsize=(8,10*2/3))

  ax = fig.add_subplot(2,2,1)
  min_time, max_time, idx_time = \
          plot_final_accuracies(ax, validation_accuracy, \
                                dirname, 'Overall accuracy', outlier_criteria, \
                                llabels=differentparameters, \
                                times=validation_time if same_time else None)

  if same_time:
    for model in validation_time.keys():
      this_min_time = min([f[-1] for f in validation_time[model].values()])
      this_min_step = next(iter(validation_step[model].values()))[-1]
      need_total_steps = max_time / this_min_time * this_min_step
      need_more_steps = (max_time-this_min_time) / this_min_time * this_min_step
      print(model+" needs "+str(round(need_more_steps))+" more steps for "+
            str(round(need_total_steps))+" steps total")

  ax = fig.add_subplot(2,2,2)
  plot_time_traces(ax, validation_time, validation_accuracy, 'Overall accuracy', dirname, \
                   outlier_crit=outlier_criteria, llabels=differentparameters, \
                   min_time=min_time)

  ax = fig.add_subplot(2,2,3)
  ldata = natsorted(nparameters_total.keys())
  xdata = range(len(ldata))
  ydata = [next(iter(nparameters_total[x].values())) - \
           next(iter(nparameters_finallayer[x].values())) for x in ldata]
  ydata2 = [next(iter(nparameters_finallayer[x].values())) for x in ldata]
  bar1 = ax.bar(xdata,ydata,color='k')
  bar2 = ax.bar(xdata,ydata2,bottom=ydata,color='gray')
  ax.legend((bar2,bar1), ('last','rest'))
  ax.set_xlabel(dirname)
  ax.set_ylabel('Trainable parameters')
  ax.set_xticks(xdata)
  ax.set_xticklabels([differentparameters[x] for x in ldata], rotation=40, ha='right')

  ax = fig.add_subplot(2,2,4)
  data = {k:list([np.median(np.diff(x)) for x in train_time[k].values()]) for k in train_time}
  ldata = jitter_plot(ax, data)
  ax.set_ylabel('time / step (ms)')
  ax.set_xlabel(dirname)
  ax.set_xticks(range(len(ldata)))
  ax.set_xticklabels([differentparameters[x] for x in ldata], rotation=40, ha='right')

  fig.suptitle(','.join(list(commonparameters)))

  fig.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.savefig(logdirs_prefix+'-compare-overall-params-speed.pdf')
  plt.close()


  summed_confusion_matrices={}
  confusion_matrices={}
  recall_confusion_matrices={}
  precision_confusion_matrices={}
  recall_summed_matrices={}
  precision_summed_matrices={}
  summed_accuracies={}
  labels=[]

  for logdir in logdirs:
    kind = next(iter(validation_time[logdir].keys())).split('_')[0]
    summed_confusion_matrices[logdir], confusion_matrices[logdir], theselabels = \
            parse_confusion_matrices(os.path.join(basename,logdir), kind, \
                                     idx_time=idx_time[logdir] if same_time else None)
    if labels:
      assert set(labels)==set(theselabels)
    else:
      labels=theselabels
    if labels!=theselabels:
      idx = [labels.index(x) for x in theselabels]
      print(idx)
      summed_confusion_matrices[logdir] = [[summed_confusion_matrices[logdir][i][j] \
                                            for j in idx] for i in idx]
      confusion_matrices[logdir] = {k: [[confusion_matrices[logdir][k][i][j] \
                                        for j in idx] for i in idx] \
                                    for k in confusion_matrices[logdir].keys()}

    recall_confusion_matrices[logdir]={}
    precision_confusion_matrices[logdir]={}
    for model in confusion_matrices[logdir].keys():
      recall_confusion_matrices[logdir][model], \
                precision_confusion_matrices[logdir][model], _ = \
            normalize_confusion_matrix(confusion_matrices[logdir][model])

    recall_summed_matrices[logdir], \
                precision_summed_matrices[logdir], _ = \
            normalize_confusion_matrix(summed_confusion_matrices[logdir])
    summed_accuracies[logdir] = 100 * np.trace(recall_summed_matrices[logdir]) / \
                      len(recall_summed_matrices[logdir])

  plot_confusion_matrices(summed_confusion_matrices,
                          precision_summed_matrices, recall_summed_matrices,
                          labels, summed_accuracies, natsorted(logdirs),
                          numbers=len(labels)<10)
  plt.savefig(logdirs_prefix+'-compare-confusion-matrices.pdf')
  plt.close()


  nrows, ncols = layout(len(logdirs))
  scale=6.4
  fig = plt.figure(figsize=(scale*ncols, scale*3/4*nrows))

  minprecision=1.0
  minrecall=1.0
  for logdir in logdirs:
    for model in recall_confusion_matrices[logdir].keys():
      for ilabel in range(len(labels)):
        minprecision = min(minprecision,
                           precision_confusion_matrices[logdir][model][ilabel][ilabel])
        minrecall = min(minrecall,
                        recall_confusion_matrices[logdir][model][ilabel][ilabel])

  for (ilogdir,logdir) in enumerate(natsorted(logdirs)):
    ax = fig.add_subplot(nrows, ncols, ilogdir+1)
    model0=list(recall_confusion_matrices[logdir].keys())[0]
    for model in recall_confusion_matrices[logdir].keys():
      ax.set_prop_cycle(None)
      for (ilabel,label) in enumerate(labels):
        line, = ax.plot(recall_confusion_matrices[logdir][model][ilabel][ilabel],
                        precision_confusion_matrices[logdir][model][ilabel][ilabel],
                        'o', markeredgecolor='k')
        if ilogdir==0 and model==model0:
          line.set_label(label)
    ax.set_xlim(left=minrecall, right=1)
    ax.set_ylim(bottom=minprecision, top=1)
    if ilogdir//ncols==nrows-1:
      ax.set_xlabel('Recall')
    if ilogdir%ncols==0:
      ax.set_ylabel('Precision')
    if ilogdir==len(logdirs)-1:
      lgd = fig.legend(bbox_to_anchor=(1.0,0.0), loc='lower left')
    ax.set_title(logdir+"   "+str(round(summed_accuracies[logdir],1))+"%")

  fig.tight_layout()
  plt.savefig(logdirs_prefix+'-compare-precision-recall.pdf',
              bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()

except Exception as e:
  print(e)

finally:
  os.sync()
  print(str(datetime.now())+": finish time")
