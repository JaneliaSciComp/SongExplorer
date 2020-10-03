#!/usr/bin/python3
# generate confusion matrices, precision-recall curves, thresholds, etc.
 
# accuracy.py <logdir> <error-ratios> <nprobabilities>

# e.g.
# accuracy.py trained-classifier 2,1,0.5 50

import sys
import os
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ion()
import csv
from natsort import natsorted, index_natsorted
import matplotlib.cm as cm

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

from scipy import interpolate
from scipy import stats
import math

_,logdir,error_ratios,nprobabilities,accuracy_parallelize = sys.argv
print('logdir: '+logdir)
print('error_ratios: '+error_ratios)
print('nprobabilities: '+nprobabilities)
print('accuracy_parallelize: '+accuracy_parallelize)
error_ratios = [float(x) for x in error_ratios.split(',')]
nprobabilities = int(nprobabilities)
accuracy_parallelize=bool(int(accuracy_parallelize))

train_accuracy, train_loss, train_time, train_step, \
      validation_precision, validation_recall, validation_accuracy, \
      validation_time, validation_step, \
      _, _, _, \
      wanted_words, word_counts, _, _, batch_size, _, _ = \
      read_logs(logdir)
training_set_size = {x: len(word_counts[x]) * np.max(list(word_counts[x].values())) \
                                for x in word_counts.keys()}

keys_to_plot = natsorted(train_step.keys())

nrows, ncols = layout(len(keys_to_plot))
fig = plt.figure(figsize=(6.4*ncols, 4.8*nrows))
fig.subplots_adjust(bottom=0.2)
for (imodel,model) in enumerate(keys_to_plot):
  ax4 = fig.add_subplot(nrows, ncols, imodel+1)
  nx = len(train_step[model])
  idx = range(nx) if nx<10000 else np.round(np.linspace(0, nx-1, 10000)).astype(np.int)
  ax4.plot(np.array(train_step[model])[idx],
           np.array(train_loss[model])[idx],
           'g', label='Cross entropy')
  ax4.set_xlabel('Step')
  ax4.set_ylabel('Loss', color='g')
  ax4.tick_params(axis='y', labelcolor='g')
  ax1 = ax4.twinx()
  ax1.plot(np.array(train_step[model])[idx],
           np.array(train_accuracy[model])[idx],
           'b', label='Train')
  validation_intervals = list(zip(validation_step[model][0:-1], \
                                  validation_step[model][1:]))
  train_accuracy_ave = [np.mean(train_accuracy[model][train_step[model].index(x): \
                                                      train_step[model].index(y)]) \
                        for (x,y) in validation_intervals]
  ax1.plot([(x+y)/2 for (x,y) in validation_intervals], train_accuracy_ave, 'c', label='Train mean')
  if validation_accuracy[model]:
    ax1.plot(validation_step[model], validation_accuracy[model], 'r', label='Validation')
    ax1.set_title(model+"   "+str(round(validation_accuracy[model][-1],1))+'%')
    ax1.set_ylim(bottom=min(validation_accuracy[model]))
  ax1.set_ylim(top=100)
  ax1.set_ylabel('Overall accuracy')
  ax1.set_xlim(0,1+len(train_step[model]))
  if imodel==0:
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    ax1.legend(handles1+handles4, labels1+labels4, loc='center right')
  ax2 = ax4.twiny()
  ax2.xaxis.set_ticks_position("bottom")
  ax2.xaxis.set_label_position("bottom")
  ax2.spines["bottom"].set_position(("axes", -0.2))
  ax2.set_frame_on(True)
  ax2.patch.set_visible(False)
  for k in ax2.spines.keys():
      ax2.spines[k].set_visible(False)
  ax2.spines["bottom"].set_visible(True)
  scaled_data,units = choose_units(train_time[model])
  ax2.set_xlim(scaled_data[0],scaled_data[-1])
  ax2.set_xlabel('Time ('+units+')')
  ax3 = ax4.twiny()
  ax3.xaxis.set_ticks_position("bottom")
  ax3.xaxis.set_label_position("bottom")
  ax3.spines["bottom"].set_position(("axes", -0.4))
  ax3.set_frame_on(True)
  ax3.patch.set_visible(False)
  for k in ax3.spines.keys():
      ax3.spines[k].set_visible(False)
  ax3.spines["bottom"].set_visible(True)
  step2epoch = batch_size[model] / training_set_size[model]
  ax3.set_xlim(train_step[model][0]*step2epoch, \
               train_step[model][-1]*step2epoch)
  ax3.set_xlabel('Epoch')

fig.tight_layout()
plt.savefig(os.path.join(logdir,'train-loss.pdf'))
plt.close()

nrows, ncols = layout(len(keys_to_plot))
for (iword,word) in enumerate(wanted_words[keys_to_plot[0]]):
  fig = plt.figure(figsize=(6.4*ncols, 4.8*nrows))
  ax = []
  minp=100
  minr=100
  for (imodel,model) in enumerate(keys_to_plot):
    ax.append(fig.add_subplot(nrows, ncols, imodel+1))
    precision = [100*x[iword] for x in validation_precision[model]]
    recall = [100*x[iword] for x in validation_recall[model]]
    minp = min(minp, min(precision))
    minr = min(minr, min(recall))
    ax[imodel].set_prop_cycle(color=[cm.viridis(1.*i/max(1,len(recall)-2)) \
                                     for i in range(len(recall)-1)])
    for i in range(len(recall)-1):
      ax[imodel].plot(recall[i:i+2], precision[i:i+2])
    ax[imodel].annotate(str(validation_step[model][0]),
                xy=(recall[0], precision[0]),
                color="r", verticalalignment='center', horizontalalignment='center')
    ax[imodel].annotate(str(validation_step[model][-1]),
                xy=(recall[-1], precision[-1]),
                color="r", verticalalignment='center', horizontalalignment='center')
    ibestF1 = np.argmax([2*p*r/(p+r) if (p+r)>0 else np.nan \
                         for (p,r) in zip(precision,recall)])
    ax[imodel].annotate(str(validation_step[model][ibestF1]),
                xy=(recall[ibestF1], precision[ibestF1]),
                color="r", verticalalignment='center', horizontalalignment='center')
    ax[imodel].set_xlabel('Recall')
    ax[imodel].set_ylabel('Precision')
    ax[imodel].set_title(model)
  for imodel in range(len(keys_to_plot)):
    ax[imodel].set_ylim([minp,100])
    ax[imodel].set_xlim([minr,100])
  fig.tight_layout()
  plt.savefig(os.path.join(logdir,'validation-PvR-'+word+'.pdf'))
  plt.close()


nrows, ncols = layout(len(wanted_words[keys_to_plot[0]]))
fig = plt.figure(figsize=(6.4*ncols, 4.8*nrows))
lines=[]
isort = index_natsorted(wanted_words[keys_to_plot[0]])
for iword in range(len(isort)):
  ax = fig.add_subplot(nrows, ncols, iword+1)
  for (imodel,model) in enumerate(keys_to_plot):
    F1 = [2*p[isort[iword]]*r[isort[iword]]/(p[isort[iword]]+r[isort[iword]]) \
          if (p[isort[iword]]+r[isort[iword]])>0 else np.nan \
          for (p,r) in zip(validation_precision[model], validation_recall[model])]
    line, = ax.plot(validation_step[model], F1)
    if iword==0:
      lines.append(line)
    ax.set_ylim(top=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('F1 = 2PR/(P+R)')
    ax.set_title(wanted_words[keys_to_plot[0]][isort[iword]])
  ax3 = ax.twiny()
  ax3.xaxis.set_ticks_position("bottom")
  ax3.xaxis.set_label_position("bottom")
  ax3.spines["bottom"].set_position(("axes", -0.15))
  ax3.set_frame_on(True)
  ax3.patch.set_visible(False)
  for k in ax3.spines.keys():
      ax3.spines[k].set_visible(False)
  ax3.spines["bottom"].set_visible(True)
  step2epoch = batch_size[model] / training_set_size[model]
  ax3.set_xlim(validation_step[model][0]*step2epoch, \
               validation_step[model][-1]*step2epoch)
  ax3.set_xlabel('Epoch')

lgd = fig.legend(lines, keys_to_plot, ncol=len(keys_to_plot),
                 bbox_to_anchor=(0.0,-0.02), loc="lower left")
fig.tight_layout()
plt.savefig(os.path.join(logdir,'validation-F1.pdf'),
            bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

if not len(validation_accuracy[keys_to_plot[0]]):
    sys.exit()

if len(keys_to_plot)>1:
  fig = plt.figure(figsize=(6.4*1.5, 4.8))

  ax = fig.add_subplot(2,3,1)
  for model in keys_to_plot:
    scaled_validation_time, units = choose_units(validation_time[model])

    line, = ax.plot(validation_step[model], validation_accuracy[model])
    line.set_label(model)
  ax.set_ylim(top=100)
  ax.set_xlabel('Step')
  ax.set_ylabel('Overall accuracy')
  #ax.legend(loc='lower right')

  ax = fig.add_subplot(2,3,2)
  for model in keys_to_plot:
    ax.plot([x*batch_size[model]/training_set_size[model] \
             for x in validation_step[model]], validation_accuracy[model])
  ax.set_ylim(top=100)
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Overall accuracy')

  ax = fig.add_subplot(2,3,3)
  for model in keys_to_plot:
    idx = min(len(scaled_validation_time), len(validation_accuracy[model]))
    line, = ax.plot(scaled_validation_time[:idx], validation_accuracy[model][:idx])
    line.set_label(model)
  ax.set_ylim(top=100)
  ax.set_xlabel('Time ('+units+')')
  ax.set_ylabel('Overall accuracy')

  ax = fig.add_subplot(2,3,4)
  for model in keys_to_plot:
    epoch = [x*batch_size[model]/training_set_size[model]
             for x in validation_step[model]]
    idx = min(len(scaled_validation_time), len(epoch))
    ax.plot(scaled_validation_time[:idx], epoch[:idx])
  ax.set_xlabel('Time ('+units+')')
  ax.set_ylabel('Epoch')

  ax = fig.add_subplot(2,3,5)
  for model in keys_to_plot:
    idx = min(len(scaled_validation_time), len(validation_step[model]))
    ax.plot(scaled_validation_time[:idx], validation_step[model][:idx])
  ax.set_xlabel('Time ('+units+')')
  ax.set_ylabel('Step')

  ax = fig.add_subplot(2,3,6)
  for model in keys_to_plot:
    line, = ax.plot([x*batch_size[model]/training_set_size[model] \
                     for x in validation_step[model]], validation_step[model])
    line.set_label(model)
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Step')
  ax.legend(loc=(1.05, 0.0))

  fig.tight_layout()

  plt.savefig(os.path.join(logdir,'train-overlay.pdf'))
  plt.close()

summed_confusion_matrix, confusion_matrices, words = \
    parse_confusion_matrices(logdir, next(iter(keys_to_plot)).split('_')[0])

recall_matrices={}
precision_matrices={}
accuracies={}
for model in keys_to_plot:
  recall_matrices[model], precision_matrices[model], accuracies[model] = \
        normalize_confusion_matrix(confusion_matrices[model])

recall_summed_matrix, precision_summed_matrix, accuracy_summed = \
      normalize_confusion_matrix(summed_confusion_matrix)

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(3*6.4, 4.8))

ax = plt.subplot(1,3,1)
plot_confusion_matrix(ax, summed_confusion_matrix, \
                      precision_summed_matrix, recall_summed_matrix, \
                      numbers=len(words)<10)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=100),
                               cmap=cm.viridis),
             cax=cax, ticks=[0,100], use_gridspec=True)
ax.set_xticklabels(words, rotation=40, ha='right')
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_xlabel('Classification')
ax.set_ylabel('Annotation')
ax.set_title(str(round(accuracy_summed,1))+"%")

ax = plt.subplot(1,3,2)
for model in keys_to_plot:
  ax.set_prop_cycle(None)
  for (iword,word) in enumerate(words):
    line, = ax.plot(100*recall_matrices[model][iword][iword],
                    100*precision_matrices[model][iword][iword],
                    'o', markeredgecolor='k')
    if model==keys_to_plot[0]:
      line.set_label(word)

ax.set_xlim(right=100)
ax.set_ylim(top=100)
ax.set_xlabel('Recall (%)')
ax.set_ylabel('Precision (%)')
ax.legend(loc=(1.05, 0.0))

ax = fig.add_subplot(1,3,3)
accuracies_ordered = [accuracies[k] for k in keys_to_plot]
print('models=', keys_to_plot)
print('accuracies=', accuracies_ordered)

x = 1
y = statistics.mean(accuracies_ordered)
h = statistics.stdev(accuracies_ordered) if len(accuracies)>1 else 0
avebox = Rectangle((y-h,-0.25),2*h,len(accuracies)-1+0.5)
ax.plot([y,y],[-0.25,len(accuracies)-1+0.25],'w-')
pc = PatchCollection([avebox], facecolor='lightgray')
ax.add_collection(pc)

ax.plot(accuracies_ordered, keys_to_plot, 'k.')
ax.set_xlabel('Overall accuracy (%)')
ax.set_ylabel('Model')

fig.tight_layout()
plt.savefig(os.path.join(logdir,'accuracy.pdf'))
plt.close()

if len(keys_to_plot)>1:
  plot_confusion_matrices(confusion_matrices,
                          precision_matrices,
                          recall_matrices,
                          words, accuracies, keys_to_plot,
                          numbers=len(words)<10)
  plt.savefig(os.path.join(logdir,'confusion-matrices.pdf'))
  plt.close()

def plot_metrics_parameterized_by_threshold(logdir, thresholds, \
                                            metric1_data, metric2_data, \
                                            metric1_label, metric2_label, \
                                            areas, words, probabilities, ratios):
  speciess = sorted(list(set([x.split('-')[0] for x in words])))
  if len(speciess)<len(words):
    ncols = len(speciess)
    nrows = max([sum([x.split('-')[0]==s for x in words]) for s in speciess])
    colcounts = np.zeros(ncols, dtype=np.uint8)
  else:
    nrows, ncols = layout(len(words))
  fig=plt.figure(figsize=(3.2*ncols, 2.4*nrows))
  for iword in range(len(words)):
    if len(speciess)<len(words):
      icol = np.argmax([words[iword].split('-')[0]==s for s in speciess])
      irow = colcounts[icol]
      colcounts[icol] += 1
      ax=plt.subplot(nrows,ncols,irow*ncols+icol+1)
    else:
      ax=plt.subplot(nrows,ncols,iword+1)
    if iword==0:
      ax.set_xlabel(metric2_label)
      ax.set_ylabel(metric1_label)
    ax.set_title(words[iword]+',  area='+str(np.round(areas[words[iword]],3)))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    line, = plt.plot(metric2_data[words[iword]], metric1_data[words[iword]])
    for (ithreshold,threshold) in enumerate(thresholds[words[iword]]):
      if np.isnan(threshold):
        continue
      iprobability = np.argmin(np.abs(probabilities[words[iword]]-threshold))
      if ratios[ithreshold]==1:
        plt.plot(metric2_data[words[iword]][iprobability], \
                 metric1_data[words[iword]][iprobability], 'ro')
      else:
        plt.plot(metric2_data[words[iword]][iprobability], \
                 metric1_data[words[iword]][iprobability], 'rx')
      whichside = 'right' if metric2_data[words[iword]][iprobability]>0.5 else 'left'
      ax.annotate(str(np.round(threshold,3))+' ',
                  xy=(metric2_data[words[iword]][iprobability], \
                      metric1_data[words[iword]][iprobability]),
                  color="r", verticalalignment='top', horizontalalignment=whichside)
  fig.tight_layout()

def plot_probability_density(test_ground_truth, test_logits, ratios, thresholds, words):
  speciess = sorted(list(set([x.split('-')[0] for x in words])))
  if len(speciess)<len(words):
    ncols = len(speciess)
    nrows = max([sum([x.split('-')[0]==s for x in words]) for s in speciess])
    colcounts = np.zeros(ncols, dtype=np.uint8)
  else:
    nrows, ncols = layout(len(words))
  x = np.arange(0, 1.01, 0.01)
  fig=plt.figure(figsize=(3.2*ncols, 2.4*nrows))
  for iword in range(len(words)):
    if len(speciess)<len(words):
      icol = np.argmax([words[iword].split('-')[0]==s for s in speciess])
      irow = colcounts[icol]
      colcounts[icol] += 1
      ax=plt.subplot(nrows,ncols,irow*ncols+icol+1)
    else:
      ax=plt.subplot(nrows,ncols,iword+1)
    if iword==0:
      ax.set_xlabel('Probability')
      ax.set_ylabel('Density')
    ax.set_title(words[iword])
    for (ithreshold,threshold) in enumerate(thresholds[words[iword]]):
      if np.isnan(threshold):
        continue
      if ratios[ithreshold]==1:
        plt.axvline(x=threshold, color='k', linestyle='dashed')
      else:
        plt.axvline(x=threshold, color='k', linestyle='dotted')
    for gt in range(len(words)):
      igt = test_ground_truth==gt
      if sum(igt)==0:
        continue
      xdata = test_logits[igt,iword]
      xdata = np.exp(xdata) / (np.exp(xdata) + 1)
      if len(xdata)<2:
        continue
      density = stats.kde.gaussian_kde(xdata)
      y = density(x)
      line, = plt.plot(x, y)
      if iword==gt:
        ax.set_ylim(0,max(y))
      if iword==0:
        line.set_label(words[gt])
  fig.tight_layout()
  return fig.legend(loc='lower left')

def doit(key_to_plot, ckpt):
  validation_samples, validation_ground_truth, validation_logits = \
        read_logits(logdir, key_to_plot, ckpt)

  probabilities, thresholds, precisions, recalls, sensitivities, specificities, \
        pr_areas, roc_areas = calculate_precision_recall_specificity( \
        validation_ground_truth, validation_logits, words, nprobabilities, \
        error_ratios)

  save_thresholds(logdir, key_to_plot, ckpt, thresholds, error_ratios, words)

  plot_metrics_parameterized_by_threshold(logdir, thresholds, precisions, recalls, \
        'precision = Tp/(Tp+Fp)', 'recall = Tp/(Tp+Fn)', pr_areas, \
        words, probabilities, error_ratios)
  plt.savefig(os.path.join(logdir,key_to_plot,'precision-recall.ckpt-'+str(ckpt)+'.pdf'))
  plt.close()

  plot_metrics_parameterized_by_threshold(logdir, thresholds, specificities, \
        sensitivities, 'specificity = Tn/(Tn+Fp)', 'sensitivity = Tp/(Tp+Fn)', \
        roc_areas, words, probabilities, error_ratios)
  plt.savefig(os.path.join(logdir, key_to_plot, \
                           'specificity-sensitivity.ckpt-'+str(ckpt)+'.pdf'))
  plt.close()

  already_written = set()
  predictions_path = os.path.join(logdir,key_to_plot,'predictions.ckpt-'+str(ckpt))
  if not os.path.isdir(predictions_path):
    os.mkdir(os.path.join(logdir,key_to_plot,'predictions.ckpt-'+str(ckpt)))
  wavfiles = set([x['file'] for x in validation_samples])
  for subdir in list(set([os.path.basename(os.path.split(x)[0]) for x in wavfiles])):
    with open(os.path.join(logdir, key_to_plot, 'predictions.ckpt-'+str(ckpt), \
                           subdir+'-mistakes.csv'), \
              'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      for i in range(len(validation_ground_truth)):
        if os.path.basename(os.path.split(validation_samples[i]['file'])[0]) != subdir:
          continue
        classified_as = np.argmax(validation_logits[i,:])
        id = validation_samples[i]['file'] + str(validation_samples[i]['ticks']) + \
             validation_samples[i]['label'] + words[classified_as]
        if id in already_written:
          continue
        already_written |= set([id])
        csvwriter.writerow([os.path.basename(validation_samples[i]['file']),
              validation_samples[i]['ticks'][0], validation_samples[i]['ticks'][1],
              'correct' if classified_as == validation_ground_truth[i] else 'mistaken',
              words[classified_as],
              validation_samples[i]['label']])

  lgd = plot_probability_density(validation_ground_truth, validation_logits, \
                                 error_ratios, thresholds, words)
  plt.savefig(os.path.join(logdir, key_to_plot, \
                           'probability-density.ckpt-'+str(ckpt)+'.pdf'), \
              bbox_extra_artists=(lgd,), bbox_inches='tight')

  plt.close()
  return 1

if accuracy_parallelize:
  from multiprocessing import Pool
  pool = Pool()
  results = []

for key_to_plot in accuracies:
  for ckpt in [int(x.split('-')[1][:-4]) for x in \
               filter(lambda x: 'validation' in x and x.endswith('.npz'), \
                      os.listdir(os.path.join(logdir,key_to_plot)))]:

    if accuracy_parallelize:
      results.append(pool.apply_async(doit, (key_to_plot,ckpt)))
    else:
      doit(key_to_plot, ckpt)

if accuracy_parallelize:
  for result in results:
    result.get()
  pool.close()
