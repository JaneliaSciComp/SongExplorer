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

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

from scipy import interpolate
from scipy import stats
import math

_,logdir,error_ratios,nprobabilities = sys.argv
print('logdir: '+logdir)
print('error_ratios: '+error_ratios)
print('nprobabilities: '+nprobabilities)
error_ratios = [float(x) for x in error_ratios.split(',')]
nprobabilities = int(nprobabilities)


labels_folders = filter(lambda x: x.startswith('train_') and \
                        os.path.isdir(os.path.join(logdir,x)), os.listdir(logdir))
labels_path = os.path.join(logdir,list(labels_folders)[0],'vgg_labels.txt')
nwanted_words = sum(1 for line in open(labels_path))


train_accuracy, train_loss, train_time, train_step, \
      validation_accuracy, validation_time, validation_step, \
      _, word_counts, _, _, batch_size, _, _ = \
      read_logs(logdir)
training_set_size = {x: np.sum([y for y in word_counts[x].values()]) for x in word_counts.keys()}

keys_to_plot = natsorted(train_step.keys())


def plot_aggregated_steps(train_accuracy, train_loss, train_time, train_step, validation_accuracy, validation_time, validation_step, models, nrows):
  ncols = math.ceil(len(models)/nrows)
  fig = plt.figure(figsize=(6.4*ncols, 4.8*nrows))
  fig.subplots_adjust(bottom=0.2)
  for (imodel,model) in enumerate(models):
    ax1 = fig.add_subplot(nrows, ncols, imodel+1)
    line, = ax1.plot(train_step[model], train_accuracy[model],'b')
    line.set_label('train')
    if validation_accuracy[model]:
      line, = ax1.plot(validation_step[model], validation_accuracy[model],'r')
      line.set_label('validation')
      ax1.set_title(model+"   "+str(round(validation_accuracy[model][-1],1))+'%')
    ax1.set_ylim([0,100])
    ax1.set_xlabel('step')
    ax1.set_ylabel('accuracy')
    ax1.set_xlim(0,1+len(train_step[model]))
    if imodel==0:
      ax1.legend(loc='center')
    ax2 = ax1.twiny()
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    for k in ax2.spines.keys():
        ax2.spines[k].set_visible(False)
    ax2.spines["bottom"].set_visible(True)
    scaled_data,units = choose_units(train_time[model])
    ax2.set_xlim(scaled_data[0],scaled_data[-1])
    ax2.set_xlabel('time ('+units+')')
    ax3 = ax1.twinx()
    line, = ax3.plot(train_step[model], train_loss[model],'g')
    line.set_label('cross entropy')
    ax3.set_ylabel('loss', color='g')
    ax3.legend(loc='right')
    ax3.tick_params(axis='y', labelcolor='g')
  fig.tight_layout()


plot_aggregated_steps(train_accuracy, train_loss, train_time, train_step, \
      validation_accuracy, validation_time, validation_step, keys_to_plot, \
      1 if len(keys_to_plot)==1 else 2)
plt.savefig(os.path.join(logdir,'training.pdf'))
plt.close()

if not len(validation_accuracy[keys_to_plot[0]]):
    sys.exit()

if len(keys_to_plot)>1:
  plot_overlayed_steps(validation_accuracy, validation_time, validation_step, keys_to_plot, batch_size, training_set_size)
  plt.savefig(os.path.join(logdir,'training-overlayed.pdf'))
  plt.close()


words={}
confusion_matrix={}
accuracy={}
row_normalized_confusion_matrix={}
col_normalized_confusion_matrix={}
for model in keys_to_plot:
  logfile = os.path.join(logdir,model+".log")
  words[model], confusion_matrix[model], accuracy[model], _ = \
        parse_confusion_matrix(logfile, nwanted_words)
  row_normalized_confusion_matrix[model], col_normalized_confusion_matrix[model] = \
        normalize_matrix(confusion_matrix[model])

if len(keys_to_plot)>1:
  accuracies = [accuracy[k] for k in keys_to_plot]
  print('models=', keys_to_plot)
  print('accuracies=', accuracies)

  fig = plt.figure(figsize=(6.4, 4.8))
  ax = fig.add_subplot(111)

  x = 1
  y = statistics.mean(accuracies)
  h = statistics.stdev(accuracies)
  avebox = Rectangle((y-h,-0.25),2*h,len(accuracies)+0.5)
  ax.plot([y,y],[-0.25,len(accuracies)+0.25],'w-')
  pc = PatchCollection([avebox], facecolor='lightgray')
  ax.add_collection(pc)

  ax.plot(accuracies, keys_to_plot, 'k.')
  ax.set_xlabel('accuracy (%)')
  ax.set_ylabel('model')
 
  fig.tight_layout()
  plt.savefig(os.path.join(logdir,'accuracy.pdf'))
  plt.close()


def plot_aggregated_triangle_matrices(abs_matrices, col_matrices, row_matrices, words, accuracies, models, numbers=True, scale=6.4, nrows=1):
  ncols = math.ceil(len(models)/nrows)
  fig = plt.figure(figsize=(scale*ncols, scale*3/4*nrows))
  for (imodel, model) in enumerate(models):
    ax = fig.add_subplot(nrows, ncols, imodel+1)
    plot_triangle_matrix(ax, abs_matrices[model], col_matrices[model], row_matrices[model], numbers)
    ax.set_xticks(range(len(words[model])))
    ax.set_yticks(range(len(words[model])))
    ax.set_xticklabels(words[model], rotation=40, ha='right')
    ax.set_yticklabels(words[model])
    ax.invert_yaxis()
    if imodel//ncols==nrows-1:
      ax.set_xlabel('classification')
    if imodel%ncols==0:
      ax.set_ylabel('annotation')
    ax.set_title(model+"   "+str(round(accuracies[model],1))+"%")
  #sm = cm.ScalarMappable(cmap=cm.viridis)
  #sm.set_array([])
  #plt.colorbar(sm, ax=[axes[x] for x in [3,7]])
  fig.tight_layout()

plot_aggregated_triangle_matrices(confusion_matrix,
                                  col_normalized_confusion_matrix,
                                  row_normalized_confusion_matrix,
                                  words, accuracy, keys_to_plot,
                                  numbers=len(words[keys_to_plot[0]])<10,
                                  nrows=1 if len(keys_to_plot)==1 else 2)
plt.savefig(os.path.join(logdir,'confusion-matrix.pdf'))
plt.close()


maxaccuracy=0.0
for model in accuracy:
  thisaccuracy=accuracy[model]
  if thisaccuracy>maxaccuracy:
    maxaccuracy=thisaccuracy
    key_to_plot=model
  

def save_thresholds(logdir, thresholds, ratios, words):
  fid = open(os.path.join(logdir,key_to_plot,'thresholds.ckpt-'+str(ckpt)+'.csv'),"w")
  fidcsv = csv.writer(fid)
  fidcsv.writerow(['precision/recall'] + ratios)
  for iword in range(len(words)):
    fidcsv.writerow([words[iword]] + thresholds[words[iword]].tolist())
  fid.close()

def plot_metrics_parameterized_by_threshold(logdir, thresholds, \
                                            metric1_data, metric2_data, \
                                            metric1_label, metric2_label, \
                                            areas, words, probabilities, ratios):
  speciess = sorted(list(set([x.split('-')[0] for x in words])))
  ncols = len(speciess)
  nrows = max([sum([x.split('-')[0]==s for x in words]) for s in speciess])
  colcounts = np.zeros(ncols, dtype=np.uint8)
  fig=plt.figure(figsize=(3.2*ncols, 2.4*nrows))
  for iword in range(len(words)):
    icol = np.argmax([words[iword].split('-')[0]==s for s in speciess])
    irow = colcounts[icol]
    colcounts[icol] += 1
    ax=plt.subplot(nrows,ncols,irow*ncols+icol+1)
    if icol==0 and irow==0:
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
  ncols = len(speciess)
  nrows = max([sum([x.split('-')[0]==s for x in words]) for s in speciess])
  colcounts = np.zeros(ncols, dtype=np.uint8)
  x = np.arange(0, 1.01, 0.01)
  fig=plt.figure(figsize=(3.2*ncols, 2.4*nrows))
  for iword in range(len(words)):
    icol = np.argmax([words[iword].split('-')[0]==s for s in speciess])
    irow = colcounts[icol]
    colcounts[icol] += 1
    ax=plt.subplot(nrows,ncols,irow*ncols+icol+1)
    if icol==0 and irow==0:
      ax.set_xlabel('probability')
      ax.set_ylabel('density')
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

ckpts = [int(x.split('-')[1][:-4]) for x in filter(lambda x: \
                             'validation' in x and x.endswith('.npz'), \
                             os.listdir(os.path.join(logdir,key_to_plot)))]

# from multiprocessing import Pool
# with Pool(nlayers) as p:
#   hidden_clustered = p.map(do_cluster, hidden_kept)
for ckpt in ckpts:
  validation_samples, validation_ground_truth, validation_logits = \
        read_logits(logdir, key_to_plot, ckpt)


  probabilities, thresholds, precisions, recalls, sensitivities, specificities, \
        pr_areas, roc_areas = calculate_precision_recall_specificity( \
        validation_ground_truth, validation_logits, words[key_to_plot], nprobabilities, error_ratios)

  save_thresholds(logdir, thresholds, error_ratios, words[key_to_plot])

  plot_metrics_parameterized_by_threshold(logdir, thresholds, precisions, recalls, \
        'precision = Tp/(Tp+Fp)', 'recall = Tp/(Tp+Fn)', pr_areas, \
        words[key_to_plot], probabilities, error_ratios)
  plt.savefig(os.path.join(logdir,key_to_plot,'precision-recall.ckpt-'+str(ckpt)+'.pdf'))
  plt.close()

  plot_metrics_parameterized_by_threshold(logdir, thresholds, specificities, sensitivities, \
        'specificity = Tn/(Tn+Fp)', 'sensitivity = Tp/(Tp+Fn)', roc_areas, \
        words[key_to_plot], probabilities, error_ratios)
  plt.savefig(os.path.join(logdir,key_to_plot,'specificity-sensitivity.ckpt-'+str(ckpt)+'.pdf'))
  plt.close()

  lgd = plot_probability_density(validation_ground_truth, validation_logits, error_ratios, \
                                        thresholds, words[key_to_plot])
  plt.savefig(os.path.join(logdir,key_to_plot,'probability-density.ckpt-'+str(ckpt)+'.pdf'), \
              bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()

  already_written = set()
  predictions_path = os.path.join(logdir,key_to_plot,'predictions.ckpt-'+str(ckpt))
  if not os.path.isdir(predictions_path):
    os.mkdir(os.path.join(logdir,key_to_plot,'predictions.ckpt-'+str(ckpt)))
  wavfiles = list(set([x['file'] for x in validation_samples]))
  for subdir in list(set([os.path.basename(os.path.split(x)[0]) for x in wavfiles])):
    with open(os.path.join(logdir,key_to_plot,'predictions.ckpt-'+str(ckpt),subdir+'.csv'), 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      for i in range(len(validation_ground_truth)):
        if os.path.basename(os.path.split(validation_samples[i]['file'])[0]) != subdir:
          continue
        classified_as = np.argmax(validation_logits[i,:])
        id = validation_samples[i]['file']+str(validation_samples[i]['ticks'])+validation_samples[i]['label']+words[key_to_plot][classified_as]
        if id in already_written:
          continue
        already_written |= set([id])
        csvwriter.writerow([os.path.basename(validation_samples[i]['file']),
              validation_samples[i]['ticks'][0], validation_samples[i]['ticks'][1],
              'correct' if classified_as == validation_ground_truth[i] else 'mistake',
              words[key_to_plot][classified_as],
              validation_samples[i]['label']])
