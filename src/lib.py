import os
import sys
import json
import re
from subprocess import check_output
import ast
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from scipy import interpolate
import math
from natsort import realsorted
from scipy.io import wavfile
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from jitter import *

def combine_events(events1, events2, logic):
  max_time1 = np.max([int(x[2]) for x in events1])
  max_time2 = np.max([int(x[2]) for x in events2])
  max_time = max(max_time1, max_time2)

  bool1 = np.full((max_time,), False)
  for event in events1:
    bool1[int(event[1]):int(event[2])+1] = True

  bool2 = np.full((max_time,), False)
  for event in events2:
    bool2[int(event[1]):int(event[2])+1] = True

  bool12 = logic(bool1, bool2)

  diff_bool12 = np.diff(bool12)
  changes = np.where(diff_bool12)[0]
  nfeatures = int(np.ceil(len(changes)/2))
  start_times = np.empty((nfeatures,), dtype=np.int32)
  stop_times = np.empty((nfeatures,), dtype=np.int32)
  ifeature = 0
  ichange = 1
  while ichange<len(changes):
    if not bool12[changes[ichange]]:  # starts with label
       ichange += 1;
       continue
    start_times[ifeature] = changes[ichange-1]+1
    stop_times[ifeature] = changes[ichange]
    ifeature += 1
    ichange += 2

  return start_times, stop_times, ifeature

def confusion_string2matrix(arg):
  arg = arg[1:-1]
  arg = arg.replace("\n", ",")
  arg = re.sub("(?P<P>[0-9]) ","\g<P>,", arg)
  return ast.literal_eval(arg)

def parse_confusion_matrix(logfile, nlabels_touse, which_one=0, test=False):
  max_count = '-m'+str(which_one) if which_one>0 else ''
  kind = "Testing" if test else "Validation"
  cmd = "grep -B"+str(nlabels_touse+1)+" '"+kind+" accuracy' "+logfile+" "+max_count+ \
        " | tail -"+str(nlabels_touse+2)+" | head -1"
  labels_string = check_output(cmd, shell=True)
  labels_string = labels_string.decode("ascii")
  labels_string = labels_string[1:-1]
  cmd = "grep -B"+str(nlabels_touse+0)+" '"+kind+" accuracy' "+logfile+" "+max_count+ \
        " | tail -"+str(nlabels_touse+1)+" | head -"+str(nlabels_touse+0)
  confusion_string = check_output(cmd, shell=True)
  confusion_string = confusion_string.decode("ascii")
  confusion_matrix = confusion_string2matrix(confusion_string)
  nannotations = [sum(x) for x in confusion_matrix]
  return confusion_matrix, ast.literal_eval(labels_string), nannotations


def parse_confusion_matrices(logdir, kind, idx_time=None, test=False):
  models = list(filter(lambda x: x.startswith(kind+'_') and \
                          os.path.isdir(os.path.join(logdir,x)), os.listdir(logdir)))
  labels_path = os.path.join(logdir,list(models)[0],'labels.txt')
  nlabels_touse = sum(1 for line in open(labels_path))

  confusion_matrices={}
  labels=[]
  for model in models:
    logfile = os.path.join(logdir, model+'.log')
    confusion_matrices[model], theselabels, _ = \
            parse_confusion_matrix(logfile, \
                                   nlabels_touse, \
                                   which_one=1+idx_time[model] if idx_time else 0, \
                                   test=test)
    if labels:
      assert labels==theselabels
    else:
      labels=theselabels
      
  summed_confusion_matrix = np.zeros((np.shape(confusion_matrices[models[0]])))
  for model in models:
    summed_confusion_matrix += confusion_matrices[model]
  return summed_confusion_matrix, confusion_matrices, labels


def normalize_confusion_matrix(matrix):
  row_norm_matrix = [[np.nan if np.nansum(x)==0.0 else y/np.nansum(x) for y in x]
                     for x in matrix]
  transposed_matrix = list(zip(*row_norm_matrix))  # row_norm_matrix here so it is balanced
  norm_transposed_matrix = [[np.nan if np.nansum(x)==0.0 else y/np.nansum(x) for y in x]
                            for x in transposed_matrix]
  col_norm_matrix = list(zip(*norm_transposed_matrix))
  accuracy = 100 * np.mean([x[i] for (i,x) in enumerate(row_norm_matrix) if not np.isnan(x[i])])
  return row_norm_matrix, col_norm_matrix, accuracy


def plot_confusion_matrix(ax, abs_matrix, col_matrix, row_matrix, numbers):
  patches = []
  sx,sy = np.shape(col_matrix)
  for ix in range(sx):
    for iy in range(sy):
      value = row_matrix[iy][ix]
      color = cm.viridis(value)
      polygon=Polygon([[ix-0.5,iy-0.5],[ix+0.5,iy-0.5],[ix+0.5,iy+0.5]], facecolor=color)
      patches.append(polygon)
      if numbers and not np.isnan(value):
        ax.annotate("{0:.1f}".format(100*value),xy=(ix+0.5,iy-0.5),
                    color=(1-value,1-value,1-value,1),
                    horizontalalignment='right', verticalalignment='top')
      value = col_matrix[iy][ix]
      color = cm.viridis(value)
      polygon=Polygon([[ix-0.5,iy-0.5],[ix-0.5,iy+0.5],[ix+0.5,iy+0.5]], facecolor=color)
      patches.append(polygon)
      if numbers and not np.isnan(value):
        ax.annotate("{0:.1f}".format(100*value),xy=(ix-0.5,iy+0.5),
                    color=(1-value,1-value,1-value,1),
                    horizontalalignment='left', verticalalignment='bottom')
      value = abs_matrix[iy][ix]
      if numbers and not np.isnan(value):
        ax.annotate(str(int(value)),xy=(ix,iy),
                    color="fuchsia",
                    horizontalalignment='center', verticalalignment='center')
      if numbers:
        polygon=Polygon([[ix-0.5,iy-0.5],[ix-0.5,iy+0.5],[ix+0.5,iy+0.5],[ix+0.5,iy-0.5]], \
                        facecolor=(1,1,1,0), edgecolor="white")
        patches.append(polygon)
  p = PatchCollection(patches,match_original=True)
  ax.add_collection(p)
  ax.set_xlim(-0.5,sx-0.5)
  ax.set_ylim(-0.5,sy-0.5)
  ax.set_xticks(range(sx))
  ax.set_yticks(range(sy))


def layout(nplots):
  if nplots==10:
    return 2,5
  if nplots==21:
    return 3,7
  nrows = 1 if nplots==1 else np.floor(np.sqrt(nplots)).astype(np.int)
  ncols = math.ceil(nplots / nrows)
  return nrows, ncols


def plot_confusion_matrices(abs_matrices, col_matrices, row_matrices, labels, accuracies, \
                            models, numbers=True, scale=6.4):
  nrows, ncols = layout(len(models))
  fig = plt.figure(figsize=(scale*ncols, scale*3/4*nrows))
  for (imodel, model) in enumerate(models):
    ax = fig.add_subplot(nrows, ncols, imodel+1)
    plot_confusion_matrix(ax, abs_matrices[model], col_matrices[model], row_matrices[model], numbers)
    ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if imodel//ncols==nrows-1:
      ax.set_xlabel('classification')
    if imodel%ncols==0:
      ax.set_ylabel('annotation')
    ax.set_title(model+"   "+str(round(accuracies[model],1))+"%")
  fig.tight_layout()


def read_log(frompath, logfile):
  train_accuracy=[]; train_loss=[]; train_time=[]; train_step=[]
  validation_time=[]; validation_step=[]
  validation_precision=[]; validation_recall=[]; validation_accuracy=[]
  test_precision=[]; test_recall=[]; test_accuracy=[]
  nlayers=0
  with open(os.path.join(frompath,logfile),'r') as fid:
    train_restart_correction=0.0
    validation_restart_correction=0.0
    count_train_state=False
    for line in fid:
      if "num training labels" in line:
        count_train_state=True
        label_counts = {}
      elif count_train_state:
        if "Model: " in line:
          count_train_state = False
        else:
          m=re.search('\s*(\d+)\s(.*)',line)
          label_counts[m.group(2)]=int(m.group(1))
      if "labels_touse" in line:
        m=re.search('labels_touse = (.+)',line)
        labels_touse = m.group(1).split(',')
      elif "batch_size" in line:
        m=re.search('batch_size = (\d+)',line)
        batch_size = int(m.group(1))
      elif "Trainable params" in line:
        m=re.search('Trainable params: ([,\d]+)',line)
        nparameters_total = int(m.group(1).replace(',',''))
      elif 'Conv2D' in line:
        if validation_restart_correction==0.0:
          nlayers += 1
        m=re.search('[^,] (\d+)',line)
        nparameters_finallayer = int(m.group(1))
      elif "cross entropy" in line:
        m=re.search('Elapsed (.*), Step #(.*):.*accuracy (.*)%.*cross entropy (.*)$', line)
        train_time_value = float(m.group(1))
        if len(train_time)>0 and \
                (train_time_value+train_restart_correction)<train_time[-1]:
          train_restart_correction = train_time[-1]
        train_time.append(train_time_value+train_restart_correction)
        train_step.append(int(m.group(2)))
        train_accuracy.append(float(m.group(3)))
        train_loss.append(float(m.group(4)))
      elif " [" in line and " ['" not in line and 'None' not in line:
        if " [[" in line:
          confusion_string=line
        else:
          confusion_string+=line
        if "]]" in line:
          confusion_matrix = confusion_string2matrix(confusion_string)
          row_normalized_confusion_matrix, column_normalized_confusion_matrix, accuracy = \
                  normalize_confusion_matrix(confusion_matrix)
          precision = [x[i] for (i,x) in enumerate(column_normalized_confusion_matrix)]
          recall = [x[i] for (i,x) in enumerate(row_normalized_confusion_matrix)]
      elif "Validation accuracy" in line:
        validation_precision.append(precision)
        validation_recall.append(recall)
        validation_accuracy.append(accuracy)
        m=re.search('Elapsed (.*), Step (.*):.* = (.*)%',line)
        validation_time_value = float(m.group(1))
        if len(validation_time)>0 and \
                (validation_time_value+validation_restart_correction)<validation_time[-1]:
          validation_restart_correction = validation_time[-1]
        validation_time.append(validation_time_value+validation_restart_correction)
        validation_step.append(int(m.group(2)))
      elif "Testing accuracy" in line:
        test_precision.append(precision)
        test_recall.append(recall)
        test_accuracy.append(accuracy)

  return train_accuracy, train_loss, train_time, train_step, \
         validation_precision, validation_recall, validation_accuracy, \
         validation_time, validation_step, \
         test_precision, test_recall, test_accuracy, \
         labels_touse, label_counts, \
         nparameters_total, nparameters_finallayer, \
         batch_size, nlayers
         #test_accuracy, \


def read_logs(frompath):
  train_accuracy={}; train_loss={}; train_time={}; train_step={}
  validation_precision={}; validation_recall={}; validation_accuracy={}
  validation_time={}; validation_step={}
  test_precision={}; test_recall={}; test_accuracy={}
  labels_touse={}
  label_counts={}
  nparameters_total={}
  nparameters_finallayer={}
  batch_size={}
  nlayers={}
  for logfile in filter(lambda x: re.match('(train|xvalidate|generalize)_.*log',x), \
                        os.listdir(frompath)):
    model=logfile[:-4]
    train_accuracy[model], train_loss[model], train_time[model], train_step[model], \
          validation_precision[model], validation_recall[model], validation_accuracy[model], \
          validation_time[model], validation_step[model], \
          test_precision[model], test_recall[model], test_accuracy[model], \
          labels_touse[model], label_counts[model], \
          nparameters_total[model], nparameters_finallayer[model], \
          batch_size[model], nlayers[model] = \
          read_log(frompath, logfile)
          #test_accuracy[model], \

  return train_accuracy, train_loss, train_time, train_step, \
         validation_precision, validation_recall, validation_accuracy, \
         validation_time, validation_step, \
         test_precision, test_recall, test_accuracy, \
         labels_touse, label_counts, \
         nparameters_total, nparameters_finallayer, \
         batch_size, nlayers
         #test_accuracy, \


def read_logits(frompath, logdir, ckpt=None):
  if ckpt:
    logit_file = list(filter(lambda x: x.startswith('logits.validation.') and \
                             x.endswith('-'+str(ckpt)+'.npz'), \
                             os.listdir(os.path.join(frompath,logdir))))[0]
  else:
    logit_file = natsorted(list(filter(lambda x: x.startswith('logits.validation.') and \
                                x.endswith('.npz'), \
                                os.listdir(os.path.join(frompath,logdir)))))[-1]
  npzfile = np.load(os.path.join(frompath,logdir,logit_file), allow_pickle=True)
  return npzfile['sounds'], npzfile['groundtruth'], npzfile['logits']
 

def plot_time_traces(ax, validation_time, accuracy, ylabel, ltitle, \
                     outlier_crit=0, real=False, llabels=None, reverse=False, \
                     min_time=None):
  bottom=100
  sortfun = realsorted if real else natsorted
  for (iexpt,expt) in enumerate(sortfun(accuracy.keys(), reverse=reverse)):
    color = cm.viridis(iexpt/max(1,len(accuracy)-1))
    for model in validation_time[expt].keys():
      line, = ax.plot(np.array(validation_time[expt][model])/60, accuracy[expt][model], \
                      color=color, zorder=iexpt, linewidth=1)
      bottom = min([bottom]+[x for x in accuracy[expt][model] if x>outlier_crit])
    line.set_label(llabels[expt] if llabels else expt.split('-')[1])
  if min_time:
    ax.axvline(min_time/60, color='k', linestyle=':')
  ax.set_ylim(bottom=bottom-5, top=min(100, ax.get_ylim()[1]))
  ax.set_xlabel('Training time (min)')
  ax.set_ylabel(ylabel)
  ax.legend(loc='lower right', title=ltitle, ncol=2 if "Annotations" in ltitle else 1)

def plot_final_accuracies(ax, accuracy, xlabel, ylabel, \
                          outlier_crit=0, real=False, llabels=None, reverse=False, \
                          times=None):
  if times:
    min_time = np.inf
    max_time = 0
    for model in times.keys():
      for fold in times[model].keys():
        min_time = min(min_time, times[model][fold][-1])
        max_time = max(max_time, times[model][fold][-1])
    data = {}
    idx_time = {}
    for model in times.keys():
      data[model] = []
      idx_time[model] = {}
      for fold in times[model].keys():
        i = np.where(np.array(times[model][fold]) >= min_time)
        idx_time[model][fold]=i[0][0]
        data[model].append(accuracy[model][fold][idx_time[model][fold]])
  else:
    min_time = max_time = idx_time = None
    data = {k:list([v[-1] for v in accuracy[k].values()]) for k in accuracy}
  ldata = jitter_plot(ax, data, outlier_crit=outlier_crit, real=real, reverse=reverse)
  ax.set_ylabel(ylabel)
  ax.set_xlabel(xlabel)
  ax.set_xticks(range(len(ldata)))
  ax.set_xticklabels([llabels[x] if llabels else x.split('-')[1] for x in ldata])
  #if llabels:
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
  return min_time, max_time, idx_time

def choose_units(data):
  if data[-1]<60:
    return data, 'sec'
  elif data[-1]<60*60:
    return [x/60 for x in data], 'min'
  elif data[-1]<24*60*60:
    return [x/60/60 for x in data], 'hr'
  else:
    return [x/24/60/60 for x in data], 'days'

def calculate_precision_recall_specificity(validation_ground_truth, test_logits, labels, \
                                           nprobabilities, ratios):
  probabilities = {}
  thresholds = {}
  precisions = {}
  recalls = {}
  sensitivities = {}
  specificities = {}
  pr_areas = {}
  roc_areas = {}
  for ilabel in range(len(labels)):
    print(ilabel, end="\r", flush=True)
    itrue = validation_ground_truth==ilabel
    ifalse = validation_ground_truth!=ilabel
    precisions[labels[ilabel]] = np.full([nprobabilities],np.nan)
    recalls[labels[ilabel]] = np.full([nprobabilities],np.nan)
    sensitivities[labels[ilabel]] = np.full([nprobabilities],np.nan)
    specificities[labels[ilabel]] = np.full([nprobabilities],np.nan)
    if not np.any(itrue):
      pr_areas[labels[ilabel]] = np.nan
      roc_areas[labels[ilabel]] = np.nan
      probabilities[labels[ilabel]] = np.full([nprobabilities],np.nan)
      thresholds[labels[ilabel]] = np.full(len(ratios),np.nan)
      continue
    pr_areas[labels[ilabel]] = 0
    roc_areas[labels[ilabel]] = 0
    max_logit = max(test_logits[itrue,ilabel])
    #max_2nd_logit = max([x for x in test_logits[itrue,ilabel] if x!=max_logit])
    #probabilities_logit = np.linspace(
    #        min(test_logits[itrue,ilabel]), max_2nd_logit, nprobabilities)
    probabilities_logit = np.linspace(
            min(test_logits[itrue,ilabel]), max_logit, nprobabilities)
    probabilities[labels[ilabel]] = np.exp(probabilities_logit) / (np.exp(probabilities_logit) + 1)
    for (iprobability,probability_logit) in enumerate(probabilities_logit):
      #Tp = np.sum(test_logits[itrue,ilabel]>probability_logit)
      #Fp = np.sum(test_logits[ifalse,ilabel]>probability_logit)
      Tp = np.sum(test_logits[itrue,ilabel]>=probability_logit)
      Fp = np.sum(test_logits[ifalse,ilabel]>=probability_logit)
      Fn = np.sum(itrue)-Tp  # == sum(test_logits[itrue,ilabel]<=probability_logit)
      Tn = np.sum(ifalse)-Fp  # == sum(test_logits[ifalse,ilabel]<=probability_logit)
      precisions[labels[ilabel]][iprobability] = Tp/(Tp+Fp)
      recalls[labels[ilabel]][iprobability] = Tp/(Tp+Fn)
      sensitivities[labels[ilabel]][iprobability] = Tp/(Tp+Fn)
      specificities[labels[ilabel]][iprobability] = Tn/(Tn+Fp)
      if iprobability==0:
        delta_pr = precisions[labels[ilabel]][iprobability] * \
                np.abs(recalls[labels[ilabel]][iprobability] - 1)
        delta_roc = specificities[labels[ilabel]][iprobability] * \
                np.abs(sensitivities[labels[ilabel]][iprobability] - 1)
      else:
        delta_pr = precisions[labels[ilabel]][iprobability] * \
                np.abs(recalls[labels[ilabel]][iprobability] - \
                       recalls[labels[ilabel]][iprobability-1])
        delta_roc = specificities[labels[ilabel]][iprobability] * \
                np.abs(sensitivities[labels[ilabel]][iprobability] - \
                       sensitivities[labels[ilabel]][iprobability-1])
      if not np.isnan(delta_pr):  pr_areas[labels[ilabel]] += delta_pr
      if not np.isnan(delta_roc): roc_areas[labels[ilabel]] += delta_roc
      if iprobability+1==len(probabilities_logit):
        delta_pr = precisions[labels[ilabel]][iprobability] * \
                np.abs(recalls[labels[ilabel]][iprobability] - 0)
        delta_roc = specificities[labels[ilabel]][iprobability] * \
                np.abs(sensitivities[labels[ilabel]][iprobability] - 0)
        if not np.isnan(delta_pr):  pr_areas[labels[ilabel]] += delta_pr
        if not np.isnan(delta_roc): roc_areas[labels[ilabel]] += delta_roc
    f = interpolate.interp1d(precisions[labels[ilabel]]/recalls[labels[ilabel]],
                             probabilities[labels[ilabel]], fill_value="extrapolate")
    thresholds[labels[ilabel]] = f(ratios)
  return probabilities, thresholds, precisions, recalls, sensitivities, specificities, \
         pr_areas, roc_areas

def read_probabilities(basepath, labels):
  wavpath = basepath+'-'+labels[0]+'.wav'
  audio_tic_rate_probabilities, probabilities = wavfile.read(wavpath)
  half_stride_sec = 1/audio_tic_rate_probabilities/2
  probability_matrix = np.empty((len(labels), len(probabilities)))
  probability_matrix[0,:] = probabilities / np.iinfo(probabilities.dtype).max
  for ilabel in range(1,len(labels)):
    wavpath = basepath+'-'+labels[ilabel]+'.wav'
    audio_tic_rate_probabilities, probabilities = wavfile.read(wavpath)
    probability_matrix[ilabel,:] = probabilities / np.iinfo(probabilities.dtype).max
  return audio_tic_rate_probabilities, half_stride_sec, probability_matrix

def discretize_probabilites(probability_matrix, thresholds, labels,
                            audio_tic_rate_probabilities, half_stride_sec, audio_tic_rate):
  behavior = probability_matrix > thresholds
  diff_behavior = np.diff(behavior)
  ichanges, jchanges = np.where(diff_behavior)
  nfeatures = int(np.ceil(len(ichanges)/2))
  features = np.empty((nfeatures,), dtype=object)
  start_tics = np.empty((nfeatures,), dtype=np.int32)
  stop_tics = np.empty((nfeatures,), dtype=np.int32)
  ifeature = 0
  ijchange = 1
  while ijchange<len(ichanges):                 # spans classes or starts with label
    if ichanges[ijchange-1]!=ichanges[ijchange] or \
           not behavior[ichanges[ijchange],jchanges[ijchange]]:
       ijchange += 1;
       continue
    start_tics[ifeature] = jchanges[ijchange-1] + 1
    stop_tics[ifeature] = jchanges[ijchange]
    features[ifeature] = labels[ichanges[ijchange]]
    ifeature += 1
    ijchange += 2
  ifeature -= 1

  features = features[:ifeature]
  start_tics = np.round((start_tics[:ifeature] / audio_tic_rate_probabilities \
                         - half_stride_sec) \
                        * audio_tic_rate).astype(np.int)
  stop_tics = np.round((stop_tics[:ifeature] / audio_tic_rate_probabilities \
                         + half_stride_sec) \
                       * audio_tic_rate).astype(np.int)
  return features, start_tics, stop_tics

def read_thresholds(logdir, model, thresholds_file):
  precision_recall_ratios=None
  thresholds=[]
  with open(os.path.join(logdir,model,thresholds_file)) as fid:
    csvreader = csv.reader(fid)
    for row in csvreader:
      if precision_recall_ratios==None:
        precision_recall_ratios=row[1:]
      else:
        thresholds.append(row)
  return precision_recall_ratios, thresholds

def save_thresholds(logdir, model, ckpt, thresholds, ratios, labels, dense=False):
  filename = 'thresholds'+\
             ('-dense-'+datetime.strftime(datetime.now(),'%Y%m%dT%H%M%S') if dense else '')+\
             '.ckpt-'+str(ckpt)+'.csv'
  fid = open(os.path.join(logdir,model,filename),"w")
  fidcsv = csv.writer(fid)
  fidcsv.writerow(['precision/recall'] + ratios)
  for ilabel in range(len(labels)):
    fidcsv.writerow([labels[ilabel]] + thresholds[labels[ilabel]].tolist())
  fid.close()
