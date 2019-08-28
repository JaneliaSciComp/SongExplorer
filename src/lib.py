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

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from jitter import *

def plot_triangle_matrix(ax, abs_matrix, col_matrix, row_matrix, numbers):
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
        polygon=Polygon([[ix-0.5,iy-0.5],[ix-0.5,iy+0.5],[ix+0.5,iy+0.5],[ix+0.5,iy-0.5]], facecolor=(1,1,1,0), edgecolor="white")
        patches.append(polygon)
  p = PatchCollection(patches,match_original=True)
  ax.add_collection(p)
  ax.set_xlim(-0.5,sx-0.5)
  ax.set_ylim(-0.5,sy-0.5)


def normalize_matrix(matrix):
  row_normalized_matrix = [[np.nan if sum(x)==0.0 else y/sum(x) for y in x] for x in matrix]
  transposed_matrix = list(zip(*matrix))
  norm_transposed_matrix = [[np.nan if sum(x)==0.0 else y/sum(x) for y in x] for x in transposed_matrix]
  col_normalized_matrix = list(zip(*norm_transposed_matrix))
  return row_normalized_matrix, col_normalized_matrix


def sum_confusion_matrices(logdir):
  labels_folders = list(filter(lambda x: x.startswith('train_') and \
                          os.path.isdir(os.path.join(logdir,x)), os.listdir(logdir)))
  labels_path = os.path.join(logdir,list(labels_folders)[0],'vgg_labels.txt')
  nwanted_words = sum(1 for line in open(labels_path))

  words={}
  confusion_matrix={}
  for k in labels_folders:
    model=k+'.log'
    logfile = os.path.join(logdir,model)
    words[model], confusion_matrix[model], _, _ = parse_confusion_matrix(logfile, nwanted_words)

  first_label=labels_folders[0]+'.log'

  summed_confusion_matrix = np.zeros((np.shape(confusion_matrix[first_label])))
  for k in labels_folders:
    summed_confusion_matrix += confusion_matrix[k+'.log']
  return summed_confusion_matrix, words[first_label]


def parse_confusion_matrix(logfile, nwanted_words, which_one=0):
  max_count = '-m'+str(which_one) if which_one>0 else ''
  kind = "Validation"  #"Final test"
  cmd = "grep -B"+str(nwanted_words+1)+" '"+kind+" accuracy' "+logfile+" "+max_count+" | tail -"+str(nwanted_words+2)+" | head -1"
  words_string = check_output(cmd, shell=True)
  words_string = words_string.decode("ascii")
  words_string = words_string[1:-1]
  cmd = "grep -B"+str(nwanted_words+0)+" '"+kind+" accuracy' "+logfile+" "+max_count+" | tail -"+str(nwanted_words+1)+" | head -"+str(nwanted_words+0)
  confusion_string = check_output(cmd, shell=True)
  confusion_string = confusion_string.decode("ascii")
  confusion_string = confusion_string[1:-1]
  confusion_string = confusion_string.replace("\n",",")
  confusion_string = re.sub("(?P<P>[0-9]) ","\g<P>,",confusion_string)
  confusion_matrix = ast.literal_eval(confusion_string)
  nannotations = [sum(x) for x in confusion_matrix]
  equalized_confusion_matrix = [[x/n if n>0 else np.nan for x in cm] for (cm,n) in zip(confusion_matrix,nannotations)]
  accuracy = [x[i] for (i,x) in enumerate(equalized_confusion_matrix)]
  accuracy = 100 * np.nansum(accuracy) / sum([not np.isnan(x) for x in accuracy])  #len(equalized_confusion_matrix)
  #cmd = "grep '"+kind+" accuracy' "+logfile+" "+max_count+" | tail -1"
  #accuracy_string = check_output(cmd, shell=True)
  #accuracy_string = accuracy_string.decode("ascii")
  #accuracy_string = accuracy_string[:-1]
  #accuracy = re.search(r'accuracy = (.*%)', accuracy_string)
  return ast.literal_eval(words_string), confusion_matrix, accuracy, nannotations
  #return ast.literal_eval(words_string), equalized_confusion_matrix, accuracy, nannotations


def read_log(frompath, logfile):
  train_accuracy=[]; train_loss=[]; train_time=[]; train_step=[]
  validation_time=[]; validation_step=[]
  #test_accuracy=np.nan
  nlayers=0
  input_size=None
  with open(os.path.join(frompath,logfile),'r') as handle:
    train_restart_correction=0.0
    validation_restart_correction=0.0
    count_train_state=False
    for line in handle:
      if "num training labels" in line:
        count_train_state=True
        word_counts = {}
      elif count_train_state:
        if "conv layer" in line:
          count_train_state = False
        else:
          m=re.search('INFO:tensorflow:\s+(\d+)\s(.*)',line)
          word_counts[m.group(2)]=int(m.group(1))

      if "wanted_words" in line:
        m=re.search('wanted_words = (.+)',line)
        wanted_words = m.group(1).split(',')
        nwords = len(wanted_words)
      elif "batch_size" in line:
        m=re.search('batch_size = (\d+)',line)
        batch_size = int(m.group(1))
      elif "trainable parameters" in line:
        m=re.search('trainable parameters: (\d+)',line)
        nparameters_total = int(m.group(1))
      elif ' layer' in line:
        if validation_restart_correction==0.0:
          nlayers += 1
        if 'conv layer 0' in line:
          m=re.search('.*conv layer 0: in_shape = \(\d+, \?, (\d+), (\d+)\), time_size = (\d+)',line)
          input_size = int(m.group(1)) * int(m.group(2)) * int(m.group(3))
        elif "final layer" in line:
          if "weight" in line:
            m=re.search('weight_shape = \((\d+), (\d+)\)',line)
            nparameters_finallayer = int(m.group(1)) * int(m.group(2))
          if "conv" in line:
            m=re.search('conv_shape = \((\d+), (\d+), (\d+)\)',line)
            nparameters_finallayer = int(m.group(1)) * int(m.group(2)) * int(m.group(3))
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
      elif "Validation accuracy" in line:
        m=re.search('Elapsed (.*), Step (.*):.* = (.*)%',line)
        validation_time_value = float(m.group(1))
        if len(validation_time)>0 and \
                (validation_time_value+validation_restart_correction)<validation_time[-1]:
          validation_restart_correction = validation_time[-1]
        validation_time.append(validation_time_value+validation_restart_correction)
        validation_step.append(int(m.group(2)))
        #validation_accuracy.append(float(m.group(3)))
      #elif "Final test accuracy" in line:
      #  m=re.search('Elapsed .*, Step .*:.* = (.*)%',line)
      #  test_accuracy = float(m.group(1))
      #  break
  validation_accuracy=[]
  if len(word_counts)>0:
    for i in range(len(validation_step)):
      _, _, accuracy, _ = parse_confusion_matrix(os.path.join(frompath,logfile), nwords, i+1)
      validation_accuracy.append(accuracy)

  return train_accuracy, train_loss, train_time, train_step, \
         validation_accuracy, validation_time, validation_step, \
         wanted_words, word_counts, \
         nparameters_total, nparameters_finallayer, \
         batch_size, nlayers, input_size
         #test_accuracy, \


def read_logs(frompath):
  train_accuracy={}; train_loss={}; train_time={}; train_step={}
  validation_accuracy={}; validation_time={}; validation_step={}
  final_samples={}; final_ground_truth={}; final_logits={}
  #test_accuracy={};
  wanted_words={}
  word_counts={}
  nparameters_total={}
  nparameters_finallayer={}
  batch_size={}
  nlayers={}
  input_size={}
  for logfile in filter(lambda x: re.match('train_.*log',x), os.listdir(frompath)):
    model=logfile[:-4]
    train_accuracy[model], train_loss[model], train_time[model], train_step[model], \
          validation_accuracy[model], validation_time[model], validation_step[model], \
          wanted_words[model], word_counts[model], \
          nparameters_total[model], nparameters_finallayer[model], \
          batch_size[model], nlayers[model], input_size[model] = \
          read_log(frompath, logfile)
          #test_accuracy[model], \

  return train_accuracy, train_loss, train_time, train_step, \
         validation_accuracy, validation_time, validation_step, \
         wanted_words, word_counts, \
         nparameters_total, nparameters_finallayer, \
         batch_size, nlayers, input_size
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
  npzfile = np.load(os.path.join(frompath,logdir,logit_file))
  return npzfile['samples'], npzfile['groundtruth'], npzfile['logits']
 

def truncate_times(validation_time, accuracy, expt):
  max_time = max([x[-1] for x in validation_time[expt].values()])
  for (iexpt,expt) in enumerate(accuracy.keys()):
    for model in validation_time[expt].keys():
      xdata = np.array(validation_time[expt][model])
      ydata = np.array(accuracy[expt][model])
      imax_time = np.where(xdata<=max_time)
      validation_time[expt][model] = xdata[imax_time]
      accuracy[expt][model] = ydata[imax_time]
  return validation_time, accuracy


def plot_time_traces(ax,validation_time,accuracy,ylabel,ltitle,outlier_crit=0,real=False,llabels=None,reverse=False):
  bottom=100
  sortfun = realsorted if real else natsorted
  for (iexpt,expt) in enumerate(sortfun(accuracy.keys(), reverse=reverse)):
    color = cm.viridis((len(accuracy)-iexpt)/len(accuracy))
    for model in validation_time[expt].keys():
      line, = ax.plot(np.array(validation_time[expt][model])/60, accuracy[expt][model], \
                      color=color, zorder=iexpt, linewidth=1)
      bottom = min([bottom]+[x for x in accuracy[expt][model] if x>outlier_crit])
    line.set_label(llabels[expt] if llabels else expt.split('-')[1])
  ax.set_ylim(bottom=bottom-5)
  ax.set_xlabel('Training time (min)')
  ax.set_ylabel(ylabel)
  ax.legend(loc='lower right', title=ltitle, ncol=2 if "Annotations" in ltitle else 1)

def plot_final_accuracies(ax,accuracy,xlabel,ylabel,outlier_crit=0,real=False,llabels=None,reverse=False):
  data = {k:list([x[-1] for x in accuracy[k].values()]) for k in accuracy}
  ldata = jitter_plot(ax, data, outlier_crit=outlier_crit, real=real, reverse=reverse)
  ax.set_ylabel(ylabel)
  ax.set_xlabel(xlabel)
  ax.set_xticks(range(len(ldata)))
  ax.set_xticklabels([llabels[x] if llabels else x.split('-')[1] for x in ldata])
  if llabels:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

def choose_units(data):
  if data[-1]<60:
    return data, 'sec'
  elif data[-1]<60*60:
    return [x/60 for x in data], 'min'
  elif data[-1]<24*60*60:
    return [x/60/60 for x in data], 'hr'
  else:
    return [x/24/60/60 for x in data], 'days'

def plot_overlayed_steps(validation_accuracy, validation_time, validation_step, models, batch_sizes, training_set_sizes):
  colors=['r','g','b','c','m','y','k','r:','g:','b:','c:','m:','y:','k:']
  color_index=natsorted(models)
  fig = plt.figure(figsize=(6.4*3, 2*4.8))

  ax = fig.add_subplot(2,3,1)
  for model in models:
    scaled_validation_time, units = choose_units(validation_time[model])

    color=colors[color_index.index(model)]
    line, = ax.plot(validation_step[model], validation_accuracy[model], color)
    line.set_label(model)
  #ax.set_ylim([0,100])
  ax.set_xlabel('step')
  ax.set_ylabel('accuracy')
  #ax.legend(loc='lower right')

  ax = fig.add_subplot(2,3,2)
  for model in models:
    color=colors[color_index.index(model)]
    ax.plot([x*batch_sizes[model]/training_set_sizes[model] for x in validation_step[model]], validation_accuracy[model], color)
  #ax.set_ylim([0,100])
  ax.set_xlabel('epoch')
  ax.set_ylabel('accuracy')

  ax = fig.add_subplot(2,3,3)
  for model in models:
    color=colors[color_index.index(model)]
    #ax.plot(np.array(scaled_validation_time), np.power(validation_accuracy[model],3), color)
    line, = ax.plot(scaled_validation_time, validation_accuracy[model], color)
    line.set_label(model)
  #ax.set_ylim([0,100])
  ax.set_xlabel('time ('+units+')')
  ax.set_ylabel('accuracy')
  #ax.set_yticklabels(np.rint(np.cbrt(ax.get_yticks())).astype(int))
  ax.legend(loc='lower right')

  ax = fig.add_subplot(2,3,4)
  for model in models:
    color=colors[color_index.index(model)]
    ax.plot(scaled_validation_time, [x*batch_sizes[model]/training_set_sizes[model] for x in validation_step[model]], color)
  ax.set_xlabel('time ('+units+')')
  ax.set_ylabel('epoch')

  ax = fig.add_subplot(2,3,5)
  for model in models:
    color=colors[color_index.index(model)]
    ax.plot(scaled_validation_time, validation_step[model], color)
  ax.set_xlabel('time ('+units+')')
  ax.set_ylabel('step')

  ax = fig.add_subplot(2,3,6)
  for model in models:
    color=colors[color_index.index(model)]
    ax.plot([x*batch_sizes[model]/training_set_sizes[model] for x in validation_step[model]], validation_step[model], color)
  ax.set_xlabel('epoch')
  ax.set_ylabel('step')

  fig.tight_layout()


def calculate_precision_recall_specificity(test_ground_truth, test_logits, words, \
                                           nprobabilities, ratios):
  probabilities = {}
  thresholds = {}
  precisions = {}
  recalls = {}
  sensitivities = {}
  specificities = {}
  pr_areas = {}
  roc_areas = {}
  for iword in range(len(words)):
    print(iword, end="\r", flush=True)
    itrue = test_ground_truth==iword
    ifalse = test_ground_truth!=iword
    precisions[words[iword]] = np.full([nprobabilities],np.nan)
    recalls[words[iword]] = np.full([nprobabilities],np.nan)
    sensitivities[words[iword]] = np.full([nprobabilities],np.nan)
    specificities[words[iword]] = np.full([nprobabilities],np.nan)
    if not np.any(itrue):
      pr_areas[words[iword]] = np.nan
      roc_areas[words[iword]] = np.nan
      probabilities[words[iword]] = np.full([nprobabilities],np.nan)
      thresholds[words[iword]] = np.full(len(ratios),np.nan)
      continue
    pr_areas[words[iword]] = 0
    roc_areas[words[iword]] = 0
    max_logit = max(test_logits[itrue,iword])
    #max_2nd_logit = max([x for x in test_logits[itrue,iword] if x!=max_logit])
    #probabilities_logit = np.linspace(
    #        min(test_logits[itrue,iword]), max_2nd_logit, nprobabilities)
    probabilities_logit = np.linspace(
            min(test_logits[itrue,iword]), max_logit, nprobabilities)
    probabilities[words[iword]] = np.exp(probabilities_logit) / (np.exp(probabilities_logit) + 1)
    for (iprobability,probability_logit) in enumerate(probabilities_logit):
      #Tp = np.sum(test_logits[itrue,iword]>probability_logit)
      #Fp = np.sum(test_logits[ifalse,iword]>probability_logit)
      Tp = np.sum(test_logits[itrue,iword]>=probability_logit)
      Fp = np.sum(test_logits[ifalse,iword]>=probability_logit)
      Fn = np.sum(itrue)-Tp  # == sum(test_logits[itrue,iword]<=probability_logit)
      Tn = np.sum(ifalse)-Fp  # == sum(test_logits[ifalse,iword]<=probability_logit)
      precisions[words[iword]][iprobability] = Tp/(Tp+Fp)
      recalls[words[iword]][iprobability] = Tp/(Tp+Fn)
      sensitivities[words[iword]][iprobability] = Tp/(Tp+Fn)
      specificities[words[iword]][iprobability] = Tn/(Tn+Fp)
      if iprobability==0:
        delta_pr = precisions[words[iword]][iprobability] * \
                np.abs(recalls[words[iword]][iprobability] - 1)
        delta_roc = specificities[words[iword]][iprobability] * \
                np.abs(sensitivities[words[iword]][iprobability] - 1)
      elif iprobability+1==len(probabilities_logit):
        delta_pr = precisions[words[iword]][iprobability] * \
                np.abs(recalls[words[iword]][iprobability] - 0)
        delta_roc = specificities[words[iword]][iprobability] * \
                np.abs(sensitivities[words[iword]][iprobability] - 0)
      else:
        delta_pr = precisions[words[iword]][iprobability] * \
                np.abs(recalls[words[iword]][iprobability] - recalls[words[iword]][iprobability-1])
        delta_roc = specificities[words[iword]][iprobability] * \
                np.abs(sensitivities[words[iword]][iprobability] - \
                sensitivities[words[iword]][iprobability-1])
      if not np.isnan(delta_pr):
        pr_areas[words[iword]] += delta_pr
      if not np.isnan(delta_roc):
        roc_areas[words[iword]] += delta_roc
    f = interpolate.interp1d(precisions[words[iword]]/recalls[words[iword]],
                             probabilities[words[iword]], fill_value="extrapolate")
    thresholds[words[iword]] = f(ratios)
  return probabilities, thresholds, precisions, recalls, sensitivities, specificities, \
         pr_areas, roc_areas
