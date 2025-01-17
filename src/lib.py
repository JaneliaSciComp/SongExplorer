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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy import interpolate
import math
from natsort import realsorted
from scipy.io import wavfile
import csv
from datetime import datetime
import importlib

import tifffile

import tensorflow as tf
import platform
from subprocess import run, PIPE, STDOUT

def load_audio_read_plugin(audio_read_plugin, audio_read_plugin_kwargs):
    sys.path.append(os.path.dirname(audio_read_plugin))
    audio_read_module = importlib.import_module(os.path.basename(audio_read_plugin))
    global audio_read, audio_read_exts, audio_read_rec2ch, audio_read_strip_rec
    def audio_read(wav_path, start_tic=None, stop_tic=None):
        return audio_read_module.audio_read(wav_path, start_tic, stop_tic,
                                            **audio_read_plugin_kwargs)
    def audio_read_exts():
         return audio_read_module.audio_read_exts(**audio_read_plugin_kwargs)
    def audio_read_rec2ch(wavfile):
         return audio_read_module.audio_read_rec2ch(wavfile, **audio_read_plugin_kwargs)
    def audio_read_strip_rec(recfile):
         return audio_read_module.audio_read_strip_rec(recfile, **audio_read_plugin_kwargs)
    return audio_read_module.audio_read_init(**audio_read_plugin_kwargs)

def load_video_read_plugin(video_read_plugin, video_read_plugin_kwargs):
    sys.path.append(os.path.dirname(video_read_plugin))
    video_read_module = importlib.import_module(os.path.basename(video_read_plugin))
    global video_read
    def video_read(fullpath, start_frame=None, stop_frame=None):
        return video_read_module.video_read(fullpath, start_frame, stop_frame,
                                            **video_read_plugin_kwargs)

def check_config(configuration_file):
    exec(open(configuration_file).read())

    def isinteger(l, x):
        if eval("not isinstance(l['"+x+"'], int)"):
            print("ERROR: "+x+" is not an integer")
    def isreal(l, x):
        if eval("not isinstance(l['"+x+"'], int) and not isinstance(l['"+x+"'], float)"):
            print("ERROR: "+x+" is neither an int nor a float")

    isreal(locals(), "gui_time_scale")
    isreal(locals(), "gui_freq_scale")
    isinteger(locals(), "audio_tic_rate")
    isinteger(locals(), "audio_nchannels")
    isreal(locals(), "gui_snippets_width_sec")
    isinteger(locals(), "gui_snippets_nx")
    isinteger(locals(), "gui_snippets_ny")
    isinteger(locals(), "gui_nlabels")
    isinteger(locals(), "gui_gui_width_pix")
    isreal(locals(), "gui_context_width_sec")
    isreal(locals(), "gui_context_offset_sec")
    isinteger(locals(), "gui_context_waveform_height_pix")
    isinteger(locals(), "gui_context_spectrogram_height_pix")
    isinteger(locals(), "models_per_job")
    isinteger(locals(), "pca_batch_size")
    isinteger(locals(), "nprobabilities")
    isinteger(locals(), "accuracy_parallelize")
    isinteger(locals(), "cluster_parallelize")
    isinteger(locals(), "congruence_parallelize")

    all_minusone = True
    local_vars = locals().copy()
    for resource_kind in ["ncpu_cores", "ngpu_cards", "ngigabytes_memory"]:
        for job_resource_name in filter(lambda x: resource_kind in x, local_vars.keys()):
            isinteger(locals(), job_resource_name)
            job_resource_value = local_vars[job_resource_name]
            all_minusone &= job_resource_value == -1
    if all_minusone:
        print("INFO: all job resources are -1 so only one job will be run at a time")

    return not all_minusone, locals()["server_username"], locals()["server_ipaddr"], local_vars, locals()["source_path"]

def check_config2(config_vars, resource_vars, server_ipaddr):
    for resource_kind in ["ncpu_cores", "ngpu_cards", "ngigabytes_memory"]:
        for job_resource_name in filter(lambda x: resource_kind in x, config_vars.keys()):
            job_resource_value = config_vars[job_resource_name]
            local_resource_name = "local_"+resource_kind
            local_resource_value = resource_vars[local_resource_name]
            if job_resource_value > local_resource_value:
                  print("WARNING: "+job_resource_name+" exceeds "+
                        str(local_resource_value)+" "+local_resource_name)
            if server_ipaddr:
                server_resource_name = "server_"+resource_kind
                server_resource_value = resource_vars[server_resource_name]
                if job_resource_value > server_resource_value:
                      print("WARNING: "+job_resource_name+" exceeds "+
                            str(server_resource_value)+" "+server_resource_name)


def get_srcrepobindirs():
    srcdir = os.path.dirname(os.path.realpath(__file__))
    repodir = os.path.dirname(srcdir)
    bindir = os.path.dirname(repodir)
    envdir = os.path.dirname(bindir)
    if platform.system()=='Windows':
        bindirs = [
            envdir,
            os.path.join(envdir, "Library", "mingw-w64", "bin"),
            os.path.join(envdir, "Library", "usr", "bin"),
            os.path.join(envdir, "Library", "bin"),
            os.path.join(envdir, "Scripts"),
            os.path.join(envdir, "bin"),
            ]
    else:
        bindirs = [os.path.join(envdir, "bin")]
    return srcdir, repodir, bindirs

def add_plugins_to_path(srcdir):
    sys.path.insert(0, srcdir)
    sys.path.insert(0, os.path.join(srcdir, "audio-read-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "video-read-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "detect-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "doubleclick-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "architecture-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "cluster-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "augmentation-plugins"))
    sys.path.insert(0, os.path.join(srcdir, "video-findfile-plugins"))

def compute_background(vidfile, video_bkg_frames, video_data, tiffile):
    print("INFO: calculating median background for "+vidfile)
    nframes = min(video_bkg_frames, video_data.shape[0])
    iframes = np.linspace(0, video_data.shape[0]-1, num=nframes, dtype=int)
    full = np.empty((nframes, *video_data[1].shape))
    for (i,iframe) in enumerate(iframes):
      full[i] = video_data[iframe]
    bkg = np.median(full, axis=0)
    tifffile.imwrite(tiffile, bkg, photometric='rgb')

def combine_events(events1, events2, logic):
  max_time1 = np.max([int(x[2]) for x in events1]+[0])
  max_time2 = np.max([int(x[2]) for x in events2]+[0])
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

def label_precisions_recall(ax, recalls_model_mean, precisions_model_mean, title, legend=True):
    ax.autoscale_view()
    ax.set_autoscale_on(False)
    miny = ax.get_ylim()[0]
    minx = ax.get_xlim()[0]
    if len(recalls_model_mean)>1:
        x = np.nanmean(recalls_model_mean)
        w = np.nanstd(recalls_model_mean)
        avebox = Rectangle((x-w,miny),2*w,100)
        ax.plot([x,x],[miny,100],'#c0c0c0', zorder=1, alpha=0.5)
        pc = PatchCollection([avebox], facecolor='#f0f0f0', alpha=0.5)
        ax.add_collection(pc)
        y = np.nanmean(precisions_model_mean)
        h = np.nanstd(precisions_model_mean)
        avebox = Rectangle((minx,y-h),100,2*h)
        ax.plot([minx,100],[y,y],'#c0c0c0', zorder=1, alpha=0.5)
        pc = PatchCollection([avebox], facecolor='#f0f0f0', alpha=0.5)
        ax.add_collection(pc)
    else:
        x,y,w,h = recalls_model_mean[0], precisions_model_mean[0], 0, 0
        ax.plot([x,x],[miny,100], '#c0c0c0', zorder=1, alpha=0.5)
        ax.plot([minx,100],[y,y], '#c0c0c0', zorder=1, alpha=0.5)

    ax.set_xlim(right=100)
    ax.set_ylim(top=100)
    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('Precision (%)')
    ax.set_title(title+
                 "P="+str(round(y,1))+"+/-"+str(round(h,1))+"%   "+
                 "R="+str(round(x,1))+"+/-"+str(round(w,1))+"%")
    if legend: ax.legend(loc=(1.05, 0.0))
  
def confusion_string2matrix(arg):
  arg = arg[1:-1]
  arg = arg.replace("\n\n", ",")
  arg = arg.replace("\n", ",")
  arg = re.sub("(?P<P>[0-9]) ","\g<P>,", arg)
  return ast.literal_eval(arg)

def _parse_confusion_matrices(logfile):
  confusion_matrices = {}
  labels_string = None
  recent_lines = []
  loss = ""
  nlines_to_keep=np.inf
  with open(logfile, 'r') as fid:
    for line in fid:
      if "labels_touse = " in line:
        m=re.search('labels_touse = (.+)',line)
        labels_touse = m.group(1).split(',')
      if "loss = " in line:
        m=re.search('loss = (.+)',line)
        loss = m.group(1)
      if "overlapped_prefix = " in line:
        m=re.search('overlapped_prefix = (.+)',line)
        overlapped_prefix = m.group(1)
        if loss=='overlapped' and len(labels_touse)>1:
            labels_touse = list(filter(lambda x: not x.startswith(overlapped_prefix),
                                       labels_touse))
            nlines_to_keep = 3*len(labels_touse)+1
        else:
            nlines_to_keep = len(labels_touse)+2
      recent_lines.append(line)
      while len(recent_lines) > nlines_to_keep:
        recent_lines.pop(0)
      if line.rstrip().endswith("Validation"):
        if not labels_string:
          labels_string = recent_lines[0].strip()
        assert labels_string == recent_lines[0].strip()
        ckpt = line.split(',')[1]
        confusion_string = ''.join(recent_lines[1:-1])
        confusion_matrix = confusion_string2matrix(confusion_string)
        confusion_matrices[ckpt] = [confusion_matrix] if loss=="exclusive" else confusion_matrix
        recent_lines = []
  return confusion_matrices, ast.literal_eval(labels_string)

def parse_confusion_matrices(logdir, kind, idx_time=None):
  models = list(filter(lambda x: x.startswith(kind+'_') and \
                          os.path.isdir(os.path.join(logdir,x)), os.listdir(logdir)))

  confusion_matrices={}
  labels=None
  for model in models:
    logfile = os.path.join(logdir, model+'.log')
    confusion_matrices[model], theselabels = _parse_confusion_matrices(logfile)
    if not labels:
      labels=theselabels
    assert labels==theselabels
      
  return confusion_matrices, labels


def normalize_confusion_matrix(matrix):
  row_norm_matrix = [[np.nan if np.nansum(x)==0.0 else y/np.nansum(x) for y in x]
                     for x in matrix]
  transposed_matrix = list(zip(*row_norm_matrix))  # row_norm_matrix here so it is balanced
  norm_transposed_matrix = [[np.nan if np.nansum(x)==0.0 else y/np.nansum(x) for y in x]
                            for x in transposed_matrix]
  col_norm_matrix = list(zip(*norm_transposed_matrix))
  recall = 100 * np.mean([x[i] for (i,x) in enumerate(row_norm_matrix) if not np.isnan(x[i])])
  precision = 100 * np.mean([x[i] for (i,x) in enumerate(col_norm_matrix) if not np.isnan(x[i])])
  return col_norm_matrix, row_norm_matrix, precision, recall


def plot_confusion_matrix(fig, ax,
                          abs_matrix, col_matrix, row_matrix, numbers,
                          title, labels, precision, recall):
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=cm.viridis),
                 cax=cax, ticks=[0,100], use_gridspec=True)
    ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Classification')
    ax.set_ylabel('Annotation')
    ax.set_title(title+"P="+str(round(precision,1))+"%   "+"R="+str(round(recall,1))+"%")

def layout(nplots):
  if nplots==10:
    return 2,5
  if nplots==21:
    return 3,7
  nrows = 1 if nplots==1 else np.floor(np.sqrt(nplots)).astype(int)
  ncols = math.ceil(nplots / nrows)
  return nrows, ncols


def read_log(frompath, logfile, loss='exclusive'):
  train_accuracy=[]; train_loss=[]; train_time=[]; train_step=[]
  validation_time=[]; validation_step=[]
  validation_precision=[]; validation_recall=[]
  validation_precision_mean=[]; validation_recall_mean=[]; validation_loss=[]
  test_precision=[]; test_recall=[]
  test_precision_mean=[]; test_recall_mean=[]; test_loss=[]
  nlayers=0
  bottleneck=math.inf
  with open(os.path.join(frompath,logfile),'r') as fid:
    train_restart_correction=0.0
    validation_restart_correction=0.0
    count_validation_state=count_testing_state=count_training_state=False
    conf_matrix_state, confusion_string = False, ""
    for line in fid:
      if "num validation labels" in line:
          count_validation_state=True
          label_counts = {"validation":{}, "testing":{}, "training":{}, }
      elif count_validation_state:
          m=re.search('\s*(\d+)\s(.*)',line)
          if m:
              label_counts["validation"][m.group(2)]=int(m.group(1))
          else:
              count_validation_state = False
      if "num testing labels" in line:
          count_testing_state=True
      elif count_testing_state:
          m=re.search('\s*(\d+)\s(.*)',line)
          if m:
              label_counts["testing"][m.group(2)]=int(m.group(1))
          else:
              count_testing_state = False
      if "num training labels" in line:
          count_training_state=True
      elif count_training_state:
          m=re.search('\s*(\d+)\s(.*)',line)
          if m:
              label_counts["training"][m.group(2)]=int(m.group(1))
          else:
              count_training_state = False
      if "labels_touse = " in line:
        m=re.search('labels_touse = (.+)',line)
        labels_touse = m.group(1).split(',')
      elif "batch_size = " in line:
        m=re.search('batch_size = (\d+)',line)
        batch_size = int(m.group(1))
      elif "Trainable params" in line:
        m=re.search('Trainable params: ([,\d]+)',line)
        nparameters_total = int(m.group(1).replace(',',''))
      elif 'Conv1D' in line or 'Conv2D' in line or 'Conv3D' in line:
        if validation_restart_correction==0.0:
          nlayers += 1
        m=re.search('[^,] (\d+)',line)
        nparameters_finallayer = int(m.group(1))
      elif 'None' in line:
        m=re.search('None, (\d+), (\d+), (\d+)',line)
        if m:
            bottleneck = min(bottleneck, int(m.group(1)) * int(m.group(2)) * int(m.group(3)))
      elif "Confusion Matrix" in line:
          conf_matrix_state=True
      elif conf_matrix_state and " [" in line and " ['" not in line and 'None' not in line:
          confusion_string+=line
          if "]]]" in line or ("]]" in line and "[[[" not in confusion_string):
              confusion_matrix = confusion_string2matrix(confusion_string)
              if np.ndim(confusion_matrix)==3:
                  precision = []
                  recall = []
                  for i in range(np.shape(confusion_matrix)[2]):
                      column_normalized_confusion_matrix, row_normalized_confusion_matrix, precision_mean, recall_mean = \
                              normalize_confusion_matrix(confusion_matrix[i])
                      precision.append(column_normalized_confusion_matrix[0][0])
                      recall.append(row_normalized_confusion_matrix[0][0])
              else:
                  column_normalized_confusion_matrix, row_normalized_confusion_matrix, precision_mean, recall_mean = \
                          normalize_confusion_matrix(confusion_matrix)
                  precision = [x[i] for (i,x) in enumerate(column_normalized_confusion_matrix)]
                  recall = [x[i] for (i,x) in enumerate(row_normalized_confusion_matrix)]
              conf_matrix_state=False
              confusion_string=""
      elif "Validation\n" in line:
        m=re.search('^([0-9.]+),([0-9]+),([0-9.]+) Validation$',line)
        validation_time_value = float(m.group(1))
        if loss != 'autoencoder':
            validation_precision.append(precision)
            validation_recall.append(recall)
            validation_precision_mean.append(precision_mean)
            validation_recall_mean.append(recall_mean)
        else:
            validation_loss.append(float(m.group(3)))
        if len(validation_time)>0 and \
                (validation_time_value+validation_restart_correction)<validation_time[-1]:
          validation_restart_correction = validation_time[-1]
        validation_time.append(validation_time_value+validation_restart_correction)
        validation_step.append(int(m.group(2)))
      elif "Testing\n" in line:
        if loss != 'autoencoder':
            test_precision.append(precision)
            test_recall.append(recall)
            test_precision_mean.append(precision_mean)
            test_recall_mean.append(recall_mean)
        else:
            m=re.search('^([0-9.]+),([0-9]+),([0-9.]+) Testing$',line)
            test_loss_mean.append(float(m.group(3)))
      else:
        if loss != 'autoencoder':
            m=re.search('^([0-9.]+),([0-9]+),([0-9.]+),([0-9.]+)$', line)
        else:
            m=re.search('^([0-9.]+),([0-9]+),([0-9.]+)$', line)
        if m:
          train_time_value = float(m.group(1))
          if len(train_time)>0 and \
                  (train_time_value+train_restart_correction)<train_time[-1]:
            train_restart_correction = train_time[-1]
          train_time.append(train_time_value+train_restart_correction)
          train_step.append(int(m.group(2)))
          if loss != 'autoencoder':
            train_accuracy.append(float(m.group(3)))
            train_loss.append(float(m.group(4)))
          else:
            train_loss.append(float(m.group(3)))

  return train_accuracy, train_loss, train_time, train_step, \
         validation_precision, validation_recall, \
         validation_precision_mean, validation_recall_mean, \
         validation_time, validation_step, validation_loss, \
         test_precision, test_recall, test_precision_mean, test_recall_mean, test_loss, \
         labels_touse, label_counts, \
         nparameters_total, nparameters_finallayer, bottleneck, \
         batch_size, nlayers
         #test_accuracy, \


def read_logs(frompath, loss='exclusive'):
  train_accuracy={}; train_loss={}; train_time={}; train_step={}
  validation_precision={}; validation_recall={}
  validation_precision_mean={}; validation_recall_mean={}
  validation_time={}; validation_step={}; validation_loss={}
  test_precision={}; test_recall={}
  test_precision_mean={}; test_recall_mean={}; test_loss={}
  labels_touse={}
  label_counts={}
  nparameters_total={}
  nparameters_finallayer={}
  bottleneck={}
  batch_size={}
  nlayers={}
  for logfile in filter(lambda x: re.match('(train|xvalidate|generalize)_.*log',x), \
                        os.listdir(frompath)):
    model=logfile[:-4]
    train_accuracy[model], train_loss[model], train_time[model], train_step[model], \
          validation_precision[model], validation_recall[model], \
          validation_precision_mean[model], validation_recall_mean[model], \
          validation_time[model], validation_step[model], validation_loss[model], \
          test_precision[model], test_recall[model], \
          test_precision_mean[model], test_recall_mean[model], test_loss[model], \
          labels_touse[model], label_counts[model], \
          nparameters_total[model], nparameters_finallayer[model], bottleneck[model], \
          batch_size[model], nlayers[model] = \
          read_log(frompath, logfile, loss)
          #test_accuracy[model], \

  return train_accuracy, train_loss, train_time, train_step, \
         validation_precision, validation_recall, \
         validation_precision_mean, validation_recall_mean, \
         validation_time, validation_step, validation_loss, \
         test_precision, test_recall, \
         test_precision_mean, test_recall_mean, test_loss, \
         labels_touse, label_counts, \
         nparameters_total, nparameters_finallayer, bottleneck, \
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
                                           nprobabilities, ratios, loss):
  probabilities = {}
  thresholds = {}
  precisions = {}
  recalls = {}
  sensitivities = {}
  specificities = {}
  pr_areas = {}
  roc_areas = {}
  for ilabel in range(len(labels)):
    if loss=='exclusive':
        itrue = validation_ground_truth==ilabel
        ifalse = validation_ground_truth!=ilabel
    else:
        itrue = validation_ground_truth[:,:,ilabel]==1
        ifalse = validation_ground_truth[:,:,ilabel]!=1
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

def discretize_probabilities(probability_matrix, thresholds, labels,
                            audio_tic_rate_probabilities, half_stride_sec, audio_tic_rate):
  probability_matrix = np.append(probability_matrix,
                                 np.zeros((np.shape(probability_matrix)[0],1)),
                                 axis=1)
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

  features = features[:ifeature]
  start_tics = np.round((start_tics[:ifeature] / audio_tic_rate_probabilities \
                         - half_stride_sec) \
                        * audio_tic_rate).astype(int)
  stop_tics = np.round((stop_tics[:ifeature] / audio_tic_rate_probabilities \
                         + half_stride_sec) \
                       * audio_tic_rate).astype(int)
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
  fidcsv = csv.writer(fid, lineterminator='\n')
  fidcsv.writerow(['precision/recall'] + ratios)
  for ilabel in range(len(labels)):
    fidcsv.writerow([labels[ilabel]] + thresholds[labels[ilabel]].tolist())
  fid.close()

def select_GPUs(igpu):
  if igpu != "songexplorer_use_all_gpus":
    physical_devices = tf.config.list_physical_devices('GPU')
    igpu = os.environ[igpu] if igpu in os.environ else igpu
    igpu = [int(x) for x in igpu.split(',') if x]
    tf.config.set_visible_devices([physical_devices[x] for x in igpu], 'GPU')

def log_nvidia_smi_output(igpu):
  if not (igpu=='' or igpu in os.environ and os.environ[igpu]==''):
      if platform.system()=='Windows':
          cmd = 'where nvidia-smi.exe && nvidia-smi.exe'
      else:
          cmd = 'which nvidia-smi && nvidia-smi'
      if igpu in os.environ and os.environ[igpu]:
          cmd += ' -i ' + os.environ[igpu] 
      elif igpu != 'songexplorer_use_all_gpus':
          cmd += ' -i ' + igpu
      p = run(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
      print(p.stdout.decode('ascii').rstrip())
