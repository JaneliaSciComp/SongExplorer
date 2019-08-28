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
from natsort import realsorted

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *
from jitter import *

_,logdirs_prefix = sys.argv
print('logdirs_prefix: '+logdirs_prefix)

basename, dirname = os.path.split(logdirs_prefix)

train_time={}
validation_accuracy={}
validation_time={}
validation_step={}
wanted_words={}
word_counts={}
nparameters_total={}
nparameters_finallayer={}
batch_size={}
nlayers={}
input_size={}
logdirs = natsorted(filter(lambda x: re.match(dirname+'\-.*',x), os.listdir(basename)))
for logdir in logdirs:
  if not os.path.isdir(os.path.join(basename,logdir)):
    continue
  print(logdir)
  _, train_time[logdir], _, \
          validation_accuracy[logdir], validation_time[logdir], validation_step[logdir], \
          wanted_words[logdir], word_counts[logdir], \
          nparameters_total[logdir], nparameters_finallayer[logdir], \
          batch_size[logdir], nlayers[logdir], input_size[logdir] = \
          read_logs(os.path.join(basename,logdir))
  if len(set([tuple(x) for x in wanted_words[logdir].values()]))>1:
    print('WARNING: not all wanted_words are the same')
  #assert len(set(word_counts[logdir].values()))==1
  if len(set(nparameters_total[logdir].values()))>1:
    print('WARNING: not all nparameters_total are the same')
  if len(set(nparameters_finallayer[logdir].values()))>1:
    print('WARNING: not all nparameters_finallayer are the same')
  if len(set(batch_size[logdir].values()))>1:
    print('WARNING: not all batch_size are the same')
  if len(set(nlayers[logdir].values()))>1:
    print('WARNING: not all nlayers are the same')
  if len(set(input_size[logdir].values()))>1:
    print('WARNING: not all input_size are the same')
  if len(validation_accuracy)>0:
    if set([tuple(x) for x in wanted_words[logdirs[0]].values()])!=set([tuple(x) for x in wanted_words[logdir].values()]):
      print('WARNING: not all wanted_words are the same')
    #assert set(word_counts[logdirs[0]].values())==set(word_counts[logdir].values())
    if set(nparameters_total[logdirs[0]].values())!=set(nparameters_total[logdir].values()):
      print('WARNING: not all nparameters_total are the same')
    if set(nparameters_finallayer[logdirs[0]].values())!=set(nparameters_finallayer[logdir].values()):
      print('WARNING: not all nparameters_finallayer are the same')
    if set(batch_size[logdirs[0]].values())!=set(batch_size[logdir].values()):
      print('WARNING: not all batch_size are the same')
    if set(nlayers[logdirs[0]].values())!=set(nlayers[logdir].values()):
      print('WARNING: not all nlayers are the same')
    if set(input_size[logdirs[0]].values())!=set(input_size[logdir].values()):
      print('WARNING: not all input_size are the same')

fig = plt.figure(figsize=(8,10*2/3))

ax = fig.add_subplot(2,2,1)
plot_final_accuracies(ax,validation_accuracy,dirname,'Accuracy')

ax = fig.add_subplot(2,2,2)
plot_time_traces(ax,validation_time,validation_accuracy,'Accuracy',dirname)

ax = fig.add_subplot(2,2,3)
ldata = natsorted(nparameters_total.keys())
xdata = range(len(ldata))
model = list(nparameters_total[ldata[0]].keys())[0]
ydata = [nparameters_total[x][model] - nparameters_finallayer[x][model] for x in ldata]
ydata2 = [nparameters_finallayer[x][model] for x in ldata]
ax.bar(xdata,ydata,color='k')
ax.bar(xdata,ydata2,bottom=ydata,color='gray')
ax.set_xlabel(dirname)
ax.set_ylabel('Trainable parameters')
ax.set_xticks(xdata)
ax.set_xticklabels([x.split('-')[1] for x in ldata], rotation=20, ha='right')

ax = fig.add_subplot(2,2,4)
data = {k:list([np.median(np.diff(x)) for x in train_time[k].values()]) for k in train_time}
ldata = jitter_plot(ax, data)
ax.set_ylabel('Time / step (ms)')
ax.set_xlabel(dirname)
ax.set_xticks(range(len(ldata)))
ax.set_xticklabels([x.split('-')[1] for x in ldata])

fig.tight_layout()
plt.savefig(logdirs_prefix+'-compare.pdf')
