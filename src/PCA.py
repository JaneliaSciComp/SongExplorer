#!/usr/bin/env python3

# reduce dimensionality of internal activation states with PCA

# e.g. PCA.py \
#     --data_dir=`pwd`/groundtruth-data \
#     --layers=0,1,2,3,4 \
#     --pca_batch_size=5 \
#     --parallelize=0 \
#     --parameters='{"ndims":2}'
 
import argparse
import os
import numpy as np
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from natsort import natsorted
from datetime import datetime
import socket
from itertools import repeat

import json

def cluster_parameters():
    return [
          ["ndims", "# dims", ["2","3"], "2", 1, [], None, True],
    ]

def do_cluster(activations_flattened, ilayer, layers, parameters):
    if ilayer in layers:
        print("reducing dimensionality of layer "+str(ilayer)+" with PCA...")
        mu = np.mean(activations_flattened[ilayer], axis=0)
        sigma = np.std(activations_flattened[ilayer], axis=0)
        activations_scaled = (activations_flattened[ilayer] - mu) / sigma
        if FLAGS.pca_batch_size==0:
            pca = PCA()
        else:
            nfeatures = np.shape(activations_scaled)[1]
            pca = IncrementalPCA(batch_size = FLAGS.pca_batch_size * nfeatures)
        fit = pca.fit(activations_scaled)
        return fit, fit.transform(activations_scaled)[:,0:int(parameters["ndims"])]
    else:
        return None, None

FLAGS = None

def main():
  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    print('%s = %s' % (key, flags[key]))

  layers = [int(x) for x in FLAGS.layers.split(',')]

  print("loading data...")
  activations=[]
  npzfile = np.load(os.path.join(FLAGS.data_dir, 'activations.npz'),
                    allow_pickle=True)
  sounds = npzfile['sounds']
  for arr_ in natsorted(filter(lambda x: x.startswith('arr_'), npzfile.files)):
    activations.append(npzfile[arr_])

  nlayers = len(activations)

  kinds = set([x['kind'] for x in sounds])
  labels = set([x['label'] for x in sounds])
  print('label counts')
  for kind in kinds:
    print(kind)
    for label in labels:
      count = sum([label==x['label'] and kind==x['kind'] for x in sounds])
      print(count,label)

  activations_flattened = [None]*nlayers
  for ilayer in layers:
    nsounds = np.shape(activations[ilayer])[0]
    activations_flattened[ilayer] = np.reshape(activations[ilayer],(nsounds,-1))
    print("shape of layer "+str(ilayer)+" is "+str(np.shape(activations_flattened[ilayer])))

  fits_pca = [None]*nlayers
  activations_scaled = [None]*nlayers

  if FLAGS.parallelize!=0:
    from multiprocessing import Pool
    nprocs = os.cpu_count() if FLAGS.parallelize==-1 else FLAGS.parallelize
    with Pool(min(nprocs,nlayers)) as p:
      fits, activations_clustered = zip(*p.starmap(do_cluster,
                                                   zip(repeat(activations_flattened),
                                                       range(len(activations_flattened)),
                                                       repeat(layers),
                                                       repeat(FLAGS.parameters))))
  else:
    fits = [None]*nlayers
    activations_clustered = [None]*nlayers
    for ilayer in layers:
      print("reducing dimensionality of layer "+str(ilayer)+" with PCA...")
      fits[ilayer], activations_clustered[ilayer] = do_cluster(activations_flattened,
                                                               ilayer,
                                                               layers,
                                                               FLAGS.parameters)

  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  #plt.ion()

  fig = plt.figure()
  ax = fig.add_subplot(111)
  for ilayer in layers:
    cumsum = np.cumsum(fits[ilayer].explained_variance_ratio_)
    line, = ax.plot(cumsum)
    line.set_label('layer '+str(ilayer))

  ax.set_ylabel('cumsum explained variance')
  ax.set_xlabel('# of components')
  ax.legend(loc='lower right')
  plt.savefig(os.path.join(FLAGS.data_dir, 'cluster.pdf'))

  np.savez(os.path.join(FLAGS.data_dir, 'cluster'), \
           sounds = sounds,
           activations_clustered = np.array(activations_clustered, dtype=object),
           fits = np.array(fits, dtype=object) if FLAGS.save_fits else None,
           labels_touse = npzfile['labels_touse'],
           kinds_touse = npzfile['kinds_touse'])
  
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str)
  parser.add_argument(
      '--layers',
      type=str)
  parser.add_argument(
      '--pca_batch_size',
      type=int)
  parser.add_argument(
      '--parallelize',
      type=int,
      default=0)
  parser.add_argument(
      '--parameters',
      type=json.loads,
      default='{"neighbors": 10, "distance": 0.1}')
  parser.add_argument(
      '--save_fits',
      type=str2bool,
      default=False,
      help='Whether to save the cluster models')

  FLAGS, unparsed = parser.parse_known_args()

  print(str(datetime.now())+": start time")
  repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
    print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
  print("hostname = "+socket.gethostname())
  
  try:
    main()

  except Exception as e:
    print(e)
  
  finally:
    if hasattr(os, 'sync'):
      os.sync()
    print(str(datetime.now())+": finish time")
