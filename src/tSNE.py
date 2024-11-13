#!/usr/bin/env python

# reduce dimensionality of internal activation states with tSNE.  optionally
# reducing with PCA first too

# e.g. tSNE.py \
#     --data_dir=`pwd`/groundtruth-data \
#     --layers=0,1,2,3,4 \
#     --pca_batch_size=5 \
#     --parallelize=0 \
#     --parameters='{"ndims":2, "pca_fraction":1, "perplexity":30, "exaggeration":12.0}'
 
import argparse
import os
import numpy as np
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from natsort import natsorted
from datetime import datetime
import socket
from itertools import repeat
from openTSNE import TSNE

import json

def _callback(p,M,V,C):
    C.time.sleep(0.5)
    V.cluster_parameters[p].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def pca_fraction_callback(n,M,V,C):
    pca_fraction = float(V.cluster_parameters['pca-fraction'].value)
    if not 0 < pca_fraction <= 1:
        V.cluster_parameters['pca-fraction'].css_classes = ['changed']
        V.cluster_parameters['pca-fraction'].value = str(min(1, max(0, pca_fraction)))
        bokehlog.info("WARNING:  `PCA fraction` must be between 0 and 1")
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback('pca-fraction',M,V,C))
        else:
            _callback('pca-fraction',M,V,C)

def positive_callback(key,n,M,V,C):
    value = float(V.cluster_parameters[key].value)
    if value <= 0:
        V.cluster_parameters[key].css_classes = ['changed']
        V.cluster_parameters[key].value = "1"
        bokehlog.info("WARNING:  `"+key+"` must be positive")
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(key,M,V,C))
        else:
            _callback(key,M,V,C)

def cluster_parameters():
    return [
          ["ndims",        "# dims",       ["2","3"], "2",    1, [], None,                                                      True],
          ["pca-fraction", "PCA fraction", "",        "1",    1, [], pca_fraction_callback,                                     True],
          ["perplexity",   "perplexity",   "",        "30",   1, [], lambda n,M,V,C: positive_callback("neighbors",n,M,V,C),    True],
          ["exaggeration", "exaggeration", "",        "12.0", 1, [], lambda n,M,V,C: positive_callback("distance",n,M,V,C),     True],
    ]

def do_cluster(activations_tocluster, ilayer, layers, parameters):
    if ilayer in layers:
        print("reducing dimensionality of layer "+str(ilayer)+" with t-SNE...")
        fit = TSNE(verbose = 3,
                   n_components = int(parameters["ndims"]),
                   perplexity = float(parameters["perplexity"]),
                   early_exaggeration = float(parameters["exaggeration"])
                  ).fit(activations_tocluster[ilayer])
        return fit, np.array(fit)
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
  pca_fraction = float(FLAGS.parameters["pca-fraction"])
  if pca_fraction < 1:
    activations_scaled = [None]*nlayers
    for ilayer in layers:
      print("reducing dimensionality of layer "+str(ilayer)+" with PCA...")
      mu = np.mean(activations_flattened[ilayer], axis=0)
      sigma = np.std(activations_flattened[ilayer], axis=0)
      activations_scaled[ilayer] = (activations_flattened[ilayer]-mu)/sigma
      if FLAGS.pca_batch_size==0:
        pca = PCA()
      else:
        nfeatures = np.shape(activations_scaled[ilayer])[1]
        pca = IncrementalPCA(batch_size=FLAGS.pca_batch_size*nfeatures)
      fits_pca[ilayer] =  pca.fit(activations_scaled[ilayer])

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    #plt.ion()

    activations_kept = [None]*nlayers
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ilayer in layers:
      cumsum = np.cumsum(fits_pca[ilayer].explained_variance_ratio_)
      line, = ax.plot(cumsum)
      activations_transformed = fits_pca[ilayer].transform(activations_scaled[ilayer])
      ncomponents = np.where(cumsum>=pca_fraction)[0][0]
      line.set_label('layer '+str(ilayer)+', n='+str(ncomponents+1))
      activations_kept[ilayer] = activations_transformed[:,0:ncomponents+1]
      print("shape of layer "+str(ilayer)+" after PCA is "+str(np.shape(activations_kept[ilayer])))

    ax.set_ylabel('cumsum explained variance')
    ax.set_xlabel('# of components')
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(FLAGS.data_dir, 'cluster-pca.pdf'))

    activations_tocluster = activations_kept
  else:
    activations_tocluster = activations_flattened

  if FLAGS.parallelize!=0:
    from multiprocessing import Pool
    nprocs = os.cpu_count() if FLAGS.parallelize==-1 else FLAGS.parallelize
    with Pool(min(nprocs,nlayers)) as p:
      fits, activations_clustered = zip(*p.starmap(do_cluster,
                                                   zip(repeat(activations_tocluster),
                                                       range(len(activations_tocluster)),
                                                       repeat(layers),
                                                       repeat(FLAGS.parameters))))
  else:
    fits = [None]*nlayers
    activations_clustered = [None]*nlayers
    for ilayer in layers:
      print("reducing dimensionality of layer "+str(ilayer)+" with t-SNE...")
      fits[ilayer], activations_clustered[ilayer] = do_cluster(activations_tocluster, \
                                                               ilayer, \
                                                               layers, \
                                                               FLAGS.parameters)

  np.savez(os.path.join(FLAGS.data_dir, 'cluster'), \
           sounds = sounds,
           activations_clustered = np.array(activations_clustered, dtype=object),
           fits_pca = fits_pca if FLAGS.save_fits else None,
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
      default='{"perplexity": 30, "exaggeration": 12.0}')
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
