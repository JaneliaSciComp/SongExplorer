#!/usr/bin/python3

# reduce dimensionality of internal activation states

# cluster.py <groundtruth-directory> <which-layers-to-cluster> <pca-fraction-variance-to-retain> <pca-batch_size> <cluster-algorithm> <cluster-num-dimensions> <cluster-parallelize> [ <tsne-perplexity> <tsne-exaggeration> | <umap-n-neighbors> <umap-min-distance> ]

# e.g.
# cluster.py `pwd`/groundtruth-data 3 0.99 5 tsne 2 1 30 12.0
# cluster.py `pwd`/groundtruth-data 0,1,2,3,4 1 5 umap 3 0 10 0.1

import os
import numpy as np
import sys
from sys import argv
from sklearn.decomposition import PCA, IncrementalPCA
from umap import UMAP
from sklearn.manifold import TSNE
from natsort import natsorted

_, groundtruth_directory, these_layers, pca_fraction_variance_to_retain, pca_batch_size, cluster_algorithm, cluster_ndims, cluster_parallelize = argv[:8]
print('groundtruth_directory: '+groundtruth_directory)
print('these_layers: '+these_layers)
print('pca_fraction_variance_to_retain: '+pca_fraction_variance_to_retain)
print('pca_batch_size: '+pca_batch_size)
print('cluster_algorithm: '+cluster_algorithm)
print('cluster_ndims: '+cluster_ndims)
print('cluster_parallelize: '+cluster_parallelize)
these_layers = [int(x) for x in these_layers.split(',')]
assert len(these_layers)>0
pca_fraction_variance_to_retain = float(pca_fraction_variance_to_retain)
pca_batch_size = int(pca_batch_size)
cluster_algorithm = cluster_algorithm.lower()
cluster_ndims = int(cluster_ndims)
cluster_parallelize = bool(int(cluster_parallelize))
if cluster_algorithm=="pca":
  None
elif cluster_algorithm=="tsne":
  tsne_perplexity, tsne_exaggeration = argv[8:]
  print('tsne_perplexity: '+tsne_perplexity)
  print('tsne_exaggeration: '+tsne_exaggeration)
  tsne_perplexity = int(tsne_perplexity)
  tsne_exaggeration = float(tsne_exaggeration)
elif cluster_algorithm=="umap":
  umap_n_neighbors, umap_min_distance = argv[8:]
  print('umap_n_neighbors: '+umap_n_neighbors)
  print('umap_min_distance: '+umap_min_distance)
  umap_n_neighbors = int(umap_n_neighbors)
  umap_min_distance = float(umap_min_distance)
else:
  print('cluster_algorithm must be one of pca, tsne or umap')
  exit()


print("loading data...")
activations=[]
npzfile = np.load(os.path.join(groundtruth_directory, 'activations.npz'),
                  allow_pickle=True)
samples = npzfile['samples']
for arr_ in natsorted(filter(lambda x: x.startswith('arr_'), npzfile.files)):
  activations.append(npzfile[arr_])

nlayers = len(activations)

kinds = set([x['kind'] for x in samples])
labels = set([x['label'] for x in samples])
print('word counts')
for kind in kinds:
  print(kind)
  for label in labels:
    count = sum([label==x['label'] and kind==x['kind'] for x in samples])
    print(count,label)


activations_flattened = [None]*nlayers
for ilayer in range(nlayers):
  if ilayer not in these_layers:
    continue
  nsamples = np.shape(activations[ilayer])[0]
  activations_flattened[ilayer] = np.reshape(activations[ilayer],(nsamples,-1))
  print(np.shape(activations_flattened[ilayer]))


fits_pca = [None]*nlayers
if pca_fraction_variance_to_retain<1 or cluster_algorithm=="pca":
  print("reducing dimensionality with PCA...")

  activations_scaled = [None]*nlayers
  for ilayer in range(nlayers):
    if ilayer not in these_layers:
      continue
    mu = np.mean(activations_flattened[ilayer], axis=0)
    sigma = np.std(activations_flattened[ilayer], axis=0)
    activations_scaled[ilayer] = (activations_flattened[ilayer]-mu)/sigma
    if pca_batch_size==0:
      pca = PCA()
    else:
      nfeatures = np.shape(activations_scaled[ilayer])[1]
      pca = IncrementalPCA(batch_size=pca_batch_size*nfeatures)
    fits_pca[ilayer] =  pca.fit(activations_scaled[ilayer])
    print(np.shape(fits_pca[ilayer]))

  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  #plt.ion()

  activations_kept = [None]*nlayers
  fig = plt.figure()
  ax = fig.add_subplot(111)
  for ilayer in range(nlayers):
    if ilayer not in these_layers:
      continue
    cumsum = np.cumsum(fits_pca[ilayer].explained_variance_ratio_)
    ncomponents = np.where(cumsum>=pca_fraction_variance_to_retain)[0][0]
    line, = ax.plot(cumsum)
    activations_transformed = fits_pca[ilayer].transform(activations_scaled[ilayer])
    if cluster_algorithm=="pca":
      line.set_label('layer '+str(ilayer)+', n='+str(np.shape(activations_transformed)[1]))
      activations_kept[ilayer] = activations_transformed
    else:
      line.set_label('layer '+str(ilayer)+', n='+str(ncomponents+1))
      activations_kept[ilayer] = activations_transformed[:,0:ncomponents+1]

  ax.set_ylabel('cumsum explained variance')
  ax.set_xlabel('# of components')
  ax.legend(loc='lower right')
  plt.savefig(os.path.join(groundtruth_directory, 'cluster-pca.pdf'))

  activations_tocluster = activations_kept
else:
  activations_tocluster = activations_flattened


if cluster_algorithm=="pca":
  def do_cluster(ilayer):
    return None, activations_tocluster[ilayer][:,:cluster_ndims]


elif cluster_algorithm=="tsne":
  print("reducing dimensionality with t-SNE...")

  def do_cluster(ilayer):
    if ilayer in these_layers:
      return None, TSNE(n_components=cluster_ndims, verbose=3, \
                        perplexity=tsne_perplexity, \
                        early_exaggeration=tsne_exaggeration \
                       ).fit_transform(activations_tocluster[ilayer])
    else:
      return None, None


elif cluster_algorithm=="umap":
  print("reducing dimensionality with UMAP...")

  def do_cluster(ilayer):
    if ilayer in these_layers:
      fit = UMAP(n_components=cluster_ndims, verbose=3, \
                 n_neighbors=umap_n_neighbors, \
                 min_dist=umap_min_distance \
                ).fit(activations_tocluster[ilayer])
      return fit, fit.transform(activations_tocluster[ilayer])
    else:
      return None, None


if cluster_parallelize:
  from multiprocessing import Pool
  with Pool(nlayers) as p:
    result = p.map(do_cluster, range(len(activations_tocluster)))
    fits, activations_clustered = zip(*result)
else:
  fits = [None]*nlayers
  activations_clustered = [None]*nlayers
  for ilayer in range(nlayers):
    if ilayer not in these_layers:
      continue
    print('layer '+str(ilayer))
    fit, activation_clustered = do_cluster(ilayer)
    fits[ilayer] = fit
    activations_clustered[ilayer] = activation_clustered

np.savez(os.path.join(groundtruth_directory, 'cluster'), \
         samples=samples,
         activations_clustered=activations_clustered,
         fits_pca=fits_pca,
         fits=fits)
