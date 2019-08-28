#!/usr/bin/python3

# reduce dimensionality of internal activation states

# cluster.py <groundtruth-directory> <cluster-equalize-ratio> <cluster-max-samples> <pca-fraction-variance-to-retain> <tsne-perplexity> <tsne-exaggeration> <cluster-parallelize>

# e.g.
# cluster.py `pwd`/groundtruth-data 0.99 30 12.0 16 0

import os
import numpy as np
import sys
from sys import argv
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ion()
from scipy import interpolate
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

_, groundtruth_directory, cluster_equalize_ratio, cluster_max_samples, pca_fraction_variance_to_retain, tsne_perplexity, tsne_exaggeration, cluster_parallelize = argv
print('groundtruth_directory: '+groundtruth_directory)
print('cluster_equalize_ratio: '+cluster_equalize_ratio)
print('cluster_max_samples: '+cluster_max_samples)
print('pca_fraction_variance_to_retain: '+pca_fraction_variance_to_retain)
print('tsne_perplexity: '+tsne_perplexity)
print('tsne_exaggeration: '+tsne_exaggeration)
print('cluster_parallelize: '+cluster_parallelize)

pca_fraction_variance_to_retain = float(pca_fraction_variance_to_retain)
cluster_equalize_ratio = int(cluster_equalize_ratio)
cluster_max_samples = int(cluster_max_samples)
tsne_perplexity = int(tsne_perplexity)
tsne_exaggeration = float(tsne_exaggeration)
cluster_parallelize = bool(int(cluster_parallelize))


### load features from hidden.sh
print("loading data...")
hidden=[]
npzfile = np.load(os.path.join(groundtruth_directory, 'hidden.npz'))
samples = npzfile['samples']
for arr_ in natsorted(filter(lambda x: x.startswith('arr_'), npzfile.files)):
  hidden.append(npzfile[arr_])

nlayers = len(hidden)


### approximately equalize word counts
labels = set([x['label'] for x in samples])
kinds = set([x['kind'] for x in samples])
print('unequalized word counts')
smallest=np.iinfo(np.uint64).max
for kind in kinds:
  print(kind)
  for label in labels:
    count = sum([label==x['label'] and kind==x['kind'] for x in samples])
    print(count,label)
    if count>0:
      smallest = np.minimum(smallest, count).astype(np.int)

bkeep = np.ones((len(samples),), dtype=bool)
for kind in kinds:
  for label in labels:
    blabel = [label==x['label'] and kind==x['kind'] for x in samples]
    nomit = sum(blabel) - smallest*cluster_equalize_ratio
    if nomit>0:
      iomit = np.random.choice(sum(blabel), nomit, replace=False)
      bkeep[np.where(blabel)[0][iomit]]=False

equalized_samples = [samples[i] for (i,b) in enumerate(bkeep) if b]
equalized_hidden=[]
for ilayer in range(nlayers):
  equalized_hidden.append([hidden[ilayer][i] for (i,b) in enumerate(bkeep) if b])

print('equalized word counts')
for kind in kinds:
  print(kind)
  for label in labels:
    count = sum([label==x['label'] and kind==x['kind'] for x in equalized_samples])
    print(count,label)


### subsample
if len(equalized_samples) > cluster_max_samples:
    isubsample = np.random.choice(len(equalized_samples), cluster_max_samples, replace=False)
else:
    isubsample = range(len(equalized_samples))
subsampled_samples = [equalized_samples[i] for i in isubsample]
subsampled_hidden=[]
for ilayer in range(nlayers):
  subsampled_hidden.append([equalized_hidden[ilayer][i] for i in isubsample])


### PCA
print("reducing dimensionality with PCA...")

hidden_pca=[]
hidden_scaled=[]
for ilayer in range(nlayers):
  nsamples = np.shape(subsampled_hidden[ilayer])[0]
  hidden_flattened = np.reshape(subsampled_hidden[ilayer],(nsamples,-1))
  mu = np.mean(hidden_flattened, axis=0)
  sigma = np.std(hidden_flattened, axis=0)
  hidden_scaled.append((hidden_flattened-mu)/sigma)
  print(np.shape(hidden_scaled[ilayer]))
  pca = PCA()
  hidden_pca.append(pca.fit(hidden_scaled[ilayer]))

ncomponents=[]
hidden_kept=[]
fig = plt.figure()
ax = fig.add_subplot(111)
for ilayer in range(nlayers):
  cumsum = np.cumsum(hidden_pca[ilayer].explained_variance_ratio_)
  ncomponents.append(np.where(cumsum>pca_fraction_variance_to_retain)[0][0])
  line, = ax.plot(cumsum)
  line.set_label('layer '+str(ilayer)+', n='+str(ncomponents[-1]))
  hidden_transformed = hidden_pca[ilayer].transform(hidden_scaled[ilayer])
  hidden_kept.append(hidden_transformed[:,0:ncomponents[ilayer]])

ax.set_ylabel('cumsum explained variance')
ax.set_xlabel('# of components')
ax.legend(loc='lower right')
plt.savefig(os.path.join(groundtruth_directory, 'cluster-pca-hidden.pdf'))
ax.set_xlim(0,300)
ax.set_ylim(0.8,1.01)
plt.savefig(os.path.join(groundtruth_directory, 'cluster-pca-hidden-zoom.pdf'))


### t-SNE
print("reducing dimensionality with t-SNE...")

def do_cluster(x):
  return TSNE(n_components=2, verbose=3, perplexity=tsne_perplexity, early_exaggeration=tsne_exaggeration).fit_transform(x)

if cluster_parallelize:
  from multiprocessing import Pool
  with Pool(nlayers) as p:
    hidden_clustered = p.map(do_cluster, hidden_kept)
else:
  hidden_clustered = []
  for i in range(len(hidden_kept)):
    print('layer '+str(i))
    hidden_clustered.append(do_cluster(hidden_kept[i]))

np.savez(os.path.join(groundtruth_directory, 'cluster'), samples=subsampled_samples, hidden_clustered=hidden_clustered, *hidden_kept)
