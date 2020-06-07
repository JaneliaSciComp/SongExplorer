#!/bin/bash

# reduce dimensionality of internal activation states
 
# cluster.sh <groundtruth-directory> <which-layers-to-cluster> <pca-fraction-variance-to-retain> <pca-batch-size> <pca|tsne|umap> <N-dims> <parallelize> [ <tsne-perplexity> <tsne-exaggeration> | <umap-n-neighbors> <umap-min-distance> ]

# e.g.
# $DEEPSONG_BIN cluster.sh `pwd`/groundtruth-data 3 0.99 0 tsne 2 1 30 12.0
# $DEEPSONG_BIN cluster.sh `pwd`/groundtruth-data 0,1,2,3,4 1 0 umap 3 0 10 0.1

groundtruth_directory=$1
these_layers=$2
pca_fraction_variance_to_retain=$3
pca_batch_size=$4
cluster_algorithm=$5
cluster_ndims=${6:0:1}
cluster_parallelize=$7

shift 7

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cluster_algorithm="$(echo $cluster_algorithm | tr '[A-Z]' '[a-z]')"
varargs=("$@")

expr="$DIR/cluster.py \"$groundtruth_directory\" $these_layers $pca_fraction_variance_to_retain $pca_batch_size $cluster_algorithm $cluster_ndims $cluster_parallelize ${varargs[@]}"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
