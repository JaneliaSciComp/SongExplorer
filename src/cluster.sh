#!/bin/bash

# reduce dimensionality of internal activation states
 
# cluster.sh <config-file> <groundtruth-directory> <cluster-equalize-ratio> <cluster-max-samples> <pca-fraction-variance-to-retain> <tsne-perplexity> <tsne-exaggeration>

# e.g.
# deepsong cluster.sh `pwd`/configuration.sh `pwd`/groundtruth-data

config_file=$1
groundtruth_directory=$2
cluster_equalize_ratio=$3
cluster_max_samples=$4
pca_fraction_variance_to_retain=$5
tsne_perplexity=$6
tsne_exaggeration=$7

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/cluster.py \"$groundtruth_directory\" $cluster_equalize_ratio $cluster_max_samples $pca_fraction_variance_to_retain $tsne_perplexity $tsne_exaggeration $cluster_parallelize"

cmd="date; \
     hostname; \
     $expr; \
     date"
echo $cmd

logfile=${groundtruth_directory}/cluster.log
jobname=cluster

cluster_it "$cmd" "$logfile" "$jobname"
