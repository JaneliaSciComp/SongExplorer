#!/bin/bash

# generate per-class probabilities

# classify.sh <config-file> <context_ms> <shiftby_ms> <stride_ms> <logdir> <model> <check-point> <wavfile>

# e.g.
# deepsong classify.sh `pwd`/configuration.sh 204.8 0.0 1.6 `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav

config_file=$1
context_ms=$2
shiftby_ms=$3
stride_ms=$4
logdir=$5
model=$6
check_point=$7
wavfile=$8

source $config_file
clip_duration=$(dc -e "3 k $context_ms $stride_ms $nstrides 1 - * + p")
clip_stride=$(dc -e "3 k $stride_ms $nstrides * p")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr_tf="/opt/bazel/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/examples/speech_commands_custom/test_streaming_accuracy \
         --graph=$logdir/train_$model/frozen-graph.ckpt-${check_point}.pb \
         --labels=$logdir/train_$model/vgg_labels.txt \
         --wav=$wavfile \
         --ground_truth=/opt/deepsong/src/streaming_test_labels.txt \
         --verbose \
         --clip_duration_ms=$clip_duration \
         --clip_stride_ms=$clip_stride \
         --output_names=output_layer"

wavdir=`dirname $wavfile`
wavbase=`basename $wavfile`
tffile=$wavdir/${wavbase%.wav}.tf
logfile=$wavdir/${wavbase%.wav}-classify
jobname=classify-${wavbase%.wav}

cmd_tf="date;
        hostname;
        unset JAVA_HOME;
        unset TF_CPP_MIN_LOG_LEVEL;
        ulimit -c 0;
        $expr_tf &> $tffile;
        date"
        #time $expr_tf &> /dev/null"
echo $cmd_tf

expr_wav="$DIR/probabilities.py $logdir $model $tffile $context_ms $shiftby_ms $stride_ms"

cmd_wav="date;
         hostname;
         $expr_wav;
         date"
echo $cmd_wav

classify_it "$cmd_tf" "$cmd_wav" "$logfile" "$jobname"
