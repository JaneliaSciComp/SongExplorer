#!/bin/bash

# generate per-class probabilities

# classify.sh <config-file> <context_ms> <shiftby_ms> <representation> <stride_ms> <logdir> <model> <check-point> <wavfile>

# e.g.
# $DEEPSONG_BIN classify.sh `pwd`/configuration.sh 204.8 0.0 waveform 1.6 `pwd`/trained-classifier train_1 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav

config_file=$1
context_ms=$2
shiftby_ms=$3
representation=$4
stride_ms=$5
logdir=$6
model=$7
check_point=$8
wavfile=$9

source $config_file
if [ "$representation" == "waveform" ] ; then
  stride_ms=`dc -e "16 k 1000 $audio_tic_rate / p"`
fi
clip_duration=$(dc -e "3 k $context_ms $stride_ms $nstrides 1 - * + p")
clip_stride=$(dc -e "3 k $stride_ms $nstrides * p")
frozenlog=$logdir/$model/frozen-graph.ckpt-${check_point}.log
ndownsample2=`grep -e 'strides = \[1, 2' -e 'strides = 2' $frozenlog | wc -l`
if (( "$ndownsample2" > 0 )) ; then
  stride_ms=`dc -e "$stride_ms 2 $ndownsample2 ^ * p"`
  clip_stride=`dc -e "$clip_stride $stride_ms + p"`
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_str=$logdir/$model/frozen-graph.ckpt-${check_point}.pb
expr_tf="/usr/bin/python3 $DIR/speech_commands_custom/test_streaming_accuracy.py \
         --model=$model_str \
         --labels=$logdir/$model/vgg_labels.txt \
         --wav=$wavfile \
         --ground_truth=/opt/deepsong/src/streaming_test_labels.txt \
         --verbose \
         --clip_duration_ms=$clip_duration \
         --clip_stride_ms=$clip_stride \
         --output_name=output_layer:0"

wavdir=`dirname $wavfile`
wavbase=`basename $wavfile`
tffile=$wavdir/${wavbase%.*}.tf
logfile=$wavdir/${wavbase%.*}-classify

cmd_tf="date; \
        hostname; \
        echo model=$model_str; \
        unset JAVA_HOME; \
        unset TF_CPP_MIN_LOG_LEVEL; \
        ulimit -c 0; \
        $expr_tf &> $tffile; \
        sync; \
        date"
echo $cmd_tf

expr_wav="$DIR/probabilities.py $logdir $model $tffile $context_ms $shiftby_ms $stride_ms"

cmd_wav="date; hostname; $expr_wav; sync; date"
echo $cmd_wav

classify_it "$cmd_tf" "$cmd_wav" "$logfile"
