# on what computer to do the computation
default_where=local

# where DeepSong saves it's state and logs
export DEEPSONG_STATE=.


# DeepSong must be restarted for changes in the following to take effect

# sampling rate of and number of channels in the WAV files
audio_tic_rate=2500
audio_nchannels=1

# specs of the 'local' computer
local_ncpu_cores=12
local_ngpu_cards=1
local_ngigabytes_memory=32

# specs of the 'server' computer
server_ipaddr=
server_ncpu_cores=
server_ngpu_cards=
server_ngigabytes_memory=
server_export=  #export\ SINGULARITYENV_PREPEND_PATH=$PWD/deepsong/src\;

# GUI
gui_cluster_background_color='#FFFFFF'  #'#440154'
#gui_cluster_dot_colormap='Category10_10'
# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
gui_cluster_dot_colormap='("#f0a3ff","#0075dc","#993f00","#4c005c","#191919","#005c31","#2bce48","#ffcc99","#808080","#94ffb5","#8f7c00","#9dcc00","#c20088","#003380","#ffa405","#ffa8bb","#426600","#ff0010","#5ef1f2","#00998f","#e0ff66","#740aff","#990000","#ffff80","#ffff00","#ff5005")'
gui_snippet_colormap='Viridis256'
gui_snippet_ms=40
gui_nx_snippets=10
gui_ny_snippets=10
gui_nlabels=8
gui_gui_width_pix=1250
gui_context_width_ms=400
gui_context_offset_ms=0

# DeepSong must be restarted for changes in the above to take effect


# GENERIC HOOK
generic_it () {
    cmd=$1
    logfile=$2
    where=$3
    localargs=$4
    localdeps=$5
    clusterflags=$6
    if [ "$where" == "local" ] ; then
        hetero submit "{ export CUDA_VISIBLE_DEVICES=\$QUEUE1; $cmd; } &> $logfile" \
                      $localargs "$localdeps" >${logfile}.job
    elif [ "$where" == "server" ] ; then
        # bug here for classify_it because of the dependency :(
        ssh $server_ipaddr "$server_export $DEEPSONG_BIN hetero submit \"{ \
                            export CUDA_VISIBLE_DEVICES=\\\$QUEUE1; $cmd; } &> $logfile\" \
                            $localargs \"$localdeps\" >${logfile}.job"
    elif [ "$where" == "cluster" ] ; then
        ssh login1 bsub \
                   -Ne \
                   -P stern \
                   -J ${logfile//,/}.job \
                   "$clusterflags" \
                   -oo $logfile <<<"$DEEPSONG_BIN bash -c \"$cmd\""
    fi
}


# DETECT

detect_where=$default_where
detect_it () {
    generic_it "$1" "$2" "$detect_where" "1 0 32" "" "-n 2 -W 60"
}


# MISSES

misses_where=$default_where
misses_it () {
    generic_it "$1" "$2" "$misses_where" "1 0 1" "" "-W 60"
}


# TRAIN

train_gpu=1
train_where=$default_where
train_it () {
    if [ "$train_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$train_where" \
                   "2 1 1" "" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx"
    else
        generic_it "$1" "$2" "$train_where" \
                   "12 0 1" "" "-n 12 -W 1440"
    fi
}


# used by `generalize` and `xvalidate` to simultaneously train multiple models on one GPU
models_per_job=1


# GENERALIZE

generalize_gpu=1
generalize_where=$default_where
generalize_it () {
    if [ "$generalize_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$generalize_where" \
                   "2 1 1" "" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx"
    else
        generic_it "$1" "$2" "$generalize_where" \
                   "24 0 1" "" "-n 24 -W 1440"
    fi
}


# XVALIDATE

xvalidate_gpu=1
xvalidate_where=$default_where
xvalidate_it () {
    if [ "$xvalidate_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$xvalidate_where" \
                   "2 1 1" "" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx"
    else
        generic_it "$1" "$2" "$xvalidate_where" \
                   "24 0 1" "" "-n 24 -W 1440"
    fi
}


# MISTAKES

mistakes_where=$default_where
mistakes_it () {
    generic_it "$1" "$2" "$mistakes_where" "1 0 1" "" "-n 1 -W 1440"
}


# ACTIVATIONS

activations_gpu=0
activations_where=$default_where
activations_it () {
    if [ "$activations_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$activations_where" \
                   "1 1 1" "" "-W 60 -gpu \"num=1\" -q gpu_short"
    else
        generic_it "$1" "$2" "$activations_where" \
                   "12 0 1" "" "-n 24 -W 60"
    fi
}


# CLUSTER

# use incremental PCA to conserve memory if >0
pca_batch_size=0 
# simultaneosly clustering each layer is fast but requires more of RAM
cluster_parallelize=1

cluster_where=$default_where
cluster_it () {
    generic_it "$1" "$2" "$cluster_where" "8 0 1" "" "-n 8 -W 1440"
}


# ACCURACY

# how many points to use for the precision-recall and sensitivity-specificity curves
accuracy_nprobabilities=50
accuracy_parallelize=1

accuracy_where=$default_where
accuracy_it () {
    generic_it "$1" "$2" "$accuracy_where" "3 0 0" "" "-W 60"
}


# used by freeze and classify to specify how many `window`s to process in parallel
# must be an integer multiple of the effective downsampling achieved by `stride after`
nstrides=64


# FREEZE

freeze_where=$default_where
freeze_it () {
    generic_it "$1" "$2" "$freeze_where" "1 0 0" "" "-W 60"
}


# CLASSIFY

classify_gpu=0
classify_where=$default_where
classify_it () {
    currtime=`date +%s`
    if [ "$classify_gpu" -eq "1" ] ; then
        generic_it "$1" "$3-tf.log" "$classify_where" \
                   "2 1 1" "" "-n 2 -W 60 -gpu \"num=1\" -q gpu_rtx"
    else
        generic_it "$1" "$3-tf.log" "$classify_where" \
                   "12 0 1" "" "-n 8 -W 60"
    fi
    generic_it "$2" "$3-wav.log" "$classify_where" \
               "1 0 1" \
               "test ! -e $3-tf.log || \
                   ! tmp=\$(date +%s --date=\"\`tail -1 $3-tf.log\`\" 2>/dev/null) || \
                   (( \$tmp - $currtime < 0 ))" \
               "-W 60 -w \"done(${3//,/}-tf.log.job)\""
}


# ETHOGRAM

ethogram_where=$default_where
ethogram_it () {
    generic_it "$1" "$2" "$ethogram_where" "1 0 1" "" "-W 60"
}


# COMPARE

compare_where=$default_where
compare_it () {
    generic_it "$1" "$2" "$compare_where" "1 0 1" "" "-W 60"
}


# CONGRUENCE

congruence_parallelize=1
congruence_where=$default_where
congruence_it () {
    generic_it "$1" "$2" "$congruence_where" "12 0 1" "" "-n 16 -W 120"
}
