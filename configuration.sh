# on what computer to do the computation
default_where=local

# where DeepSong saves it's state and logs
export DEEPSONG_STATE=.

# define DEEPSONG_CPU_BIN to batch certain jobs to a computer without a GPU
[ -z "$DEEPSONG_CPU_BIN" ] && DEEPSONG_CPU_BIN=$DEEPSONG_BIN

# sampling rate of the recordings in the WAV files
# DeepSong must be restarted for changes to take effect
tic_rate=2500


# GENERIC HOOK
generic_it () {  # 1=cmd, 2=logfile, 3=jobname, 4=where, 5=deepsongbin, 6=bsubflags
    if [ "$4" == "local" ] ; then
        bash -c "$1" &> $2  #&
    elif [ "$4" == "server" ] ; then
        ssh c03u14 "$5 bash -c \"$1\" &> $2" #&
    elif [ "$4" == "cluster" ] ; then
        ssh login1 bsub \
                -P stern \
                -J $3 \
                "$6" \
                -oo $2 <<<"$5 bash -c \"$1\""
    fi
}


# GUI

# DeepSong must be restarted for changes to take effect

gui_snippet_ms=40
gui_nx_snippets=10
gui_ny_snippets=10
gui_nlabels=8
gui_gui_width_pix=1200
gui_context_width_ms=400
gui_context_offset_ms=0
gui_where=local
gui_it () {
    generic_it "$1" "$2" "$3" "$gui_where" "$DEEPSONG_CPU_BIN" "-W 1440"
}


# DETECT

detect_where=$default_where
detect_it () {
    generic_it "$1" "$2" "$3" "$detect_where" "$DEEPSONG_CPU_BIN" "-n 2 -W 60"
}


# MISSES

misses_where=$default_where
misses_it () {
    generic_it "$1" "$2" "$3" "$misses_where" "$DEEPSONG_CPU_BIN" "-W 60"
}


# TRAIN

train_gpu=1
train_where=$default_where
train_it () {
    if [ "$train_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$3" "$train_where" "$DEEPSONG_BIN" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx" #&
    else
        generic_it "$1" "$2" "$3" "$train_where" "$DEEPSONG_CPU_BIN" "-n 24 -W 1440" #&
    fi
}


# if you have multiple GPU cards in your computer, set CUDA_VISIBLE_DEVICES to $4,
# instead of the default 0, in `generalize_it()` and `xvalidate_it()`, and be careful
# not to run more jobs at once than you have cards.

# used by `generalize` and `xvalidate` to simultaneously train multiple models on a GPU
models_per_job=1


# GENERALIZE

generalize_gpu=1
generalize_where=$default_where
generalize_it () {
    if [ "$generalize_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$3" "$generalize_where" "CUDA_VISIBLE_DEVICES=0 $DEEPSONG_BIN" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx" #&
    else
        generic_it "$1" "$2" "$3" "$generalize_where" "$DEEPSONG_CPU_BIN" "-n 24 -W 1440" #&
    fi
}


# XVALIDATE

xvalidate_gpu=1
xvalidate_where=$default_where
xvalidate_it () {
    if [ "$xvalidate_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$3" "$xvalidate_where" "CUDA_VISIBLE_DEVICES=0 $DEEPSONG_BIN" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx" #&
    else
        generic_it "$1" "$2" "$3" "$xvalidate_where" "$DEEPSONG_CPU_BIN" "-n 24 -W 1440" #&
    fi
}


# HIDDEN

hidden_gpu=1
hidden_where=$default_where
hidden_it () {
    if [ "$hidden_gpu" -eq "1" ] ; then
        generic_it "$1" "$2" "$3" "$hidden_where" "$DEEPSONG_BIN" "-W 60 -gpu \"num=1\" -q gpu_short"
    else
        generic_it "$1" "$2" "$3" "$hidden_where" "$DEEPSONG_CPU_BIN" "-n 24 -W 60"
    fi
}


# CLUSTER

# simultaneosly clustering each hidden layer is fast but requires a lot of RAM
cluster_parallelize=1

cluster_where=$default_where
cluster_it () {
    generic_it "$1" "$2" "$3" "$cluster_where" "$DEEPSONG_CPU_BIN" "-n 8 -W 1440"
}


# ACCURACY

# how many points to use for the precision-recall and sensitivity-specificity curves
accuracy_nprobabilities=50

accuracy_where=$default_where
accuracy_it () {
    generic_it "$1" "$2" "$3" "$accuracy_where" "$DEEPSONG_CPU_BIN" "-W 60"
}


# used by freeze and classify to specify how many `window`s to process in parallel
nstrides=64


# FREEZE

freeze_where=$default_where
freeze_it () {
    generic_it "$1" "$2" "$3" "$freeze_where" "$DEEPSONG_CPU_BIN" "-W 60"
}


# CLASSIFY

classify_gpu=0
classify_where=$default_where
classify_it () {
    if [ "$classify_gpu" -eq "1" ] ; then
        generic_it "$1" "$3-tf.log" "$4-tf" "$classify_where" "$DEEPSONG_BIN" "-n 2 -W 60 -gpu \"num=1\" -q gpu_rtx"
    else
        generic_it "$1" "$3-tf.log" "$4-tf" "$classify_where" "$DEEPSONG_CPU_BIN" "-n 8 -W 60"
    fi
    wait
    generic_it "$2" "$3-wav.log" "$4-wav" "$classify_where" "$DEEPSONG_CPU_BIN" "-W 60 -w \"done($4-tf)\""
}


# ETHOGRAM

ethogram_where=$default_where
ethogram_it () {
    generic_it "$1" "$2" "$3" "$ethogram_where" "$DEEPSONG_CPU_BIN" "-n 1 -W 60"
}


# COMPARE

compare_where=$default_where
compare_it () {
    generic_it "$1" "$2" "$3" "$compare_where" "$DEEPSONG_CPU_BIN" "-n 1 -W 60"
}


# DENSE

dense_where=$default_where
dense_it () {
    generic_it "$1" "$2" "$3" "$dense_where" "$DEEPSONG_CPU_BIN" "-n 1 -W 120"
}
