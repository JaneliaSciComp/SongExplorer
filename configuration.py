# this file must be valid python code

# SongExplorer must be restarted for changes to take effect

# where to save the GUI text boxes
state_dir=""

# sampling rate of and number of channels in the WAV files
audio_tic_rate=2500
audio_nchannels=1
audio_read_plugin="load-wav"
audio_read_plugin_kwargs={}

# sampling rate and frame size of the AVI, MP4, or MOV files (if any)
video_frame_rate=0
video_frame_width=0
video_frame_height=0
video_channels=0  # comma-separated list of which colors to use
video_bkg_frames=0  # how many frames to use when computing median background image to subtract
video_findfile_plugin="same-basename"  # given a directory and a WAV file, return the corresponding video file
video_read_plugin="template"
video_read_plugin_kwargs={}

# URL of the 'server' computer
server_username=""
server_ipaddr=""

# how to dispatch jobs to the 'cluster'
cluster_username=""
cluster_ipaddr=""
cluster_cmd=""
cluster_logfile_flag=""

# GUI
gui_nlabels=7
gui_gui_width_pix=1350
#gui_label_palette="Category10_10"
# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
gui_label_palette="('#0075dc','#993f00','#4c005c','#191919','#005c31','#2bce48','#ffcc99','#808080','#94ffb5','#8f7c00','#9dcc00','#c20088','#003380','#ffa405','#ffa8bb','#426600','#ff0010','#5ef1f2','#00998f','#e0ff66','#740aff','#990000','#ffff80','#ffff00','#ff5005')"
gui_cluster_circle_color="#f0a3ff"
gui_time_units="ms"  # units and scales for model parameters, e.g. context, width, stride
gui_time_scale=0.001
gui_freq_units="Hz"
gui_freq_scale=1
gui_snippets_colormap="Viridis256"
gui_snippets_width_sec=0.04
gui_snippets_nx=10
gui_snippets_ny=5
gui_snippets_waveform=1   # comma-separated list of channels to display, or () if none
gui_snippets_spectrogram=1   # comma-separated list of channels to display, or () if none
gui_context_time_units="sec"
gui_context_time_scale=1
gui_context_freq_units="kHz"
gui_context_freq_scale=1000
gui_context_width_sec=0.4
gui_context_offset_sec=0
gui_context_waveform=1   # comma-separated list of channels to display, or () if none
gui_context_waveform_height_pix=150
gui_context_spectrogram=1   # comma-separated list of channels to display, or () if none
gui_context_spectrogram_height_pix=150
gui_context_probability_height_pix=75
gui_context_undo_proximity_pix=3
gui_context_doubleclick_plugin="point"  # or snap-to
gui_spectrogram_colormap="Viridis256"
gui_spectrogram_window="hann"
gui_spectrogram_length_sec=0.010
gui_spectrogram_overlap=0.5
gui_spectrogram_low_hz=0
gui_spectrogram_high_hz=1250
gui_spectrogram_clip=[1,99]
gui_probability_style="lines" # either "lines" or "bars"

# neural network architecture to use
architecture_plugin="convolutional"
overlapped_prefix="not_"
augmentation_plugin="volume-noise-dc-reverse-invert"

# on what computer to do the computation
default_where="local"

# action buttons

detect_where=default_where
detect_ncpu_cores=-1
detect_ngpu_cards=-1
detect_ngigabytes_memory=-1
detect_cluster_flags=""
detect_plugin="time-freq-threshold"

misses_where=default_where
misses_ncpu_cores=-1
misses_ngpu_cards=-1
misses_ngigabytes_memory=-1
misses_cluster_flags=""

train_where=default_where
train_ncpu_cores=-1
train_ngpu_cards=-1
train_ngigabytes_memory=-1
train_cluster_flags=""

generalize_where=default_where
generalize_ncpu_cores=-1
generalize_ngpu_cards=-1
generalize_ngigabytes_memory=-1
generalize_cluster_flags=""

xvalidate_where=default_where
xvalidate_ncpu_cores=-1
xvalidate_ngpu_cards=-1
xvalidate_ngigabytes_memory=-1
xvalidate_cluster_flags=""

mistakes_where=default_where
mistakes_ncpu_cores=-1
mistakes_ngpu_cards=-1
mistakes_ngigabytes_memory=-1
mistakes_cluster_flags=""

activations_where=default_where
activations_ncpu_cores=-1
activations_ngpu_cards=-1
activations_ngigabytes_memory=-1
activations_cluster_flags=""

cluster_where=default_where
cluster_ncpu_cores=-1
cluster_ngpu_cards=-1
cluster_ngigabytes_memory=-1
cluster_cluster_flags=""
cluster_plugin="UMAP"  # or tSNE, PCA

accuracy_where=default_where
accuracy_ncpu_cores=-1
accuracy_ngpu_cards=-1
accuracy_ngigabytes_memory=-1
accuracy_cluster_flags=""

delete_ckpts_where=default_where
delete_ckpts_ncpu_cores=-1
delete_ckpts_ngpu_cards=-1
delete_ckpts_ngigabytes_memory=-1
delete_ckpts_cluster_flags=""

freeze_where=default_where
freeze_ncpu_cores=-1
freeze_ngpu_cards=-1
freeze_ngigabytes_memory=-1
freeze_cluster_flags=""

ensemble_where=default_where
ensemble_ncpu_cores=-1
ensemble_ngpu_cards=-1
ensemble_ngigabytes_memory=-1
ensemble_cluster_flags=""

classify_where=default_where
classify_ncpu_cores=-1
classify_ngpu_cards=-1
classify_ngigabytes_memory=-1
classify_cluster_flags=""

ethogram_where=default_where
ethogram_ncpu_cores=-1
ethogram_ngpu_cards=-1
ethogram_ngigabytes_memory=-1
ethogram_cluster_flags=""

compare_where=default_where
compare_ncpu_cores=-1
compare_ngpu_cards=-1
compare_ngigabytes_memory=-1
compare_cluster_flags=""

congruence_where=default_where
congruence_ncpu_cores=-1
congruence_ngpu_cards=-1
congruence_ngigabytes_memory=-1
congruence_cluster_flags=""

# simultaneously train multiple models on one GPU
models_per_job=1

# use incremental PCA to conserve memory when clustering if >0
pca_batch_size=0 

# parallelizing is faster but requires more RAM.
# use -1 to automatically use all CPU cores, 0 to not parallelize at all, 
# or a positive integer to manually specify how many CPU cores to use.
cluster_parallelize=-1
accuracy_parallelize=-1
congruence_parallelize=-1

# how many points to use for the precision-recall, sensitivity-specificity, and congruence curves
nprobabilities=20

# used by train, generalize, xvalidate, and activations
data_loader_maxprocs=0  # 0 = num CPU cores
data_loader_queuesize=1  # 0 = infinite


# what follows is for developers only

# the location of writable code files
source_path=""

# a boolean to make the output (more) reproducible
deterministic=0
