import sys
import os
import yaml
import numpy as np
from datetime import datetime
import logging 
import csv
import re
import pandas as pd

bokehlog = logging.getLogger("songexplorer") 

import view as V

import importlib

def parse_model_file(modelstr):
    filepath, filename = os.path.split(modelstr)
    if 'ckpt-' not in filename:
        filepath, filename = os.path.split(filepath)
    logdir, modeldir = os.path.split(filepath)
    prefix = filename.split('.')[0]
    m=re.search('ckpt-([\d,]+)',filename)
    check_point = m.group(1)
    return logdir, modeldir, prefix, check_point

def save_state_callback():
    with open(statepath, 'w') as fid:
       yaml.dump({**{'logs_folder': V.logs_folder.value,
                     'model_file': V.model_file.value,
                     'wavcsv_files': V.wavcsv_files.value,
                     'groundtruth_folder': V.groundtruth_folder.value,
                     'validation_files': V.validation_files.value,
                     'test_files': V.test_files.value,
                     'labels_touse': V.labels_touse.value,
                     'kinds_touse': V.kinds_touse.value,
                     'prevalences': V.prevalences.value,
                     'circle_radius': V.circle_radius.value,
                     'dot_size': V.dot_size.value,
                     'dot_alpha': V.dot_alpha.value,
                     'nsteps': V.nsteps.value,
                     'restore_from': V.restore_from.value,
                     'save_and_validate_period': V.save_and_validate_period.value,
                     'validate_percentage': V.validate_percentage.value,
                     'mini_batch': V.mini_batch.value,
                     'kfold': V.kfold.value,
                     'activations_equalize_ratio': V.activations_equalize_ratio.value,
                     'activations_max_sounds': V.activations_max_sounds.value,
                     'pca_fraction_variance_to_retain': \
                             V.pca_fraction_variance_to_retain.value,
                     'tsne_perplexity': V.tsne_perplexity.value,
                     'tsne_exaggeration': V.tsne_exaggeration.value,
                     'umap_neighbors': V.umap_neighbors.value,
                     'umap_distance': V.umap_distance.value,
                     'cluster_algorithm': V.cluster_algorithm.value,
                     'cluster_these_layers': [x for x in V.cluster_these_layers.value],
                     'precision_recall_ratios': V.precision_recall_ratios.value,
                     'congruence_portion': V.congruence_portion.value,
                     'congruence_convolve': V.congruence_convolve.value,
                     'congruence_measure': V.congruence_measure.value,
                     'nreplicates': V.nreplicates.value,
                     'batch_seed': V.batch_seed.value,
                     'weights_seed': V.weights_seed.value,
                     'labels': str.join(',',[x.value for x in V.label_texts]),
                     'file_dialog_string': V.file_dialog_string.value,
                     'context_ms': V.context_ms.value,
                     'shiftby_ms': V.shiftby_ms.value,
                     'optimizer': V.optimizer.value,
                     'learning_rate': V.learning_rate.value},
                  **{k:v.value for k,v in V.detect_parameters.items()},
                  **{k:v.value for k,v in V.doubleclick_parameters.items()},
                  **{k:v.value for k,v in V.model_parameters.items()}},
                 fid)

def isannotated(sound):
    return np.where([x['file']==sound['file'] and x['ticks']==sound['ticks'] \
                     for x in annotated_sounds])[0]

def isclustered(sound):
    return np.where([x['file']==sound['file'] and x['ticks']==sound['ticks'] \
                     for x in clustered_sounds])[0]

def save_annotations():
    global nrecent_annotations
    if nrecent_annotations>0:
        fids = {}
        csvwriters = {}
        csvfiles_current = set([])
        for wavfile in set([x['file'] for x in annotated_sounds if x["label"]!=""]):
            csvfile = wavfile[:-4]+"-annotated-"+songexplorer_starttime+".csv"
            annotated_csvfiles_all.add(csvfile)
            csvfiles_current.add(csvfile)
            fids[wavfile] = open(os.path.join(V.groundtruth_folder.value, csvfile),
                                 "w", newline='')
            csvwriters[wavfile] = csv.writer(fids[wavfile])
        for filename in annotated_csvfiles_all - csvfiles_current:
            filepath = os.path.join(V.groundtruth_folder.value, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        corrected_sounds=[]
        for annotation in annotated_sounds:
            if annotation['label']!="" and not annotation['label'].isspace():
                csvwriters[annotation['file']].writerow(
                        [os.path.basename(annotation['file']),
                        annotation['ticks'][0], annotation['ticks'][1],
                        'annotated', annotation['label']])
            iclustered = isclustered(annotation)
            if len(iclustered)>0 and clustered_sounds[iclustered[0]]['kind']=='annotated':
                corrected_sounds.append(annotation)
        if corrected_sounds:
            df_corrected = pd.DataFrame([[os.path.basename(x['file']), x['ticks'][0], \
                                          x['ticks'][1], 'annotated', x['label']] \
                                         for x in corrected_sounds], \
                                        columns=['file','start','stop','kind','label'])
            for wavfile in set([x['file'] for x in corrected_sounds]):
                wavdir, wavbase = os.path.split(wavfile)
                wavpath = os.path.join(V.groundtruth_folder.value, wavdir)
                for csvbase in filter(lambda x: x.startswith(wavbase[:-4]) and
                                                x.endswith(".csv") and
                                                "-annotated-" in x and
                                                songexplorer_starttime not in x,
                                      os.listdir(wavpath)):
                    csvfile = os.path.join(wavpath, csvbase)
                    df_clustered = pd.read_csv(csvfile, \
                                               names=['file','start','stop','kind','label'], \
                                               header=None, index_col=False)
                    df_all = df_clustered.merge(df_corrected, \
                                                on=['file', 'start', 'stop', 'kind'], \
                                                how='left', indicator=True)
                    isetdiff = ~(df_all['_merge']=='both').values
                    if not all(isetdiff):
                        if any(isetdiff):
                            df_clustered.loc[isetdiff,:].to_csv(csvfile, header=False, \
                                                                index=False)
                        else:
                            os.remove(csvfile)
        for fid in fids.values():
            fid.close()
        nrecent_annotations=0
        if bokeh_document:
            bokeh_document.add_next_tick_callback(lambda n=nrecent_annotations:  V.save_update(n))

def add_annotation(sound, addto_history=True):
    global annotated_sounds, annotated_starts_sorted
    global annotated_stops, iannotated_stops_sorted
    global history_stack, history_idx
    iannotated = isannotated(sound)
    if len(iannotated)>0:
        del annotated_sounds[iannotated[0]]
    idx = np.searchsorted(annotated_starts_sorted, sound['ticks'][0])
    annotated_sounds.insert(idx, sound.copy())
    if addto_history:
        del history_stack[history_idx:]
        history_stack.append(['add',sound])
        history_idx+=1
    annotated_starts_sorted = [x['ticks'][0] for x in annotated_sounds]
    annotated_stops = [x['ticks'][1] for x in annotated_sounds]
    iannotated_stops_sorted = np.argsort(annotated_stops)
    if sound['label'] in state['labels']:
        thislabel = state['labels'].index(sound['label'])
        count = int(V.nsounds_per_label_buttons[thislabel].label)
        V.nsounds_per_label_buttons[thislabel].label = str(count+1)
    finalize_annotation(addto_history)
    return idx

def delete_annotation(isound, addto_history=True):
    global annotated_sounds, annotated_starts_sorted
    global annotated_stops, iannotated_stops_sorted
    global history_stack, history_idx
    if addto_history:
        del history_stack[history_idx:]
        history_stack.append(['delete',annotated_sounds[isound].copy()])
        history_idx+=1
    if annotated_sounds[isound]['label'] in state['labels']:
        thislabel = state['labels'].index(annotated_sounds[isound]['label'])
        count = int(V.nsounds_per_label_buttons[thislabel].label)
        V.nsounds_per_label_buttons[thislabel].label = str(count-1)
    iclustered = isclustered(annotated_sounds[isound])
    if len(iclustered)>0 and clustered_sounds[iclustered[0]]['kind']=='annotated':
        annotated_sounds[isound]['label'] = clustered_sounds[iclustered[0]]['label']
    else:
        del annotated_sounds[isound]
        annotated_starts_sorted = [x['ticks'][0] for x in annotated_sounds]
        annotated_stops = [x['ticks'][1] for x in annotated_sounds]
        iannotated_stops_sorted = np.argsort(annotated_stops)
    finalize_annotation(addto_history)

def finalize_annotation(redraw_snippets=True):
    global nrecent_annotations
    nrecent_annotations+=1
    V.save_update(nrecent_annotations)
    if history_idx<1:
        V.undo.disabled=True
    else:
        V.undo.disabled=False
    if history_idx==len(history_stack):
        V.redo.disabled=True
    else:
        V.redo.disabled=False
    if redraw_snippets:
        V.snippets_update(False)
    V.context_update()

def next_pow2_ms(x_ms):
    x = round(x_ms/1000*audio_tic_rate)
    if not (x & (x-1) == 0) or x == 0:
        next_higher = np.power(2, np.ceil(np.log2(x))).astype(np.int)
        return True, next_higher/audio_tic_rate*1000
    return False, x_ms

def init(_bokeh_document, _configuration_file):
    global bokeh_document, configuration_file
    global audio_tic_rate, audio_nchannels
    global nlabels, gui_width_pix
    global cluster_circle_color, cluster_dot_palette
    global snippets_colormap, snippets_width_ms, snippets_nx, snippets_ny, snippets_waveform, snippets_spectrogram
    global context_width_ms, context_offset_ms, context_waveform, context_waveform_height_pix, context_spectrogram, context_spectrogram_height_pix, context_spectrogram_units, context_probability_height_pix, context_undo_proximity_pix
    global context_waveform_low, context_waveform_high, context_spectrogram_freq_scale, cluster_dot_colors
    global spectrogram_colormap, spectrogram_window, spectrogram_length_ms, spectrogram_overlap, spectrogram_low_hz, spectrogram_high_hz
    global deterministic
    global context_width_ms0, context_offset_ms0
    global xcluster, ycluster, zcluster, ndcluster, tic2pix_max, snippet_width_pix, ilayer, ispecies, iword, inohyphen, ikind, nlayers, layers, species, words, nohyphens, kinds, clustered_labels, snippets_gap_ms, snippets_tic, snippets_gap_tic, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_decimate_by, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint_tic, ilabel, annotated_sounds, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, clustered_sounds, clustered_activations, clustered_recording2firstsound, clustered_starts_sorted, clustered_stops, iclustered_stops_sorted, songexplorer_starttime, history_stack, history_idx, wizard, action, function, statepath, state, file_dialog_root, file_dialog_filter, nearest_sounds, status_ticker_queue, waitfor_job
    global user_changed_recording 
    global detect_parameters, detect_labels, doubleclick_parameters, doubleclick_annotation, context_data, context_data_istart, model_parameters, video_findfile

    bokeh_document = _bokeh_document

    exec(open(_configuration_file).read(), globals())

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sys.path.append(os.path.dirname(detect_plugin))
    tmp = importlib.import_module(os.path.basename(detect_plugin))
    detect_parameters = tmp.detect_parameters
    detect_labels = tmp.detect_labels(int(audio_nchannels))

    sys.path.append(os.path.dirname(gui_context_doubleclick_plugin))
    tmp = importlib.import_module(os.path.basename(gui_context_doubleclick_plugin))
    doubleclick_parameters = tmp.doubleclick_parameters
    doubleclick_annotation = tmp.doubleclick_annotation

    sys.path.append(os.path.dirname(architecture_plugin))
    model = importlib.import_module(os.path.basename(architecture_plugin))
    model_parameters = model.model_parameters

    sys.path.append(os.path.dirname(video_findfile_plugin))
    video_findfile = importlib.import_module(os.path.basename(video_findfile_plugin)).video_findfile

    def is_local_server_or_cluster(varname, varvalue):
        if varvalue not in ['local', 'server', 'cluster']:
            bokehlog.info("WARNING: "+varname+" is '"+varvalue+"' but needs to be one of 'local', 'server', or 'cluster'")

    is_local_server_or_cluster("detect_where", detect_where)
    is_local_server_or_cluster("misses_where", misses_where)
    is_local_server_or_cluster("train_where", train_where)
    is_local_server_or_cluster("generalize_where", generalize_where)
    is_local_server_or_cluster("xvalidate_where", xvalidate_where)
    is_local_server_or_cluster("mistakes_where", mistakes_where)
    is_local_server_or_cluster("activations_where", activations_where)
    is_local_server_or_cluster("cluster_where", cluster_where)
    is_local_server_or_cluster("accuracy_where", accuracy_where)
    is_local_server_or_cluster("freeze_where", freeze_where)
    is_local_server_or_cluster("classify_where", classify_where)
    is_local_server_or_cluster("ethogram_where", ethogram_where)
    is_local_server_or_cluster("compare_where", compare_where)
    is_local_server_or_cluster("congruence_where", congruence_where)

    configuration_file = _configuration_file
    audio_tic_rate=int(audio_tic_rate)
    audio_nchannels=int(audio_nchannels)
    snippets_width_ms=float(gui_snippets_width_ms)
    snippets_nx=int(gui_snippets_nx)
    snippets_ny=int(gui_snippets_ny)

    snippets_waveform=gui_snippets_waveform if type(gui_snippets_waveform) is tuple \
                      else [gui_snippets_waveform]
    snippets_spectrogram=gui_snippets_spectrogram if type(gui_snippets_spectrogram) is tuple \
                         else [gui_snippets_spectrogram]
    if snippets_waveform and audio_nchannels < max(snippets_waveform):
        print("ERROR: max(snippets_waveform) exceeds audio_nchannels ")
        exit()
    if snippets_spectrogram and audio_nchannels < max(snippets_spectrogram):
        print("ERROR: max(snippets_waveform) exceeds audio_nchannels ")
        exit()

    nlabels=int(gui_nlabels)
    gui_width_pix=int(gui_gui_width_pix)
    context_width_ms0=float(gui_context_width_ms)
    context_offset_ms0=float(gui_context_offset_ms)
    context_width_ms=float(gui_context_width_ms)
    context_offset_ms=float(gui_context_offset_ms)

    context_waveform_height_pix=int(gui_context_waveform_height_pix)

    context_waveform=gui_context_waveform if type(gui_context_waveform) is tuple \
                      else [gui_context_waveform]
    context_spectrogram=gui_context_spectrogram if type(gui_context_spectrogram) is tuple \
                      else [gui_context_spectrogram]
    context_spectrogram_height_pix=int(gui_context_spectrogram_height_pix)
    if context_waveform and audio_nchannels < max(context_waveform):
        print("ERROR: max(snippets_waveform) exceeds audio_nchannels ")
        exit()
    if context_spectrogram and audio_nchannels < max(context_spectrogram):
        print("ERROR: max(snippets_waveform) exceeds audio_nchannels ")
        exit()

    context_spectrogram_units=gui_context_spectrogram_units
    spectrogram_colormap=gui_spectrogram_colormap
    spectrogram_window=gui_spectrogram_window
    spectrogram_length_ms=[next_pow2_ms(float(gui_spectrogram_length_ms))[1]]*audio_nchannels
    spectrogram_overlap=float(gui_spectrogram_overlap)
    tmp = max(0, min(audio_tic_rate/2, gui_spectrogram_low_hz))
    if tmp != gui_spectrogram_low_hz:
        print('WARNING: gui_spectrogram_low_hz should be between 0 and audio_tic_rate/2')
    spectrogram_low_hz=[float(tmp)]*audio_nchannels
    tmp = max(0, min(audio_tic_rate/2, gui_spectrogram_high_hz))
    if tmp != gui_spectrogram_high_hz:
        print('WARNING: gui_spectrogram_high_hz should be between 0 and audio_tic_rate/2')
    spectrogram_high_hz=[float(tmp)]*audio_nchannels
    context_spectrogram_freq_scale = 0.001 if context_spectrogram_units=='mHz' else \
                                     1 if context_spectrogram_units=='Hz' else \
                                  1000 if context_spectrogram_units=='kHz' else \
                               1000000

    context_probability_height_pix=int(gui_context_probability_height_pix)

    context_undo_proximity_pix=int(gui_context_undo_proximity_pix)

    context_waveform_low = [-1]*audio_nchannels
    context_waveform_high = [1]*audio_nchannels

    context_data = [None]*audio_nchannels
    context_data_istart = None

    cluster_dot_palette = gui_cluster_dot_palette
    snippets_colormap = gui_snippets_colormap
    cluster_circle_color = gui_cluster_circle_color
    cluster_dot_colors = {}

    user_changed_recording=True

    deterministic = str(deterministic)

    xcluster = ycluster = zcluster = np.nan
    ndcluster = 0

    tic2pix_max=4

    snippet_width_pix = gui_width_pix/2/snippets_nx

    ilayer=0
    ispecies=0
    iword=0
    inohyphen=0
    ikind=0

    nlayers = 0
    layers = []
    species = []
    words = []
    nohyphens = []
    kinds = []

    clustered_labels = []

    snippets_gap_ms=snippets_width_ms/10

    snippets_tic = int(np.rint(snippets_width_ms/1000*audio_tic_rate))
    snippets_gap_tic = int(np.rint(snippets_gap_ms/1000*audio_tic_rate))
    tic2pix = (snippets_gap_tic+snippets_tic) / snippet_width_pix
    snippets_decimate_by = round(tic2pix/tic2pix_max) if tic2pix>tic2pix_max else 1
    snippets_pix = round(snippets_tic / snippets_decimate_by)
    snippets_gap_pix = round(snippets_gap_tic / snippets_decimate_by)

    context_width_tic = int(np.rint(context_width_ms/1000*audio_tic_rate))
    context_offset_tic = int(np.rint(context_offset_ms/1000*audio_tic_rate))

    isnippet = -1
    xsnippet = -1
    ysnippet = -1

    file_nframes = -1
    context_midpoint_tic = -1
    context_decimate_by = -1

    ilabel=0

    annotated_sounds=[]
    annotated_starts_sorted=[]
    annotated_stops=[]
    iannotated_stops_sorted=[]
    annotated_csvfiles_all = set([])
    nrecent_annotations=0

    songexplorer_starttime = datetime.strftime(datetime.now(),'%Y%m%dT%H%M%S')

    history_stack=[]
    history_idx=0

    wizard=None
    action=None
    function=None

    statepath = os.path.join(state_dir, 'songexplorer.state.yml')

    if not os.path.exists(statepath):
        with open(statepath, 'w') as fid:
            yaml.dump({**{'logs_folder':'', \
                          'model_file':'', \
                          'wavcsv_files':'', \
                          'groundtruth_folder':'', \
                          'validation_files':'', \
                          'test_files':'', \
                          'labels_touse':'', \
                          'kinds_touse':'', \
                          'prevalences':'', \
                          'circle_radius':1, \
                          'dot_size':6, \
                          'dot_alpha':0.1, \
                          'nsteps':'0', \
                          'restore_from':'', \
                          'save_and_validate_period':'0', \
                          'validate_percentage':'0', \
                          'mini_batch':'32', \
                          'kfold':'4', \
                          'activations_equalize_ratio':'10', \
                          'activations_max_sounds':'1000', \
                          'pca_fraction_variance_to_retain':'1.0', \
                          'tsne_perplexity':'30', \
                          'tsne_exaggeration':'12.0', \
                          'umap_neighbors':'10', \
                          'umap_distance':'0.1', \
                          # https://github.com/plotly/plotly.js/issues/5158
                          'cluster_algorithm':'UMAP 2D', \
                          'cluster_these_layers':['0'], \
                          'precision_recall_ratios':'1.0', \
                          'congruence_portion':'union', \
                          'congruence_convolve':'0.0', \
                          'congruence_measure':'label', \
                          'nreplicates':'1', \
                          'batch_seed':'-1', \
                          'weights_seed':'-1', \
                          'labels':','*(nlabels-1), \
                          'file_dialog_string':os.getcwd(), \
                          'context_ms':'204.8', \
                          'shiftby_ms':'0.0', \
                          'optimizer':'Adam', \
                          'learning_rate':'0.0002'}, \
                       **{x[0]:x[3] for x in detect_parameters}, \
                       **{x[0]:x[3] for x in doubleclick_parameters}, \
                       **{x[0]:x[3] for x in model_parameters}},
                      fid)

    with open(statepath, 'r') as fid:
        state = yaml.load(fid, Loader=yaml.Loader)
        state['labels'] = state['labels'].split(',')

    file_dialog_root, file_dialog_filter = None, None

    clustered_sounds, clustered_activations, clustered_starts_sorted = None, None, None
    clustered_stops, iclustered_stops_sorted = None, None
    clustered_recording2firstsound = {}
    nearest_sounds=[-1]*snippets_nx*snippets_ny

    status_ticker_queue = {}

    waitfor_job = []
