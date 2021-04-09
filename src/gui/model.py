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

bokeh_document, configuration_file, architecture, audio_tic_rate, audio_nchannels, snippets_width_ms, snippets_nx, snippets_ny, snippets_waveform, snippets_spectrogram, nlabels, gui_width_pix, context_width_ms0, context_offset_ms0, context_width_ms, context_offset_ms, context_waveform, context_waveform_height_pix, context_spectrogram, context_spectrogram_height_pix, context_probability_height_pix, spectrogram_window, spectrogram_length_ms, spectrogram_overlap, spectrogram_colormap, context_spectrogram_units, spectrogram_low_hz, spectrogram_high_hz, context_waveform_low, context_waveform_high, context_spectrogram_freq_scale, cluster_dot_colors, xcluster, ycluster, zcluster, ndcluster, filter_order, filter_ratio_max, snippet_width_pix, layer, specie, word, nohyphen, kind, nlayers, layers, species, words, nohyphens, kinds, clustered_labels, snippets_gap_ms, snippets_tic, snippets_gap_tic, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint_tic, context_decimate_by, ilabel, annotated_samples, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, clustered_samples, clustered_activations, clustered_starts_sorted, clustered_stops, iclustered_stops_sorted, songexplorer_starttime, history_stack, history_idx, wizard, action, function, statepath, state, file_dialog_root, file_dialog_filter, nearest_samples, status_ticker_queue, waitfor_job, server_ipaddr, cluster_ipaddr, cluster_cmd, cluster_logfile_flag, source_path, cluster_circle_color, cluster_dot_palette, snippets_colormap = [None]*98
detect_where, detect_ncpu_cores, detect_ngpu_cards, detect_ngigabytes_memory, detect_cluster_flags, misses_where, misses_ncpu_cores, misses_ngpu_cards, misses_ngigabytes_memory, misses_cluster_flags, train_gpu, train_where, train_ncpu_cores, train_ngpu_cards, train_gpu_ngigabytes_memory, train_gpu_cluster_flags, train_ncpu_cores, train_ngpu_cards, train_cpu_ngigabytes_memory, train_cpu_cluster_flags, models_per_job, generalize_gpu, generalize_where, generalize_ncpu_cores, generalize_ngpu_cards, generalize_gpu_ngigabytes_memory, generalize_gpu_cluster_flags, generalize_ncpu_cores, generalize_ngpu_cards, generalize_cpu_ngigabytes_memory, generalize_cpu_cluster_flags, xvalidate_gpu, xvalidate_where, xvalidate_ncpu_cores, xvalidate_ngpu_cards, xvalidate_gpu_ngigabytes_memory, xvalidate_gpu_cluster_flags, xvalidate_ncpu_cores, xvalidate_ngpu_cards, xvalidate_cpu_ngigabytes_memory, xvalidate_cpu_cluster_flags, mistakes_where, mistakes_ncpu_cores, mistakes_ngpu_cards, mistakes_ngigabytes_memory, mistakes_cluster_flags, activations_gpu, activations_where, activations_ncpu_cores, activations_ngpu_cards, activations_gpu_ngigabytes_memory, activations_gpu_cluster_flags, activations_ncpu_cores, activations_ngpu_cards, activations_cpu_ngigabytes_memory, activations_cpu_cluster_flags, cluster_where, cluster_ncpu_cores, cluster_ngpu_cards, cluster_ngigabytes_memory, cluster_cluster_flags, accuracy_where, accuracy_ncpu_cores, accuracy_ngpu_cards, accuracy_ngigabytes_memory, accuracy_cluster_flags, freeze_where, freeze_ncpu_cores, freeze_ngpu_cards, freeze_ngigabytes_memory, freeze_cluster_flags, classify_gpu, classify_where, classify1_ncpu_cores, classify1_ngpu_cards, classify1_gpu_ngigabytes_memory, classify1_gpu_cluster_flags, classify1_ncpu_cores, classify1_ngpu_cards, classify1_cpu_ngigabytes_memory, classify1_cluster_cpu_flags, classify2_ncpu_cores, classify2_ngpu_cards, classify2_ngigabytes_memory, classify2_cluster_flags, ethogram_where, ethogram_ncpu_cores, ethogram_ngpu_cards, ethogram_ngigabytes_memory, ethogram_cluster_flags, compare_where, compare_ncpu_cores, compare_ngpu_cards, compare_ngigabytes_memory, compare_cluster_flags, congruence_where, congruence_ncpu_cores, congruence_ngpu_cards, congruence_ngigabytes_memory, congruence_cluster_flags, pca_batch_size, cluster_parallelize, accuracy_parallelize, congruence_parallelize, nprobabilities, nwindows, model_parameters = [None]*107

def parse_model_file(modelstr):
    filepath, filename = os.path.split(modelstr)
    if 'ckpt-' not in filename:
        filepath, filename = os.path.split(filepath)
    logdir, modeldir = os.path.split(filepath)
    prefix = filename.split('.')[0]
    m=re.search('ckpt-(\d+)\.',filename)
    check_point = m.group(1)
    return logdir, modeldir, prefix, check_point

def save_state_callback():
    with open(statepath, 'w') as fid:
       yaml.dump({**{'logs': V.logs_folder.value,
                     'model': V.model_file.value,
                     'wavtfcsvfiles': V.wavtfcsvfiles_string.value,
                     'groundtruth': V.groundtruth_folder.value,
                     'validationfiles': V.validationfiles_string.value,
                     'testfiles': V.testfiles_string.value,
                     'wantedwords': V.wantedwords_string.value,
                     'labeltypes': V.labeltypes_string.value,
                     'prevalences': V.prevalences_string.value,
                     'time_sigma': V.time_sigma_string.value,
                     'time_smooth_ms': V.time_smooth_ms_string.value,
                     'frequency_n_ms': V.frequency_n_ms_string.value,
                     'frequency_nw': V.frequency_nw_string.value,
                     'frequency_p': V.frequency_p_string.value,
                     'frequency_smooth_ms': V.frequency_smooth_ms_string.value,
                     'circle_radius': V.circle_radius.value,
                     'dot_size': V.dot_size.value,
                     'dot_alpha': V.dot_alpha.value,
                     'nsteps': V.nsteps_string.value,
                     'restore_from': V.restore_from_string.value,
                     'save_and_validate_interval': V.save_and_validate_period_string.value,
                     'validate_percentage': V.validate_percentage_string.value,
                     'mini_batch': V.mini_batch_string.value,
                     'kfold': V.kfold_string.value,
                     'activations_equalize_ratio': V.activations_equalize_ratio_string.value,
                     'activations_max_samples': V.activations_max_samples_string.value,
                     'pca_fraction_variance_to_retain': \
                             V.pca_fraction_variance_to_retain_string.value,
                     'tsne_perplexity': V.tsne_perplexity_string.value,
                     'tsne_exaggeration': V.tsne_exaggeration_string.value,
                     'umap_neighbors': V.umap_neighbors_string.value,
                     'umap_distance': V.umap_distance_string.value,
                     'cluster_algorithm': V.cluster_algorithm.value,
                     'cluster_these_layers': [x for x in V.cluster_these_layers.value],
                     'precision_recall_ratios': V.precision_recall_ratios_string.value,
                     'replicates': V.replicates_string.value,
                     'batch_seed': V.batch_seed_string.value,
                     'weights_seed': V.weights_seed_string.value,
                     'labels': str.join(',',[x.value for x in V.label_text_widgets]),
                     'file_dialog_string': V.file_dialog_string.value,
                     'context_ms': V.context_ms_string.value,
                     'shiftby_ms': V.shiftby_ms_string.value,
                     'representation': V.representation.value,
                     'window_ms': V.window_ms_string.value,
                     'mel&dct': V.mel_dct_string.value,
                     'stride_ms': V.stride_ms_string.value,
                     'optimizer': V.optimizer.value,
                     'learning_rate': V.learning_rate_string.value},
                  **{k:v.value for k,v in V.model_parameters.items()}},
                 fid)

def isannotated(sample):
    return np.where([x['file']==sample['file'] and x['ticks']==sample['ticks'] \
                     for x in annotated_samples])[0]

def isclustered(sample):
    return np.where([x['file']==sample['file'] and x['ticks']==sample['ticks'] \
                     for x in clustered_samples])[0]

def save_annotations():
    global nrecent_annotations
    if nrecent_annotations>0:
        fids = {}
        csvwriters = {}
        csvfiles_current = set([])
        for wavfile in set([x['file'] for x in annotated_samples if x["label"]!=""]):
            csvfile = wavfile[:-4]+"-annotated-"+songexplorer_starttime+".csv"
            annotated_csvfiles_all.add(csvfile)
            csvfiles_current.add(csvfile)
            fids[wavfile] = open(csvfile, "w", newline='')
            csvwriters[wavfile] = csv.writer(fids[wavfile])
        for file in annotated_csvfiles_all - csvfiles_current:
            if os.path.exists(file):
                os.remove(file)
        corrected_samples=[]
        for annotation in annotated_samples:
            if annotation['label']!="" and not annotation['label'].isspace():
                csvwriters[annotation['file']].writerow(
                        [os.path.basename(annotation['file']),
                        annotation['ticks'][0], annotation['ticks'][1],
                        'annotated', annotation['label']])
            iclustered = isclustered(annotation)
            if len(iclustered)>0 and clustered_samples[iclustered[0]]['kind']=='annotated':
                corrected_samples.append(annotation)
        if corrected_samples:
            df_corrected = pd.DataFrame([[os.path.basename(x['file']), x['ticks'][0], \
                                          x['ticks'][1], 'annotated', x['label']] \
                                         for x in corrected_samples], \
                                        columns=['file','start','stop','kind','label'])
            for wavfile in set([x['file'] for x in corrected_samples]):
                wavdir, wavbase = os.path.split(wavfile)
                for csvbase in filter(lambda x: x.startswith(wavbase[:-4]) and
                                                x.endswith(".csv") and
                                                "-annotated-" in x and
                                                songexplorer_starttime not in x,
                                      os.listdir(wavdir)):
                    csvfile = os.path.join(wavdir, csvbase)
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

def add_annotation(sample, addto_history=True):
    global annotated_samples, annotated_starts_sorted
    global annotated_stops, iannotated_stops_sorted
    global history_stack, history_idx
    iannotated = isannotated(sample)
    if len(iannotated)>0:
        del annotated_samples[iannotated[0]]
    idx = np.searchsorted(annotated_starts_sorted, sample['ticks'][0])
    annotated_samples.insert(idx, sample.copy())
    if addto_history:
        del history_stack[history_idx:]
        history_stack.append(['add',sample])
        history_idx+=1
    annotated_starts_sorted = [x['ticks'][0] for x in annotated_samples]
    annotated_stops = [x['ticks'][1] for x in annotated_samples]
    iannotated_stops_sorted = np.argsort(annotated_stops)
    if sample['label'] in state['labels']:
        thislabel = state['labels'].index(sample['label'])
        count = int(V.label_count_widgets[thislabel].label)
        V.label_count_widgets[thislabel].label = str(count+1)
    finalize_annotation(addto_history)
    return idx

def delete_annotation(isample, addto_history=True):
    global annotated_samples, annotated_starts_sorted
    global annotated_stops, iannotated_stops_sorted
    global history_stack, history_idx
    if addto_history:
        del history_stack[history_idx:]
        history_stack.append(['delete',annotated_samples[isample].copy()])
        history_idx+=1
    if annotated_samples[isample]['label'] in state['labels']:
        thislabel = state['labels'].index(annotated_samples[isample]['label'])
        count = int(V.label_count_widgets[thislabel].label)
        V.label_count_widgets[thislabel].label = str(count-1)
    iclustered = isclustered(annotated_samples[isample])
    if len(iclustered)>0 and clustered_samples[iclustered[0]]['kind']=='annotated':
        annotated_samples[isample]['label'] = clustered_samples[iclustered[0]]['label']
    else:
        del annotated_samples[isample]
        annotated_starts_sorted = [x['ticks'][0] for x in annotated_samples]
        annotated_stops = [x['ticks'][1] for x in annotated_samples]
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

def init(_bokeh_document, _configuration_file):
    global bokeh_document, configuration_file, architecture, audio_tic_rate, audio_nchannels, snippets_width_ms, snippets_nx, snippets_ny, snippets_waveform, snippets_spectrogram, nlabels, gui_width_pix, context_width_ms0, context_offset_ms0, context_width_ms, context_offset_ms, context_waveform, context_waveform_height_pix, context_spectrogram, context_spectrogram_height_pix, context_probability_height_pix, context_spectrogram_units, spectrogram_window, spectrogram_length_ms, spectrogram_overlap, spectrogram_colormap, spectrogram_low_hz, spectrogram_high_hz, context_waveform_low, context_waveform_high, context_spectrogram_freq_scale, cluster_dot_colors, xcluster, ycluster, zcluster, ndcluster, filter_order, filter_ratio_max, snippet_width_pix, ilayer, ispecies, iword, inohyphen, ikind, nlayers, layers, species, words, nohyphens, kinds, clustered_labels, snippets_gap_ms, snippets_tic, snippets_gap_tic, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint_tic, context_decimate_by, ilabel, annotated_samples, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, clustered_samples, clustered_activations, clustered_starts_sorted, clustered_stops, iclustered_stops_sorted, songexplorer_starttime, history_stack, history_idx, wizard, action, function, statepath, state, file_dialog_root, file_dialog_filter, nearest_samples, status_ticker_queue, waitfor_job, server_ipaddr, cluster_ipaddr, cluster_cmd, cluster_logfile_flag, source_path, cluster_circle_color, cluster_dot_palette, snippets_colormap
    global detect_where, detect_local_resources, detect_cluster_flags, misses_where, misses_local_resources, misses_cluster_flags, train_gpu, train_where, train_local_resources_gpu, train_cluster_flags_gpu, train_local_resources_cpu, train_cluster_flags_cpu, models_per_job, generalize_gpu, generalize_where, generalize_local_resources_gpu, generalize_cluster_flags_gpu, generalize_local_resources_cpu, generalize_cluster_flags_cpu, xvalidate_gpu, xvalidate_where, xvalidate_local_resources_gpu, xvalidate_cluster_flags_gpu, xvalidate_local_resources_cpu, xvalidate_cluster_flags_cpu, mistakes_where, mistakes_local_resources, mistakes_cluster_flags, activations_gpu, activations_where, activations_local_resources_gpu, activations_cluster_flags_gpu, activations_local_resources_cpu, activations_cluster_flags_cpu, cluster_where, cluster_local_resources, cluster_cluster_flags, accuracy_where, accuracy_local_resources, accuracy_cluster_flags, freeze_where, freeze_local_resources, freeze_cluster_flags, classify_gpu, classify_where, classify1_local_resources_gpu, classify1_cluster_flags_gpu, classify1_local_resources_cpu, classify1_cluster_flags_cpu, classify2_local_resources, classify2_cluster_flags, ethogram_where, ethogram_local_resources, ethogram_cluster_flags, compare_where, compare_local_resources, compare_cluster_flags, congruence_where, congruence_local_resources, congruence_cluster_flags, pca_batch_size, cluster_parallelize, accuracy_parallelize, congruence_parallelize, nprobabilities, nwindows, model_parameters

    bokeh_document = _bokeh_document

    exec(open(_configuration_file).read(), globals())

    sys.path.append(os.path.dirname(architecture))
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "speech_commands_custom"))
    model = importlib.import_module(os.path.basename(architecture))
    model_parameters = model.model_parameters

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
    snippets_waveform=gui_snippets_waveform
    snippets_spectrogram=gui_snippets_spectrogram
    nlabels=int(gui_nlabels)
    gui_width_pix=int(gui_gui_width_pix)
    context_width_ms0=float(gui_context_width_ms)
    context_offset_ms0=float(gui_context_offset_ms)
    context_width_ms=float(gui_context_width_ms)
    context_offset_ms=float(gui_context_offset_ms)

    context_waveform=gui_context_waveform
    context_waveform_height_pix=int(gui_context_waveform_height_pix)

    context_spectrogram=gui_context_spectrogram
    context_spectrogram_height_pix=int(gui_context_spectrogram_height_pix)
    context_spectrogram_units=gui_context_spectrogram_units
    spectrogram_colormap=gui_spectrogram_colormap
    spectrogram_window=gui_spectrogram_window
    spectrogram_length_ms=[float(gui_spectrogram_length_ms)]*audio_nchannels
    spectrogram_overlap=float(gui_spectrogram_overlap)
    spectrogram_low_hz=[float(gui_spectrogram_low_hz)]*audio_nchannels
    spectrogram_high_hz=[float(gui_spectrogram_high_hz)]*audio_nchannels
    context_spectrogram_freq_scale = 0.001 if context_spectrogram_units=='mHz' else \
                                     1 if context_spectrogram_units=='Hz' else \
                                  1000 if context_spectrogram_units=='kHz' else \
                               1000000

    context_probability_height_pix=int(gui_context_probability_height_pix)

    context_waveform_low = [-1]*audio_nchannels
    context_waveform_high = [1]*audio_nchannels

    cluster_dot_palette = gui_cluster_dot_palette
    snippets_colormap = gui_snippets_colormap
    cluster_circle_color = gui_cluster_circle_color
    cluster_dot_colors = {}

    xcluster = ycluster = zcluster = np.nan
    ndcluster = 0

    filter_order=2
    filter_ratio_max=4

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
    snippets_decimate_by = round(tic2pix/filter_ratio_max) if tic2pix>filter_ratio_max else 1
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

    annotated_samples=[]
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
            yaml.dump({**{'logs':'', \
                          'model':'', \
                          'wavtfcsvfiles':'', \
                          'groundtruth':'', \
                          'validationfiles':'', \
                          'testfiles':'', \
                          'wantedwords':'', \
                          'labeltypes':'', \
                          'prevalences':'', \
                          'time_sigma':'6,3', \
                          'time_smooth_ms':'6.4', \
                          'frequency_n_ms':'25.6', \
                          'frequency_nw':'4', \
                          'frequency_p':'0.1,1.0', \
                          'frequency_smooth_ms':'25.6', \
                          'circle_radius':1, \
                          'dot_size':6, \
                          'dot_alpha':0.1, \
                          'nsteps':'0', \
                          'restore_from':'', \
                          'save_and_validate_interval':'0', \
                          'validate_percentage':'0', \
                          'mini_batch':'32', \
                          'kfold':'4', \
                          'activations_equalize_ratio':'10', \
                          'activations_max_samples':'1000', \
                          'pca_fraction_variance_to_retain':'1.0', \
                          'tsne_perplexity':'30', \
                          'tsne_exaggeration':'12.0', \
                          'umap_neighbors':'10', \
                          'umap_distance':'0.1', \
                          # https://github.com/plotly/plotly.js/issues/5158
                          'cluster_algorithm':'UMAP 2D', \
                          'cluster_these_layers':['0'], \
                          'precision_recall_ratios':'1.0', \
                          'replicates':'1', \
                          'batch_seed':'-1', \
                          'weights_seed':'-1', \
                          'labels':','*(nlabels-1), \
                          'file_dialog_string':os.getcwd(), \
                          'context_ms':'204.8', \
                          'shiftby_ms':'0.0', \
                          'representation':'mel-cepstrum', \
                          'window_ms':'6.4', \
                          'mel&dct':'7,7', \
                          'stride_ms':'1.6', \
                          'optimizer':'adam', \
                          'learning_rate':'0.0002'}, \
                       **{x[0]:x[3] for x in model_parameters}},
                      fid)

    with open(statepath, 'r') as fid:
        state = yaml.load(fid, Loader=yaml.Loader)
        state['labels'] = state['labels'].split(',')

    file_dialog_root, file_dialog_filter = None, None

    clustered_samples, clustered_activations, clustered_starts_sorted = None, None, None
    clustered_stops, iclustered_stops_sorted = None, None

    nearest_samples=[-1]*snippets_nx*snippets_ny

    status_ticker_queue = {}

    waitfor_job = []
