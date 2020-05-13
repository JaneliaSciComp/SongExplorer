import os
import yaml
import numpy as np
from datetime import datetime
import logging 
import csv
import re
import pandas as pd

bokehlog = logging.getLogger("deepsong") 

import view as V

configuration_file, audio_tic_rate, audio_nchannels, snippets_ms, nx, ny, nlabels, gui_width_pix, context_width_ms0, context_offset_ms0, context_width_ms, context_offset_ms, cluster_circle_color, cluster_dot_colors, xcluster, ycluster, zcluster, ndcluster, filter_order, filter_ratio_max, snippet_width_pix, layer, specie, word, nohyphen, kind, nlayers, layers, species, words, nohyphens, kinds, snippets_gap_ms, snippets_tic, snippets_gap_tic, tic2pix, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint_tic, context_decimate_by, panned_sample, ipanned_quad, ilabel, annotated_samples, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, clustered_samples, clustered_activations, clustered_starts_sorted, clustered_stops, iclustered_stops_sorted, deepsong_starttime, history_stack, history_idx, wizard, action, function, statepath, state, file_dialog_root, file_dialog_filter, nearest_samples, status_ticker_queue = [None]*73

def parse_model_file(filepath):
    head, check_point_str = os.path.split(filepath)
    logdir, model = os.path.split(head)
    m=re.search('ckpt-(\d+)\.',check_point_str)
    check_point = m.group(1)
    return logdir, model, check_point

def save_state_callback():
    with open(statepath, 'w') as fid:
        yaml.dump({'logs': V.logs_folder.value,
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
                   'connection_type': V.connection_type.value,
                   'precision_recall_ratios': V.precision_recall_ratios_string.value,
                   'context_ms': V.context_ms_string.value,
                   'shiftby_ms': V.shiftby_ms_string.value,
                   'representation': V.representation.value,
                   'window_ms': V.window_ms_string.value,
                   'mel&dct': V.mel_dct_string.value,
                   'stride_ms': V.stride_ms_string.value,
                   'dropout': V.dropout_string.value,
                   'optimizer': V.optimizer.value,
                   'learning_rate': V.learning_rate_string.value,
                   'kernel_sizes': V.kernel_sizes_string.value,
                   'last_conv_width': V.last_conv_width_string.value,
                   'nfeatures': V.nfeatures_string.value,
                   'dilate_after_layer': V.dilate_after_layer_string.value,
                   'stride_after_layer': V.stride_after_layer_string.value,
                   'labels': str.join(',',[x.value for x in V.label_text_widgets]),
                   'file_dialog_string': V.file_dialog_string.value},
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
            csvfile = wavfile[:-4]+"-annotated-"+deepsong_starttime+".csv"
            annotated_csvfiles_all.add(csvfile)
            csvfiles_current.add(csvfile)
            fids[wavfile] = open(csvfile, "w", newline='')
            csvwriters[wavfile] = csv.writer(fids[wavfile])
        for file in annotated_csvfiles_all - csvfiles_current:
            if os.path.exists(file):
                os.remove(file)
        corrected_samples=[]
        for annotation in annotated_samples:
            if annotation['label']!="":
                csvwriters[annotation['file']].writerow(
                        [os.path.basename(annotation['file']),
                        annotation['ticks'][0], annotation['ticks'][1],
                        'annotated', annotation['label']])
            if len(isclustered(annotation))>0:
                corrected_samples.append(annotation)
        if corrected_samples:
            df_corrected = pd.DataFrame([[os.path.basename(x['file']), x['ticks'][0], \
                                          x['ticks'][1], x['kind'], x['label']] \
                                         for x in corrected_samples], \
                                        columns=['file','start','stop','kind','label'])
            for wavfile in set([x['file'] for x in corrected_samples]):
                wavdir, wavbase = os.path.split(wavfile)
                for csvbase in filter(lambda x: x.startswith(wavbase[:-4]) and
                                                x.endswith(".csv") and
                                                "-annotated-" in x and
                                                deepsong_starttime not in x,
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
                        df_clustered.loc[isetdiff,:].to_csv(csvfile, header=False, \
                                                            index=False)
        for fid in fids.values():
            fid.close()
        nrecent_annotations=0
        V.save_update(nrecent_annotations)

def add_annotation(sample, addto_history=True):
    global annotated_samples, annotated_starts_sorted
    global annotated_stops, iannotated_stops_sorted
    global history_stack, history_idx
    iannotated = isannotated(sample)
    if len(iannotated)>0:
        del annotated_samples[iannotated[0]]
    idx = np.searchsorted(annotated_starts_sorted, sample['ticks'][0])
    annotated_samples.insert(idx, sample)
    if addto_history:
        del history_stack[history_idx:]
        history_stack.append(['add',sample.copy()])
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
    if len(iclustered)>0:
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
    V.context_update(redraw_snippets)

def init(_configuration_file, _audio_tic_rate, _audio_nchannels,
         _snippets_ms, _nx, _ny, _nlabels, _gui_width_pix,
         _context_width_ms, _context_offset_ms):
    global configuration_file, audio_tic_rate, audio_nchannels, snippets_ms, nx, ny, nlabels, gui_width_pix, context_width_ms0, context_offset_ms0, context_width_ms, context_offset_ms, cluster_circle_color, cluster_dot_colors, xcluster, ycluster, zcluster, ndcluster, filter_order, filter_ratio_max, snippet_width_pix, ilayer, ispecies, iword, inohyphen, ikind, nlayers, layers, species, words, nohyphens, kinds, snippets_gap_ms, snippets_tic, snippets_gap_tic, tic2pix, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint_tic, context_decimate_by, panned_sample, ipanned_quad, ilabel, annotated_samples, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, clustered_samples, clustered_activations, clustered_starts_sorted, clustered_stops, iclustered_stops_sorted, deepsong_starttime, history_stack, history_idx, wizard, action, function, statepath, state, file_dialog_root, file_dialog_filter, nearest_samples, status_ticker_queue

    configuration_file = _configuration_file
    audio_tic_rate=int(_audio_tic_rate)
    audio_nchannels=int(_audio_nchannels)
    snippets_ms=float(_snippets_ms)
    nx=int(_nx)
    ny=int(_ny)
    nlabels=int(_nlabels)
    gui_width_pix=int(_gui_width_pix)
    context_width_ms0=float(_context_width_ms)
    context_offset_ms0=float(_context_offset_ms)
    context_width_ms=float(_context_width_ms)
    context_offset_ms=float(_context_offset_ms)

    cluster_circle_color = ''
    cluster_dot_colors = {}

    xcluster = ycluster = zcluster = np.nan
    ndcluster = 0

    filter_order=2
    filter_ratio_max=4

    snippet_width_pix = gui_width_pix/2/nx

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

    snippets_gap_ms=snippets_ms/10

    snippets_tic = int(np.rint(snippets_ms/1000*audio_tic_rate))
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

    panned_sample = None
    ipanned_quad = -1

    ilabel=0

    annotated_samples=[]
    annotated_starts_sorted=[]
    annotated_stops=[]
    iannotated_stops_sorted=[]
    annotated_csvfiles_all = set([])
    nrecent_annotations=0

    deepsong_starttime = datetime.strftime(datetime.now(),'%Y%m%dT%H%M%S')

    history_stack=[]
    history_idx=0

    wizard=None
    action=None
    function=None

    statepath = os.path.join(os.environ["DEEPSONG_STATE"], 'deepsong.state.yml')

    if not os.path.exists(statepath):
        with open(statepath, 'w') as fid:
            yaml.dump({'logs':'', \
                       'model':'', \
                       'wavtfcsvfiles':'', \
                       'groundtruth':'', \
                       'validationfiles':'', \
                       'testfiles':'', \
                       'wantedwords':'', \
                       'labeltypes':'', \
                       'prevalences':'', \
                       'time_sigma':'6', \
                       'time_smooth_ms':'6.4', \
                       'frequency_n_ms':'25.6', \
                       'frequency_nw':'4', \
                       'frequency_p':'0.1', \
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
                       'cluster_algorithm':'UMAP 3D', \
                       'connection_type':'plain', \
                       'precision_recall_ratios':'0.5,1,2', \
                       'context_ms':'204.8', \
                       'shiftby_ms':'0.0', \
                       'representation':'mel-cepstrum', \
                       'window_ms':'6.4', \
                       'mel&dct':'7,7', \
                       'stride_ms':'1.6', \
                       'dropout':'0.5', \
                       'optimizer':'adam', \
                       'learning_rate':'0.0002', \
                       'kernel_sizes':'5,3,3', \
                       'last_conv_width':'130', \
                       'nfeatures':'64,64,64', \
                       'dilate_after_layer':'65535', \
                       'stride_after_layer':'65535', \
                       'labels':','*(nlabels-1), \
                       'file_dialog_string':os.getcwd()}, \
                       fid)

    with open(statepath, 'r') as fid:
        state = yaml.load(fid, Loader=yaml.Loader)
        state['labels'] = state['labels'].split(',')

    file_dialog_root, file_dialog_filter = None, None

    clustered_samples, clustered_activations, clustered_starts_sorted = None, None, None
    clustered_stops, iclustered_stops_sorted = None, None

    nearest_samples=[-1]*nx*ny

    status_ticker_queue = {}
