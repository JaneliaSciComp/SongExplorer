import os
import yaml
from bokeh.palettes import viridis
import numpy as np
from datetime import datetime
import logging 
import csv
import re

log = logging.getLogger("deepsong-model") 

import view as V

Fs, snippets_ms, nx, ny, nlabels, gui_width_pix, context_width_ms, context_offset_ms, xtsne, ytsne, filter_order, filter_ratio_max, radius, hex_size, snippet_width_pix, layer, specie, word, nohyphen, kind, palette, nlayers, layers, species, words, nohyphens, kinds, snippets_gap_ms, snippets_tic, snippets_gap_tic, tic2pix, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint, context_decimate_by, panned_sample, ipanned_quad, ilabel, annotated_samples, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, deepsong_starttime, history_stack, history_idx, wizard, action, function, logdir, model, check_point, statepath, state, file_dialog_root, file_dialog_filter, nearest_samples= [None]*65

def parse_model_file():
    global logdir, model, check_point
    if V.model_file.value is not '':
        head,check_point_str = os.path.split(V.model_file.value)
        logdir,model_str = os.path.split(head)
        model = model_str.split('_')[1]
        m=re.search('ckpt-(\d+)\.',check_point_str)
        check_point = m.group(1)

def save_state_callback():
    with open(statepath, 'w') as fid:
        yaml.dump({'configuration': V.configuration_file.value,
                   'logs': V.logs_folder.value,
                   'model': V.model_file.value,
                   'wavtfcsvfiles': V.wavtfcsvfiles_string.value,
                   'groundtruth': V.groundtruth_folder.value,
                   'validationfiles': V.validationfiles_string.value,
                   'testfiles': V.testfiles_string.value,
                   'wantedwords': V.wantedwords_string.value,
                   'labeltypes': V.labeltypes_string.value,
                   'time_sigma': V.time_sigma_string.value,
                   'time_smooth_ms': V.time_smooth_ms_string.value,
                   'frequency_n_ms': V.frequency_n_ms_string.value,
                   'frequency_nw': V.frequency_nw_string.value,
                   'frequency_p': V.frequency_p_string.value,
                   'frequency_smooth_ms': V.frequency_smooth_ms_string.value,
                   'nsteps': V.nsteps_string.value,
                   'save_and_validate_interval': V.save_and_validate_period_string.value,
                   'validate_percentage': V.validate_percentage_string.value,
                   'mini_batch': V.mini_batch_string.value,
                   'kfold': V.kfold_string.value,
                   'cluster_equalize_ratio': V.cluster_equalize_ratio_string.value,
                   'cluster_max_samples': V.cluster_max_samples_string.value,
                   'pca_fraction_variance_to_retain': V.pca_fraction_variance_to_retain_string.value,
                   'tsne_perplexity': V.tsne_perplexity_string.value,
                   'tsne_exaggeration': V.tsne_exaggeration_string.value,
                   'precision_recall_ratios': V.precision_recall_ratios_string.value,
                   'context_ms': V.context_ms_string.value,
                   'shiftby_ms': V.shiftby_ms_string.value,
                   'window_ms': V.window_ms_string.value,
                   'mel&dct': V.mel_dct_string.value,
                   'stride_ms': V.stride_ms_string.value,
                   'dropout': V.dropout_string.value,
                   'optimizer': V.optimizer.value,
                   'learning_rate': V.learning_rate_string.value,
                   'kernel_sizes': V.kernel_sizes_string.value,
                   'last_conv_width': V.last_conv_width_string.value,
                   'nfeatures': V.nfeatures_string.value,
                   'labels': str.join(',',[x.value for x in V.label_text_widgets]),
                   'file_dialog_string': V.file_dialog_string.value},
                  fid)

def save_annotations():
    global nrecent_annotations
    if nrecent_annotations>0:
        fids = {}
        csvwriters = {}
        csvfiles_current = set([])
        for wavfile in set([x['file'] for x in annotated_samples]):
            csvfile = wavfile[:-4]+"-annotated-"+deepsong_starttime+".csv"
            annotated_csvfiles_all.add(csvfile)
            csvfiles_current.add(csvfile)
            fids[wavfile] = open(csvfile, "w", newline='')
            csvwriters[wavfile] = csv.writer(fids[wavfile])
        for file in annotated_csvfiles_all - csvfiles_current:
            if os.path.exists(file):
                os.remove(file)
        for annotation in annotated_samples:
            csvwriters[annotation['file']].writerow(
                    [os.path.basename(annotation['file']),
                    annotation['ticks'][0], annotation['ticks'][1],
                    'annotated', annotation['label']])
        for fid in fids.values():
            fid.close()
        nrecent_annotations=0
        V.save_update(nrecent_annotations)

def add_annotation(sample, addto_history=True):
    global annotated_samples
    global annotated_starts_sorted
    global annotated_stops, iannotated_stops_sorted
    global history_stack, history_idx
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
    global annotated_samples
    global annotated_starts_sorted
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


def init(_configuration_inarg, _Fs, _snippets_ms, _nx, _ny, _nlabels, _gui_width_pix, _context_width_ms, _context_offset_ms):
    global Fs, snippets_ms, nx, ny, nlabels, gui_width_pix, context_width_ms, context_offset_ms, xtsne, ytsne, filter_order, filter_ratio_max, radius, hex_size, snippet_width_pix, ilayer, ispecies, iword, inohyphen, ikind, palette, nlayers, layers, species, words, nohyphens, kinds, snippets_gap_ms, snippets_tic, snippets_gap_tic, tic2pix, snippets_decimate_by, snippets_pix, snippets_gap_pix, context_width_tic, context_offset_tic, isnippet, xsnippet, ysnippet, file_nframes, context_midpoint, context_decimate_by, panned_sample, ipanned_quad, ilabel, annotated_samples, annotated_starts_sorted, annotated_stops, iannotated_stops_sorted, annotated_csvfiles_all, nrecent_annotations, deepsong_starttime, history_stack, history_idx, wizard, action, function, logdir, model, check_point, statepath, state, file_dialog_root, file_dialog_filter, nearest_samples

    Fs=int(_Fs)
    snippets_ms=float(_snippets_ms)
    nx=int(_nx)
    ny=int(_ny)
    nlabels=int(_nlabels)
    gui_width_pix=int(_gui_width_pix)
    context_width_ms=int(_context_width_ms)
    context_offset_ms=int(_context_offset_ms)

    xtsne, ytsne = np.nan, np.nan

    filter_order=2
    filter_ratio_max=4
    radius=3
    hex_size=1

    snippet_width_pix = gui_width_pix/2/nx

    ilayer=0
    ispecies=0
    iword=0
    inohyphen=0
    ikind=0
    palette = viridis(256)

    nlayers = 0
    layers = []
    species = []
    words = []
    nohyphens = []
    kinds = []

    snippets_gap_ms=snippets_ms/10

    snippets_tic = int(np.rint(snippets_ms/1000*Fs))
    snippets_gap_tic = int(np.rint(snippets_gap_ms/1000*Fs))
    tic2pix = (snippets_gap_tic+snippets_tic) / snippet_width_pix
    snippets_decimate_by = round(tic2pix/filter_ratio_max) if tic2pix>filter_ratio_max else 1
    snippets_pix = round(snippets_tic / snippets_decimate_by)
    snippets_gap_pix = round(snippets_gap_tic / snippets_decimate_by)

    context_width_tic = int(np.rint(context_width_ms/1000*Fs))
    context_offset_tic = int(np.rint(context_offset_ms/1000*Fs))

    isnippet = -1
    xsnippet = -1
    ysnippet = -1

    file_nframes = -1
    context_midpoint = -1
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

    logdir, model, check_point = '', '', ''

    statepath = os.path.join(os.environ["DEEPSONG_STATE"], 'deepsong.state.yml')

    if not os.path.exists(statepath):
        with open(statepath, 'w') as fid:
            yaml.dump({'configuration':os.path.abspath(_configuration_inarg), \
                       'logs':'', \
                       'model':'', \
                       'wavtfcsvfiles':'', \
                       'groundtruth':'', \
                       'validationfiles':'', \
                       'testfiles':'', \
                       'wantedwords':'', \
                       'labeltypes':'', \
                       'time_sigma':'4', \
                       'time_smooth_ms':'6.4', \
                       'frequency_n_ms':'25.6', \
                       'frequency_nw':'4', \
                       'frequency_p':'0.1', \
                       'frequency_smooth_ms':'25.6', \
                       'nsteps':'0', \
                       'save_and_validate_interval':'0', \
                       'validate_percentage':'0', \
                       'mini_batch':'32', \
                       'kfold':'8', \
                       'cluster_equalize_ratio':'10000', \
                       'cluster_max_samples':'100000', \
                       'pca_fraction_variance_to_retain':'0.99', \
                       'tsne_perplexity':'30', \
                       'tsne_exaggeration':'12.0', \
                       'precision_recall_ratios':'0.5,1,2', \
                       'context_ms':'204.8', \
                       'shiftby_ms':'0.0', \
                       'window_ms':'6.4', \
                       'mel&dct':'7,7', \
                       'stride_ms':'1.6', \
                       'dropout':'0.5', \
                       'optimizer':'adam', \
                       'learning_rate':'0.0002', \
                       'kernel_sizes':'5,3,3', \
                       'last_conv_width':'130', \
                       'nfeatures':'256,256,256', \
                       'labels':','*(nlabels-1), \
                       'file_dialog_string':os.getcwd()}, \
                       fid)

    with open(statepath, 'r') as fid:
        state = yaml.load(fid, Loader=yaml.Loader)
        state['labels'] = state['labels'].split(',')

    file_dialog_root, file_dialog_filter = None, None

    clustered_samples, clustered_hidden, clustered_starts_sorted = None, None, None

    nearest_samples=[-1]*nx*ny
