import os
from bokeh.models.widgets import RadioButtonGroup, TextInput, Button, Div, DateFormatter, TextAreaInput, Select, NumberFormatter, Slider, Toggle, ColorPicker
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.plotting import figure
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap
from bokeh.events import Tap, DoubleTap, PanStart, Pan, PanEnd, ButtonClick
from bokeh.models.callbacks import CustomJS
from bokeh.models.markers import Circle
import numpy as np
import glob
from datetime import datetime
import markdown
import pandas as pd
import wave
import scipy.io.wavfile as spiowav
from scipy.signal import decimate
import logging 
import base64
import io
from natsort import natsorted
import pims
import av
from bokeh import palettes
from itertools import cycle, product
import ast

bokehlog = logging.getLogger("deepsong") 

import model as M
import controller as C

bokeh_document, cluster_dot_palette, snippet_palette, p_cluster, cluster_hexs, cluster_dots, cluster_extent, p_snippets, label_sources, label_sources_new, wav_sources, line_glyphs, quad_grey_snippets, circle_fuchsia_cluster, p_cluster_circle, p_context, p_line_red_context, line_red_context, quad_grey_context_old, quad_grey_context_new, quad_grey_context_pan, quad_fuchsia_context, quad_fuchsia_snippets, wav_source, line_glyph, label_source, label_source_new, which_layer, which_species, which_word, which_nohyphen, which_kind, color_picker, circle_radius_string, hex_size_string, dot_alpha, cluster_style, zoom_context, zoom_offset, zoomin, zoomout, reset, panleft, panright, allleft, allout, allright, save_indicator, label_text_widgets, label_count_widgets, play, play_callback, video_toggle, video_div, undo, redo, detect, misses, configuration, configuration_file, train, leaveoneout, leaveallout, xvalidate, mistakes, activations, cluster, visualize, accuracy, freeze, classify, ethogram, compare, congruence, status_ticker, file_dialog_source, file_dialog_source, configuration_contents, logs, logs_folder, model, model_file, wavtfcsvfiles, wavtfcsvfiles_string, groundtruth, groundtruth_folder, validationfiles, testfiles, validationfiles_string, testfiles_string, wantedwords, wantedwords_string, labeltypes, labeltypes_string, copy, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, generalize, tunehyperparameters, findnovellabels, examineerrors, testdensely, doit, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string, nsteps_string, restore_from_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string, kfold_string, activations_equalize_ratio_string, activations_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, umap_neighbors_string, umap_distance_string, cluster_algorithm, connection_type, precision_recall_ratios_string, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, editconfiguration, file_dialog_string, file_dialog_table, readme_contents, wordcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs = [None]*153

def cluster_initialize(newcolors=True):
    global precomputed_hexs, precomputed_dots
    global p_cluster_xmax, p_cluster_xmin, p_cluster_ymax, p_cluster_ymin

    npzfile = np.load(os.path.join(groundtruth_folder.value,'cluster.npz'),
                      allow_pickle=True)
    M.clustered_samples = npzfile['samples']
    M.clustered_activations = npzfile['activations_clustered']

    M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_samples]
    isort = np.argsort(M.clustered_starts_sorted)
    for i in range(len(M.clustered_activations)):
        M.clustered_activations[i] = M.clustered_activations[i][isort,:]
    M.clustered_samples = [M.clustered_samples[x] for x in isort]
    M.clustered_starts_sorted = [M.clustered_starts_sorted[x] for x in isort]

    M.clustered_stops = [x['ticks'][1] for x in M.clustered_samples]
    M.iclustered_stops_sorted = np.argsort(M.clustered_stops)

    cluster_isnotnan = [not np.isnan(x[0]) and not np.isnan(x[1]) \
                        for x in M.clustered_activations[0]]

    M.nlayers=len(M.clustered_activations)

    M.layers = ["input"]+["hidden #"+str(i) for i in range(1,M.nlayers-1)]+["output"]
    M.species = set([x['label'].split('-')[0]+'-' \
                     for x in M.clustered_samples if '-' in x['label']])
    M.species |= set([''])
    M.species = natsorted(list(M.species))
    M.words = set(['-'+x['label'].split('-')[1] \
                   for x in M.clustered_samples if '-' in x['label']])
    M.words |= set([''])
    M.words = natsorted(list(M.words))
    M.nohyphens = set([x['label'] for x in M.clustered_samples if '-' not in x['label']])
    M.nohyphens |= set([''])
    M.nohyphens = natsorted(list(M.nohyphens))
    M.kinds = natsorted(list(set([x['kind'] for x in M.clustered_samples])))

    if newcolors:
        allcombos = [x[0][:-1]+x[1] for x in product(M.species[1:], M.words[1:])]
        M.cluster_dot_colors = { l:c for l,c in zip(allcombos+ M.nohyphens[1:],
                                                    cycle(cluster_dot_palette)) }

    p_cluster_xmax = [np.iinfo(np.int64).min]*M.nlayers
    p_cluster_xmin = [np.iinfo(np.int64).max]*M.nlayers
    p_cluster_ymax = [np.iinfo(np.int64).min]*M.nlayers
    p_cluster_ymin = [np.iinfo(np.int64).max]*M.nlayers
    precomputed_hexs = [None]*M.nlayers
    precomputed_dots = [None]*M.nlayers
    for ilayer in range(M.nlayers):
        precomputed_hexs[ilayer] = [None]*len(M.species)
        precomputed_dots[ilayer] = [None]*len(M.species)
        p_cluster_xmin[ilayer] = np.minimum(p_cluster_xmin[ilayer], \
                                            np.min(M.clustered_activations[ilayer][:,0]))
        p_cluster_xmax[ilayer] = np.maximum(p_cluster_xmax[ilayer], \
                                            np.max(M.clustered_activations[ilayer][:,0]))
        p_cluster_ymin[ilayer] = np.minimum(p_cluster_ymin[ilayer], \
                                            np.min(M.clustered_activations[ilayer][:,1]))
        p_cluster_ymax[ilayer] = np.maximum(p_cluster_ymax[ilayer], \
                                            np.max(M.clustered_activations[ilayer][:,1]))
        for (ispecies,specie) in enumerate(M.species):
            precomputed_hexs[ilayer][ispecies] = [None]*len(M.words)
            precomputed_dots[ilayer][ispecies] = [None]*len(M.words)
            for (iword,word) in enumerate(M.words):
                precomputed_hexs[ilayer][ispecies][iword] = [None]*len(M.nohyphens)
                precomputed_dots[ilayer][ispecies][iword] = [None]*len(M.nohyphens)
                for (inohyphen,nohyphen) in enumerate(M.nohyphens):
                    precomputed_hexs[ilayer][ispecies][iword][inohyphen] = \
                            [None]*len(M.kinds)
                    precomputed_dots[ilayer][ispecies][iword][inohyphen] = \
                            [None]*len(M.kinds)
                    for (ikind,kind) in enumerate(M.kinds):
                        if inohyphen!=0 and (ispecies!=0 or iword!=0):
                            continue
                        bidx = np.logical_and([specie in x['label'] and \
                                               word in x['label'] and \
                                               (nohyphen=="" or nohyphen==x['label']) and \
                                               kind==x['kind'] \
                                               for x in M.clustered_samples], \
                                               cluster_isnotnan)
                        if not any(bidx):
                            continue
                        precomputed_hexs[ilayer][ispecies][iword][inohyphen][ikind] = \
                                hexbin(M.clustered_activations[ilayer][bidx,0], \
                                       M.clustered_activations[ilayer][bidx,1], \
                                       size=float(M.state["hex_size"]))
                        if inohyphen>0:
                            colors = [M.cluster_dot_colors[nohyphen] for b in bidx if b]
                        else:
                            colors = [M.cluster_dot_colors[x['label']] \
                                      if x['label'] in M.cluster_dot_colors else "black" \
                                      for x,b in zip(M.clustered_samples,bidx) if b]
                        precomputed_dots[ilayer][ispecies][iword][inohyphen][ikind] = \
                                {'x': M.clustered_activations[ilayer][bidx,0], \
                                 'y': M.clustered_activations[ilayer][bidx,1], \
                                 'color': colors }

    for ilayer in range(M.nlayers):
        for (ispecies,specie) in enumerate(M.species):
            for (iword,word) in enumerate(M.words):
                for (inohyphen,nohyphen) in enumerate(M.nohyphens):
                    for (ikind,kind) in enumerate(M.kinds):
                        if precomputed_hexs[ilayer][ispecies][iword][inohyphen][ikind] is not None:
                            cmax = max(precomputed_hexs[ilayer][ispecies][iword][inohyphen][ikind]['counts'])
                            precomputed_hexs[ilayer][ispecies][iword][inohyphen][ikind]['counts'] = precomputed_hexs[ilayer][ispecies][iword][inohyphen][ikind]['counts'].apply(lambda x: x/cmax)

    which_layer.options = M.layers
    which_species.options = M.species
    which_word.options = M.words
    which_nohyphen.options = M.nohyphens
    which_kind.options = M.kinds

    p_cluster_hexs.glyph.size = float(M.state["hex_size"])
    circle_radius_string.disabled=False
    hex_size_string.disabled=False
    dot_alpha.disabled=False
    cluster_style.disabled=False

    M.ilayer=0
    M.ispecies=0
    M.iword=0
    M.inohyphen=0
    M.ikind=0

def cluster_update():
    global cluster_hexs, cluster_dots, cluster_extent
    global p_cluster, p_cluster_xmax, p_cluster_xmin, p_cluster_ymax, p_cluster_ymin
    p_cluster_hexs.visible = False
    if M.state['cluster_style']=="hexs":
        cluster_dots.data.update(x=[], y=[], fill_color=[])
        hex_size_string.disabled=False
        dot_alpha.disabled=True
        if precomputed_hexs == None:
            return
        selected_hexs = precomputed_hexs[M.ilayer][M.ispecies][M.iword][M.inohyphen][M.ikind]
        if selected_hexs is not None:
            cluster_hexs.data.update(q=selected_hexs['q'], \
                                     r=selected_hexs['r'], \
                                     counts=selected_hexs['counts'], \
                                     index=range(len(selected_hexs['counts'])))
        else:
            cluster_hexs.data.update(q=[], r=[], counts=[], index=[])
        p_cluster_hexs.visible = True
    else:
        hex_size_string.disabled=True
        dot_alpha.disabled=False
        if precomputed_dots == None:
            return
        selected_dots = precomputed_dots[M.ilayer][M.ispecies][M.iword][M.inohyphen][M.ikind]
        if selected_dots is not None:
            cluster_dots.data.update(x=selected_dots['x'],
                                     y=selected_dots['y'],
                                     fill_color=selected_dots['color'])
        else:
            cluster_dots.data.update(x=[], y=[], fill_color=[])
        extent = min(p_cluster_xmax[M.ilayer] - p_cluster_xmin[M.ilayer],
                     p_cluster_ymax[M.ilayer] - p_cluster_ymin[M.ilayer])
        npoints = np.shape(M.clustered_activations[M.ilayer])[0]
        p_cluster_dots.radius = extent / np.sqrt(npoints)
    cluster_extent.data.update(x=[p_cluster_xmin[M.ilayer], p_cluster_xmax[M.ilayer]],
                               y=[p_cluster_ymin[M.ilayer], p_cluster_ymax[M.ilayer]])

def within_an_annotation(sample):
    if len(M.annotated_starts_sorted)>0:
        ifrom = np.searchsorted(M.annotated_starts_sorted, sample['ticks'][0],
                                side='right') - 1
        if 0 <= ifrom and ifrom < len(M.annotated_starts_sorted) and \
                    M.annotated_samples[ifrom]['ticks'][1] >= sample['ticks'][1]:
            return ifrom
    return -1

def snippets_update(redraw_wavs):
    if M.isnippet>0 and not np.isnan(M.xcluster) and not np.isnan(M.ycluster):
        quad_fuchsia_snippets.data.update(
                left=[M.xsnippet*(M.snippets_gap_pix+M.snippets_pix)],
                right=[(M.xsnippet+1)*(M.snippets_gap_pix+M.snippets_pix)-
                       M.snippets_gap_pix],
                top=[-M.ysnippet*2+1], bottom=[-M.ysnippet*2-1])
    else:
        quad_fuchsia_snippets.data.update(left=[], right=[], top=[], bottom=[])

    isubset = np.where([M.species[M.ispecies] in x['label'] and
                      M.words[M.iword] in x['label'] and
                      (M.nohyphens[M.inohyphen]=="" or \
                       M.nohyphens[M.inohyphen]==x['label']) and
                      M.kinds[M.ikind]==x['kind'] for x in M.clustered_samples])[0]
    distance = np.linalg.norm(M.clustered_activations[M.ilayer][isubset,:] - \
                                  [M.xcluster,M.ycluster], \
                              axis=1)
    isort = np.argsort(distance)
    songs, labels, labels_new, scales = [], [], [], []
    for isnippet in range(M.nx*M.ny):
        if isnippet<len(distance) and \
                    distance[isort[isnippet]] < float(M.state["circle_radius"]):
            M.nearest_samples[isnippet] = isubset[isort[isnippet]]
            thissample = M.clustered_samples[M.nearest_samples[isnippet]]
            labels.append(thissample['label'])
            midpoint = np.mean(thissample['ticks'], dtype=int)
            if redraw_wavs:
                _, wavs = spiowav.read(thissample['file'], mmap=True)
                if np.ndim(wavs)==1:
                  wavs = np.expand_dims(wavs, axis=1)
                start_frame = max(0, midpoint-M.snippets_tic//2)
                nframes_to_get = min(np.shape(wavs)[0] - start_frame,
                                     M.snippets_tic+1,
                                     M.snippets_tic+1+(midpoint-M.snippets_tic//2))
                left_pad = max(0, M.snippets_pix-nframes_to_get if start_frame==0 else 0)
                right_pad = max(0, M.snippets_pix-nframes_to_get if start_frame>0 else 0)
                song = [[]]*M.audio_nchannels
                scale = [[]]*M.audio_nchannels
                for ichannel in range(M.audio_nchannels):
                    wavi = wavs[start_frame : start_frame+nframes_to_get, ichannel]
                    wavi = decimate(wavi, M.snippets_decimate_by,
                                    n=M.filter_order,
                                    ftype='iir', zero_phase=True)
                    np.pad(wavi, ((left_pad, right_pad),),
                           'constant', constant_values=(np.nan,))
                    wavi = wavi[:M.snippets_pix]
                    scale[ichannel]=np.minimum(np.iinfo(np.int16).max-1,
                                               np.max(np.abs(wavi)))
                    song[ichannel]=wavi/scale[ichannel]
                songs.append(song)
                scales.append(scale)
            else:
                songs.append([[]])
            iannotated = within_an_annotation(thissample)
            if iannotated == -1:
                labels_new.append('')
            else:
                labels_new.append(M.annotated_samples[iannotated]['label'])
        else:
            M.nearest_samples[isnippet] = -1
            labels.append('')
            labels_new.append('')
            scales.append([0]*M.audio_nchannels)
            songs.append([np.full(M.snippets_pix,np.nan)]*M.audio_nchannels)
    label_sources.data.update(text=labels)
    label_sources_new.data.update(text=labels_new)
    left, right, top, bottom = [], [], [], []
    for (isong,song) in enumerate(songs):
        ix, iy = isong%M.nx, isong//M.nx
        if redraw_wavs:
            xdata = range(ix*(M.snippets_gap_pix+M.snippets_pix),
                          (ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            for ichannel in range(M.audio_nchannels):
                ydata = -iy*2 + \
                        (M.audio_nchannels-1-2*ichannel)/M.audio_nchannels + \
                        song[ichannel]/M.audio_nchannels
                wav_sources[isong][ichannel].data.update(x=xdata, y=ydata)
                ipalette = int(np.floor(scales[isong][ichannel] /
                                        np.iinfo(np.int16).max *
                                        len(snippet_palette)))
                line_glyphs[isong][ichannel].glyph.line_color = snippet_palette[ipalette]
        if labels_new[isong]!='':
            left.append(ix*(M.snippets_gap_pix+M.snippets_pix))
            right.append((ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            top.append(-iy*2+1)
            bottom.append(-iy*2-1)
    quad_grey_snippets.data.update(left=left, right=right, top=top, bottom=bottom)

    xcluster_last, ycluster_last = M.xcluster, M.ycluster

def nparray2base64wav(data, samplerate):
    fid=io.BytesIO()
    wav=wave.open(fid, "w")
    wav.setframerate(samplerate)
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.writeframes(data.tobytes())
    wav.close()
    fid.seek(0)
    ret_val = base64.b64encode(fid.read()).decode('utf-8')
    fid.close()
    return ret_val

def nparray2base64mp4(filename, start_sec, stop_sec):
    vid = pims.open(filename)

    start_frame = round(start_sec * vid.frame_rate).astype(np.int)
    stop_frame = round(stop_sec * vid.frame_rate).astype(np.int)

    fid=io.BytesIO()
    container = av.open(fid, mode='w', format='mp4')

    stream = container.add_stream('h264', rate=vid.frame_rate)
    stream.width = video_div.width = vid.frame_shape[0]
    stream.height = video_div.height = vid.frame_shape[1]
    stream.pix_fmt = 'yuv420p'

    for iframe in range(start_frame, stop_frame):
        frame = av.VideoFrame.from_ndarray(np.array(vid[iframe]), format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    fid.seek(0)
    ret_val = base64.b64encode(fid.read()).decode('utf-8')
    fid.close()
    return ret_val

# _context_update() might be able to be folded back in to context_update() with bokeh 2.0
# ditto for _doit_callback() and _groundtruth_update()
# see https://discourse.bokeh.org/t/bokeh-server-is-it-possible-to-push-updates-to-js-in-the-middle-of-a-python-callback/3455/4

def __context_update(wavi, tapped_sample, istart_bounded, ilength):
    if video_toggle.active:
        sample_basename=os.path.basename(tapped_sample)
        sample_dirname=os.path.dirname(tapped_sample)
        vids = list(filter(lambda x: x!=sample_basename and
                                     os.path.splitext(x)[0] == \
                                         os.path.splitext(sample_basename)[0] and
                                     os.path.splitext(x)[1].lower() in \
                                         ['.avi','.mp4','.mov'],
                           os.listdir(sample_dirname)))
        base64vid = nparray2base64mp4(os.path.join(sample_dirname,vids[0]),
                                      istart_bounded / M.audio_tic_rate,
                                      (istart_bounded+ilength) / M.audio_tic_rate) \
                    if len(vids)==1 else ""
        video_toggle.button_type="default"
    else:
        base64vid = ""

    play_callback.code = C.play_callback_code % \
                         (nparray2base64wav(wavi, M.audio_tic_rate), \
                          base64vid)

def _context_update(wavi, tapped_sample, istart_bounded, ilength):
    if video_toggle.active:
        video_toggle.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: \
            __context_update(wavi, tapped_sample, istart_bounded, ilength))

def context_update(highlight_tapped_snippet=True):
    p_context.title.text = ''
    tapped_ticks = [np.nan, np.nan]
    istart = np.nan
    scales = [0]*M.audio_nchannels
    ywavs = [np.full(1,np.nan)]*M.audio_nchannels
    xwavs = [[np.full(1,np.nan)]]*M.audio_nchannels
    xlabel, ylabel, tlabel = [], [], []
    xlabel_new, ylabel_new, tlabel_new = [], [], []
    left, right, top, bottom = [], [], [], []
    left_new, right_new, top_new, bottom_new = [], [], [], []

    if M.isnippet>=0:
        play.disabled=False
        video_toggle.disabled=False
        zoom_context.disabled=False
        zoom_offset.disabled=False
        zoomin.disabled=False
        zoomout.disabled=False
        reset.disabled=False
        panleft.disabled=False
        panright.disabled=False
        allleft.disabled=False
        allout.disabled=False
        allright.disabled=False
        tapped_sample = M.clustered_samples[M.isnippet]
        tapped_ticks = tapped_sample['ticks']
        M.context_midpoint_tic = np.mean(tapped_ticks, dtype=int)
        istart = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
        p_context.title.text = tapped_sample['file']
        _, wavs = spiowav.read(tapped_sample['file'], mmap=True)
        if np.ndim(wavs)==1:
            wavs = np.expand_dims(wavs, axis=1)
        M.file_nframes = np.shape(wavs)[0]
        if istart+M.context_width_tic>0 and istart<M.file_nframes:
            istart_bounded = np.maximum(0, istart)
            context_tic_adjusted = M.context_width_tic+1-(istart_bounded-istart)
            ilength = np.minimum(M.file_nframes-istart_bounded, context_tic_adjusted)

            tic2pix = M.context_width_tic / M.gui_width_pix
            context_decimate_by = round(tic2pix/M.filter_ratio_max) if \
                     tic2pix>M.filter_ratio_max else 1
            context_pix = round(M.context_width_tic / context_decimate_by)

            for ichannel in range(M.audio_nchannels):
                wavi = wavs[istart_bounded : istart_bounded+ilength, ichannel]
                if len(wavi)<M.context_width_tic+1:
                    npad = M.context_width_tic+1-len(wavi)
                    if istart<0:
                        wavi = np.concatenate((np.full((npad,),0), wavi))
                    if istart+M.context_width_tic>M.file_nframes:
                        wavi = np.concatenate((wavi, np.full((npad,),0)))

                if ichannel==0:
                    bokeh_document.add_next_tick_callback(lambda: \
                            _context_update(wavi,
                                            tapped_sample['file'],
                                            istart_bounded,
                                            ilength))

                wavi_downsampled = decimate(wavi, context_decimate_by, n=M.filter_order,
                                            ftype='iir', zero_phase=True)
                wavi_trimmed = wavi_downsampled[:context_pix]

                scales[ichannel]=np.minimum(np.iinfo(np.int16).max-1,
                                            np.max(np.abs(wavi_trimmed)))
                ywavs[ichannel]=wavi_trimmed/scales[ichannel]
                xwavs[ichannel]=[(istart+i*context_decimate_by)/M.audio_tic_rate \
                                 for i in range(len(wavi_trimmed))]
                if ichannel==0:
                    song_max = np.max(wavi_trimmed) / scales[ichannel]
                    song_max /= M.audio_nchannels
                    song_max += (M.audio_nchannels-1-2*ichannel)/M.audio_nchannels
                if ichannel==M.audio_nchannels-1:
                    song_min = np.min(wavi_trimmed) / scales[ichannel]
                    song_min /= M.audio_nchannels
                    song_min += (M.audio_nchannels-1-2*ichannel)/M.audio_nchannels

            line_red_context.data.update(x=[xwavs[0][0],xwavs[0][0]])
            p_line_red_context.visible=True

            ileft = np.searchsorted(M.clustered_starts_sorted, istart+M.context_width_tic)
            samples_to_plot = set(range(0,ileft))
            iright = np.searchsorted(M.clustered_stops, istart,
                                    sorter=M.iclustered_stops_sorted)
            samples_to_plot &= set([M.iclustered_stops_sorted[i] for i in \
                    range(iright, len(M.iclustered_stops_sorted))])

            tapped_wav_in_view = False
            for isample in samples_to_plot:
                if tapped_sample['file']!=M.clustered_samples[isample]['file']:
                    continue
                L = np.max([istart, M.clustered_samples[isample]['ticks'][0]])
                R = np.min([istart+M.context_width_tic,
                            M.clustered_samples[isample]['ticks'][1]])
                xlabel.append((L+R)/2/M.audio_tic_rate)
                tlabel.append(M.clustered_samples[isample]['kind']+'\n'+\
                              M.clustered_samples[isample]['label'])
                ylabel.append(song_max)
                left.append(L/M.audio_tic_rate)
                right.append(R/M.audio_tic_rate)
                top.append(song_max)
                bottom.append(0)
                if tapped_sample==M.clustered_samples[isample] and highlight_tapped_snippet:
                    quad_fuchsia_context.data.update(left=[L/M.audio_tic_rate],
                                                     right=[R/M.audio_tic_rate],
                                                     top=[song_max], bottom=[0])
                    tapped_wav_in_view = True

            if not tapped_wav_in_view:
                quad_fuchsia_context.data.update(left=[], right=[], top=[], bottom=[])

            if len(M.annotated_starts_sorted)>0:
                ileft = np.searchsorted(M.annotated_starts_sorted,
                                        istart+M.context_width_tic)
                samples_to_plot = set(range(0,ileft))
                iright = np.searchsorted(M.annotated_stops, istart,
                                         sorter=M.iannotated_stops_sorted)
                samples_to_plot &= set([M.iannotated_stops_sorted[i] for i in \
                        range(iright, len(M.iannotated_stops_sorted))])

                for isample in samples_to_plot:
                    iclustered = M.isclustered(M.annotated_samples[isample])
                    if len(iclustered)>0 and M.clustered_samples[iclustered[0]] == \
                                             M.annotated_samples[isample]:
                        continue
                    if tapped_sample['file']!=M.annotated_samples[isample]['file']:
                        continue
                    L = np.max([istart, M.annotated_samples[isample]['ticks'][0]])
                    R = np.min([istart+M.context_width_tic,
                                M.annotated_samples[isample]['ticks'][1]])
                    xlabel_new.append((L+R)/2/M.audio_tic_rate)
                    tlabel_new.append(M.annotated_samples[isample]['label'])
                    ylabel_new.append(song_min)
                    left_new.append(L/M.audio_tic_rate)
                    right_new.append(R/M.audio_tic_rate)
                    top_new.append(0)
                    bottom_new.append(song_min)
    else:
        play.disabled=True
        video_toggle.disabled=True
        zoom_context.disabled=True
        zoom_offset.disabled=True
        zoomin.disabled=True
        zoomout.disabled=True
        reset.disabled=True
        panleft.disabled=True
        panright.disabled=True
        allleft.disabled=True
        allout.disabled=True
        allright.disabled=True
        quad_fuchsia_context.data.update(left=[], right=[], top=[], bottom=[])
        line_red_context.data.update(x=[0,0])
        p_line_red_context.visible=False
        play_callback.code = C.play_callback_code % ("", "")

    for ichannel in range(M.audio_nchannels):
        xdata = xwavs[ichannel]
        ydata = ywavs[ichannel]/M.audio_nchannels + \
                (M.audio_nchannels-1-2*ichannel) / M.audio_nchannels
        wav_source[ichannel].data.update(x=xdata, y=ydata)
        ipalette = int(np.floor(scales[ichannel] /
                                np.iinfo(np.int16).max *
                                len(snippet_palette)))
        line_glyph[ichannel].glyph.line_color = snippet_palette[ipalette]
    quad_grey_context_old.data.update(left=left, right=right, top=top, bottom=bottom)
    quad_grey_context_new.data.update(left=left_new, right=right_new,
                                      top=top_new, bottom=bottom_new)
    label_source.data.update(x=xlabel, y=ylabel, text=tlabel)
    label_source_new.data.update(x=xlabel_new, y=ylabel_new, text=tlabel_new)

def save_update(n):
    save_indicator.label=str(n)
    if n==0:
        save_indicator.button_type="default"
    elif n<10:
        save_indicator.button_type="warning"
    else:
        save_indicator.button_type="danger"

def configuration_contents_update():
    if configuration_file.value:
        with open(configuration_file.value, 'r') as fid:
            configuration_contents.value = fid.read()

def model_file_update(attr, old, new):
    M.save_state_callback()
    buttons_update()

def groundtruth_update():
    groundtruth.button_type="warning"
    groundtruth.disabled=True
    bokeh_document.add_next_tick_callback(_groundtruth_update)

def _groundtruth_update():
    wordcounts_update()
    M.save_state_callback()
    groundtruth.button_type="default"
    groundtruth.disabled=True
    buttons_update()

def buttons_update():
    for button in wizard_buttons:
        button.button_type="success" if button==M.wizard else "default"
    for button in action_buttons:
        button.button_type="primary" if button==M.action else "default"
        button.disabled=False if button in wizard2actions[M.wizard] else True
    if M.action in [detect,classify]:
        wavtfcsvfiles.label='wav files'
    elif M.action==ethogram:
        wavtfcsvfiles.label='tf files'
    elif M.action==misses:
        wavtfcsvfiles.label='csv files'
    else:
        wavtfcsvfiles.label='wav,tf,csv files'
    for button in parameter_buttons:
        button.disabled=False if button in action2parameterbuttons[M.action] else True
    if M.wizard==findnovellabels:
        wantedwords = [x.value for x in label_text_widgets if x.value!='']
        for i in range(M.audio_nchannels):
            i_str = str(i) if M.audio_nchannels>1 else ''
            if 'time'+i_str not in wantedwords:
                wantedwords.append('time'+i_str)
            if 'frequency'+i_str not in wantedwords:
                wantedwords.append('frequency'+i_str)
        wantedwords_string.value=str.join(',',wantedwords)
        if M.action==train:
            labeltypes_string.value="annotated"
        elif M.action==activations:
            labeltypes_string.value="annotated,detected"
    okay=True if M.action else False
    for textinput in parameter_textinputs:
        if textinput in action2parametertextinputs[M.action]:
            if textinput==window_ms_string:
                window_ms_string.disabled=True \
                        if representation.value=='waveform' else False
            elif textinput==stride_ms_string:
                stride_ms_string.disabled=True \
                        if representation.value=='waveform' else False
            elif textinput==mel_dct_string:
                mel_dct_string.disabled=False \
                        if representation.value=='mel-cepstrum' else True
            elif textinput==pca_fraction_variance_to_retain_string:
                pca_fraction_variance_to_retain_string.disabled=False \
                        if cluster_algorithm.value[:5] in ['t-SNE','UMAP '] else True
            elif textinput==tsne_perplexity_string:
                tsne_perplexity_string.disabled=False \
                        if cluster_algorithm.value.startswith('t-SNE') else True
            elif textinput==tsne_exaggeration_string:
                tsne_exaggeration_string.disabled=False \
                        if cluster_algorithm.value.startswith('t-SNE') else True
            elif textinput==umap_neighbors_string:
                umap_neighbors_string.disabled=False \
                        if cluster_algorithm.value.startswith('UMAP') else True
            elif textinput==umap_distance_string:
                umap_distance_string.disabled=False \
                        if cluster_algorithm.value.startswith('UMAP') else True
            else:
                textinput.disabled=False
            if textinput.disabled==False and textinput.value=='' and \
                    textinput not in [testfiles_string, restore_from_string]:
                okay=False
        else:
            textinput.disabled=True
    if M.action==congruence and \
            (validationfiles_string.value!='' or testfiles_string.value!=''):
        okay=True
    doit.button_type="default"
    if okay:
        doit.disabled=False
        doit.button_type="danger"
    else:
        doit.disabled=True
        doit.button_type="default"

def file_dialog_update():
    thispath = os.path.join(M.file_dialog_root,M.file_dialog_filter)
    files = glob.glob(thispath)
    uniqdirnames = set([os.path.dirname(x) for x in files])
    files = natsorted(['.', '..', *files])
    if len(uniqdirnames)==1:
        names=[os.path.basename(x) + ('/' if os.path.isdir(x) else '') for x in files]
    else:
        names=[x + ('/' if os.path.isdir(x) else '') for x in files]
    file_dialog = dict(
        names=names,
        sizes=[os.path.getsize(f) for f in files],
        dates=[datetime.fromtimestamp(os.path.getmtime(f)) for f in files],
    )
    file_dialog_source.data = file_dialog

def wordcounts_update():
    if not os.path.isdir(groundtruth_folder.value):
        return
    dfs = []
    for subdir in filter(lambda x: os.path.isdir(os.path.join(groundtruth_folder.value,x)), \
                         os.listdir(groundtruth_folder.value)):
        for csvfile in filter(lambda x: x.endswith('.csv'), \
                              os.listdir(os.path.join(groundtruth_folder.value, subdir))):
            filepath = os.path.join(groundtruth_folder.value, subdir, csvfile)
            if os.path.getsize(filepath) > 0:
                dfs.append(pd.read_csv(filepath, header=None, index_col=False))
    if dfs:
        df = pd.concat(dfs)
        M.kinds = sorted(set(df[3]))
        bkinds = {}
        table_str = '<table><tr><th></th><th>'+'</th><th>'.join(M.kinds)+'</th></tr>'
        for word in sorted(set(df[4])):
            bword = np.array(df[4]==word)
            table_str += '<tr><th>'+word+'</th>'
            for kind in M.kinds:
                if kind not in bkinds:
                    bkinds[kind] = np.array(df[3]==kind)
                table_str += '<td align="center">'+str(np.sum(np.logical_and(bkinds[kind], bword)))+'</td>'
            table_str += '</tr>'
        table_str += '<tr><th>TOTAL</th>'
        for kind in M.kinds:
            table_str += '<td align="center">'+str(np.sum(bkinds[kind]))+'</td>'
        table_str += '</tr>'
        table_str += '</table>'
        wordcounts.text = table_str
    else:
        wordcounts.text = ""

from tornado import gen

@gen.coroutine
def status_ticker_update():
    if len(M.status_ticker_queue)>0:
        newtext = []
        for k in M.status_ticker_queue.keys():
            if M.status_ticker_queue[k]=="pending":
                color = "gray"
            elif M.status_ticker_queue[k]=="running":
                color = "black"
            elif M.status_ticker_queue[k]=="succeeded":
                color = "blue"
            elif M.status_ticker_queue[k]=="failed":
                color = "red"
            newtext.append("<span style='color:"+color+"'>"+k+"</span>")
        newtext = (', ').join(newtext)
    else:
        newtext = ''
    status_ticker.text = status_ticker_pre+newtext+status_ticker_post

def init(_bokeh_document, _cluster_background_color, _cluster_dot_colormap, _cluster_hex_colormap, _cluster_hex_range_low, _cluster_hex_range_high, _snippet_colormap):
    global bokeh_document, cluster_dot_palette, snippet_palette, p_cluster, cluster_hexs, cluster_dots, cluster_extent, p_cluster_hexs, p_cluster_dots, precomputed_hexs, precomputed_dots, p_snippets, label_sources, label_sources_new, wav_sources, line_glyphs, quad_grey_snippets, circle_fuchsia_cluster, p_cluster_circle, p_context, p_line_red_context, line_red_context, quad_grey_context_old, quad_grey_context_new, quad_grey_context_pan, quad_fuchsia_context, quad_fuchsia_snippets, wav_source, line_glyph, label_source, label_source_new, which_layer, which_species, which_word, which_nohyphen, which_kind, color_picker, circle_radius_string, hex_size_string, dot_alpha, cluster_style, zoom_context, zoom_offset, zoomin, zoomout, reset, panleft, panright, allleft, allout, allright, save_indicator, label_text_widgets, label_count_widgets, play, play_callback, video_toggle, video_div, undo, redo, detect, misses, configuration, configuration_file, train, leaveoneout, leaveallout, xvalidate, mistakes, activations, cluster, visualize, accuracy, freeze, classify, ethogram, compare, congruence, status_ticker, file_dialog_source, file_dialog_source, configuration_contents, logs, logs_folder, model, model_file, wavtfcsvfiles, wavtfcsvfiles_string, groundtruth, groundtruth_folder, validationfiles, testfiles, validationfiles_string, testfiles_string, wantedwords, wantedwords_string, labeltypes, labeltypes_string, copy, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, generalize, tunehyperparameters, findnovellabels, examineerrors, testdensely, doit, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string, nsteps_string, restore_from_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string, kfold_string, activations_equalize_ratio_string, activations_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, umap_neighbors_string, umap_distance_string, cluster_algorithm, connection_type, precision_recall_ratios_string, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, editconfiguration, file_dialog_string, file_dialog_table, readme_contents, wordcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs, status_ticker_update, status_ticker_pre, status_ticker_post

    bokeh_document = _bokeh_document

    if '#' in _cluster_dot_colormap:
      cluster_dot_palette = ast.literal_eval(_cluster_dot_colormap)
    else:
      cluster_dot_palette = getattr(palettes, _cluster_dot_colormap)

    snippet_palette = getattr(palettes, _snippet_colormap)

    p_cluster = figure(plot_width=M.gui_width_pix//2, \
                       background_fill_color=_cluster_background_color, \
                       toolbar_location=None, match_aspect=True)
    p_cluster.toolbar.active_drag = None
    p_cluster.grid.visible = False
    p_cluster.xaxis.visible = False
    p_cluster.yaxis.visible = False
     
    cluster_hexs = ColumnDataSource(hexbin(np.array([]), np.array([]), 1))
    init_fill_color = linear_cmap('counts', _cluster_hex_colormap,
                                  float(_cluster_hex_range_low),
                                  float(_cluster_hex_range_high))
    p_cluster_hexs = p_cluster.hex_tile(q="q", r="r", size=1,
                                        line_color=None, source=cluster_hexs,
                                        fill_color=init_fill_color)
    p_cluster_hexs.visible = False

    cluster_dots = ColumnDataSource(data=dict(x=[], y=[], fill_color=[]))
    p_cluster_dots = Circle(x='x', y='y', fill_color='fill_color',
                            fill_alpha=M.state["dot_alpha"], line_width=0)
    p_cluster.add_glyph(cluster_dots, p_cluster_dots)

    cluster_extent = ColumnDataSource(data=dict(x=[], y=[]))
    p_cluster_extent = p_cluster.line('x','y',source=cluster_extent, \
                                      level='underlay', color=_cluster_background_color)
    p_cluster_extent.visible = True

    precomputed_hexs = None
    precomputed_dots = None

    p_snippets = figure(plot_width=M.gui_width_pix//2, \
                        background_fill_color='#FFFFFF', toolbar_location=None)
    p_snippets.toolbar.active_drag = None
    p_snippets.grid.visible = False
    p_snippets.xaxis.visible = False
    p_snippets.yaxis.visible = False

    xdata = [(i%M.nx)*(M.snippets_gap_pix+M.snippets_pix) for i in range(M.nx*M.ny)]
    ydata = [-(i//M.nx*2-1) for i in range(M.nx*M.ny)]
    text = ['' for i in range(M.nx*M.ny)]
    label_sources = ColumnDataSource(data=dict(x=xdata, y=ydata, text=text))
    p_snippets.text('x', 'y', source=label_sources, text_font_size='6pt',
                    text_baseline='top')

    xdata = [(i%M.nx)*(M.snippets_gap_pix+M.snippets_pix) for i in range(M.nx*M.ny)]
    ydata = [-(i//M.nx*2+1) for i in range(M.nx*M.ny)]
    text_new = ['' for i in range(M.nx*M.ny)]
    label_sources_new = ColumnDataSource(data=dict(x=xdata, y=ydata, text=text_new))
    p_snippets.text('x', 'y', source=label_sources_new, text_font_size='6pt')

    wav_sources=[None]*(M.nx*M.ny)
    line_glyphs=[None]*(M.nx*M.ny)
    for ixy in range(M.nx*M.ny):
        wav_sources[ixy]=[None]*M.audio_nchannels
        line_glyphs[ixy]=[None]*M.audio_nchannels
        for ichannel in range(M.audio_nchannels):
            wav_sources[ixy][ichannel]=ColumnDataSource(data=dict(x=[], y=[]))
            line_glyphs[ixy][ichannel]=p_snippets.line('x', 'y',
                                                       source=wav_sources[ixy][ichannel])

    quad_grey_snippets = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=quad_grey_snippets,
                fill_color="lightgrey", line_color="lightgrey", level='underlay')

    circle_fuchsia_cluster = ColumnDataSource(data=dict(x=[], y=[]))
    p_cluster_circle = p_cluster.circle('x','y',source=circle_fuchsia_cluster, \
                                  radius=float(M.state["circle_radius"]),
                                  fill_color=None, line_color="fuchsia", level='overlay')

    p_cluster.on_event(Tap, C.cluster_tap_callback)

    p_context = figure(plot_width=M.gui_width_pix, plot_height=150,
                       background_fill_color='#FFFFFF', toolbar_location=None)
    p_context.toolbar.active_drag = None
    p_context.grid.visible = False
    p_context.xaxis.axis_label = 'time (sec)'
    p_context.yaxis.visible = False
    p_context.x_range.range_padding = 0.0
    p_context.title.text=' '

    line_red_context = ColumnDataSource(data=dict(x=[0,0], y=[-1,0]))
    p_line_red_context = p_context.line('x','y',source=line_red_context, color="red")
    p_line_red_context.visible=False

    quad_grey_context_old = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_grey_context_old,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    quad_grey_context_new = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_grey_context_new,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    quad_grey_context_pan = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_grey_context_pan,
                fill_color="lightgrey", fill_alpha=0.5, line_color="lightgrey",
                level='underlay')
    quad_fuchsia_context = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_context.quad('left','right','top','bottom',source=quad_fuchsia_context,
                fill_color=None, line_color="fuchsia", level='underlay')
    quad_fuchsia_snippets = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=quad_fuchsia_snippets,
                fill_color=None, line_color="fuchsia", level='underlay')

    wav_source=[None]*M.audio_nchannels
    line_glyph=[None]*M.audio_nchannels
    for ichannel in range(M.audio_nchannels):
        wav_source[ichannel] = ColumnDataSource(data=dict(x=[], y=[]))
        line_glyph[ichannel] = p_context.line('x', 'y', source=wav_source[ichannel])

    label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_context.text('x', 'y', source=label_source,
                   text_font_size='6pt', text_align='center', text_baseline='top',
                   text_line_height=0.8, level='underlay')
    label_source_new = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_context.text('x', 'y', source=label_source_new,
                   text_font_size='6pt', text_align='center', text_baseline='bottom',
                   text_line_height=0.8, level='underlay')

    p_snippets.on_event(Tap, C.snippets_tap_callback)

    p_context.on_event(DoubleTap, C.context_doubletap_callback)

    p_context.on_event(PanStart, C.context_pan_start_callback)
    p_context.on_event(Pan, C.context_pan_callback)
    p_context.on_event(PanEnd, C.context_pan_end_callback)

    p_snippets.on_event(DoubleTap, C.snippets_doubletap_callback)

    which_layer = Select(title="layer:")
    which_layer.on_change('value', lambda a,o,n: C.layer_callback(n))

    which_species = Select(title="species:")
    which_species.on_change('value', lambda a,o,n: C.species_callback(n))

    which_word = Select(title="word:")
    which_word.on_change('value', lambda a,o,n: C.word_callback(n))

    which_nohyphen = Select(title="no hyphen:")
    which_nohyphen.on_change('value', lambda a,o,n: C.nohyphen_callback(n))

    which_kind = Select(title="kind:")
    which_kind.on_change('value', lambda a,o,n: C.kind_callback(n))

    color_picker = ColorPicker(disabled=True)
    color_picker.on_change("color", lambda a,o,n: C.color_picker_callback(n))

    circle_radius_string = TextInput(value=M.state["circle_radius"], \
                                     title="circle radius:", \
                                     disabled=True)
    circle_radius_string.on_change("value", C.circle_radius_callback)

    hex_size_string = TextInput(value=M.state["hex_size"], \
                                title="hex size:", \
                                disabled=True)
    hex_size_string.on_change("value", C.hex_size_callback)

    dot_alpha = Slider(start=0.01, end=1.0, step=0.01, \
                       value=M.state["dot_alpha"], \
                       title="dot alpha", \
                       disabled=True)
    dot_alpha.on_change("value", C.dot_alpha_callback)

    cluster_style = RadioButtonGroup(labels=["hexs", "dots"], \
                                     active=0 if M.state['cluster_style']=='hexs' else 1, \
                                     disabled=True)
    cluster_style.on_change("active", C.cluster_style_callback)

    cluster_update()

    zoom_context = TextInput(value=str(M.context_width_ms),
                             title="context (msec):",
                             disabled=True)
    zoom_context.on_change("value", C.zoom_context_callback)

    zoom_offset = TextInput(value=str(M.context_offset_ms),
                            title="offset (msec):",
                            disabled=True)
    zoom_offset.on_change("value", C.zoom_offset_callback)

    zoomin = Button(label='\u2191', disabled=True, width=40)
    zoomin.on_click(C.zoomin_callback)

    zoomout = Button(label='\u2193', disabled=True, width=40)
    zoomout.on_click(C.zoomout_callback)

    reset = Button(label='\u25ef', disabled=True, width=40)
    reset.on_click(C.zero_callback)

    panleft = Button(label='\u2190', disabled=True, width=40)
    panleft.on_click(C.panleft_callback)

    panright = Button(label='\u2192', disabled=True, width=40)
    panright.on_click(C.panright_callback)

    allleft = Button(label='\u21e4', disabled=True, width=40)
    allleft.on_click(C.allleft_callback)

    allout = Button(label='\u2913', disabled=True, width=40)
    allout.on_click(C.allout_callback)

    allright = Button(label='\u21e5', disabled=True, width=40)
    allright.on_click(C.allright_callback)

    save_indicator = Button(label='0', width=40)

    label_text_callbacks=[]
    label_text_widgets=[]
    label_count_callbacks=[]
    label_count_widgets=[]

    for i in range(M.nlabels):
        label_text_callbacks.append(lambda a,o,n,i=i: C.label_text_callback(n,i))
        label_text_widgets.append(TextInput(value=M.state['labels'][i],
                                            css_classes=['hide-label']))
        label_text_widgets[-1].on_change("value", label_text_callbacks[-1])
        label_count_callbacks.append(lambda i=i: C.label_count_callback(i))
        label_count_widgets.append(Button(label='0', css_classes=['hide-label'], width=40))
        label_count_widgets[-1].on_click(label_count_callbacks[-1])

    C.label_count_callback(M.ilabel)

    play = Button(label='play', disabled=True)
    play_callback = CustomJS(args=dict(line_red_context=line_red_context),
                                 code=C.play_callback_code % ("",""))
    play.js_on_event(ButtonClick, play_callback)

    video_toggle = Toggle(label='video', active=False, disabled=True)
    video_toggle.on_click(context_update)

    video_div = Div(text="""<video id="context_video"></video>""", width=0, height=0)

    undo = Button(label='undo', disabled=True)
    undo.on_click(C.undo_callback)

    redo = Button(label='redo', disabled=True)
    redo.on_click(C.redo_callback)

    detect = Button(label='detect')
    detect.on_click(lambda: C.action_callback(detect, C.detect_actuate))

    misses = Button(label='misses')
    misses.on_click(lambda: C.action_callback(misses, C.misses_actuate))

    configuration = Button(label='configuration:', width=110)
    configuration.on_click(C.configuration_button_callback)

    configuration_file = TextInput(value=M.state['configuration'], title="", disabled=False)
    configuration_file.on_change('value', C.configuration_text_callback)

    train = Button(label='train')
    train.on_click(lambda: C.action_callback(train, C.train_actuate))

    leaveoneout = Button(label='omit one')
    leaveoneout.on_click(lambda: C.action_callback(leaveoneout,
                                                   lambda: C.leaveout_actuate(False)))

    leaveallout = Button(label='omit all')
    leaveallout.on_click(lambda: C.action_callback(leaveallout,
                                                   lambda: C.leaveout_actuate(True)))

    xvalidate = Button(label='x-validate')
    xvalidate.on_click(lambda: C.action_callback(xvalidate, C.xvalidate_actuate))

    mistakes = Button(label='mistakes')
    mistakes.on_click(lambda: C.action_callback(mistakes, C.mistakes_actuate))

    activations = Button(label='activations')
    activations.on_click(lambda: C.action_callback(activations, C.activations_actuate))

    cluster = Button(label='cluster')
    cluster.on_click(lambda: C.action_callback(cluster, C.cluster_actuate))

    visualize = Button(label='visualize')
    visualize.on_click(lambda: C.action_callback(visualize, C.visualize_actuate))

    accuracy = Button(label='accuracy')
    accuracy.on_click(lambda: C.action_callback(accuracy, C.accuracy_actuate))

    freeze = Button(label='freeze')
    freeze.on_click(lambda: C.action_callback(freeze, C.freeze_actuate))

    classify = Button(label='classify')
    classify.on_click(lambda: C.action_callback(classify, C.classify_actuate))

    ethogram = Button(label='ethogram')
    ethogram.on_click(lambda: C.action_callback(ethogram, C.ethogram_actuate))

    compare = Button(label='compare')
    compare.on_click(lambda: C.action_callback(compare, C.compare_actuate))

    congruence = Button(label='congruence')
    congruence.on_click(lambda: C.action_callback(congruence, C.congruence_actuate))

    status_ticker_pre="<div style='overflow:auto; white-space:nowrap; width:"+str(M.gui_width_pix-10)+"px'>status: "
    status_ticker_post="</div>"
    status_ticker = Div(text=status_ticker_pre+status_ticker_post)

    file_dialog_source = ColumnDataSource(data=dict(names=[], sizes=[], dates=[]))
    file_dialog_source.selected.on_change('indices', C.file_dialog_callback)

    file_dialog_columns = [
        TableColumn(field="names", title="Name", width=M.gui_width_pix//2-50-115-10),
        TableColumn(field="sizes", title="Size", width=50, \
                    formatter=NumberFormatter(format="0 b")),
        TableColumn(field="dates", title="Date", width=115, \
                    formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
    ]
    file_dialog_table = DataTable(source=file_dialog_source, \
                                  columns=file_dialog_columns, \
                                  height=660, width=M.gui_width_pix//2-10, \
                                  index_position=None,
                                  fit_columns=False)

    configuration_contents = TextAreaInput(rows=46, max_length=50000, \
                                        disabled=True, css_classes=['fixedwidth'])
    configuration_contents_update()
    configuration_contents.on_change('value', C.configuration_textarea_callback)

    logs = Button(label='logs folder:', width=110)
    logs.on_click(C.logs_callback)
    logs_folder = TextInput(value=M.state['logs'], title="", disabled=False)
    logs_folder.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    model = Button(label='model:', width=110)
    model.on_click(C.model_callback)
    model_file = TextInput(value=M.state['model'], title="", disabled=False)
    model_file.on_change('value', model_file_update)

    wavtfcsvfiles = Button(label='wav,tf,csv files:', width=110)
    wavtfcsvfiles.on_click(C.wavtfcsvfiles_callback)
    wavtfcsvfiles_string = TextInput(value=M.state['wavtfcsvfiles'], title="", disabled=False)
    wavtfcsvfiles_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    groundtruth = Button(label='ground truth:', width=110)
    groundtruth.on_click(C.groundtruth_callback)
    groundtruth_folder = TextInput(value=M.state['groundtruth'], title="", disabled=False)
    groundtruth_folder.on_change('value', lambda a,o,n: groundtruth_update())

    validationfiles = Button(label='validation files:', width=110)
    validationfiles.on_click(C.validationfiles_callback)
    validationfiles_string = TextInput(value=M.state['validationfiles'], title="", disabled=False)
    validationfiles_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    testfiles = Button(label='test files:', width=110)
    testfiles.on_click(C.testfiles_callback)
    testfiles_string = TextInput(value=M.state['testfiles'], title="", disabled=False)
    testfiles_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    wantedwords = Button(label='wanted words:', width=110)
    wantedwords.on_click(C.wantedwords_callback)
    wantedwords_string = TextInput(value=M.state['wantedwords'], title="", disabled=False)
    wantedwords_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    labeltypes = Button(label='label types:', width=110)
    labeltypes_string = TextInput(value=M.state['labeltypes'], title="", disabled=False)
    labeltypes_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    copy = Button(label='copy')
    copy.on_click(C.copy_callback)

    labelsounds = Button(label='label sounds')
    labelsounds.on_click(C.labelsounds_callback)

    makepredictions = Button(label='make predictions')
    makepredictions.on_click(C.makepredictions_callback)

    fixfalsepositives = Button(label='fix false positives')
    fixfalsepositives.on_click(C.fixfalsepositives_callback)

    fixfalsenegatives = Button(label='fix false negatives')
    fixfalsenegatives.on_click(C.fixfalsenegatives_callback)

    generalize = Button(label='test generalization')
    generalize.on_click(C.generalize_callback)

    tunehyperparameters = Button(label='tune h-parameters')
    tunehyperparameters.on_click(C.tunehyperparameters_callback)

    findnovellabels = Button(label='find novel labels')
    findnovellabels.on_click(C.findnovellabels_callback)

    examineerrors = Button(label='examine errors')
    examineerrors.on_click(C.examineerrors_callback)

    testdensely = Button(label='test densely')
    testdensely .on_click(C.testdensely_callback)

    doit = Button(label='do it!', disabled=True)
    doit.on_click(C.doit_callback)

    time_sigma_string = TextInput(value=M.state['time_sigma'], \
                                  title="time σ", \
                                  disabled=False)
    time_sigma_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    time_smooth_ms_string = TextInput(value=M.state['time_smooth_ms'], \
                                      title="time smooth", \
                                      disabled=False)
    time_smooth_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    frequency_n_ms_string = TextInput(value=M.state['frequency_n_ms'], \
                                      title="freq N (msec)", \
                                      disabled=False)
    frequency_n_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    frequency_nw_string = TextInput(value=M.state['frequency_nw'], \
                                    title="freq NW", \
                                    disabled=False)
    frequency_nw_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    frequency_p_string = TextInput(value=M.state['frequency_p'], \
                                   title="freq ρ", \
                                   disabled=False)
    frequency_p_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    frequency_smooth_ms_string = TextInput(value=M.state['frequency_smooth_ms'], \
                                           title="freq smooth", \
                                           disabled=False)
    frequency_smooth_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    nsteps_string = TextInput(value=M.state['nsteps'], title="# steps", disabled=False)
    nsteps_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    restore_from_string = TextInput(value=M.state['restore_from'], title="restore from", disabled=False)
    restore_from_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    save_and_validate_period_string = TextInput(value=M.state['save_and_validate_interval'], \
                                                title="validate period", \
                                                disabled=False)
    save_and_validate_period_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    validate_percentage_string = TextInput(value=M.state['validate_percentage'], \
                                           title="validate %", \
                                           disabled=False)
    validate_percentage_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    mini_batch_string = TextInput(value=M.state['mini_batch'], \
                                  title="mini-batch", \
                                  disabled=False)
    mini_batch_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    kfold_string = TextInput(value=M.state['kfold'], title="k-fold",  disabled=False)
    kfold_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    activations_equalize_ratio_string = TextInput(value=M.state['activations_equalize_ratio'], \
                                             title="equalize ratio", \
                                             disabled=False)
    activations_equalize_ratio_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    activations_max_samples_string = TextInput(value=M.state['activations_max_samples'], \
                                          title="max samples", \
                                          disabled=False)
    activations_max_samples_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    pca_fraction_variance_to_retain_string = TextInput(value=M.state['pca_fraction_variance_to_retain'], \
                                                       title="PCA fraction", \
                                                       disabled=False)
    pca_fraction_variance_to_retain_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    tsne_perplexity_string = TextInput(value=M.state['tsne_perplexity'], \
                                       title="perplexity", \
                                       disabled=False)
    tsne_perplexity_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    tsne_exaggeration_string = TextInput(value=M.state['tsne_exaggeration'], \
                                        title="exaggeration", \
                                        disabled=False)
    tsne_exaggeration_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    umap_neighbors_string = TextInput(value=M.state['umap_neighbors'], \
                                      title="neighbors", \
                                      disabled=False)
    umap_neighbors_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    umap_distance_string = TextInput(value=M.state['umap_distance'], \
                                     title="distance", \
                                     disabled=False)
    umap_distance_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    precision_recall_ratios_string = TextInput(value=M.state['precision_recall_ratios'], \
                                               title="P/Rs", \
                                               disabled=False)
    precision_recall_ratios_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    context_ms_string = TextInput(value=M.state['context_ms'], \
                                  title="context (msec)", \
                                  disabled=False)
    context_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    shiftby_ms_string = TextInput(value=M.state['shiftby_ms'], \
                                  title="shift by (msec)", \
                                  disabled=False)
    shiftby_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    representation = Select(title="representation", height=50, \
                            value=M.state['representation'], \
                            options=["waveform", "spectrogram", "mel-cepstrum"])
    representation.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    cluster_algorithm = Select(title="cluster", height=50, \
                               value=M.state['cluster_algorithm'], \
                               options=["PCA 2D", "PCA 3D", \
                                        "t-SNE 2D", "t-SNE 3D", \
                                        "UMAP 2D", "UMAP 3D"])
    cluster_algorithm.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    connection_type = Select(title="connection", height=50, \
                             value=M.state['connection_type'], \
                             options=["plain", "residual"])
    connection_type.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    window_ms_string = TextInput(value=M.state['window_ms'], \
                                 title="window (msec)", \
                                 disabled=False)
    window_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    stride_ms_string = TextInput(value=M.state['stride_ms'], \
                                 title="stride (msec)", \
                                 disabled=False)
    stride_ms_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    mel_dct_string = TextInput(value=M.state['mel&dct'], \
                               title="Mel & DCT", \
                               disabled=False)
    mel_dct_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    dropout_string = TextInput(value=M.state['dropout'], \
                               title="dropout", \
                               disabled=False)
    dropout_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    optimizer = Select(title="optimizer", height=50, \
                       value=M.state['optimizer'], \
                       options=[("sgd","SGD"), ("adam","Adam"), ("adagrad","AdaGrad"), ("rmsprop","RMSProp")])
    optimizer.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    learning_rate_string = TextInput(value=M.state['learning_rate'], \
                                     title="learning rate", \
                                     disabled=False)
    learning_rate_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    kernel_sizes_string = TextInput(value=M.state['kernel_sizes'], \
                                    title="kernels", \
                                    disabled=False)
    kernel_sizes_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    last_conv_width_string = TextInput(value=M.state['last_conv_width'], \
                                       title="last conv width", \
                                       disabled=False)
    last_conv_width_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    nfeatures_string = TextInput(value=M.state['nfeatures'], \
                                 title="# features", \
                                 disabled=False)
    nfeatures_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    dilate_after_layer_string = TextInput(value=M.state['dilate_after_layer'], \
                                          title="dilate after", \
                                          disabled=False)
    dilate_after_layer_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    stride_after_layer_string = TextInput(value=M.state['stride_after_layer'], \
                                          title="stride after", \
                                          disabled=False)
    stride_after_layer_string.on_change('value', lambda a,o,n: C.generic_parameters_callback())

    editconfiguration = Button(label='edit', button_type="default")
    editconfiguration.on_click(C.editconfiguration_callback)

    file_dialog_string = TextInput(disabled=False)
    file_dialog_string.on_change("value", C.file_dialog_path_callback)
    file_dialog_string.value = M.state['file_dialog_string']
     
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','README.md'), 'r', encoding='utf-8') as fid:
        contents = fid.read()
    html = markdown.markdown(contents, extensions=['tables','toc'])
    readme_contents = Div(text=html, style={'overflow':'scroll','width':'600px','height':'1390px'})

    wordcounts = Div(text="")
    wordcounts_update()

    wizard_buttons = set([
        labelsounds,
        makepredictions,
        fixfalsepositives,
        fixfalsenegatives,
        generalize,
        tunehyperparameters,
        findnovellabels,
        examineerrors,
        testdensely])

    action_buttons = set([
        detect,
        train,
        leaveoneout,
        leaveallout,
        xvalidate,
        mistakes,
        activations,
        cluster,
        visualize,
        accuracy,
        freeze,
        classify,
        ethogram,
        misses,
        compare,
        congruence])

    parameter_buttons = set([
        configuration,
        logs,
        model,
        wavtfcsvfiles,
        groundtruth,
        validationfiles,
        testfiles,
        wantedwords,
        labeltypes])

    parameter_textinputs = set([
        configuration_file,
        logs_folder,
        model_file,
        wavtfcsvfiles_string,
        groundtruth_folder,
        validationfiles_string,
        testfiles_string,
        wantedwords_string,
        labeltypes_string,

        time_sigma_string,
        time_smooth_ms_string,
        frequency_n_ms_string,
        frequency_nw_string,
        frequency_p_string,
        frequency_smooth_ms_string,
        nsteps_string,
        restore_from_string,
        save_and_validate_period_string,
        validate_percentage_string,
        mini_batch_string,
        kfold_string,
        activations_equalize_ratio_string,
        activations_max_samples_string,
        pca_fraction_variance_to_retain_string,
        tsne_perplexity_string,
        tsne_exaggeration_string,
        umap_neighbors_string,
        umap_distance_string,
        cluster_algorithm,
        connection_type,
        precision_recall_ratios_string,
        context_ms_string,
        shiftby_ms_string,
        representation,
        window_ms_string,
        stride_ms_string,
        mel_dct_string,
        dropout_string,
        optimizer,
        learning_rate_string,
        kernel_sizes_string,
        last_conv_width_string,
        nfeatures_string,
        dilate_after_layer_string,
        stride_after_layer_string])

    wizard2actions = {
            labelsounds: [detect,train,activations,cluster,visualize],
            makepredictions: [train, accuracy, freeze, classify, ethogram],
            fixfalsepositives: [activations, cluster, visualize],
            fixfalsenegatives: [detect, misses, activations, cluster, visualize],
            generalize: [leaveoneout, leaveallout, accuracy],
            tunehyperparameters: [xvalidate, accuracy, compare],
            findnovellabels: [detect, train, activations, cluster, visualize],
            examineerrors: [detect, mistakes, activations, cluster, visualize],
            testdensely: [train, leaveoneout, leaveallout, xvalidate, accuracy, freeze, classify, ethogram, congruence],
            None: action_buttons }

    action2parameterbuttons = {
            detect: [configuration,wavtfcsvfiles],
            train: [configuration, logs, groundtruth, wantedwords, testfiles, labeltypes],
            leaveoneout: [configuration, logs, groundtruth, validationfiles, testfiles, wantedwords, labeltypes],
            leaveallout: [configuration, logs, groundtruth, validationfiles, testfiles, wantedwords, labeltypes],
            xvalidate: [configuration, logs, groundtruth, testfiles, wantedwords, labeltypes],
            mistakes: [configuration, groundtruth],
            activations: [configuration, logs, model, groundtruth, wantedwords, labeltypes],
            cluster: [configuration, groundtruth],
            visualize: [groundtruth],
            accuracy: [configuration, logs],
            freeze: [configuration, logs, model],
            classify: [configuration, logs, model, wavtfcsvfiles],
            ethogram: [configuration, model, wavtfcsvfiles],
            misses: [configuration, wavtfcsvfiles],
            compare: [configuration, logs],
            congruence: [configuration, groundtruth, validationfiles, testfiles],
            None: parameter_buttons }

    action2parametertextinputs = {
            detect: [configuration_file, wavtfcsvfiles_string, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string],
            train: [configuration_file, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string],
            leaveoneout: [configuration_file, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, validationfiles_string, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, mini_batch_string],
            leaveallout: [configuration_file, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, validationfiles_string, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, mini_batch_string],
            xvalidate: [configuration_file, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, groundtruth_folder, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, restore_from_string, save_and_validate_period_string, mini_batch_string, kfold_string],
            mistakes: [configuration_file, groundtruth_folder],
            activations: [configuration_file, context_ms_string, shiftby_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, model_file, groundtruth_folder, wantedwords_string, labeltypes_string, activations_equalize_ratio_string, activations_max_samples_string, mini_batch_string],
            cluster: [configuration_file, groundtruth_folder, cluster_algorithm, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, umap_neighbors_string, umap_distance_string],
            visualize: [groundtruth_folder],
            accuracy: [configuration_file, logs_folder, precision_recall_ratios_string],
            freeze: [configuration_file, context_ms_string, representation, window_ms_string, stride_ms_string, mel_dct_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, dilate_after_layer_string, stride_after_layer_string, connection_type, logs_folder, model_file],
            classify: [configuration_file, context_ms_string, shiftby_ms_string, representation, stride_ms_string, logs_folder, model_file, wavtfcsvfiles_string],
            ethogram: [configuration_file, model_file, wavtfcsvfiles_string],
            misses: [configuration_file, wavtfcsvfiles_string],
            compare: [configuration_file, logs_folder],
            congruence: [configuration_file, groundtruth_folder, validationfiles_string, testfiles_string],
            None: parameter_textinputs }