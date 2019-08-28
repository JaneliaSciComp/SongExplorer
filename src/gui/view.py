import os
from bokeh.models.widgets import RadioButtonGroup, TextInput, Button, Div, DateFormatter, TextAreaInput, Select
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.plotting import figure
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap
from bokeh.events import Tap, DoubleTap, PanStart, Pan, PanEnd
import numpy as np
import glob
from datetime import datetime
import markdown
import pandas as pd
import wave
from scipy.signal import decimate
import logging 

log = logging.getLogger("deepsong-view") 

import model as M
import controller as C

p_tsne, hex_bins, p_snippets, label_sources, label_sources_new, wav_sources, line_glyphs, quad_grey_snippets, circle_fuchsia_tsne, p_tsne_circle, p_context, quad_grey_context_old, quad_grey_context_new, quad_grey_context_pan, quad_fuchsia_context, quad_fuchsia_snippets, wav_source, line_glyph, label_source, label_source_new, which_layer, which_species, which_word, which_nohyphen, which_kind, radius_size, zoom_context, zoom_offset, zoomin, zoomout, reset, panleft, panright, save_indicator, label_text_widgets, label_count_widgets, undo, redo, detect, misses, configuration, configuration_file, train, generalize, xvalidate, hidden, cluster, visualize, accuracy, freeze, classify, ethogram, compare, dense, file_dialog_source, file_dialog_source, configuration_contents, logs, logs_folder, model, model_file, wavtfcsvfiles, wavtfcsvfiles_string, groundtruth, groundtruth_folder, validationfiles, testfiles, validationfiles_string, testfiles_string, wantedwords, wantedwords_string, labeltypes, labeltypes_string, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, leaveoneout, tunehyperparameters, findnovellabels, examineerrors, doit, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string, nsteps_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string, kfold_string, cluster_equalize_ratio_string, cluster_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, precision_recall_ratios_string, context_ms_string, shiftby_ms_string, window_ms_string, mel_dct_string, stride_ms_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, editconfiguration, file_dialog_string, file_dialog_table, readme_contents, wordcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs = [None]*122

def tsne_initialize():
    global precomputed_hex_bins 
    global p_tsne_qmax, p_tsne_qmin, p_tsne_rmax, p_tsne_rmin

    npzfile = np.load(os.path.join(groundtruth_folder.value,'cluster.npz'))
    M.clustered_samples = npzfile['samples']
    M.clustered_hidden = npzfile['hidden_clustered']
    #norm_hidden_pca = []
    #for arr in sorted(filter(lambda x: x.startswith('arr_'), list(npzfile.keys()))):
    #  norm_hidden_pca.append(npzfile[arr])

    M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_samples]
    isort = np.argsort(M.clustered_starts_sorted)
    for i in range(len(M.clustered_hidden)):
        M.clustered_hidden[i] = [M.clustered_hidden[i][x] for x in isort]
    M.clustered_samples = [M.clustered_samples[x] for x in isort]
    M.clustered_starts_sorted = [M.clustered_starts_sorted[x] for x in isort]

    M.clustered_stops = [x['ticks'][1] for x in M.clustered_samples]
    M.iclustered_stops_sorted = np.argsort(M.clustered_stops)

    tsne_isnotnan = [not np.isnan(x[0]) and not np.isnan(x[1]) for x in M.clustered_hidden[0]]

    M.nlayers=len(M.clustered_hidden)

    M.layers = ["layer "+str(i) for i in range(M.nlayers)]
    M.species = set([x['label'].split('-')[0]+'-' for x in M.clustered_samples if '-' in x['label']])
    M.species |= set([''])
    M.species = sorted(list(M.species))
    M.words = set(['-'+x['label'].split('-')[1] for x in M.clustered_samples if '-' in x['label']])
    M.words |= set([''])
    M.words = sorted(list(M.words))
    M.nohyphens = set([x['label'] for x in M.clustered_samples if '-' not in x['label']])
    M.nohyphens |= set([''])
    M.nohyphens = sorted(list(M.nohyphens))
    M.kinds = sorted(list(set([x['kind'] for x in M.clustered_samples])))

    circle_fuchsia_tsne.data.update(x=[], y=[])
    M.xtsne, M.ytsne = np.nan, np.nan
    M.isnippet = -1
    snippets_update(True)
    context_update()

    precomputed_hex_bins = [None]*M.nlayers
    for ilayer in range(M.nlayers):
        precomputed_hex_bins[ilayer] = [None]*len(M.species)
        p_tsne_qmax[ilayer] = np.iinfo(np.int64).min
        p_tsne_qmin[ilayer] = np.iinfo(np.int64).max 
        p_tsne_rmax[ilayer] = np.iinfo(np.int64).min
        p_tsne_rmin[ilayer] = np.iinfo(np.int64).max
        for (ispecies,specie) in enumerate(M.species):
            precomputed_hex_bins[ilayer][ispecies] = [None]*len(M.words)
            for (iword,word) in enumerate(M.words):
                precomputed_hex_bins[ilayer][ispecies][iword] = [None]*len(M.nohyphens)
                for (inohyphen,nohyphen) in enumerate(M.nohyphens):
                    precomputed_hex_bins[ilayer][ispecies][iword][inohyphen] = \
                            [None]*len(M.kinds)
                    for (ikind,kind) in enumerate(M.kinds):
                        if inohyphen!=0 and (ispecies!=0 or iword!=0):
                            continue
                        bidx = np.logical_and([specie in x['label'] and \
                                               word in x['label'] and \
                                               nohyphen in x['label'] and \
                                               kind==x['kind'] \
                                               for x in M.clustered_samples], tsne_isnotnan)
                        if sum(bidx)==0:
                            continue
                        these_hex_bins = hexbin(M.clustered_hidden[ilayer][bidx,0], \
                                                M.clustered_hidden[ilayer][bidx,1], \
                                                size=M.hex_size)
                        precomputed_hex_bins[ilayer][ispecies][iword][inohyphen][ikind] = \
                                these_hex_bins
                        hex_bins.data.update(q=these_hex_bins['q'], \
                                             r=these_hex_bins['r'], \
                                             counts=these_hex_bins['counts'],\
                                             index=range(len(these_hex_bins['counts'])))
                        p_tsne_qmax[ilayer] = np.maximum(p_tsne_qmax[ilayer], \
                                                         np.max(these_hex_bins['q']))
                        p_tsne_qmin[ilayer] = np.minimum(p_tsne_qmin[ilayer], \
                                                         np.min(these_hex_bins['q']))
                        p_tsne_rmax[ilayer] = np.maximum(p_tsne_rmax[ilayer], \
                                                         np.max(these_hex_bins['r']))
                        p_tsne_rmin[ilayer] = np.minimum(p_tsne_rmin[ilayer], \
                                                         np.min(these_hex_bins['r']))

    which_layer.labels = M.layers
    which_layer.active = M.ilayer = 0
    which_species.labels = M.species
    which_species.active = M.ispecies = 0
    which_word.labels = M.words
    which_word.active = M.iword = 0
    which_nohyphen.labels = M.nohyphens
    which_nohyphen.active = M.inohyphen = 0
    which_kind.labels = M.kinds
    which_kind.active = M.ikind = 0

    tsne_update()
    radius_size.disabled=False

def tsne_update():
    global hex_bins
    if precomputed_hex_bins == None:
        return
    selected_hex_bins = precomputed_hex_bins[M.ilayer][M.ispecies][M.iword][M.inohyphen][M.ikind]
    if selected_hex_bins is not None:
        qlim=pd.Series([p_tsne_qmin[M.ilayer]-1,p_tsne_qmax[M.ilayer]+1], name='q')
        q=selected_hex_bins['q'].append(qlim, ignore_index=True)
        rlim=pd.Series([p_tsne_rmin[M.ilayer]-1,p_tsne_rmax[M.ilayer]+1], name='r')
        r=selected_hex_bins['r'].append(rlim, ignore_index=True)
        countslim=pd.Series([0,0])
        counts=selected_hex_bins['counts'].append(countslim, ignore_index=True)
        counts/=max(selected_hex_bins['counts'])
        index=range(2+len(selected_hex_bins['counts']))
        hex_bins.data.update(q=q, r=r, counts=counts, index=index)
    else:
        hex_bins.data.update(q=[], r=[], counts=[], index=[])

def within_an_annotation(sample):
    if len(M.annotated_starts_sorted)>0:
        ifrom = np.searchsorted(M.annotated_starts_sorted, sample['ticks'][0],
                                side='right') - 1
        if 0 <= ifrom and ifrom < len(M.annotated_starts_sorted) and \
                    M.annotated_samples[ifrom]['ticks'][1] >= sample['ticks'][1]:
            return ifrom
    return -1

def snippets_update(redraw_wavs):
    if M.isnippet>0 and not np.isnan(M.xtsne) and not np.isnan(M.ytsne):
        quad_fuchsia_snippets.data.update(
                left=[M.xsnippet*(M.snippets_gap_pix+M.snippets_pix)],
                right=[(M.xsnippet+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix],
                top=[-M.ysnippet*2+1], bottom=[-M.ysnippet*2-1])
    else:
        quad_fuchsia_snippets.data.update(left=[], right=[], top=[], bottom=[])

    isubset = np.where([M.species[M.ispecies] in x['label'] and
                      M.words[M.iword] in x['label'] and
                      M.nohyphens[M.inohyphen] in x['label'] and
                      M.kinds[M.ikind]==x['kind'] for x in M.clustered_samples])[0]
    distance = np.linalg.norm(M.clustered_hidden[M.ilayer][isubset,:]-[M.xtsne,M.ytsne], axis=1)
    isort = np.argsort(distance)
    songs, labels, labels_new, scales = [], [], [], []
    for iwav in range(M.nx*M.ny):
        if iwav<len(distance) and distance[isort[iwav]]<M.radius*M.hex_size:
            M.nearest_samples[iwav] = isubset[isort[iwav]]
            thissample = M.clustered_samples[M.nearest_samples[iwav]]
            labels.append(thissample['label'])
            midpoint = np.mean(thissample['ticks'], dtype=int)
            if redraw_wavs:
                with wave.open(thissample['file']) as fid:
                    start_frame = max(0, midpoint-M.snippets_tic//2)
                    nframes_to_get = min(fid.getnframes() - start_frame,
                                         M.snippets_tic+1,
                                         M.snippets_tic+1+(midpoint-M.snippets_tic//2))
                    fid.setpos(start_frame)
                    song = np.frombuffer(fid.readframes(nframes_to_get), dtype=np.int16)
                    song = decimate(song, M.snippets_decimate_by, n=M.filter_order,
                                    ftype='iir', zero_phase=True)
                    left_pad = max(0, M.snippets_pix-nframes_to_get if start_frame==0 else 0)
                    right_pad = max(0, M.snippets_pix-nframes_to_get if start_frame>0 else 0)
                    np.pad(song, ((left_pad, right_pad),), 'constant', constant_values=(np.nan,))
                    song = song[:M.snippets_pix]
                    scales.append(np.minimum(np.iinfo(np.int16).max, np.max(np.abs(song))))
                    songs.append(song/scales[-1])
            else:
                songs.append([])
            iannotated = within_an_annotation(thissample)
            if iannotated == -1:
                labels_new.append('')
            else:
                labels_new.append(M.annotated_samples[iannotated]['label'])
        else:
            M.nearest_samples[iwav] = -1
            labels.append('')
            labels_new.append('')
            scales.append(0)
            songs.append(np.full(M.snippets_pix,np.nan))
    label_sources.data.update(text=labels)
    label_sources_new.data.update(text=labels_new)
    left, right, top, bottom = [], [], [], []
    for (isong,song) in enumerate(songs):
        ix, iy = isong%M.nx, isong//M.nx
        if redraw_wavs:
            xdata = range(ix*(M.snippets_gap_pix+M.snippets_pix),
                          (ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            ydata = -iy*2+song
            wav_sources[isong].data.update(x=xdata, y=ydata)
            line_glyphs[isong].glyph.line_color = M.palette[int(np.floor(scales[isong]/2**7))]
        if labels_new[isong]!='':
            left.append(ix*(M.snippets_gap_pix+M.snippets_pix))
            right.append((ix+1)*(M.snippets_gap_pix+M.snippets_pix)-M.snippets_gap_pix)
            top.append(-iy*2+1)
            bottom.append(-iy*2-1)
    quad_grey_snippets.data.update(left=left, right=right, top=top, bottom=bottom)

    xtsne_last, ytsne_last = M.xtsne, M.ytsne

def context_update(highlight_tapped_snippet=True):
    p_context.title.text = ''
    tapped_ticks = [np.nan, np.nan]
    istart = np.nan
    scale = 0
    ywav, xwav = [], []
    xlabel, ylabel, tlabel = [], [], []
    xlabel_new, ylabel_new, tlabel_new = [], [], []
    left, right, top, bottom = [], [], [], []
    left_new, right_new, top_new, bottom_new = [], [], [], []

    if M.isnippet>=0:
        zoom_context.disabled=False
        zoom_offset.disabled=False
        zoomin.disabled=False
        zoomout.disabled=False
        reset.disabled=False
        panleft.disabled=False
        panright.disabled=False
        tapped_sample = M.clustered_samples[M.isnippet]
        tapped_ticks = tapped_sample['ticks']
        M.context_midpoint = np.mean(tapped_ticks, dtype=int)
        istart = M.context_midpoint-M.context_width_tic//2 + M.context_offset_tic
        p_context.title.text = tapped_sample['file']
        fid = wave.open(tapped_sample['file'])
        M.file_nframes = fid.getnframes()
        if istart+M.context_width_tic>0 and istart<M.file_nframes:
            istart_bounded = np.maximum(0, istart)
            fid.setpos(istart_bounded)
            context_tic_adjusted = M.context_width_tic+1-(istart_bounded-istart)
            ilength = np.minimum(M.file_nframes-istart_bounded, context_tic_adjusted)

            song = np.frombuffer(fid.readframes(ilength), dtype=np.int16)
            if len(song)<M.context_width_tic+1:
                npad = M.context_width_tic+1-len(song)
                if istart<0:
                    song = np.concatenate((np.full((npad,),0), song))
                if istart+M.context_width_tic>M.file_nframes:
                    song = np.concatenate((song, np.full((npad,),0)))

            tic2pix = M.context_width_tic / M.gui_width_pix
            context_decimate_by = round(tic2pix/M.filter_ratio_max) if \
                     tic2pix>M.filter_ratio_max else 1
            context_pix = round(M.context_width_tic / context_decimate_by)
            song = decimate(song, context_decimate_by, n=M.filter_order,
                            ftype='iir', zero_phase=True)
            song = song[:context_pix]

            scale = np.max(np.abs(song))
            ywav = song/scale
            xwav = [(istart+i*context_decimate_by)/M.Fs for i in range(len(song))]
            song_max = np.max(song) / scale
            song_min = np.min(song) / scale

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
                xlabel.append((L+R)/2/M.Fs)
                tlabel.append(M.clustered_samples[isample]['kind'][:1]+
                              '. '+M.clustered_samples[isample]['label'])
                ylabel.append(song_max)
                left.append(L/M.Fs)
                right.append(R/M.Fs)
                top.append(song_max)
                bottom.append(0)
                if tapped_sample==M.clustered_samples[isample] and highlight_tapped_snippet:
                    quad_fuchsia_context.data.update(left=[L/M.Fs], right=[R/M.Fs],
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
                    if tapped_sample['file']!=M.annotated_samples[isample]['file']:
                        continue
                    L = np.max([istart, M.annotated_samples[isample]['ticks'][0]])
                    R = np.min([istart+M.context_width_tic,
                                M.annotated_samples[isample]['ticks'][1]])
                    xlabel_new.append((L+R)/2/M.Fs)
                    tlabel_new.append(M.annotated_samples[isample]['label'])
                    ylabel_new.append(song_min)
                    left_new.append(L/M.Fs)
                    right_new.append(R/M.Fs)
                    top_new.append(0)
                    bottom_new.append(song_min)
        fid.close()
    else:
        zoom_context.disabled=True
        zoom_offset.disabled=True
        zoomin.disabled=True
        zoomout.disabled=True
        reset.disabled=True
        panleft.disabled=True
        panright.disabled=True
        quad_fuchsia_context.data.update(left=[], right=[], top=[], bottom=[])

    wav_source.data.update(x=xwav, y=ywav)
    line_glyph.glyph.line_color = M.palette[int(np.floor(scale/2**7))]
    quad_grey_context_old.data.update(left=left, right=right, top=top, bottom=bottom)
    quad_grey_context_new.data.update(left=left_new, right=right_new, top=top_new, bottom=bottom_new)
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

def generic_parameters_update():
    M.save_state_callback()
    buttons_update()

def configuration_contents_update():
    if configuration_file.value:
        with open(configuration_file.value, 'r') as fid:
            configuration_contents.value = fid.read()

def model_file_update(attr, old, new):
    M.save_state_callback()
    M.parse_model_file()
    buttons_update()

def groundtruth_update():
    groundtruth.button_type="warning"
    groundtruth.disabled=True
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
        if 'time' not in wantedwords:
            wantedwords.append('time')
        if 'frequency' not in wantedwords:
            wantedwords.append('frequency')
        wantedwords_string.value=str.join(',',wantedwords)
        if M.action==train:
            labeltypes_string.value="annotated"
        elif M.action==hidden:
            labeltypes_string.value="annotated,detected"
    okay=True if M.action else False
    for textinput in parameter_textinputs:
        if textinput in action2parametertextinputs[M.action]:
            textinput.disabled=False
            if textinput.value=='' and textinput is not testfiles_string:
                okay=False
        else:
            textinput.disabled=True
    doit.button_type="default"
    if okay:
        doit.disabled=False
        doit.button_type="danger"
    else:
        doit.disabled=True
        doit.button_type="default"

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def file_dialog_update():
    thispath = os.path.join(M.file_dialog_root,M.file_dialog_filter)
    files = sorted(['.', '..', *glob.glob(thispath)])
    file_dialog = dict(
        names=[os.path.basename(x) + ('/' if os.path.isdir(x) else '') for x in files],
        sizes=[sizeof_fmt(os.path.getsize(f)) for f in files],
        dates=[datetime.fromtimestamp(os.path.getmtime(f)) for f in files],
    )
    file_dialog_source.data = file_dialog
    file_dialog_source.selected.indices = []

def wordcounts_update():
    if not os.path.isdir(groundtruth_folder.value):
        return
    dfs = []
    for subdir in filter(lambda x: os.path.isdir(os.path.join(groundtruth_folder.value,x)), \
                         os.listdir(groundtruth_folder.value)):
        for csvfile in filter(lambda x: x.endswith('.csv'), \
                              os.listdir(os.path.join(groundtruth_folder.value , subdir))):
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

def init():
    global p_tsne, hex_bins, precomputed_hex_bins, p_tsne_qmax, p_tsne_qmin, p_tsne_rmax, p_tsne_rmin, p_snippets, label_sources, label_sources_new, wav_sources, line_glyphs, quad_grey_snippets, circle_fuchsia_tsne, p_tsne_circle, p_context, quad_grey_context_old, quad_grey_context_new, quad_grey_context_pan, quad_fuchsia_context, quad_fuchsia_snippets, wav_source, line_glyph, label_source, label_source_new, which_layer, which_species, which_word, which_nohyphen, which_kind, radius_size, zoom_context, zoom_offset, zoomin, zoomout, reset, panleft, panright, save_indicator, label_text_widgets, label_count_widgets, undo, redo, detect, misses, configuration, configuration_file, train, generalize, xvalidate, hidden, cluster, visualize, accuracy, freeze, classify, ethogram, compare, dense, file_dialog_source, file_dialog_source, configuration_contents, logs, logs_folder, model, model_file, wavtfcsvfiles, wavtfcsvfiles_string, groundtruth, groundtruth_folder, validationfiles, testfiles, validationfiles_string, testfiles_string, wantedwords, wantedwords_string, labeltypes, labeltypes_string, labelsounds, makepredictions, fixfalsepositives, fixfalsenegatives, leaveoneout, tunehyperparameters, findnovellabels, examineerrors, doit, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string, nsteps_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string, kfold_string, cluster_equalize_ratio_string, cluster_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string, precision_recall_ratios_string, context_ms_string, shiftby_ms_string, window_ms_string, mel_dct_string, stride_ms_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, editconfiguration, file_dialog_string, file_dialog_table, readme_contents, wordcounts, wizard_buttons, action_buttons, parameter_buttons, parameter_textinputs, wizard2actions, action2parameterbuttons, action2parametertextinputs

    p_tsne = figure(background_fill_color='#440154', toolbar_location=None,
                    match_aspect=True)
    p_tsne.toolbar.active_drag = None
    p_tsne.grid.visible = False
    p_tsne.xaxis.visible = False
    p_tsne.yaxis.visible = False
     
    hex_bins = ColumnDataSource(hexbin(np.array([]), np.array([]), M.hex_size))
    p_tsne.hex_tile(q="q", r="r", size=M.hex_size, line_color=None, source=hex_bins,
               fill_color=linear_cmap('counts', 'Viridis256', 0, 1))

    precomputed_hex_bins = None
    p_tsne_qmax = {}
    p_tsne_qmin = {}
    p_tsne_rmax = {}
    p_tsne_rmin = {}

    p_snippets = figure(background_fill_color='#FFFFFF', toolbar_location=None)
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

    wav_sources=[]
    line_glyphs=[]
    for iwav in range(M.nx*M.ny):
        wav_sources.append(ColumnDataSource(data=dict(x=[], y=[])))
        line_glyphs.append(p_snippets.line('x', 'y', source=wav_sources[-1]))

    quad_grey_snippets = ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    p_snippets.quad('left','right','top','bottom',source=quad_grey_snippets,
                fill_color="lightgrey", line_color="lightgrey", level='underlay')

    circle_fuchsia_tsne = ColumnDataSource(data=dict(x=[], y=[]))
    p_tsne_circle = p_tsne.circle('x','y',source=circle_fuchsia_tsne, radius=M.radius,
                fill_color=None, line_color="fuchsia", level='overlay')

    p_tsne.on_event(Tap, C.tsne_tap_callback)

    p_context = figure(plot_width=M.gui_width_pix, plot_height=150,
                       background_fill_color='#FFFFFF', toolbar_location=None)
    p_context.toolbar.active_drag = None
    p_context.grid.visible = False
    p_context.xaxis.axis_label = 'time (sec)'
    p_context.yaxis.visible = False
    p_context.x_range.range_padding = 0.0

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

    wav_source = ColumnDataSource(data=dict(x=[], y=[]))
    line_glyph = p_context.line('x', 'y', source=wav_source)

    label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_context.text('x', 'y', source=label_source,
                text_font_size='6pt', text_align='center', text_baseline='top')
    label_source_new = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    p_context.text('x', 'y', source=label_source_new,
                text_font_size='6pt', text_align='center', text_baseline='bottom')

    p_snippets.on_event(Tap, C.snippets_tap_callback)

    p_context.on_event(DoubleTap, C.context_doubletap_callback)

    p_context.on_event(PanStart, C.context_pan_start_callback)
    p_context.on_event(Pan, C.context_pan_callback)
    p_context.on_event(PanEnd, C.context_pan_end_callback)

    p_snippets.on_event(DoubleTap, C.snippets_doubletap_callback)

    tsne_update()

    which_layer = RadioButtonGroup(labels=['.'], active=M.ilayer)
    which_layer.on_click(C.layer_callback)

    which_species = RadioButtonGroup(labels=['.'], active=M.ispecies)
    which_species.on_click(C.species_callback)

    which_word = RadioButtonGroup(labels=['.'], active=M.iword)
    which_word.on_click(C.word_callback)

    which_nohyphen = RadioButtonGroup(labels=['.'], active=M.inohyphen)
    which_nohyphen.on_click(C.nohyphen_callback)

    which_kind = RadioButtonGroup(labels=['.'], active=M.ikind)
    which_kind.on_click(C.kind_callback)

    radius_size = TextInput(value=str(M.radius), title="radius:", disabled=True)
    radius_size.on_change("value", C.radius_callback)

    zoom_context = TextInput(value=str(M.context_width_ms), title="context (msec):", disabled=True)
    zoom_context.on_change("value", C.zoom_context_callback)

    zoom_offset = TextInput(value=str(M.context_offset_ms), title="offset (msec):", disabled=True)
    zoom_offset.on_change("value", C.zoom_offset_callback)

    zoomin = Button(label='+', disabled=True, width=40)
    zoomin.on_click(C.zoomin_callback)

    zoomout = Button(label='-', disabled=True, width=40)
    zoomout.on_click(C.zoomout_callback)

    reset = Button(label='0', disabled=True, width=40)
    reset.on_click(C.zero_callback)

    panleft = Button(label='<', disabled=True, width=40)
    panleft.on_click(C.panleft_callback)

    panright = Button(label='>', disabled=True, width=40)
    panright.on_click(C.panright_callback)

    save_indicator = Button(label='0', width=40)

    label_text_callbacks=[]
    label_text_widgets=[]
    label_count_callbacks=[]
    label_count_widgets=[]

    for i in range(M.nlabels):
        label_text_callbacks.append(lambda a,o,n,i=i: C.label_text_callback(n,i))
        label_text_widgets.append(TextInput(value=M.state['labels'][i], css_classes=['hide-label']))
        label_text_widgets[-1].on_change("value", label_text_callbacks[-1])
        label_count_callbacks.append(lambda i=i: C.label_count_callback(i))
        label_count_widgets.append(Button(label='0', css_classes=['hide-label'], width=40))
        label_count_widgets[-1].on_click(label_count_callbacks[-1])

    C.label_count_callback(M.ilabel)

    undo = Button(label='undo', disabled=True, width=50)
    undo.on_click(C.undo_callback)

    redo = Button(label='redo', disabled=True, width=50)
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

    generalize = Button(label='generalize')
    generalize.on_click(lambda: C.action_callback(generalize, C.generalize_actuate))

    xvalidate = Button(label='x-validate')
    xvalidate.on_click(lambda: C.action_callback(xvalidate, C.xvalidate_actuate))

    hidden = Button(label='hidden')
    hidden.on_click(lambda: C.action_callback(hidden, C.hidden_actuate))

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

    dense = Button(label='dense')
    dense.on_click(lambda: C.action_callback(dense, C.dense_actuate))

    file_dialog_source = ColumnDataSource(data=dict(names=[], sizes=[], dates=[]))
    file_dialog_source.selected.on_change('indices', C.file_dialog_callback)

    file_dialog_columns = [
        TableColumn(field="names", title="Name"),
        TableColumn(field="sizes", title="Size", width=40),
        TableColumn(field="dates", title="Date", width=80, formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
    ]
    file_dialog_table = DataTable(source=file_dialog_source, columns=file_dialog_columns, height=560, width=560, index_position=None)

    configuration_contents = TextAreaInput(rows=33, width=620, max_length=50000, \
                                        disabled=True, css_classes=['fixedwidth'])
    configuration_contents_update()
    configuration_contents.on_change('value', C.configuration_textarea_callback)

    logs = Button(label='logs folder:', width=110)
    logs.on_click(C.logs_callback)
    logs_folder = TextInput(value=M.state['logs'], title="", disabled=False)
    logs_folder.on_change('value', lambda a,o,n: generic_parameters_update())

    model = Button(label='model:', width=110)
    model.on_click(C.model_callback)
    model_file = TextInput(value=M.state['model'], title="", disabled=False)
    model_file.on_change('value', model_file_update)
    M.parse_model_file()

    wavtfcsvfiles = Button(label='wav,tf,csv files:', width=110)
    wavtfcsvfiles.on_click(C.wavtfcsvfiles_callback)
    wavtfcsvfiles_string = TextInput(value=M.state['wavtfcsvfiles'], title="", disabled=False)
    wavtfcsvfiles_string.on_change('value', lambda a,o,n: generic_parameters_update())

    groundtruth = Button(label='ground truth:', width=110)
    groundtruth.on_click(C.groundtruth_callback)
    groundtruth_folder = TextInput(value=M.state['groundtruth'], title="", disabled=False)
    groundtruth_folder.on_change('value', lambda a,o,n: groundtruth_update())

    validationfiles = Button(label='validation files:', width=110)
    validationfiles.on_click(C.validationfiles_callback)
    validationfiles_string = TextInput(value=M.state['validationfiles'], title="", disabled=False)
    validationfiles_string.on_change('value', lambda a,o,n: generic_parameters_update())

    testfiles = Button(label='test files:', width=110)
    testfiles.on_click(C.testfiles_callback)
    testfiles_string = TextInput(value=M.state['testfiles'], title="", disabled=False)
    testfiles_string.on_change('value', lambda a,o,n: generic_parameters_update())

    wantedwords = Button(label='wanted words:', width=110)
    wantedwords.on_click(C.wantedwords_callback)
    wantedwords_string = TextInput(value=M.state['wantedwords'], title="", disabled=False)
    wantedwords_string.on_change('value', lambda a,o,n: generic_parameters_update())

    labeltypes = Button(label='label types:', width=110)
    labeltypes_string = TextInput(value=M.state['labeltypes'], title="", disabled=False)
    labeltypes_string.on_change('value', lambda a,o,n: generic_parameters_update())

    labelsounds = Button(label='label sounds')
    labelsounds.on_click(C.labelsounds_callback)

    makepredictions = Button(label='make predictions')
    makepredictions.on_click(C.makepredictions_callback)

    fixfalsepositives = Button(label='fix false positives')
    fixfalsepositives.on_click(C.fixfalsepositives_callback)

    fixfalsenegatives = Button(label='fix false negatives')
    fixfalsenegatives.on_click(C.fixfalsenegatives_callback)

    leaveoneout = Button(label='leave one out')
    leaveoneout.on_click(C.leaveoneout_callback)

    tunehyperparameters = Button(label='tune h-parameters')
    tunehyperparameters.on_click(C.tunehyperparameters_callback)

    findnovellabels = Button(label='find novel labels')
    findnovellabels.on_click(C.findnovellabels_callback)

    examineerrors = Button(label='examine errors')
    examineerrors.on_click(C.examineerrors_callback)

    doit = Button(label='do it!', disabled=True)
    doit.on_click(C.doit_callback)

    time_sigma_string = TextInput(value=M.state['time_sigma'], title="σ time", disabled=False)
    time_sigma_string.on_change('value', lambda a,o,n: generic_parameters_update())

    time_smooth_ms_string = TextInput(value=M.state['time_smooth_ms'], title="time smooth", disabled=False)
    time_smooth_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    frequency_n_ms_string = TextInput(value=M.state['frequency_n_ms'], title="freq N", disabled=False)
    frequency_n_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    frequency_nw_string = TextInput(value=M.state['frequency_nw'], title="freq NW", disabled=False)
    frequency_nw_string.on_change('value', lambda a,o,n: generic_parameters_update())

    frequency_p_string = TextInput(value=M.state['frequency_p'], title="ρ freq", disabled=False)
    frequency_p_string.on_change('value', lambda a,o,n: generic_parameters_update())

    frequency_smooth_ms_string = TextInput(value=M.state['frequency_smooth_ms'], title="freq smooth", disabled=False)
    frequency_smooth_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    nsteps_string = TextInput(value=M.state['nsteps'], title="# steps", disabled=False)
    nsteps_string.on_change('value', lambda a,o,n: generic_parameters_update())

    save_and_validate_period_string = TextInput(value=M.state['save_and_validate_interval'], \
                                              title="validate period", \
                                              disabled=False)
    save_and_validate_period_string.on_change('value', lambda a,o,n: generic_parameters_update())

    validate_percentage_string = TextInput(value=M.state['validate_percentage'], \
                                       title="validate %", \
                                       disabled=False)
    validate_percentage_string.on_change('value', lambda a,o,n: generic_parameters_update())

    mini_batch_string = TextInput(value=M.state['mini_batch'], \
                                       title="mini-batch", \
                                       disabled=False)
    mini_batch_string.on_change('value', lambda a,o,n: generic_parameters_update())

    kfold_string = TextInput(value=M.state['kfold'], title="k-fold",  disabled=False)
    kfold_string.on_change('value', lambda a,o,n: generic_parameters_update())

    cluster_equalize_ratio_string = TextInput(value=M.state['cluster_equalize_ratio'], \
                                       title="equalize ratio", \
                                       disabled=False)
    cluster_equalize_ratio_string.on_change('value', lambda a,o,n: generic_parameters_update())

    cluster_max_samples_string = TextInput(value=M.state['cluster_max_samples'], \
                                       title="max samples", \
                                       disabled=False)
    cluster_max_samples_string.on_change('value', lambda a,o,n: generic_parameters_update())

    pca_fraction_variance_to_retain_string = TextInput(value=M.state['pca_fraction_variance_to_retain'], \
                                       title="PCA fraction", \
                                       disabled=False)
    pca_fraction_variance_to_retain_string.on_change('value', lambda a,o,n: generic_parameters_update())

    tsne_perplexity_string = TextInput(value=M.state['tsne_perplexity'], \
                                       title="perplexity", \
                                       disabled=False)
    tsne_perplexity_string.on_change('value', lambda a,o,n: generic_parameters_update())

    tsne_exaggeration_string = TextInput(value=M.state['tsne_exaggeration'], \
                                       title="exaggeration", \
                                       disabled=False)
    tsne_exaggeration_string.on_change('value', lambda a,o,n: generic_parameters_update())

    precision_recall_ratios_string = TextInput(value=M.state['precision_recall_ratios'], \
                                       title="P/Rs", \
                                       disabled=False)
    precision_recall_ratios_string.on_change('value', lambda a,o,n: generic_parameters_update())

    context_ms_string = TextInput(value=M.state['context_ms'], \
                                       title="context", \
                                       disabled=False)
    context_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    shiftby_ms_string = TextInput(value=M.state['shiftby_ms'], \
                                       title="shift by", \
                                       disabled=False)
    shiftby_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    window_ms_string = TextInput(value=M.state['window_ms'], \
                                       title="window", \
                                       disabled=False)
    window_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    mel_dct_string = TextInput(value=M.state['mel&dct'], \
                                       title="Mel & DCT", \
                                       disabled=False)
    mel_dct_string.on_change('value', lambda a,o,n: generic_parameters_update())

    stride_ms_string = TextInput(value=M.state['stride_ms'], \
                                       title="stride", \
                                       disabled=False)
    stride_ms_string.on_change('value', lambda a,o,n: generic_parameters_update())

    dropout_string = TextInput(value=M.state['dropout'], \
                                       title="dropout", \
                                       disabled=False)
    dropout_string.on_change('value', lambda a,o,n: generic_parameters_update())

    optimizer = Select(title="optimizer", height=50, \
                       value=M.state['optimizer'], \
                       options=[("sgd","SGD"), ("adam","Adam"), ("adagrad","AdaGrad"), ("rmsprop","RMSProp")])
    optimizer.on_change('value', lambda a,o,n: generic_parameters_update())

    learning_rate_string = TextInput(value=M.state['learning_rate'], \
                                       title="learning rate", \
                                       disabled=False)
    learning_rate_string.on_change('value', lambda a,o,n: generic_parameters_update())

    kernel_sizes_string = TextInput(value=M.state['kernel_sizes'], \
                                       title="kernels", \
                                       disabled=False)
    kernel_sizes_string.on_change('value', lambda a,o,n: generic_parameters_update())

    last_conv_width_string = TextInput(value=M.state['last_conv_width'], \
                                       title="last conv width", \
                                       disabled=False)
    last_conv_width_string.on_change('value', lambda a,o,n: generic_parameters_update())

    nfeatures_string = TextInput(value=M.state['nfeatures'], \
                                       title="# features", \
                                       disabled=False)
    nfeatures_string.on_change('value', lambda a,o,n: generic_parameters_update())

    editconfiguration = Button(label='edit', button_type="default", width=50)
    editconfiguration.on_click(C.editconfiguration_callback)

    file_dialog_string = TextInput(disabled=False)
    file_dialog_string.on_change("value", C.file_dialog_path_callback)
    file_dialog_string.value = M.state['file_dialog_string']
     
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','README.md'), 'r', encoding='utf-8') as fid:
        contents = fid.read()
    html = markdown.markdown(contents, extensions=['tables','toc'])
    readme_contents = Div(text=html, style={'overflow':'scroll','width':'600px','height':'1275px'})

    wordcounts = Div(text="")
    wordcounts_update()

    wizard_buttons = set([
        labelsounds,
        makepredictions,
        fixfalsepositives,
        fixfalsenegatives,
        leaveoneout,
        tunehyperparameters,
        findnovellabels,
        examineerrors])

    action_buttons = set([
        detect,
        train,
        generalize,
        xvalidate,
        hidden,
        cluster,
        visualize,
        accuracy,
        freeze,
        classify,
        ethogram,
        misses,
        compare,
        dense])

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
        save_and_validate_period_string,
        validate_percentage_string,
        mini_batch_string,
        kfold_string,
        cluster_equalize_ratio_string,
        cluster_max_samples_string,
        pca_fraction_variance_to_retain_string,
        tsne_perplexity_string,
        tsne_exaggeration_string,
        precision_recall_ratios_string,
        context_ms_string,
        shiftby_ms_string,
        window_ms_string,
        mel_dct_string,
        stride_ms_string,
        dropout_string,
        optimizer,
        learning_rate_string,
        kernel_sizes_string,
        last_conv_width_string,
        nfeatures_string])

    wizard2actions = {
            labelsounds: [detect,train,hidden,cluster,visualize],
            makepredictions: [train, accuracy, freeze, classify, ethogram],
            fixfalsepositives: [hidden, cluster, visualize],
            fixfalsenegatives: [detect, misses, hidden, cluster, visualize],
            leaveoneout: [generalize, accuracy],
            tunehyperparameters: [xvalidate, accuracy, compare],
            findnovellabels: [detect, train, hidden, cluster, visualize],
            examineerrors: [detect, hidden, cluster, visualize],
            None: action_buttons }

    action2parameterbuttons = {
            detect: [configuration,wavtfcsvfiles],
            train: [configuration, logs, groundtruth, wantedwords, testfiles, labeltypes],
            generalize: [configuration, logs, groundtruth, validationfiles, testfiles, wantedwords, labeltypes],
            xvalidate: [configuration, logs, groundtruth, testfiles, wantedwords, labeltypes],
            hidden: [configuration, logs, model, groundtruth, labeltypes],
            cluster: [configuration, groundtruth],
            visualize: [groundtruth],
            accuracy: [configuration, logs],
            freeze: [configuration, logs, model],
            classify: [configuration, logs, model, wavtfcsvfiles],
            ethogram: [configuration, model, wavtfcsvfiles],
            misses: [configuration, wavtfcsvfiles],
            compare: [configuration, logs],
            dense: [configuration, testfiles],
            None: parameter_buttons }

    action2parametertextinputs = {
            detect: [configuration_file, wavtfcsvfiles_string, time_sigma_string, time_smooth_ms_string, frequency_n_ms_string, frequency_nw_string, frequency_p_string, frequency_smooth_ms_string],
            train: [configuration_file, context_ms_string, shiftby_ms_string, window_ms_string, mel_dct_string, stride_ms_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, logs_folder, groundtruth_folder, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, save_and_validate_period_string, validate_percentage_string, mini_batch_string],
            generalize: [configuration_file, context_ms_string, shiftby_ms_string, window_ms_string, mel_dct_string, stride_ms_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, logs_folder, groundtruth_folder, validationfiles_string, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, save_and_validate_period_string, mini_batch_string],
            xvalidate: [configuration_file, context_ms_string, shiftby_ms_string, window_ms_string, mel_dct_string, stride_ms_string, dropout_string, optimizer, learning_rate_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, logs_folder, groundtruth_folder, testfiles_string, wantedwords_string, labeltypes_string, nsteps_string, save_and_validate_period_string, mini_batch_string, kfold_string],
            hidden: [configuration_file, context_ms_string, shiftby_ms_string, window_ms_string, mel_dct_string, stride_ms_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, logs_folder, model_file, groundtruth_folder, labeltypes_string, mini_batch_string],
            cluster: [configuration_file, groundtruth_folder, cluster_equalize_ratio_string, cluster_max_samples_string, pca_fraction_variance_to_retain_string, tsne_perplexity_string, tsne_exaggeration_string],
            visualize: [groundtruth_folder],
            accuracy: [configuration_file, logs_folder, precision_recall_ratios_string],
            freeze: [configuration_file, context_ms_string, window_ms_string, mel_dct_string, stride_ms_string, kernel_sizes_string, last_conv_width_string, nfeatures_string, logs_folder, model_file],
            classify: [configuration_file, context_ms_string, shiftby_ms_string, stride_ms_string, logs_folder, model_file, wavtfcsvfiles_string],
            ethogram: [configuration_file, model_file, wavtfcsvfiles_string],
            misses: [configuration_file, wavtfcsvfiles_string],
            compare: [configuration_file, logs_folder],
            dense: [configuration_file, testfiles_string],
            None: parameter_textinputs }
