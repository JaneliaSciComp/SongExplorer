import os
from subprocess import run
import pandas as pd
from datetime import datetime
import numpy as np
import time
import logging 

log = logging.getLogger("deepsong") 

import model as M
import view as V

def layer_callback(new):
    M.ilayer=int(new)
    V.circle_fuchsia_tsne.data.update(x=[], y=[])
    V.tsne_update()
    M.xtsne, M.ytsne = np.nan, np.nan
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def species_callback(new):
    M.ispecies=int(new)
    if M.ispecies>0:
        V.which_nohyphen.active=0
    V.tsne_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def word_callback(new):
    M.iword=int(new)
    if M.iword>0:
        V.which_nohyphen.active=0
    V.tsne_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def nohyphen_callback(new):
    M.inohyphen=int(new)
    if M.inohyphen>0:
        V.which_word.active=0
        V.which_species.active=0
    V.tsne_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def kind_callback(new):
    M.ikind=int(new)
    V.tsne_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def radius_callback(attr, old, new):
    M.radius=float(new)
    V.p_tsne_circle.glyph.radius=M.radius
    V.snippets_update(True)
    M.isnippet = -1
    V.context_update()

def tsne_tap_callback(event):
    M.xtsne, M.ytsne = event.x, event.y
    V.circle_fuchsia_tsne.data.update(x=[M.xtsne], y=[M.ytsne])
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()

def snippets_tap_callback(event):
    M.xsnippet = int(np.rint(event.x/(M.snippets_gap_pix+M.snippets_pix)-0.5))
    M.ysnippet = int(np.rint(-event.y/2))
    M.isnippet = M.nearest_samples[M.ysnippet*M.nx + M.xsnippet]
    V.snippets_update(False)
    V.context_update()

def context_doubletap_callback(event):
    x_tic = int(np.rint(event.x*M.Fs))
    currfile = M.clustered_samples[M.isnippet]['file']
    if event.y<0:
        idouble_tapped_sample=-1
        if len(M.annotated_starts_sorted)>0:
            idouble_tapped_sample = np.searchsorted(M.annotated_starts_sorted, x_tic,
                                                      side='right') - 1
            while (idouble_tapped_sample > 0) and \
                  (M.annotated_samples[idouble_tapped_sample]['file'] != currfile):
                idouble_tapped_sample -= 1
            if (M.annotated_samples[idouble_tapped_sample]['file'] != currfile) or \
               (x_tic > M.annotated_samples[idouble_tapped_sample]['ticks'][1]):
                idouble_tapped_sample = -1
        if idouble_tapped_sample >= 0:
            M.delete_annotation(idouble_tapped_sample)
        elif M.state['labels'][M.ilabel] != '':
            thissample = {'file':currfile,
                          'ticks':[x_tic,x_tic],
                          'label':M.state['labels'][M.ilabel]}
            M.add_annotation(thissample)
    else:
        ileft = np.searchsorted(M.clustered_starts_sorted, x_tic)
        samples_righthere = set(range(0,ileft))
        iright = np.searchsorted(M.clustered_stops, x_tic,
                                 sorter=M.iclustered_stops_sorted)
        samples_righthere &= set([M.iclustered_stops_sorted[i] for i in \
                range(iright, len(M.iclustered_stops_sorted))])
        samples_inthisfile = filter(lambda x: M.clustered_samples[x]['file'] == currfile,
                                    samples_righthere)
        samples_shortest = sorted(samples_inthisfile, key=lambda x: \
                M.clustered_samples[x]['ticks'][1]-M.clustered_samples[x]['ticks'][0])
        if len(samples_shortest)>0:
            toggle_annotation(samples_shortest[0])

def context_pan_start_callback(event):
    if M.state['labels'][M.ilabel]=='' or event.y>0:
        return
    x_tic = int(np.rint(event.x*M.Fs))
    currfile = M.clustered_samples[M.isnippet]['file']
    M.panned_sample = {'file':currfile, 'ticks':[x_tic,x_tic],
                       'label':M.state['labels'][M.ilabel]}
    V.quad_grey_context_pan.data.update(left=[x_tic/M.Fs], right=[x_tic/M.Fs], top=[0],
            bottom=[V.p_context.y_range.start + V.p_context.y_range.range_padding])

def context_pan_callback(event):
    if M.state['labels'][M.ilabel]=='':
        return
    x_tic = int(np.rint(event.x*M.Fs))
    left_limit_tic = M.context_midpoint-M.context_width_tic//2 + M.context_offset_tic
    right_limit_tic = left_limit_tic + M.context_width_tic
    if x_tic < left_limit_tic or x_tic > right_limit_tic:
        return
    M.panned_sample['ticks'][1]=x_tic
    V.quad_grey_context_pan.data.update(right=[x_tic/M.Fs])

def context_pan_end_callback(event):
    if M.state['labels'][M.ilabel]=='':
        return
    M.panned_sample['ticks'] = sorted(M.panned_sample['ticks'])
    V.quad_grey_context_pan.data.update(left=[], right=[], top=[], bottom=[])
    M.add_annotation(M.panned_sample)

def zoom_context_callback(attr, old, new):
    M.context_width_ms = float(new)
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.Fs))
    V.context_update()

def zoom_offset_callback(attr, old, new):
    M.context_offset_ms = float(new)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.Fs))
    V.context_update()

def zoomin_callback():
    if M.context_width_tic>20:
        M.context_width_ms /= 2
        M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.Fs))
        V.zoom_context.value = str(M.context_width_ms)
        V.context_update()
    
def zoomout_callback():
    limit = M.file_nframes/M.Fs*1000
    M.context_width_ms = np.minimum(limit, 2*M.context_width_ms)
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.Fs))
    V.zoom_context.value = str(M.context_width_ms)
    limit_lo = (M.context_width_tic//2-M.context_midpoint)/M.Fs*1000
    limit_hi = (M.file_nframes-M.context_width_tic//2-M.context_midpoint)/M.Fs*1000
    M.context_offset_ms = np.clip(M.context_offset_ms, limit_lo, limit_hi)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.Fs))
    V.zoom_offset.value = str(M.context_offset_ms)
    V.context_update()
    
def zero_callback():
    M.context_width_ms = 400
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.Fs))
    V.zoom_context.value = str(M.context_width_ms)
    M.context_offset_ms = 0
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.Fs))
    V.zoom_offset.value = str(M.context_offset_ms)
    V.context_update()
    
def panleft_callback():
    limit = (M.context_width_tic//2-M.context_midpoint)/M.Fs*1000
    M.context_offset_ms = np.maximum(limit, M.context_offset_ms-M.context_width_ms//2)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.Fs))
    V.zoom_offset.value = str(M.context_offset_ms)
    V.context_update()
    
def panright_callback():
    limit = (M.file_nframes-M.context_width_tic//2-M.context_midpoint)/M.Fs*1000
    M.context_offset_ms = np.minimum(limit, M.context_offset_ms+M.context_width_ms//2)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.Fs))
    V.zoom_offset.value = str(M.context_offset_ms)
    V.context_update()
    
def isannotated(sample):
    return np.where([x['file']==sample['file'] and x['ticks']==sample['ticks'] for x in M.annotated_samples])[0]

def toggle_annotation(idouble_tapped_sample):
    iannotated = isannotated(M.clustered_samples[idouble_tapped_sample])
    if np.size(iannotated)!=0:
        M.delete_annotation(iannotated[0])
    elif M.state['labels'][M.ilabel] != '':
        thissample = M.clustered_samples[idouble_tapped_sample].copy()
        thissample['label'] = M.state['labels'][M.ilabel]
        M.add_annotation(thissample)

def snippets_doubletap_callback(event):
    x_tic = int(np.rint(event.x/(M.snippets_gap_pix+M.snippets_pix)-0.5))
    y_tic = int(np.rint(-event.y/2))
    idouble_tapped_sample = M.nearest_samples[y_tic*M.nx + x_tic]
    toggle_annotation(idouble_tapped_sample)

def undo_callback():
    if M.history_idx>0:
        M.history_idx-=1
        V.circle_fuchsia_tsne.data.update(x=[], y=[])
        M.xtsne, M.ytsne = np.nan, np.nan
        M.isnippet = -1
        V.snippets_update(True)
        M.isnippet = np.searchsorted(M.clustered_starts_sorted, \
                                   M.history_stack[M.history_idx][1]['ticks'][0])
        M.context_offset_tic = M.history_stack[M.history_idx][1]['ticks'][0] - \
                             M.clustered_starts_sorted[M.isnippet]
        M.context_offset_ms = M.context_offset_tic/M.Fs*1000
        V.zoom_offset.value = str(M.context_offset_ms)
        V.context_update(False)
        time.sleep(0.5)
        if M.history_stack[M.history_idx][0]=='add':
            iannotated = np.searchsorted(M.annotated_starts_sorted, \
                                         M.history_stack[M.history_idx][1]['ticks'][0])
            while M.annotated_samples[iannotated]!=M.history_stack[M.history_idx][1]:
                iannotated += 1
            M.delete_annotation(iannotated, addto_history=False)
        elif M.history_stack[M.history_idx][0]=='delete':
            M.add_annotation(M.history_stack[M.history_idx][1], addto_history=False)
    
def redo_callback():
    if M.history_idx<len(M.history_stack):
        M.history_idx+=1
        V.circle_fuchsia_tsne.data.update(x=[], y=[])
        M.xtsne, M.ytsne = np.nan, np.nan
        M.isnippet = -1
        V.snippets_update(True)
        M.isnippet = np.searchsorted(M.clustered_starts_sorted, \
                                   M.history_stack[M.history_idx-1][1]['ticks'][0])
        M.context_offset_tic = M.history_stack[M.history_idx-1][1]['ticks'][0] - \
                             M.clustered_starts_sorted[M.isnippet]
        M.context_offset_ms = M.context_offset_tic/M.Fs*1000
        V.zoom_offset.value = str(M.context_offset_ms)
        V.context_update(False)
        time.sleep(0.5)
        if M.history_stack[M.history_idx-1][0]=='add':
            M.add_annotation(M.history_stack[M.history_idx-1][1], addto_history=False)
        elif M.history_stack[M.history_idx-1][0]=='delete':
            iannotated = np.searchsorted(M.annotated_starts_sorted, \
                                         M.history_stack[M.history_idx-1][1]['ticks'][0])
            while M.annotated_samples[iannotated]!=M.history_stack[M.history_idx-1][1]:
                iannotated += 1
            M.delete_annotation(iannotated, addto_history=False)
    
def action_callback(thisaction, thisactuate):
    M.action=None if M.action is thisaction else thisaction
    M.function=thisactuate
    V.buttons_update()

def detect_actuate():
    for wavfile in V.wavtfcsvfiles_string.value.split(','):
        run(["detect.sh", V.configuration_file.value, wavfile, V.time_sigma_string.value, V.time_smooth_ms_string.value, V.frequency_n_ms_string.value, V.frequency_nw_string.value, V.frequency_p_string.value, V.frequency_smooth_ms_string.value])
    if True:  # detect.sh exited without error
        V.wordcounts_update()

def misses_actuate():
    run(["misses.sh", V.configuration_file.value, V.wavtfcsvfiles_string.value])
    if True:  # misses.sh exited without error
        V.wordcounts_update()

def isoldfile(x,subdir,basewavs):
    return \
        x.endswith('.tf') or \
        np.any([x.startswith(b+'-') and x.endswith('.wav') for b in basewavs]) or \
        x.endswith('-classify.log') or \
        '-predicted' in x or \
        x.endswith('-ethogram.log') or \
        '-missed' in x or \
        x.endswith('-missed.log') or \
        x == subdir+'.csv'

def _validation_test_files(files_string):
    if files_string.rstrip('/') == V.groundtruth_folder.value.rstrip('/'):
        dfs = []
        for subdir in filter(lambda x: os.path.isdir(os.path.join(V.groundtruth_folder.value,x)), \
                             os.listdir(V.groundtruth_folder.value)):
            for csvfile in filter(lambda x: '-annotated-' in x and x.endswith('.csv'), \
                                  os.listdir(os.path.join(V.groundtruth_folder.value, \
                                                          subdir))):
                filepath = os.path.join(V.groundtruth_folder.value, subdir, csvfile)
                if os.path.getsize(filepath) > 0:
                    dfs.append(pd.read_csv(filepath, header=None, index_col=False))
        if dfs:
            df = pd.concat(dfs)
            return list(set(df[0]))
    elif os.path.dirname(files_string.rstrip('/')) == V.groundtruth_folder.value.rstrip('/'):
        dfs = []
        for csvfile in filter(lambda x: '-annotated-' in x and x.endswith('.csv'), \
                              os.listdir(files_string)):
            filepath = os.path.join(files_string, csvfile)
            if os.path.getsize(filepath) > 0:
                dfs.append(pd.read_csv(filepath, header=None, index_col=False))
        if dfs:
            df = pd.concat(dfs)
            return [','.join(set(df[0]))]
    elif files_string.lower().endswith('.wav'):
        return [files_string]
    elif files_string!='':
        with open(files_string, "r") as fid:
            wavfiles = fid.readlines()
        return [str.join(',',[x.strip() for x in wavfiles])]
    else:
        return ['']

def train_actuate():
    M.save_annotations()
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    run(["train.sh", V.configuration_file.value, V.context_ms_string.value, \
         V.shiftby_ms_string.value, V.window_ms_string.value, \
         *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
         V.dropout_string.value, V.optimizer.value, V.learning_rate_string.value, \
         V.kernel_sizes_string.value, V.last_conv_width_string.value, \
         V.nfeatures_string.value, V.logs_folder.value, '1', \
         V.groundtruth_folder.value, V.wantedwords_string.value, \
         V.labeltypes_string.value, V.nsteps_string.value, \
         V.save_and_validate_period_string.value, \
         V.validate_percentage_string.value, V.mini_batch_string.value, test_files])
    if True:  # train.sh exited without error
        M.deepsong_starttime = datetime.strftime(datetime.now(),'%Y%m%dT%H%M%S')
        M.annotated_samples=[]
        M.annotated_starts_sorted=[]
        M.annotated_stops=[]
        M.iannotated_stops_sorted=[]
        M.annotated_csvfiles_all=set([])
        for label_count_widget in V.label_count_widgets:
            label_count_widget.label = str(0)
        for subdir in filter(lambda x: os.path.isdir(os.path.join(V.groundtruth_folder.value,x)), \
                             os.listdir(V.groundtruth_folder.value)):
            dfs = []
            for csvfile in filter(lambda x: '-annotated-' in x and x.endswith('.csv'), \
                                  os.listdir(os.path.join(V.groundtruth_folder.value, \
                                                          subdir))):
                filepath = os.path.join(V.groundtruth_folder.value, subdir, csvfile)
                if os.path.getsize(filepath) > 0:
                    dfs.append(pd.read_csv(filepath, header=None, index_col=False))
            if dfs:
                df = pd.concat(dfs)
                basewavs = set([os.path.splitext(x)[0] for x in df[0]])
                oldfiles = []
                for oldfile in filter(lambda x: isoldfile(x,subdir,basewavs), \
                                      os.listdir(os.path.join(V.groundtruth_folder.value, \
                                                              subdir))):
                    oldfiles.append(oldfile)
                if len(oldfiles)>0:
                    topath = os.path.join(V.groundtruth_folder.value, \
                                          subdir, \
                                          'oldfiles-'+M.deepsong_starttime)
                    os.mkdir(topath)
                    for oldfile in oldfiles:
                        os.rename(os.path.join(V.groundtruth_folder.value, subdir, oldfile), \
                                  os.path.join(topath, oldfile))
        V.wordcounts_update()

def generalize_actuate():
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    validation_files = list(filter(lambda x: not any([y!='' and y in x for y in test_files.split('m')]),
            _validation_test_files(V.validationfiles_string.value)))
    run(["generalize.sh", V.configuration_file.value, V.context_ms_string.value, \
         V.shiftby_ms_string.value, V.window_ms_string.value, \
         *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
         V.dropout_string.value, V.optimizer.value, \
         V.learning_rate_string.value, V.kernel_sizes_string.value, \
         V.last_conv_width_string.value, V.nfeatures_string.value, \
         V.logs_folder.value, V.groundtruth_folder.value, \
         V.wantedwords_string.value, V.labeltypes_string.value, \
         V.nsteps_string.value, V.save_and_validate_period_string.value, \
         V.mini_batch_string.value, test_files, *validation_files])

def xvalidate_actuate():
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    run(["xvalidate.sh", V.configuration_file.value, V.context_ms_string.value, \
         V.shiftby_ms_string.value, V.window_ms_string.value, \
         *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
         V.dropout_string.value, V.optimizer.value, \
         V.learning_rate_string.value, V.kernel_sizes_string.value, \
         V.last_conv_width_string.value, V.nfeatures_string.value, \
         V.logs_folder.value, V.groundtruth_folder.value, \
         V.wantedwords_string.value, V.labeltypes_string.value, \
         V.nsteps_string.value, V.save_and_validate_period_string.value, \
         V.mini_batch_string.value, V.kfold_string.value, test_files])

def hidden_actuate():
    run(["hidden.sh", V.configuration_file.value, V.context_ms_string.value, \
         V.shiftby_ms_string.value, V.window_ms_string.value, \
         *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
         V.kernel_sizes_string.value, \
         V.last_conv_width_string.value, V.nfeatures_string.value, \
         M.logdir, M.model, M.check_point, V.groundtruth_folder.value, \
         V.labeltypes_string.value, V.mini_batch_string.value])

def cluster_actuate():
    run(["cluster.sh", V.configuration_file.value, V.groundtruth_folder.value, V.cluster_equalize_ratio_string.value, V.cluster_max_samples_string.value, V.pca_fraction_variance_to_retain_string.value, V.tsne_perplexity_string.value, V.tsne_exaggeration_string.value])

def visualize_actuate():
    V.tsne_initialize()

def accuracy_actuate():
    run(["accuracy.sh", V.configuration_file.value, V.logs_folder.value, V.precision_recall_ratios_string.value])

def freeze_actuate():
    run(["freeze.sh", V.configuration_file.value, V.context_ms_string.value, \
         V.window_ms_string.value, *V.mel_dct_string.value.split(','), \
         V.stride_ms_string.value, V.kernel_sizes_string.value, \
         V.last_conv_width_string.value, V.nfeatures_string.value, \
         M.logdir, M.model, M.check_point])

def classify_actuate():
    for wavfile in V.wavtfcsvfiles_string.value.split(','):
        run(["classify.sh", V.configuration_file.value, V.context_ms_string.value, \
             V.shiftby_ms_string.value, V.stride_ms_string.value, \
             M.logdir, M.model, M.check_point, wavfile])

def ethogram_actuate():
    for tffile in V.wavtfcsvfiles_string.value.split(','):
        if tffile.lower().endswith('.wav'):
            tffile = os.path.splitext(tffile)[0]+'.tf'
        run(["ethogram.sh", V.configuration_file.value, M.logdir, M.model, M.check_point, tffile])
    if True:  # ethogram.sh exited without error
        V.wordcounts_update()

def compare_actuate():
    run(["compare.sh", V.configuration_file.value, V.logs_folder.value])

def dense_actuate():
    run(["dense.sh", V.configuration_file.value, V.testfiles_string.value])

def configuration_button_callback():
    assert len(V.file_dialog_source.selected.indices)==1
    configuration_file.value = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]])
    V.configuration_contents_update()

def configuration_text_callback(attr, old, new):
    M.save_state_callback()
    V.configuration_contents_update()
    V.buttons_update()

def configuration_textarea_callback(attr, old, new):
    with open(V.configuration_file.value, 'w') as fid:
        fid.write(V.configuration_contents.value)
    V.editconfiguration.button_type="default"
    V.configuration_contents.disabled=True

def logs_callback():
    assert len(V.file_dialog_source.selected.indices)<2 
    if len(V.file_dialog_source.selected.indices)==1:
        assert V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] == '.'
    V.logs_folder.value = M.file_dialog_root

def model_callback():
    assert len(V.file_dialog_source.selected.indices)==1
    V.model_file.value = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]])

def wavtfcsvfiles_callback():
    assert len(V.file_dialog_source.selected.indices)>0 
    filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]]
    files = os.path.join(M.file_dialog_root, filename)
    for i in range(1, len(V.file_dialog_source.selected.indices)):
        filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[i]]
        files += ','+os.path.join(M.file_dialog_root, filename)
    V.wavtfcsvfiles_string.value = files

def groundtruth_callback():
    assert len(V.file_dialog_source.selected.indices)<2 
    if len(V.file_dialog_source.selected.indices)==1:
        assert V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] == './'
    V.groundtruth_folder.value = M.file_dialog_root

def _validation_test_files_callback():
    nindices = len(V.file_dialog_source.selected.indices)
    if nindices==0 or (nindices==1 and V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]]=='.'):
      filepath = M.file_dialog_root
    else:
      filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]]
      filepath = os.path.join(M.file_dialog_root, filename)
    if nindices<2:
        if filepath.lower().endswith('.wav'):
            return os.path.basename(filepath)
        else:
            return filepath
    else:
        files = os.path.basename(filepath)
        for i in range(1, nindices):
            files += ','+ V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[i]]
        return files

def validationfiles_callback():
  V.validationfiles_string.value = _validation_test_files_callback()

def testfiles_callback():
  V.testfiles_string.value = _validation_test_files_callback()

def wantedwords_callback():
    assert len(V.file_dialog_source.selected.indices)==1
    labels_file = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]])
    with open(labels_file, "r") as fid:
        labels = fid.readlines()
    V.wantedwords_string.value = str.join(',',[x.strip() for x in labels])

def labelsounds_callback():
    M.wizard=None if M.wizard is V.labelsounds else V.labelsounds
    M.action=None
    V.wantedwords_string.value="time,frequency"
    V.labeltypes_string.value="detected"
    V.nsteps_string.value="0"
    V.save_and_validate_period_string.value="0"
    V.validate_percentage_string.value="0"
    V.buttons_update()
  
def makepredictions_callback():
    M.wizard=None if M.wizard is V.makepredictions else V.makepredictions
    M.action=None
    wantedwords = [x.value for x in V.label_text_widgets if x.value!='']
    if 'other' not in wantedwords:
        wantedwords.append('other')
    V.wantedwords_string.value=str.join(',',wantedwords)
    V.labeltypes_string.value="annotated"
    V.buttons_update()
  
def fixfalsepositives_callback():
    M.wizard=None if M.wizard is V.fixfalsepositives else V.fixfalsepositives
    M.action=None
    wantedwords = [x.value for x in V.label_text_widgets if x.value!='']
    if 'other' not in wantedwords:
        wantedwords.append('other')
    V.wantedwords_string.value=str.join(',',wantedwords)
    V.labeltypes_string.value="annotated,predicted"
    V.buttons_update()
    
def fixfalsenegatives_callback():
    M.wizard=None if M.wizard is V.fixfalsenegatives else V.fixfalsenegatives
    M.action=None
    wantedwords = [x.value for x in V.label_text_widgets if x.value!='']
    if 'other' not in wantedwords:
        wantedwords.append('other')
    V.wantedwords_string.value=str.join(',',wantedwords)
    V.labeltypes_string.value="annotated,missed"
    V.buttons_update()
  
def leaveoneout_callback():
    M.wizard=None if M.wizard is V.leaveoneout else V.leaveoneout
    M.action=None
    wantedwords = [x.value for x in V.label_text_widgets if x.value!='']
    if 'other' not in wantedwords:
        wantedwords.append('other')
    V.wantedwords_string.value=str.join(',',wantedwords)
    V.labeltypes_string.value="annotated"
    V.buttons_update()
  
def tunehyperparameters_callback():
    M.wizard=None if M.wizard is V.tunehyperparameters else V.tunehyperparameters
    M.action=None
    wantedwords = [x.value for x in V.label_text_widgets if x.value!='']
    if 'other' not in wantedwords:
        wantedwords.append('other')
    V.wantedwords_string.value=str.join(',',wantedwords)
    V.labeltypes_string.value="annotated"
    V.buttons_update()
  
def findnovellabels_callback():
    M.wizard=None if M.wizard is V.findnovellabels else V.findnovellabels
    M.action=None
    V.buttons_update()
  
def examineerrors_callback():
    M.wizard=None if M.wizard is V.examineerrors else V.examineerrors
    M.action=None
    V.labeltypes_string.value="annotated,correct,mistake"
    V.buttons_update()
  
def doit_callback():
    V.doit.button_type="warning"
    V.doit.disabled=True
    M.function()
    V.doit.button_type="default"
    V.doit.disabled=True
  
def editconfiguration_callback():
    if V.editconfiguration.button_type=="default":
        V.editconfiguration.button_type="danger"
        V.configuration_contents.disabled=False
    else:
        V.editconfiguration.button_type="default"
        V.configuration_contents.disabled=True

def file_dialog_callback(attr, old, new):
    if len(V.file_dialog_source.selected.indices)==1:
        iselected = V.file_dialog_source.selected.indices[0]
        selected_file = V.file_dialog_source.data['names'][iselected]
        if selected_file=='./':
            V.file_dialog_update()
        elif selected_file=='../':
            if M.file_dialog_root.endswith('/'):
                M.file_dialog_root = M.file_dialog_root[:-1]
            M.file_dialog_root = os.path.dirname(M.file_dialog_root)
            V.file_dialog_string.value = M.file_dialog_root
            V.file_dialog_update()
        else:
            selected_path = os.path.join(M.file_dialog_root, selected_file)
            if os.path.isdir(selected_path):
                M.file_dialog_root = selected_path
                V.file_dialog_string.value = M.file_dialog_root
                V.file_dialog_update()

def file_dialog_path_callback(attr, old, new):
    if os.path.isdir(new):
        M.file_dialog_root = new
        M.file_dialog_filter = '*'
    else:
        M.file_dialog_root = os.path.dirname(new)
        M.file_dialog_filter = os.path.basename(new)
    
    V.file_dialog_update()
    M.save_state_callback()

def label_count_callback(i):
    M.ilabel=i
    for button in V.label_count_widgets:
        button.button_type="default"
    V.label_count_widgets[i].button_type="primary"

def label_text_callback(new, i):
    if M.state['labels'][i]!='':
        V.label_count_widgets[i].label='0'
        M.history_stack=[]
        M.history_idx=0
        undo.disabled=True
        redo.disabled=True
    label_count_callback(i)
    M.state['labels'][i]=new
    M.nrecent_annotations+=1
    V.save_update(M.nrecent_annotations)
    M.save_state_callback()

