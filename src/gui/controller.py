import os
from subprocess import run, PIPE, STDOUT, Popen
import pandas as pd
from datetime import datetime
import numpy as np
import time
import logging 
import threading
import csv
import re
import asyncio
import math

bokehlog = logging.getLogger("songexplorer") 
#class Object(object):
#  pass
#bokehlog=Object()
#bokehlog.info=print

import model as M
import view as V

def generic_actuate(cmd, logfile, where,
                    ncpu_cores, ngpu_cards, ngigabyes_memory, localdeps, clusterflags, *args):
    args = ["\'\'" if x=="" else x for x in args]
    if V.waitfor.active:
        if localdeps=="":
            localdeps = ""
            for job in M.waitfor_job:
                localdeps += "hetero job "+job+" || "
            localdeps = localdeps[:-4]
        if " -w \"done(" not in clusterflags:
            clusterflags = " -w "
            for job in M.waitfor_job:
               clusterflags += "\"done("+job+")\"&&"
            clusterflags = clusterflags[:-2]
    if where == "local":
        p = run(["hetero", "submit",
                 "{ export CUDA_VISIBLE_DEVICES=$QUEUE1; "+cmd+" "+' '.join(args)+"; } &> "+logfile,
                 str(ncpu_cores), str(ngpu_cards), str(ngigabyes_memory), localdeps],
                stdout=PIPE, stderr=STDOUT)
        jobid = p.stdout.decode('ascii').rstrip()
        bokehlog.info(jobid)
    elif where == "server":
        p = run(["ssh", M.server_ipaddr, "export SINGULARITYENV_PREPEND_PATH="+M.source_path+";",
                 "$SONGEXPLORER_BIN", "hetero", "submit",
                 "\"{ export CUDA_VISIBLE_DEVICES=\$QUEUE1; "+cmd+" "+' '.join(args)+"; } &> "+logfile+"\"",
                 str(ncpu_cores), str(ngpu_cards), str(ngigabyes_memory), "'"+localdeps+"'"],
                stdout=PIPE, stderr=STDOUT)
        jobid = p.stdout.decode('ascii').rstrip()
        bokehlog.info(jobid)
    elif where == "cluster":
        pe = Popen(["echo",
                    "export SINGULARITYENV_PREPEND_PATH="+M.source_path+";",
                    os.environ["SONGEXPLORER_BIN"]+" "+cmd+" "+' '.join(args)],
                   stdout=PIPE)
        ps = Popen(["ssh", M.cluster_ipaddr, M.cluster_cmd,
                    #"-J ${logfile//,/}.job",
                    clusterflags,
                    M.cluster_logfile_flag+" "+logfile],
                   stdin=pe.stdout, stdout=PIPE, stderr=STDOUT)
        pe.stdout.close()
        jobinfo = ps.communicate()[0].decode('ascii').rstrip()
        jobid = re.search('([0-9]+)',jobinfo).group(1)
        bokehlog.info(jobinfo)
    return jobid

def generic_parameters_callback(n):
    if ' ' in n:
      bokehlog.info('ERROR: textboxes should not contain spaces')
    M.save_state_callback()
    V.buttons_update()

def layer_callback(new):
    M.ilayer=M.layers.index(new)
    V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
    V.cluster_update()
    M.xcluster = M.ycluster = M.zcluster = np.nan
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def species_callback(new):
    M.ispecies=M.species.index(new)
    if M.ispecies>0:
        V.which_nohyphen.value=M.nohyphens[0]
        if M.iword>0:
            V.color_picker.disabled=False
    else:
        V.color_picker.disabled=True
    V.cluster_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def word_callback(new):
    M.iword=M.words.index(new)
    if M.iword>0:
        V.which_nohyphen.value=M.nohyphens[0]
        if M.ispecies>0:
            V.color_picker.disabled=False
    else:
        V.color_picker.disabled=True
    V.cluster_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def nohyphen_callback(new):
    M.inohyphen=M.nohyphens.index(new)
    if M.inohyphen>0:
        V.which_word.value=M.words[0]
        V.which_species.value=M.species[0]
        V.color_picker.disabled=False
    else:
        V.color_picker.disabled=True
    V.cluster_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def kind_callback(new):
    M.ikind=M.kinds.index(new)
    V.cluster_update()
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()
    
def color_picker_callback(new):
    if M.inohyphen>0:
        if new.lower()=='#ffffff':
            del M.cluster_dot_colors[M.nohyphens[M.inohyphen]]
        else:
            M.cluster_dot_colors[M.nohyphens[M.inohyphen]]=new
    elif M.ispecies>0 and M.iword>0:
        if new.lower()=='#ffffff':
            del M.cluster_dot_colors[M.species[M.ispecies][:-1]+M.words[M.iword]]
        else:
            M.cluster_dot_colors[M.species[M.ispecies][:-1]+M.words[M.iword]]=new
    #elif M.ispecies>0:
    #    if new.lower()=='#ffffff':
    #        del M.cluster_dot_colors[M.species[M.ispecies]]
    #    else:
    #        M.cluster_dot_colors[M.species[M.ispecies]]=new
    #elif M.iword>0:
    #    if new.lower()=='#ffffff':
    #        del M.cluster_dot_colors[M.words[M.iword]]
    #    else:
    #        M.cluster_dot_colors[M.words[M.iword]]=new
    if not V.cluster_initialize():
        return
    V.cluster_update()
    
def circle_radius_callback(attr, old, new):
    M.state["circle_radius"]=new
    if len(V.cluster_circle_fuchsia.data['cx'])==1:
        V.cluster_circle_fuchsia.data.update(cr=[M.state["circle_radius"]])
    M.save_state_callback()
    V.snippets_update(True)
    M.isnippet = -1
    V.context_update()

def dot_size_callback(attr, old, new):
    M.state["dot_size"]=new
    V.dot_size_cluster.data.update(ds=[M.state["dot_size"]])
    M.save_state_callback()

def dot_alpha_callback(attr, old, new):
    M.state["dot_alpha"]=new
    V.dot_alpha_cluster.data.update(da=[M.state["dot_alpha"]])
    M.save_state_callback()

play_callback_code="""
const aud = document.getElementById("context_audio")
aud.src="data:audio/wav;base64,"+%r

var x0 = waveform_span_red.location;

aud.ontimeupdate = function() {
  waveform_span_red.location = x0+aud.currentTime
  spectrogram_span_red.location = x0+aud.currentTime
  probablity_span_red.location = x0+aud.currentTime
};

aud.onended = function() {
  waveform_span_red.location = x0
  spectrogram_span_red.location = x0
  probability_span_red.location = x0
};

const vid = document.getElementById("context_video")
vid.src="data:video/mp4;base64,"+%r

aud.play()
vid.play()
"""

def cluster_tap_callback(event):
    M.xcluster, M.ycluster = event[0], event[1]
    if M.ndcluster==3:
      M.zcluster = event[2]
    V.cluster_circle_fuchsia.data.update(cx=[M.xcluster],
                                         cy=[M.ycluster],
                                         cz=[M.zcluster],
                                         cr=[M.state["circle_radius"]],
                                         cc=[M.cluster_circle_color])
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()

def snippets_tap_callback(event):
    if np.isnan(M.xcluster) or np.isnan(M.ycluster):
        return
    M.xsnippet = int(np.rint(event.x/(M.snippets_gap_pix+M.snippets_pix)-0.5))
    M.ysnippet = int(np.floor(-(event.y-1)/V.snippets_dy))
    M.isnippet = M.nearest_samples[M.ysnippet*M.snippets_nx + M.xsnippet]
    V.snippets_update(False)
    V.context_update()

def context_doubletap_callback(event, midpoint):
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    currfile = M.clustered_samples[M.isnippet]['file']
    if event.y<midpoint:
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
        if M.state['labels'][M.ilabel]=='':
            return
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

pan_start_x = pan_start_y = pan_start_sx = pan_start_sy = None

def waveform_pan_start_callback(event):
    global pan_start_x, pan_start_y, pan_start_sx, pan_start_sy
    pan_start_x, pan_start_y = event.x, event.y
    pan_start_sx, pan_start_sy = event.sx, event.sy
    V.waveform_quad_grey_pan.data.update(left=[event.x], right=[event.x],
                                            top=[event.y], bottom=[event.y])

def waveform_pan_callback(event):
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    left_limit_tic = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
    right_limit_tic = left_limit_tic + M.context_width_tic
    if x_tic < left_limit_tic or x_tic > right_limit_tic:
        return
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        ichannel_start = round((pan_start_y * M.audio_nchannels - M.audio_nchannels + 1) / -2)
        ichannel_end = round((event.y * M.audio_nchannels - M.audio_nchannels + 1) / -2)
        if ichannel_start != ichannel_end:
            return
        V.waveform_quad_grey_pan.data.update(left=[V.p_waveform.x_range.start],
                                                right=[V.p_waveform.x_range.end],
                                                bottom=[pan_start_y],
                                                top=[event.y])
    elif pan_start_y<0 and M.state['labels'][M.ilabel]!='':
        if event.y < V.p_waveform.y_range.start or event.y > V.p_waveform.y_range.end:
            return
        V.waveform_quad_grey_pan.data.update(left=[pan_start_x],
                                                right=[event.x],
                                                bottom=[V.p_waveform.y_range.start],
                                                top=[0])
    
def _waveform_scale(low_y, high_y, ichannel):
    delta = M.context_waveform_high[ichannel] - M.context_waveform_low[ichannel]
    return [(x+1)/2 * delta + M.context_waveform_low[ichannel] for x in [low_y, high_y]]

def waveform_pan_end_callback(event):
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        ichannel = round((pan_start_y * M.audio_nchannels - M.audio_nchannels + 1) / -2)
        event_wy_start = pan_start_y * M.audio_nchannels - M.audio_nchannels + 1 + 2*ichannel
        event_wy_end = event.y * M.audio_nchannels - M.audio_nchannels + 1 + 2*ichannel
        if event.y > pan_start_y:
            M.context_waveform_low[ichannel], M.context_waveform_high[ichannel] = \
                    _waveform_scale(event_wy_start, min(event_wy_end, 1), ichannel)
        else:
            M.context_waveform_low[ichannel], M.context_waveform_high[ichannel] = \
                    _waveform_scale(max(event_wy_end, -1), event_wy_start, ichannel)
    elif pan_start_y<0 and M.state['labels'][M.ilabel]!='':
        x_tic0 = int(np.rint(event.x*M.audio_tic_rate))
        x_tic1 = int(np.rint(pan_start_x*M.audio_tic_rate))
        M.add_annotation({'file':M.clustered_samples[M.isnippet]['file'],
                          'ticks':sorted([x_tic0,x_tic1]),
                          'label':M.state['labels'][M.ilabel]})
    V.waveform_quad_grey_pan.data.update(left=[], right=[], top=[], bottom=[])
    V.context_update()

def waveform_tap_callback(event):
    M.context_waveform_low = [-1]*M.audio_nchannels
    M.context_waveform_high = [1]*M.audio_nchannels
    V.context_update()
    
def zoom_context_callback(attr, old, new):
    M.context_width_ms = float(new)
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    V.context_update()

def zoom_offset_callback(attr, old, new):
    M.context_offset_ms = float(new)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.context_update()

def zoomin_callback():
    if M.context_width_tic>20:
        M.context_width_ms /= 2
        M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
        V.zoom_context.value = str(M.context_width_ms)
    
def zoomout_callback():
    limit = M.file_nframes/M.audio_tic_rate*1000
    M.context_width_ms = np.minimum(limit, 2*M.context_width_ms)
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    V.zoom_context.value = str(M.context_width_ms)
    limit_lo = (M.context_width_tic//2 - M.context_midpoint_tic) / M.audio_tic_rate*1000
    limit_hi = (M.file_nframes - M.context_width_tic//2 - M.context_midpoint_tic) / \
               M.audio_tic_rate*1000
    M.context_offset_ms = np.clip(M.context_offset_ms, limit_lo, limit_hi)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def zero_callback():
    M.context_width_ms = M.context_width_ms0
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    V.zoom_context.value = str(M.context_width_ms)
    M.context_offset_ms = M.context_offset_ms0
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def panleft_callback():
    limit = (M.context_width_tic//2-M.context_midpoint_tic)/M.audio_tic_rate*1000
    M.context_offset_ms = np.maximum(limit, M.context_offset_ms-M.context_width_ms//2)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def panright_callback():
    limit = (M.file_nframes - M.context_width_tic//2 - M.context_midpoint_tic) / \
            M.audio_tic_rate*1000
    M.context_offset_ms = np.minimum(limit, M.context_offset_ms+M.context_width_ms//2)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def allleft_callback():
    M.context_offset_ms = (M.context_width_tic//2 - M.context_midpoint_tic) / \
                          M.audio_tic_rate*1000
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def allout_callback():
    M.context_width_ms = M.file_nframes/M.audio_tic_rate*1000
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    V.zoom_context.value = str(M.context_width_ms)
    M.context_offset_ms = M.context_width_ms//2 - \
                          M.context_midpoint_tic/M.audio_tic_rate*1000
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def allright_callback():
    M.context_offset_ms = (M.file_nframes-M.context_width_tic//2 - \
                           M.context_midpoint_tic) / \
                          M.audio_tic_rate*1000
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)

spectrogram_mousewheel_last_change = time.time()

def spectrogram_mousewheel_callback(event):
    global spectrogram_mousewheel_last_change
    this_change = time.time()
    if this_change-spectrogram_mousewheel_last_change < 0.5:
        return
    if not event.y:
        return
    ichannel = M.audio_nchannels-1 - int(np.floor(event.y))
    if ichannel < 0 or ichannel >= M.audio_nchannels:
        return
    spectrogram_mousewheel_last_change = this_change
    if event.delta<0:
      M.spectrogram_length_ms[ichannel] = min(float(V.zoom_context.value),
                                              M.spectrogram_length_ms[ichannel]*2)
    else:
      M.spectrogram_length_ms[ichannel] = max(2, M.spectrogram_length_ms[ichannel]/2)
    V.context_update()

def spectrogram_pan_start_callback(event):
    global pan_start_x, pan_start_y, pan_start_sx, pan_start_sy
    pan_start_x, pan_start_y = event.x, event.y
    pan_start_sx, pan_start_sy = event.sx, event.sy
    V.spectrogram_quad_grey_pan.data.update(left=[event.x], right=[event.x],
                                            top=[event.y], bottom=[event.y])

def spectrogram_pan_callback(event):
    if event.y < math.floor(pan_start_y) or event.y > math.ceil(pan_start_y):
        return
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    left_limit_tic = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
    right_limit_tic = left_limit_tic + M.context_width_tic
    if x_tic < left_limit_tic or x_tic > right_limit_tic:
        return
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        V.spectrogram_quad_grey_pan.data.update(left=[V.p_spectrogram.x_range.start],
                                                right=[V.p_spectrogram.x_range.end],
                                                bottom=[pan_start_y],
                                                top=[event.y])
    elif M.state['labels'][M.ilabel]!='':
        V.spectrogram_quad_grey_pan.data.update(left=[pan_start_x],
                                                right=[event.x],
                                                bottom=[V.p_spectrogram.y_range.start],
                                                top=[M.audio_nchannels/2])
    
def spectrogram_pan_end_callback(event):
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        ichannel = M.audio_nchannels-1 - int(np.floor(pan_start_y))
        freq_range = M.spectrogram_high_hz[ichannel] - M.spectrogram_low_hz[ichannel]
        old_low_hz = M.spectrogram_low_hz[ichannel]
        if event.y > pan_start_y:
            M.spectrogram_low_hz[ichannel] = (pan_start_y - int(pan_start_y)) * \
                                             freq_range + old_low_hz
            M.spectrogram_high_hz[ichannel] = min(1, event.y - int(pan_start_y)) * \
                                              freq_range + old_low_hz
        else:
            M.spectrogram_low_hz[ichannel] = max(0, event.y - int(pan_start_y)) * \
                                             freq_range + old_low_hz
            M.spectrogram_high_hz[ichannel] = (pan_start_y - int(pan_start_y)) * \
                                              freq_range + old_low_hz
    elif M.state['labels'][M.ilabel]!='':
      x_tic0 = int(np.rint(event.x*M.audio_tic_rate))
      x_tic1 = int(np.rint(pan_start_x*M.audio_tic_rate))
      M.add_annotation({'file':M.clustered_samples[M.isnippet]['file'],
                        'ticks':sorted([x_tic0,x_tic1]),
                        'label':M.state['labels'][M.ilabel]})
    V.spectrogram_quad_grey_pan.data.update(left=[], right=[], top=[], bottom=[])
    V.context_update()
    
def spectrogram_tap_callback(event):
    ichannel = M.audio_nchannels-1 - int(np.floor(event.y))
    M.spectrogram_low_hz[ichannel] = 0
    M.spectrogram_high_hz[ichannel] = M.audio_tic_rate/2
    V.context_update()
    
def toggle_annotation(idouble_tapped_sample):
    iannotated = M.isannotated(M.clustered_samples[idouble_tapped_sample])
    if len(iannotated)>0:
        M.delete_annotation(iannotated[0])
    else:
        thissample = M.clustered_samples[idouble_tapped_sample].copy()
        thissample['label'] = M.state['labels'][M.ilabel]
        thissample.pop('kind', None)
        M.add_annotation(thissample)

def snippets_doubletap_callback(event):
    x_tic = int(np.rint(event.x/(M.snippets_gap_pix+M.snippets_pix)-0.5))
    y_tic = int(np.floor(-(event.y-1)/V.snippets_dy))
    idouble_tapped_sample = M.nearest_samples[y_tic*M.snippets_nx + x_tic]
    toggle_annotation(idouble_tapped_sample)

def _find_nearest_clustered_sample(history_idx):
    M.isnippet = np.searchsorted(M.clustered_starts_sorted, \
                                 M.history_stack[history_idx][1]['ticks'][0])
    delta=0
    while True:
        if M.isnippet+delta < len(M.clustered_samples) and \
                  M.clustered_samples[M.isnippet+delta]['file'] == \
                  M.history_stack[history_idx][1]['file']:
            M.isnippet += delta
            break
        elif M.isnippet-delta >= 0 and \
                  M.clustered_samples[M.isnippet-delta]['file'] == \
                  M.history_stack[history_idx][1]['file']:
            M.isnippet -= delta
            break
        if M.isnippet+delta >= len(M.clustered_samples) and M.isnippet-delta < 0:
            break
        delta += 1
    if M.clustered_samples[M.isnippet]['file'] != \
            M.history_stack[history_idx][1]['file']:
        bokehlog.info("WARNING: can't jump to undone annotation")

def _undo_callback():
    time.sleep(0.5)
    if M.history_stack[M.history_idx][0]=='add':
        iannotated = np.searchsorted(M.annotated_starts_sorted, \
                                     M.history_stack[M.history_idx][1]['ticks'][0])
        while M.annotated_samples[iannotated]!=M.history_stack[M.history_idx][1]:
            iannotated += 1
        M.delete_annotation(iannotated, addto_history=False)
    elif M.history_stack[M.history_idx][0]=='delete':
        M.add_annotation(M.history_stack[M.history_idx][1], addto_history=False)
    
def undo_callback():
    if M.history_idx>0:
        M.history_idx-=1
        V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
        M.xcluster = M.ycluster = M.zcluster = np.nan
        M.isnippet = -1
        V.snippets_update(True)
        _find_nearest_clustered_sample(M.history_idx)
        M.context_offset_tic = M.history_stack[M.history_idx][1]['ticks'][0] - \
                               M.clustered_starts_sorted[M.isnippet]
        M.context_offset_ms = M.context_offset_tic/M.audio_tic_rate*1000
        V.zoom_offset.value = str(M.context_offset_ms)
        V.context_update()
        if bokeh_document:
            bokeh_document.add_next_tick_callback(_undo_callback)
        else:
            _undo_callback()

def _redo_callback():
    time.sleep(0.5)
    if M.history_stack[M.history_idx-1][0]=='add':
        M.add_annotation(M.history_stack[M.history_idx-1][1], addto_history=False)
    elif M.history_stack[M.history_idx-1][0]=='delete':
        iannotated = np.searchsorted(M.annotated_starts_sorted, \
                                     M.history_stack[M.history_idx-1][1]['ticks'][0])
        while M.annotated_samples[iannotated]!=M.history_stack[M.history_idx-1][1]:
            iannotated += 1
        M.delete_annotation(iannotated, addto_history=False)

def redo_callback():
    if M.history_idx<len(M.history_stack):
        M.history_idx+=1
        V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
        M.xcluster = M.ycluster = M.zcluster = np.nan
        M.isnippet = -1
        V.snippets_update(True)
        _find_nearest_clustered_sample(M.history_idx-1)
        M.context_offset_tic = M.history_stack[M.history_idx-1][1]['ticks'][0] - \
                             M.clustered_starts_sorted[M.isnippet]
        M.context_offset_ms = M.context_offset_tic/M.audio_tic_rate*1000
        V.zoom_offset.value = str(M.context_offset_ms)
        V.context_update()
        if bokeh_document:
            bokeh_document.add_next_tick_callback(_redo_callback)
        else:
            _redo_callback()

def action_callback(thisaction, thisactuate):
    M.action=None if M.action is thisaction else thisaction
    M.function=thisactuate
    V.buttons_update()

def classify_callback():
    labels_file = os.path.dirname(V.model_file.value)
    wantedwords_update(os.path.join(labels_file, "vgg_labels.txt"))
    action_callback(V.classify, classify_actuate)

async def actuate_monitor(displaystring, results, idx, isrunningfun, isdonefun, succeededfun):
    M.status_ticker_queue = {k:v for k,v in M.status_ticker_queue.items() if v!="succeeded"}
    M.status_ticker_queue[displaystring] = "pending"
    if bokeh_document: 
        bokeh_document.add_next_tick_callback(V.status_ticker_update)
    while not isrunningfun():
        await asyncio.sleep(1)
    M.status_ticker_queue[displaystring] = "running"
    if bokeh_document: 
        bokeh_document.add_next_tick_callback(V.status_ticker_update)
    while not isdonefun():
        await asyncio.sleep(1)
    for sec in [3,10,30,100,300,1000]:
        M.status_ticker_queue[displaystring] = "succeeded" if succeededfun() else "failed"
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(V.status_ticker_update)
        if M.status_ticker_queue[displaystring] == "succeeded":
            if results:
                results[idx]=True
            return
        await asyncio.sleep(sec)
    if results:
        results[idx]=False

async def actuate_finalize(threads, results, finalizefun):
    for i in range(len(threads)):
        await threads[i]
    if any(results) and bokeh_document: 
        bokeh_document.add_next_tick_callback(finalizefun)

def recent_file_exists(thisfile, reftime, verbose):
    if not os.path.exists(thisfile):
        if verbose:
            bokehlog.info("ERROR: "+thisfile+" does not exist.")
        return False
    if reftime > os.path.getmtime(thisfile):
        if verbose:
            bokehlog.info("ERROR: existing "+thisfile+" is old.")
        return False
    return True

def contains_two_timestamps(thisfile):
    with open(thisfile) as fid:
        ntimestamps=0
        for line in fid:
            try:
                datetime.strptime(line.rstrip('\n'), "%a %b %d %H:%M:%S %Z %Y")
                ntimestamps += 1
            except:
                pass
    return ntimestamps==2

def logfile_succeeded(logfile, reftime):
    if not recent_file_exists(logfile, reftime, True):
        return False
    with open(logfile) as fid:
        for line in fid:
            # https://github.com/tensorflow/tensorflow/issues/37689
            if 'Error' in line or ('E tensorflow' in line and \
                                   not "failed call to cuInit" in line and \
                                   not "kernel version" in line):
                bokehlog.info("ERROR: "+logfile+" contains Errors.")
                return False
    return True

def csvfile_succeeded(csvfile, reftime):
    return recent_file_exists(csvfile, reftime, True)
    
def pdffile_succeeded(pdffile, reftime):
    return recent_file_exists(pdffile, reftime, True)
    
def npzfile_succeeded(npzfile, reftime):
    return recent_file_exists(npzfile, reftime, True)
    
def pbfile_succeeded(pbfile, reftime):
    return recent_file_exists(pbfile, reftime, True)
    
def tffile_succeeded(tffile, reftime):
    return recent_file_exists(tffile, reftime, True)
    
def detect_succeeded(wavfile, reftime):
    logfile = wavfile[:-4]+'-detect.log'
    if not logfile_succeeded(logfile, reftime):
        return False
    csvfile = wavfile[:-4]+'-detected.csv'
    if not csvfile_succeeded(csvfile, reftime):
        return False
    return True

async def detect_actuate():
    M.waitfor_job = []
    wavfiles = V.wavtfcsvfiles_string.value.split(',')
    threads = [None] * len(wavfiles)
    results = [None] * len(wavfiles)
    await _detect_actuate(0, wavfiles, threads, results)

async def _detect_actuate(i, wavfiles, threads, results):
    wavfile = wavfiles.pop(0)
    currtime = time.time()
    logfile = os.path.splitext(wavfile)[0]+'-detect.log'
    jobid = generic_actuate("detect.sh", logfile, \
                            M.detect_where,
                            M.detect_ncpu_cores,
                            M.detect_ngpu_cards,
                            M.detect_ngigabytes_memory,
                            "",
                            M.detect_cluster_flags,
                            wavfile, \
                            *V.time_sigma_string.value.split(','), \
                            V.time_smooth_ms_string.value, \
                            V.frequency_n_ms_string.value, \
                            V.frequency_nw_string.value, \
                            *V.frequency_p_string.value.split(','), \
                            V.frequency_smooth_ms_string.value,
                            str(M.audio_tic_rate), str(M.audio_nchannels))
    M.waitfor_job.append(jobid)
    displaystring = "DETECT "+os.path.basename(wavfile)+" ("+jobid+")"
    threads[i] = asyncio.create_task(actuate_monitor(displaystring, results, i, \
                                     lambda l=logfile, t=currtime: \
                                            recent_file_exists(l, t, False), \
                                     lambda l=logfile: contains_two_timestamps(l), \
                                     lambda w=wavfile, t=currtime: detect_succeeded(w, t)))
    await asyncio.sleep(0.001)
    if wavfiles:
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(lambda i=i+1, w=wavfiles, t=threads, \
                                                  r=results: _detect_actuate(i,w,t,r))
        else:
            _detect_actuate(i+1, wavfiles, threads, results)
    else:
        asyncio.create_task(actuate_finalize(threads, results, V.groundtruth_update))
        V.waitfor_update()

def misses_succeeded(wavfile, reftime):
    logfile = wavfile[:-4]+'-misses.log'
    if not logfile_succeeded(logfile, reftime):
        return False
    csvfile = wavfile[:-4]+'-missed.csv'
    if not csvfile_succeeded(csvfile, reftime):
        return False
    return True

async def misses_actuate():
    currtime = time.time()
    csvfile1 = V.wavtfcsvfiles_string.value.split(',')[0]
    basepath = os.path.dirname(csvfile1)
    with open(csvfile1) as fid:
        csvreader = csv.reader(fid)
        row1 = next(csvreader)
    wavfile = row1[0]
    noext = os.path.join(basepath, os.path.splitext(wavfile)[0])
    logfile = noext+'-misses.log'
    jobid = generic_actuate("misses.sh", logfile, \
                            M.misses_where,
                            M.misses_ncpu_cores,
                            M.misses_ngpu_cards,
                            M.misses_ngigabytes_memory,
                            "",
                            M.misses_cluster_flags, \
                            V.wavtfcsvfiles_string.value)
    displaystring = "MISSES "+wavfile+" ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    threads, results = [None], [None]
    threads[0] = asyncio.create_task(actuate_monitor(displaystring, results, 0, \
                                     lambda l=logfile, t=currtime: \
                                            recent_file_exists(l, t, False), \
                                     lambda l=logfile: contains_two_timestamps(l), \
                                     lambda w=os.path.join(basepath, wavfile), t=currtime: \
                                            misses_succeeded(w, t)))
    asyncio.create_task(actuate_finalize(threads, results, V.groundtruth_update))

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

def _validation_test_files(files_string, comma=True):
    if files_string.rstrip('/') == V.groundtruth_folder.value.rstrip('/'):
        dfs = []
        for subdir in filter(lambda x: os.path.isdir(os.path.join(V.groundtruth_folder.value,x)), \
                             os.listdir(V.groundtruth_folder.value)):
            for csvfile in filter(lambda x: x.endswith('.csv'), \
                                  os.listdir(os.path.join(V.groundtruth_folder.value, \
                                                          subdir))):
                filepath = os.path.join(V.groundtruth_folder.value, subdir, csvfile)
                if os.path.getsize(filepath) > 0:
                    dfs.append(pd.read_csv(filepath, header=None, index_col=False))
        if dfs:
            df = pd.concat(dfs)
            wavfiles = set(df.loc[df[3]=="annotated"][0])
            return [','.join(wavfiles)] if comma else list(wavfiles)
    elif os.path.dirname(files_string.rstrip('/')) == V.groundtruth_folder.value.rstrip('/'):
        dfs = []
        for csvfile in filter(lambda x: x.endswith('.csv'), os.listdir(files_string)):
            filepath = os.path.join(files_string, csvfile)
            if os.path.getsize(filepath) > 0:
                dfs.append(pd.read_csv(filepath, header=None, index_col=False))
        if dfs:
            df = pd.concat(dfs)
            wavfiles = set(df.loc[df[3]=="annotated"][0])
            return [','.join(wavfiles)] if comma else list(wavfiles)
    elif files_string.lower().endswith('.wav'):
        return [files_string] if comma else files_string.split(',')
    elif files_string!='':
        with open(files_string, "r") as fid:
            wavfiles = fid.readlines()
        wavfiles = [x.strip() for x in wavfiles]
        return [','.join(wavfiles)] if comma else wavfiles
    else:
        return ['']

def _train_succeeded(logdir, kind, model, reftime):
    train_dir = os.path.join(logdir, kind+"_"+model)
    if not logfile_succeeded(train_dir+".log", reftime):
        return False
    if not os.path.isdir(os.path.join(logdir, "summaries_"+model)):
        bokehlog.info("ERROR: summaries_"+model+"/ does not exist.")
        return False
    if not os.path.isdir(train_dir):
        bokehlog.info("ERROR: "+train_dir+"/ does not exist.")
        return False
    train_files = os.listdir(train_dir)
    if "vgg_labels.txt" not in train_files:
        bokehlog.info("ERROR: "+train_dir+"/vgg_labels.txt does not exist.")
        return False
    eval_step_interval = save_step_interval = how_many_training_steps = None
    with open(train_dir+".log") as fid:
        for line in fid:
            if "eval_step_interval" in line:
                m=re.search('eval_step_interval = (\d+)',line)
                eval_step_interval = int(m.group(1))
            if "save_step_interval" in line:
                m=re.search('save_step_interval = (\d+)',line)
                save_step_interval = int(m.group(1))
            if "how_many_training_steps" in line:
                m=re.search('how_many_training_steps = (\d+)',line)
                how_many_training_steps = int(m.group(1))
    if eval_step_interval is None or save_step_interval is None or how_many_training_steps is None:
        bokehlog.info("ERROR: "+train_dir+".log should contain `eval_step_interval`, `save_step_interval`, and `how_many_training_steps`")
        return False
    if save_step_interval>0:
        nckpts = how_many_training_steps // save_step_interval + 1
        if len(list(filter(lambda x: x.startswith("vgg.ckpt-"), \
                           train_files))) != 3*nckpts:
            bokehlog.info("ERROR: "+train_dir+"/ should contain "+ \
                          str(3*nckpts)+" vgg.ckpt-* files.")
            return False
    if eval_step_interval>0:
        nevals = how_many_training_steps // eval_step_interval 
        if len(list(filter(lambda x: x.startswith("logits.validation.ckpt-"), \
                           train_files))) != nevals:
            bokehlog.info("ERROR: "+train_dir+"/ should contain "+str(nevals)+\
                  " logits.validation.ckpt-* files.")
            return False
    return True

def train_succeeded(logdir, nreplicates, reftime):
    for ireplicate in range(0, nreplicates):
        if not _train_succeeded(logdir, "train", str(1+ireplicate)+"r", reftime):
            return False
    return True

def train_generalize_xvalidate_finished(lastlogfile, reftime):
    if recent_file_exists(lastlogfile, reftime, False):
        return contains_two_timestamps(lastlogfile)
    return False

def sequester_stalefiles():
    M.songexplorer_starttime = datetime.strftime(datetime.now(),'%Y%m%dT%H%M%S')
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
                                      'oldfiles-'+M.songexplorer_starttime)
                os.mkdir(topath)
                for oldfile in oldfiles:
                    os.rename(os.path.join(V.groundtruth_folder.value, subdir, oldfile), \
                              os.path.join(topath, oldfile))
    V.groundtruth_update()

async def train_actuate():
    M.save_annotations()
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    currtime = time.time()
    jobids = []
    os.makedirs(V.logs_folder.value, exist_ok=True)
    nreplicates = int(V.replicates_string.value)
    for ireplicate in range(1, 1+nreplicates, M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "train"+str(ireplicate)+".log")
        args = [V.context_ms_string.value, V.shiftby_ms_string.value, \
                V.representation.value, V.window_ms_string.value, \
                *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
                V.dropout_string.value, V.optimizer.value, V.learning_rate_string.value, \
                V.kernel_sizes_string.value, V.last_conv_width_string.value, \
                V.nfeatures_string.value, V.dilate_after_layer_string.value, \
                V.stride_after_layer_string.value, \
                V.connection_type.value, V.logs_folder.value, \
                V.groundtruth_folder.value, V.wantedwords_string.value, \
                V.labeltypes_string.value, V.nsteps_string.value, V.restore_from_string.value, \
                V.save_and_validate_period_string.value, \
                V.validate_percentage_string.value, V.mini_batch_string.value, test_files, \
                str(M.audio_tic_rate), str(M.audio_nchannels), \
                V.batch_seed_string.value, V.weights_seed_string.value, \
                ','.join([str(x) for x in range(ireplicate, min(1+nreplicates, \
                                                                ireplicate+M.models_per_job))])]
        if M.train_gpu == 1:
            jobid = generic_actuate("train.sh", logfile, M.train_where,
                                    M.train_gpu_ncpu_cores,
                                    M.train_gpu_ngpu_cards,
                                    M.train_gpu_ngigabytes_memory,
                                    "", M.train_gpu_cluster_flags,
                                    *args)
        else:
            jobid = generic_actuate("train.sh", logfile, M.train_where,
                                    M.train_cpu_ncpu_cores,
                                    M.train_cpu_ngpu_cards,
                                    M.train_cpu_ngigabytes_memory,
                                    "", M.train_cpu_cluster_flags,
                                    *args)
        jobids.append(jobid)
    displaystring = "TRAIN "+os.path.basename(V.logs_folder.value.rstrip('/'))+ \
                    " ("+','.join([str(x) for x in jobids])+")"
    M.waitfor_job = jobids
    V.waitfor_update()
    threads, results = [None], [None]
    logfile1 = os.path.join(V.logs_folder.value, "train1.log")
    logfileN = os.path.join(V.logs_folder.value, "train"+str(len(jobids))+".log")
    threads[0] = asyncio.create_task(actuate_monitor(displaystring, results, 0, \
                                     lambda l=logfile1, t=currtime: \
                                            recent_file_exists(l, t, False), \
                                     lambda l=logfileN, t=currtime: \
                                            train_generalize_xvalidate_finished(l, t), \
                                     lambda l=V.logs_folder.value, r=nreplicates, t=currtime: \
                                            train_succeeded(l, r, t)))
    asyncio.create_task(actuate_finalize(threads, results, sequester_stalefiles))

def generalize_xvalidate_succeeded(kind, logdir, currtime):
    summary_dirs = list(filter(lambda x: os.path.isdir(os.path.join(logdir,x)) and \
                          x.startswith("summaries_"), os.listdir(logdir)))
    kind_dirs = list(filter(lambda x: os.path.isdir(os.path.join(logdir,x)) and \
                       x.startswith(kind+"_"), os.listdir(logdir)))
    kind_logs = list(filter(lambda x: x.startswith(kind+"_") and x.endswith(".log"), \
                       os.listdir(logdir)))
    if len(summary_dirs) != len(kind_dirs) != len(kind_logs):
        bokehlog.info("ERROR: different numbers of summary directories, "+kind+\
              " directories, and "+kind+" log files.")
        return False
    for wk in [x.split('_')[1] for x in summary_dirs]:
        if not _train_succeeded(logdir, kind, wk, currtime):
            return False
    return True

async def leaveout_actuate(comma):
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    validation_files = list(filter(
            lambda x: not any([y!='' and y in x for y in test_files.split(',')]),
            _validation_test_files(V.validationfiles_string.value, comma)))
    currtime = time.time()
    jobids = []
    os.makedirs(V.logs_folder.value, exist_ok=True)
    for ivalidation_file in range(0, len(validation_files), M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "generalize"+str(1+ivalidation_file)+".log")
        args = [V.context_ms_string.value, \
                V.shiftby_ms_string.value, V.representation.value, \
                V.window_ms_string.value, \
                *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
                V.dropout_string.value, V.optimizer.value, \
                V.learning_rate_string.value, V.kernel_sizes_string.value, \
                V.last_conv_width_string.value, V.nfeatures_string.value, \
                V.dilate_after_layer_string.value, V.stride_after_layer_string.value, \
                V.connection_type.value, \
                V.logs_folder.value, V.groundtruth_folder.value, \
                V.wantedwords_string.value, V.labeltypes_string.value, \
                V.nsteps_string.value, V.restore_from_string.value, \
                V.save_and_validate_period_string.value, \
                V.mini_batch_string.value, test_files, \
                str(M.audio_tic_rate), str(M.audio_nchannels), \
                V.batch_seed_string.value, V.weights_seed_string.value, \
                str(ivalidation_file),
                *validation_files[ivalidation_file:ivalidation_file+M.models_per_job]]
        if M.generalize_gpu == 1:
            jobid = generic_actuate("generalize.sh", logfile, M.generalize_where,
                                    M.generalize_gpu_ncpu_cores,
                                    M.generalize_gpu_ngpu_cards,
                                    M.generalize_gpu_ngigabytes_memory,
                                    "", \
                                    M.generalize_gpu_cluster_flags, \
                                    *args)
        else:
            jobid = generic_actuate("generalize.sh", logfile, M.generalize_where,
                                    M.generalize_cpu_ncpu_cores,
                                    M.generalize_cpu_ngpu_cards,
                                    M.generalize_cpu_ngigabytes_memory,
                                    "", \
                                    M.generalize_cpu_cluster_flags, \
                                    *args)
        jobids.append(jobid)
    displaystring = "GENERALIZE "+os.path.basename(V.logs_folder.value.rstrip('/'))+ \
                    " ("+','.join([str(x) for x in jobids])+")"
    M.waitfor_job = jobids
    V.waitfor_update()
    logfile1 = os.path.join(V.logs_folder.value, "generalize1.log")
    logfileN = os.path.join(V.logs_folder.value, "generalize"+str(len(jobids))+".log")
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile1, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfileN, t=currtime: \
                               train_generalize_xvalidate_finished(l, t), \
                        lambda l=V.logs_folder.value, t=currtime: \
                               generalize_xvalidate_succeeded("generalize", l, t)))

async def xvalidate_actuate():
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    currtime = time.time()
    jobids = []
    os.makedirs(V.logs_folder.value, exist_ok=True)
    kfolds = int(V.kfold_string.value)
    for ifold in range(1, 1+kfolds, M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "xvalidate"+str(ifold)+".log")
        args = [V.context_ms_string.value, \
                V.shiftby_ms_string.value, V.representation.value, \
                V.window_ms_string.value, \
                *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
                V.dropout_string.value, V.optimizer.value, \
                V.learning_rate_string.value, V.kernel_sizes_string.value, \
                V.last_conv_width_string.value, V.nfeatures_string.value, \
                V.dilate_after_layer_string.value, V.stride_after_layer_string.value, \
                V.connection_type.value, \
                V.logs_folder.value, V.groundtruth_folder.value, \
                V.wantedwords_string.value, V.labeltypes_string.value, \
                V.nsteps_string.value, V.restore_from_string.value, \
                V.save_and_validate_period_string.value, \
                V.mini_batch_string.value, test_files, \
                str(M.audio_tic_rate), str(M.audio_nchannels), \
                V.batch_seed_string.value, V.weights_seed_string.value, \
                V.kfold_string.value, \
                ','.join([str(x) for x in range(ifold, min(1+kfolds, ifold+M.models_per_job))])]
        if M.xvalidate_gpu == 1:
            jobid = generic_actuate("xvalidate.sh", logfile, M.xvalidate_where,
                                    M.xvalidate_gpu_ncpu_cores,
                                    M.xvalidate_gpu_ngpu_cards,
                                    M.xvalidate_gpu_ngigabytes_memory,
                                    "", \
                                    M.xvalidate_gpu_cluster_flags, \
                                    *args)
        else:
            jobid = generic_actuate("xvalidate.sh", logfile, M.xvalidate_where,
                                    M.xvalidate_cpu_ncpu_cores,
                                    M.xvalidate_cpu_ngpu_cards,
                                    M.xvalidate_cpu_ngigabytes_memory,
                                    "", \
                                    M.xvalidate_cpu_cluster_flags, \
                                    *args)
        jobids.append(jobid)

    displaystring = "XVALIDATE "+os.path.basename(V.logs_folder.value.rstrip('/'))+ \
                    " ("+','.join([str(x) for x in jobids])+")"
    M.waitfor_job = jobids
    V.waitfor_update()
    logfile1 = os.path.join(V.logs_folder.value, "xvalidate1.log")
    logfileN = os.path.join(V.logs_folder.value, "xvalidate"+str(len(jobids))+".log")
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile1, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfileN, t=currtime: \
                               train_generalize_xvalidate_finished(l, t), \
                        lambda l=V.logs_folder.value, t=currtime: \
                               generalize_xvalidate_succeeded("xvalidate", l, t)))

def mistakes_succeeded(groundtruthdir, reftime):
    logfile = os.path.join(groundtruthdir, "mistakes.log")
    if not logfile_succeeded(logfile, reftime):
        return False
    return True

async def mistakes_actuate():
    currtime = time.time()
    logfile = os.path.join(V.groundtruth_folder.value, "mistakes.log")
    jobid = generic_actuate("mistakes.sh", logfile,
                            M.mistakes_where,
                            M.mistakes_ncpu_cores,
                            M.mistakes_ngpu_cards,
                            M.mistakes_ngigabytes_memory,
                            "",
                            M.mistakes_cluster_flags,
                            V.groundtruth_folder.value)
    displaystring = "MISTAKES "+os.path.basename(V.groundtruth_folder.value.rstrip('/'))+ \
                    " ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda g=V.groundtruth_folder.value, t=currtime: \
                               mistakes_succeeded(g, t)))

def activations_cluster_succeeded(kind, groundtruthdir, reftime):
    logfile = os.path.join(groundtruthdir, kind+".log")
    if not logfile_succeeded(logfile, reftime):
        return False
    npzfile = os.path.join(groundtruthdir, kind+".npz")
    if not npzfile_succeeded(npzfile, reftime):
        return False
    if bokeh_document: 
        bokeh_document.add_next_tick_callback(V.cluster_these_layers_update)
    return True

async def activations_actuate():
    currtime = time.time()
    logdir, model, _, check_point = M.parse_model_file(V.model_file.value)
    logfile = os.path.join(V.groundtruth_folder.value, "activations.log")
    args = [V.context_ms_string.value, \
            V.shiftby_ms_string.value, V.representation.value, \
            V.window_ms_string.value, \
            *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
            V.kernel_sizes_string.value, V.last_conv_width_string.value, \
            V.nfeatures_string.value, V.dilate_after_layer_string.value, \
            V.stride_after_layer_string.value, \
            V.connection_type.value, \
            logdir, model, check_point, V.groundtruth_folder.value, \
            V.wantedwords_string.value, V.labeltypes_string.value, \
            V.activations_equalize_ratio_string.value, \
            V.activations_max_samples_string.value, \
            V.mini_batch_string.value, \
            str(M.audio_tic_rate), str(M.audio_nchannels)]
    if M.activations_gpu:
        jobid = generic_actuate("activations.sh", logfile, M.activations_where,
                                M.activations_gpu_ncpu_cores,
                                M.activations_gpu_ngpu_cards,
                                M.activations_gpu_ngigabytes_memory,
                                "",
                                M.activations_gpu_cluster_flags,
                                *args)
    else:
        jobid = generic_actuate("activations.sh", logfile, M.activations_where,
                                M.activations_cpu_ncpu_cores,
                                M.activations_cpu_ngpu_cards,
                                M.activations_cpu_ngigabytes_memory,
                                "",
                                M.activations_cpu_cluster_flags,
                                *args)

    displaystring = "ACTIVATIONS " + \
                    os.path.join(os.path.basename(logdir), model, "ckpt-"+check_point) + \
                    " ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda g=V.groundtruth_folder.value, t=currtime: \
                               activations_cluster_succeeded("activations", g, t)))

async def cluster_actuate():
    currtime = time.time()
    algorithm, ndims = V.cluster_algorithm.value.split(' ')
    these_layers = ','.join([x for x in V.cluster_these_layers.value])
    logfile = os.path.join(V.groundtruth_folder.value, "cluster.log")
    args = [V.groundtruth_folder.value, \
            these_layers, \
            V.pca_fraction_variance_to_retain_string.value, \
            str(M.pca_batch_size), \
            algorithm, ndims, str(M.cluster_parallelize)]
    if algorithm == "tSNE":
        args.extend([V.tsne_perplexity_string.value, V.tsne_exaggeration_string.value])
    elif algorithm == "UMAP":
        args.extend([V.umap_neighbors_string.value, V.umap_distance_string.value])
    jobid = generic_actuate("cluster.sh", logfile,
                            M.cluster_where,
                            M.cluster_ncpu_cores,
                            M.cluster_ngpu_cards,
                            M.cluster_ngigabytes_memory,
                            "",
                            M.cluster_cluster_flags,
                            *args)

    displaystring = "CLUSTER "+os.path.basename(V.groundtruth_folder.value.rstrip('/'))+ \
                    " ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda g=V.groundtruth_folder.value, t=currtime: \
                               activations_cluster_succeeded("cluster", g, t)))

async def visualize_actuate():
    if not V.cluster_initialize():
        return
    V.which_layer.value = M.layers[M.ilayer]
    V.which_species.value = M.species[M.ispecies]
    V.which_word.value = M.words[M.iword]
    V.which_nohyphen.value = M.nohyphens[M.inohyphen]
    V.which_kind.value = M.kinds[M.ikind]
    V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
    V.cluster_update()
    M.xcluster = M.ycluster = M.zcluster = np.nan
    M.isnippet = -1
    V.snippets_update(True)
    V.context_update()

def accuracy_succeeded(logdir, reftime):
    logfile = os.path.join(logdir, 'accuracy.log')
    if not logfile_succeeded(logfile, reftime):
        return False
    traindirs = list(filter(lambda x: os.path.isdir(os.path.join(logdir,x)) and \
                            not x.startswith('summaries_'), os.listdir(logdir)))
    toplevelfiles = ["accuracy.pdf", "train-loss.pdf", "validation-F1.pdf"]
    if len(traindirs)>1:
        toplevelfiles.append("train-overlay.pdf")
        toplevelfiles.append("confusion-matrices.pdf")
    with open(os.path.join(logdir, traindirs[0], 'vgg_labels.txt'), 'r') as fid:
        labels = fid.read().splitlines()
    for label in labels:
        toplevelfiles.append("validation-PvR-"+label+".pdf")
    for toplevelfile in toplevelfiles:
        if not pdffile_succeeded(os.path.join(logdir, toplevelfile), reftime):
            return False
    one_fold_has_thresholds = False
    for traindir in traindirs:
        trainfiles = os.listdir(os.path.join(logdir,traindir))
        nckpts = len(list(filter( \
                lambda x: x.startswith('logits.validation.ckpt-'), trainfiles)))
        nprecision = len(list(filter( \
                lambda x: x.startswith('precision-recall.ckpt-') and \
                          reftime < os.path.getmtime(os.path.join(logdir,traindir,x)), \
                trainfiles)))
        #npredictions = len(list(filter( \
        #        lambda x: x.startswith('predictions.ckpt-') and \
        #                  reftime < os.path.getmtime(os.path.join(logdir,traindir,x)), \
        #        trainfiles)))
        nprobability = len(list(filter( \
                lambda x: x.startswith('probability-density.ckpt-') and \
                          reftime < os.path.getmtime(os.path.join(logdir,traindir,x)), \
                trainfiles)))
        nspecificity = len(list(filter( \
                lambda x: x.startswith('specificity-sensitivity.ckpt-') and \
                          reftime < os.path.getmtime(os.path.join(logdir,traindir,x)), \
                trainfiles)))
        nthresholds = len(list(filter( \
                lambda x: x.startswith('thresholds.ckpt-') and \
                          reftime < os.path.getmtime(os.path.join(logdir,traindir,x)), \
                trainfiles)))
        if not nprecision==nprobability==nspecificity==nthresholds:  #==npredictions
            bokehlog.info("ERROR: number of files are not equal to each other"+str(nprecision)+' '+' '+str(nprobability)+' '+str(nspecificity)+' '+str(nthresholds))  #+str(npredictions)
            return False
        if nthresholds>0:
            one_fold_has_thresholds = True
            if nthresholds!=nckpts:
                bokehlog.info("ERROR: number of files are not equal to number of checkpoints in "+traindir)
                return False
    if not one_fold_has_thresholds:
        bokehlog.info("ERROR: no fold has thresholds")
        return False
    return True

async def accuracy_actuate():
    currtime = time.time()
    logfile = os.path.join(V.logs_folder.value, "accuracy.log")
    jobid = generic_actuate("accuracy.sh", logfile,
                            M.accuracy_where,
                            M.accuracy_ncpu_cores,
                            M.accuracy_ngpu_cards,
                            M.accuracy_ngigabytes_memory,
                            "",
                            M.accuracy_cluster_flags,
                            V.logs_folder.value, \
                            V.precision_recall_ratios_string.value, \
                            str(M.nprobabilities), str(M.accuracy_parallelize))
    displaystring = "ACCURACY "+os.path.basename(V.logs_folder.value.rstrip('/'))+ \
                    " ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda l=V.logs_folder.value, t=currtime: accuracy_succeeded(l, t)))
 
def freeze_succeeded(modeldir, ckpt, reftime):
    logfile = os.path.join(modeldir, "freeze.ckpt-"+str(ckpt)+".log")
    if not logfile_succeeded(logfile, reftime):
        return False
    logfile = os.path.join(modeldir, "frozen-graph.ckpt-"+str(ckpt)+".log")
    if not logfile_succeeded(logfile, reftime):
        return False
    pbfile = os.path.join(modeldir, "frozen-graph.ckpt-"+str(ckpt)+".pb")
    if not pbfile_succeeded(pbfile, reftime):
        return False
    return True

async def freeze_actuate():
    M.waitfor_job = []
    await _freeze_actuate(V.model_file.value.split(','))

async def _freeze_actuate(ckpts):
    ckpt = ckpts.pop(0)
    currtime = time.time()
    logdir, model, _, check_point = M.parse_model_file(ckpt)
    logfile = os.path.join(logdir, model, "freeze.ckpt-"+str(check_point)+".log")
    jobid = generic_actuate("freeze.sh", logfile,
                            M.freeze_where,
                            M.freeze_ncpu_cores,
                            M.freeze_ngpu_cards,
                            M.freeze_ngigabytes_memory,
                            "",
                            M.freeze_cluster_flags,
                            V.context_ms_string.value, \
                            V.representation.value, V.window_ms_string.value, \
                            V.stride_ms_string.value, *V.mel_dct_string.value.split(','), \
                            V.kernel_sizes_string.value, \
                            V.last_conv_width_string.value, V.nfeatures_string.value, \
                            V.dilate_after_layer_string.value, \
                            V.stride_after_layer_string.value, \
                            V.connection_type.value, \
                            logdir, model, check_point, str(M.nwindows), \
                            str(M.audio_tic_rate), str(M.audio_nchannels))
    displaystring = "FREEZE " + \
                    os.path.join(os.path.basename(logdir), model, "ckpt-"+check_point) + \
                    " ("+jobid+")"
    M.waitfor_job.append(jobid)
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda m=os.path.join(logdir, model), c=check_point, t=currtime: \
                               freeze_succeeded(m, c, t)))
    await asyncio.sleep(0.001)
    if ckpts:
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(lambda c=ckpts: _freeze_actuate(c))
        else:
            _freeze_actuate(ckpts)
    else:
        V.waitfor_update()

def classify_isdone(wavlogfile, reftime):
    return recent_file_exists(wavlogfile, reftime, False) and \
           contains_two_timestamps(wavlogfile)

def classify_succeeded(modeldir, wavfile, reftime):
    tffile = wavfile[:-4]+".tf"
    if not tffile_succeeded(tffile, reftime):
        return False
    with open(os.path.join(modeldir, 'vgg_labels.txt'), 'r') as fid:
        labels = fid.read().splitlines()
    for x in labels:
        if not recent_file_exists(wavfile[:-4]+'-'+x+'.wav', reftime, True):
            return False
    return True

async def classify_actuate():
    M.waitfor_job = []
    await _classify_actuate(V.wavtfcsvfiles_string.value.split(','))

async def _classify_actuate(wavfiles):
    wavfile = wavfiles.pop(0)
    currtime = time.time()
    logdir, model, _, check_point = M.parse_model_file(V.model_file.value)
    logfile0 = os.path.splitext(wavfile)[0]+'-classify'
    logfile1 = logfile0+'1.log'
    logfile2 = logfile0+'2.log'
    args = [V.context_ms_string.value, \
            V.shiftby_ms_string.value, V.representation.value, \
            V.stride_ms_string.value, \
            logdir, model, check_point, wavfile, str(M.audio_tic_rate), str(M.nwindows)]
    if V.prevalences_string.value!='':
        args += [V.wantedwords_string.value, V.prevalences_string.value]
    p = run(["date", "+%s"], stdout=PIPE, stderr=STDOUT)
    if M.classify_gpu:
        jobid1 = generic_actuate("classify1.sh", logfile1, M.classify_where,
                                  M.classify1_gpu_ncpu_cores,
                                  M.classify1_gpu_ngpu_cards,
                                  M.classify1_gpu_ngigabytes_memory,
                                  "",
                                  M.classify1_gpu_cluster_flags,
                                  *args)
    else:
        jobid1 = generic_actuate("classify1.sh", logfile1, M.classify_where,
                                  M.classify1_cpu_ncpu_cores,
                                  M.classify1_cpu_ngpu_cards,
                                  M.classify1_cpu_ngigabytes_memory,
                                  "",
                                  M.classify1_cpu_cluster_flags,
                                  *args)
    jobid2 = generic_actuate("classify2.sh", logfile2, M.classify_where,
                            M.classify2_ncpu_cores,
                            M.classify2_ngpu_cards,
                            M.classify2_ngigabytes_memory,
                            "hetero job "+jobid1,
                            M.classify2_cluster_flags+" -w \"done("+jobid1+")\"", *args)
    displaystring = "CLASSIFY "+os.path.basename(wavfile)+" ("+jobid1+','+jobid2+")"
    M.waitfor_job.append(jobid2)
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                     lambda l=logfile1, t=currtime: recent_file_exists(l, t, False), \
                     lambda w=logfile2, t=currtime: classify_isdone(w, t), \
                     lambda m=os.path.join(logdir, model), w=wavfile, t=currtime: \
                            classify_succeeded(m, w, t)))
    await asyncio.sleep(0.001)
    if wavfiles:
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(lambda w=wavfiles: _classify_actuate(w))
        else:
            _classify_actuate(wavfiles)
    else:
        V.waitfor_update()

def ethogram_succeeded(modeldir, ckpt, tffile, reftime):
    thresholds_file = os.path.join(modeldir, 'thresholds.ckpt-'+str(ckpt)+'.csv')
    if not os.path.exists(thresholds_file):
        return False
    with open(thresholds_file) as fid:
        csvreader = csv.reader(fid)
        row1 = next(csvreader)
    precision_recalls = row1[1:]
    for x in precision_recalls:
        if not recent_file_exists(tffile[:-3]+'-predicted-'+x+'pr.csv', reftime, True):
            return False
    return True

async def ethogram_actuate():
    tffiles = V.wavtfcsvfiles_string.value.split(',')
    threads = [None] * len(tffiles)
    results = [None] * len(tffiles)
    await _ethogram_actuate(0, tffiles, threads, results)

async def _ethogram_actuate(i, tffiles, threads, results):
    tffile = tffiles.pop(0)
    currtime = time.time()
    logdir, model, prefix, check_point = M.parse_model_file(V.model_file.value)
    if 'thresholds' in prefix:
        thresholds_file =  prefix+'.ckpt-'+check_point+'.csv'
    else:
        thresholds_file =  'thresholds.ckpt-'+check_point+'.csv'
    if tffile.lower().endswith('.wav'):
        tffile = os.path.splitext(tffile)[0]+'.tf'
    logfile = tffile[:-3]+'-ethogram.log'
    jobid = generic_actuate("ethogram.sh", logfile, M.ethogram_where,
                            M.ethogram_ncpu_cores,
                            M.ethogram_ngpu_cards,
                            M.ethogram_ngigabytes_memory,
                            "", M.ethogram_cluster_flags,
                            logdir, model, thresholds_file, tffile,
                            str(M.audio_tic_rate))
    displaystring = "ETHOGRAM "+os.path.basename(tffile)+" ("+jobid+")"
    M.waitfor_job.append(jobid)
    threads[i] = asyncio.create_task(actuate_monitor(displaystring, results, i, \
                                     lambda l=logfile, t=currtime: \
                                            recent_file_exists(l, t, False), \
                                     lambda l=logfile: contains_two_timestamps(l), \
                                     lambda m=os.path.join(logdir, model), \
                                            c=check_point, l=tffile, t=currtime: \
                                            ethogram_succeeded(m, c, l, t)))
    await asyncio.sleep(0.001)
    if tffiles:
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(lambda i=i+1, tf=tffiles, th=threads, \
                                                  r=results: _ethogram_actuate(i,tf,th,r))
        else:
            _ethogram_actuate(i+1, tffiles, threads, results)
    else:
        asyncio.create_task(actuate_finalize(threads, results, V.groundtruth_update))
        V.waitfor_update()

def compare_succeeded(logdirprefix, reftime):
    logfile = logdirprefix+'-compare.log'
    if not logfile_succeeded(logfile, reftime):
        return False
    for pdffile in ["-compare-overall-params-speed.pdf", 
                    "-compare-confusion-matrices.pdf", 
                    "-compare-precision-recall.pdf"]:
        if not pdffile_succeeded(logdirprefix+pdffile, reftime):
            return False
    return True

async def compare_actuate():
    currtime = time.time()
    logfile = V.logs_folder.value+'-compare.log'
    jobid = generic_actuate("compare.sh", logfile,
                            M.compare_where,
                            M.compare_ncpu_cores,
                            M.compare_ngpu_cards,
                            M.compare_ngigabytes_memory,
                            "",
                            M.compare_cluster_flags,
                            V.logs_folder.value)
    displaystring = "COMPARE "+os.path.basename(V.logs_folder.value.rstrip('/'))+ \
                    " ("+jobid+")"
    M.waitfor_job = jobid
    V.waitfor_update()
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda l=V.logs_folder.value, t=currtime: compare_succeeded(l, t)))

def congruence_succeeded(groundtruth_folder, reftime, regex_files):
    logfile = os.path.join(groundtruth_folder,'congruence.log')
    if not logfile_succeeded(logfile, reftime):
        return False
    listfiles = os.listdir(groundtruth_folder)
    csvfiles = list(filter(lambda x: x.endswith(".csv"), listfiles))
    pdffiles = list(filter(lambda x: x.endswith(".pdf"), listfiles))
    for (suffix, allfiles) in zip(["CSV", "PDF"], [csvfiles, pdffiles]):
        ntic = len(list(filter(lambda x: x.startswith("congruence.tic") and reftime <=
                                         os.path.getmtime(os.path.join(groundtruth_folder,x)),
                               allfiles)))
        nword = len(list(filter(lambda x: x.startswith("congruence.word") and reftime <=
                                          os.path.getmtime(os.path.join(groundtruth_folder,x)),
                                allfiles)))
        if ntic != nword or ntic==0:
            bokehlog.info("ERROR: missing congruence-{tic,word} "+suffix+" files.")
            return False
    for subdir in filter(lambda x: os.path.isdir(os.path.join(groundtruth_folder,x)), \
                         os.listdir(groundtruth_folder)):
        listfiles = os.listdir(os.path.join(groundtruth_folder,subdir))
        csvfiles = list(filter(lambda x: re.match(regex_files, x) and reftime <=
                                         os.path.getmtime(os.path.join(groundtruth_folder,subdir,x)),
                               listfiles))
        if len(csvfiles)==0:
          continue
        ndisjoint_everyone = len(list(filter(lambda x: "disjoint-everyone" in x, csvfiles)))
        for disjointstr in ["disjoint-tic-only", "disjoint-tic-not", \
                            "disjoint-word-only", "disjoint-word-not"]:
            n = len(list(filter(lambda x: disjointstr in x, csvfiles)))
            if n % ndisjoint_everyone != 0:
                bokehlog.info("ERROR: # of "+k+ \
                              " CSV files does not match # of disjoint-everyone CSV files.")
                return False
    return True

async def congruence_actuate():
    currtime = time.time()
    validation_files = _validation_test_files(V.validationfiles_string.value, False)
    test_files = _validation_test_files(V.testfiles_string.value, False)
    all_files = validation_files + test_files
    if '' in all_files:
        all_files.remove('')
    logfile = os.path.join(V.groundtruth_folder.value,'congruence.log')
    jobid = generic_actuate("congruence.sh", logfile,
                            M.congruence_where,
                            M.congruence_ncpu_cores,
                            M.congruence_ngpu_cards,
                            M.congruence_ngigabytes_memory,
                            "", \
                            M.congruence_cluster_flags,
                            V.groundtruth_folder.value, ','.join(all_files),
                            str(M.nprobabilities),
                            str(M.audio_tic_rate),
                            str(M.congruence_parallelize))
    displaystring = "CONGRUENCE "+os.path.basename(all_files[0])+" ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    regex_files = '('+'|'.join([os.path.splitext(x)[0] for x in all_files])+')*csv'
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda l=V.groundtruth_folder.value, t=currtime,
                               r=regex_files: congruence_succeeded(l, t, r)))

def waitfor_callback(arg):
    if V.waitfor.active:
        V.waitfor.button_type="warning"
    else:
        V.waitfor.button_type="default"

def configuration_textarea_callback(attr, old, new):
    with open(M.configuration_file, 'w') as fid:
        fid.write(V.configuration_contents.value)
    V.editconfiguration.button_type="default"
    V.configuration_contents.disabled=True

def logs_callback():
    if len(V.file_dialog_source.selected.indices)>=2:
        bokehlog.info('ERROR: a directory must be selected in the file browser')
        return
    if len(V.file_dialog_source.selected.indices)==1 and \
            V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] != '.':
        bokehlog.info('ERROR: a directory must be selected in the file browser')
        return
    V.logs_folder.value = M.file_dialog_root

def _dialog2list():
    if len(V.file_dialog_source.selected.indices)==0:
        bokehlog.info('ERROR: a file(s) must be selected in the file browser')
        return
    filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]]
    files = os.path.join(M.file_dialog_root, filename)
    for i in range(1, len(V.file_dialog_source.selected.indices)):
        filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[i]]
        files += ','+os.path.join(M.file_dialog_root, filename)
    return files

def model_callback():
    V.model_file.value = _dialog2list()

def wavtfcsvfiles_callback():
    V.wavtfcsvfiles_string.value = _dialog2list()

def groundtruth_callback():
    if len(V.file_dialog_source.selected.indices)>=2:
        bokehlog.info('ERROR: a directory must be selected in the file browser')
        return
    if len(V.file_dialog_source.selected.indices)==1 and \
            V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] != '.':
        bokehlog.info('ERROR: a directory must be selected in the file browser')
        return
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

def wantedwords_update(labels_file):
    if os.path.isfile(labels_file):
        with open(labels_file, "r") as fid:
            labels = fid.readlines()
        V.wantedwords_string.value = str.join(',',[x.strip() for x in labels])

def wantedwords_callback():
    if len(V.file_dialog_source.selected.indices)!=1:
        bokehlog.info('ERROR: a file must be selected in the file browser')
        return
    idx = V.file_dialog_source.selected.indices[0]
    labels_file = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    wantedwords_update(labels_file)

def prevalences_callback():
    if len(V.file_dialog_source.selected.indices)!=1:
        bokehlog.info('ERROR: a file must be selected in the file browser')
        return
    idx = V.file_dialog_source.selected.indices[0]
    classify_wav_log = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    with open(classify_wav_log,'r') as fid:
        for line in fid:
            if "prevalences: " in line:
                m=re.search('prevalences: \[(.*)\]', line)
                prevalences = m.group(1)
                break
    V.prevalences_string.value = prevalences.replace(' ',',')

def _copy_callback_finalize():
    V.copy.button_type="default"
    V.copy.disabled=False

def _copy_callback():
    if len(V.file_dialog_source.selected.indices)!=1:
        bokehlog.info('ERROR: a file must be selected in the file browser')
        _copy_callback_finalize()
        return
    idx = V.file_dialog_source.selected.indices[0]
    logfile = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    if not os.path.isfile(logfile):
        bokehlog.info('ERROR: '+logfile+' does not exist')
        _copy_callback_finalize()
        return
    with open(logfile, "r") as fid:
        for line in fid:
            if "batch_size" in line:
                m=re.search('batch_size = (\d+)', line)
                V.mini_batch_string.value = m.group(1)
            elif "clip_duration_ms" in line:
                m=re.search('clip_duration_ms = (.*)', line)
                V.context_ms_string.value = m.group(1)
            elif "time_shift_ms" in line:
                m=re.search('time_shift_ms = (.*)', line)
                V.shiftby_ms_string.value = m.group(1)
            elif "connection_type" in line:
                m=re.search('connection_type = (.*)', line)
                V.connection_type.value = m.group(1)
            elif "data_dir" in line:
                m=re.search('data_dir = (.*)', line)
                V.groundtruth_folder.value = m.group(1)
            elif "dct_coefficient_count" in line:
                m=re.search('dct_coefficient_count = (\d+)', line)
                currmel, currdct = V.mel_dct_string.value.split(',')
                V.mel_dct_string.value = ','.join([currmel, m.group(1)])
            elif "dilate_after_layer" in line:
                m=re.search('dilate_after_layer = (\d+)', line)
                V.dilate_after_layer_string.value = m.group(1)
            elif "stride_after_layer" in line:
                m=re.search('stride_after_layer = (\d+)', line)
                V.stride_after_layer_string.value = m.group(1)
            elif "dropout_prob" in line:
                m=re.search('dropout_prob = (.*)', line)
                V.dropout_string.value = m.group(1)
            elif "random_seed_batch" in line:
                m=re.search('random_seed_batch = (.*)', line)
                V.batch_seed_string.value = m.group(1)
            elif "random_seed_weights" in line:
                m=re.search('random_seed_weights = (.*)', line)
                V.weights_seed_string.value = m.group(1)
            elif "eval_step_interval" in line:
                m=re.search('eval_step_interval = (\d+)', line)
                V.save_and_validate_period_string.value = m.group(1)
            elif "filter_counts" in line:
                m=re.search('filter_counts = (.*)', line)
                V.nfeatures_string.value = m.group(1)
            elif "filter_sizes" in line:
                m=re.search('filter_sizes = (.*)', line)
                V.kernel_sizes_string.value = m.group(1)
            elif "filterbank_channel_count" in line:
                m=re.search('filterbank_channel_count = (\d+)', line)
                currmel, currdct = V.mel_dct_string.value.split(',')
                V.mel_dct_string.value = ','.join([m.group(1), currdct])
            elif "final_filter_len" in line:
                m=re.search('final_filter_len = (\d+)', line)
                V.last_conv_width_string.value = m.group(1)
            elif "how_many_training_steps" in line:
                m=re.search('how_many_training_steps = (\d+)', line)
                V.nsteps_string.value = m.group(1)
            elif "labels_touse" in line:
                m=re.search('labels_touse = (.*)', line)
                V.labeltypes_string.value = m.group(1)
            elif "learning_rate" in line:
                m=re.search('learning_rate = (.*)', line)
                V.learning_rate_string.value = m.group(1)
            elif "optimizer" in line:
                m=re.search('optimizer = (.*)', line)
                V.optimizer.value = m.group(1)
            elif "representation = " in line:
                m=re.search('representation = (.*)', line)
                V.representation.value = m.group(1)
            elif "start_checkpoint = " in line:
                m=re.search('start_checkpoint = .*ckpt-([0-9]+)', line)
                if m:
                    V.restore_from_string.value = m.group(1)
                else:
                    V.restore_from_string.value = ''
            elif "testing_files" in line:
                m=re.search('testing_files = (.*)', line)
                V.testfiles_string.value = m.group(1)
            elif "validation_files" in line:
                m=re.search('validation_files = (.*)', line)
                V.validationfiles_string.value = m.group(1)
            elif "validation_percentage" in line:
                m=re.search('validation_percentage = (.*)', line)
                V.validate_percentage_string.value = m.group(1)
                validate_perentage_float = float(V.validate_percentage_string.value)
                if validate_perentage_float!=0:
                    V.kfold_string.value = str(int(round(100/validate_perentage_float)))
                else:
                    V.kfold_string.value = "0"
            elif "wanted_words" in line:
                m=re.search('wanted_words = (.*)', line)
                V.wantedwords_string.value = m.group(1)
            elif "window_size_ms" in line:
                m=re.search('window_size_ms = (.*)', line)
                V.window_ms_string.value = m.group(1)
            elif "window_stride_ms" in line:
                m=re.search('window_stride_ms = (.*)', line)
                V.stride_ms_string.value = m.group(1)
    _copy_callback_finalize()
    
def copy_callback():
    V.copy.button_type="warning"
    V.copy.disabled=True
    bokeh_document.add_next_tick_callback(_copy_callback)

def wizard_callback(wizard):
    M.wizard=None if M.wizard is wizard else wizard
    M.action=None
    if M.wizard==V.labelsounds:
        wantedwords=[]
        for i in range(M.audio_nchannels):
            i_str = str(i) if M.audio_nchannels>1 else ''
            wantedwords.append("time"+i_str+",frequency"+i_str+",neither"+i_str)
        V.wantedwords_string.value=','.join(wantedwords)
        V.labeltypes_string.value="detected"
        V.nsteps_string.value="0"
        V.save_and_validate_period_string.value="0"
        V.validate_percentage_string.value="0"
    elif M.wizard==V.makepredictions:
        V.wantedwords_update_other()
        V.labeltypes_string.value="annotated"
    elif M.wizard==V.fixfalsepositives:
        V.wantedwords_update_other()
        V.labeltypes_string.value="annotated,predicted"
    elif M.wizard==V.fixfalsenegatives:
        V.wantedwords_update_other()
        V.labeltypes_string.value="annotated,missed"
    elif M.wizard==V.generalize:
        V.wantedwords_update_other()
        V.labeltypes_string.value="annotated"
    elif M.wizard==V.tunehyperparameters:
        V.wantedwords_update_other()
        V.labeltypes_string.value="annotated"
    elif M.wizard==V.examineerrors:
        V.labeltypes_string.value="detected,mistaken"
    elif M.wizard==V.testdensely:
        V.wantedwords_update_other()
        V.labeltypes_string.value="annotated"
    elif M.wizard==V.findnovellabels:
        wantedwords = [x.value for x in V.label_text_widgets if x.value!='']
        for i in range(M.audio_nchannels):
            i_str = str(i) if M.audio_nchannels>1 else ''
            if 'time'+i_str not in wantedwords:
                wantedwords.append('time'+i_str)
            if 'frequency'+i_str not in wantedwords:
                wantedwords.append('frequency'+i_str)
        V.wantedwords_string.value=str.join(',',wantedwords)
        if M.action==V.train:
            V.labeltypes_string.value="annotated"
        elif M.action==V.activations:
            V.labeltypes_string.value="annotated,detected"
    V.buttons_update()
  
async def _doit_callback():
    await M.function()
    V.doit.button_type="default"
    V.doit.disabled=True
  
def doit_callback():
    V.doit.button_type="warning"
    V.doit.disabled=True
    bokeh_document.add_next_tick_callback(_doit_callback)

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
            V.file_dialog_source.selected.indices = []
            V.file_dialog_update()
        elif selected_file=='../':
            if M.file_dialog_root.endswith('/'):
                M.file_dialog_root = M.file_dialog_root[:-1]
            M.file_dialog_root = os.path.dirname(M.file_dialog_root)
            V.file_dialog_string.value = M.file_dialog_root
            V.file_dialog_source.selected.indices = []
            V.file_dialog_update()
        else:
            selected_path = os.path.join(M.file_dialog_root, selected_file)
            if os.path.isdir(selected_path):
                M.file_dialog_root = selected_path
                V.file_dialog_string.value = M.file_dialog_root
                V.file_dialog_source.selected.indices = []
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
        V.undo.disabled=True
        V.redo.disabled=True
    label_count_callback(i)
    M.state['labels'][i]=new
    M.nrecent_annotations+=1
    V.save_update(M.nrecent_annotations)
    M.save_state_callback()

def init(_bokeh_document):
    global bokeh_document
    bokeh_document = _bokeh_document
