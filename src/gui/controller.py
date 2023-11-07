import os
from subprocess import run, PIPE, Popen
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
import json
import shutil
import operator
import platform
import sys

bokehlog = logging.getLogger("songexplorer") 
#class Object(object):
#  pass
#bokehlog=Object()
#bokehlog.info=print

import model as M
import view as V

def generic_actuate(cmd, logfile, where,
                    ncpu_cores, ngpu_cards, ngigabyes_memory, clusterflags, *args):
    # https://github.com/rust-lang/rust/issues/94743
    if platform.system()=='Windows':
        cmd += ".bat"
        args = [x.replace('<','^^^<').replace('>','^^^>') for x in args]
    args = ["\'\'" if x=="" else "'"+x+"'" for x in args]
    with open(logfile, 'w') as fid:
        fid.write(cmd+" "+' '.join(args)+'\n')
    localdeps = []
    if V.waitfor.active:
        for job in M.waitfor_job:
            localdeps.append("-d "+job)
        if " -w \"done(" not in clusterflags:
            clusterflags = " -w "
            for job in M.waitfor_job:
               clusterflags += "\"done("+job+")\"&&"
            clusterflags = clusterflags[:-2]
    if where == "local":
        kwargs = {"process_group": 0} if sys.version_info.major == 3 and sys.version_info.minor >= 11 else {}
        p = Popen(["hsubmit",
                   str(ncpu_cores)+','+str(ngpu_cards)+','+str(ngigabyes_memory),
                   "-o", logfile, "-e", logfile, "-a",
                   *localdeps,
                   cmd+" "+' '.join(args)],
                  stdout=PIPE, **kwargs)
        jobid = p.stdout.readline().decode('ascii').rstrip()
        bokehlog.info(jobid)
    elif where == "server":
        p = Popen(["ssh", "-l", M.server_username, M.server_ipaddr, "export SINGULARITYENV_PREPEND_PATH="+M.source_path+";",
                   "$SONGEXPLORER_BIN", "hsubmit",
                   str(ncpu_cores)+','+str(ngpu_cards)+','+str(ngigabyes_memory),
                   "-o", logfile, "-e", logfile, "-a",
                   *localdeps,
                   cmd+" "+' '.join(args).replace('"','\\"')],
                  stdout=PIPE)
        jobid = p.stdout.readline().decode('ascii').rstrip()
        bokehlog.info(jobid)
    elif where == "cluster":
        pe = Popen(["echo",
                    "export SINGULARITYENV_PREPEND_PATH="+M.source_path+";",
                    "$SONGEXPLORER_BIN "+cmd+" "+' '.join(args)],
                   stdout=PIPE)
        if M.cluster_ipaddr:
            ssh_cmd = ["ssh", M.cluster_ipaddr]
            if M.cluster_username:
                ssh_cmd = [*ssh_cmd, "-l", M.cluster_username]
        else:
            ssh_cmd = ["bash", "-c"]
        ps = Popen([*ssh_cmd, M.cluster_cmd, clusterflags, M.cluster_logfile_flag+" "+logfile],
                   stdin=pe.stdout, stdout=PIPE)
        pe.stdout.close()
        jobinfo = ps.communicate()[0].decode('ascii').rstrip()
        tmp = re.search('([0-9]+)',jobinfo)
        jobid = tmp.group(1) if hasattr(tmp,'group') else "ERROR"
        bokehlog.info(jobinfo)
    return jobid

def generic_parameters_callback(n):
    if ' ' in n:
        bokehlog.info('ERROR: textboxes should not contain spaces')
    M.save_state_callback()
    V.buttons_update()

def context_callback(n):
    V.model_summary_update()
    generic_parameters_callback(n)

def layer_callback(new):
    M.ilayer=M.layers.index(new)
    V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
    V.cluster_update()
    M.xcluster = M.ycluster = M.zcluster = np.nan
    M.isnippet = -1
    M.context_sound = None
    if V.recordings.value != '':
        M.user_changed_recording=False
    V.recordings.value = ''
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
    M.context_sound = None
    if V.recordings.value != '':
        M.user_changed_recording=False
    V.recordings.value = ''
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
    M.context_sound = None
    if V.recordings.value != '':
        M.user_changed_recording=False
    V.recordings.value = ''
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
    M.context_sound = None
    if V.recordings.value != '':
        M.user_changed_recording=False
    V.recordings.value = ''
    V.snippets_update(True)
    V.context_update()
    
def kind_callback(new):
    M.ikind=M.kinds.index(new)
    V.cluster_update()
    M.isnippet = -1
    M.context_sound = None
    if V.recordings.value != '':
        M.user_changed_recording=False
    V.recordings.value = ''
    V.snippets_update(True)
    V.context_update()
    
def color_picker_callback(new):
    if M.inohyphen>0:
        if new.lower()=='#ffffff':
            del M.label_colors[M.nohyphens[M.inohyphen]]
        else:
            M.label_colors[M.nohyphens[M.inohyphen]]=new
    elif M.ispecies>0 and M.iword>0:
        if new.lower()=='#ffffff':
            del M.label_colors[M.species[M.ispecies][:-1]+M.words[M.iword]]
        else:
            M.label_colors[M.species[M.ispecies][:-1]+M.words[M.iword]]=new
    #elif M.ispecies>0:
    #    if new.lower()=='#ffffff':
    #        del M.label_colors[M.species[M.ispecies]]
    #    else:
    #        M.label_colors[M.species[M.ispecies]]=new
    #elif M.iword>0:
    #    if new.lower()=='#ffffff':
    #        del M.label_colors[M.words[M.iword]]
    #    else:
    #        M.label_colors[M.words[M.iword]]=new
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
    M.context_sound = None
    V.context_update()

def dot_size_callback(attr, old, new):
    M.state["dot_size"]=new
    V.dot_size_cluster.data.update(ds=[M.state["dot_size"]])
    M.save_state_callback()

def dot_alpha_callback(attr, old, new):
    M.state["dot_alpha"]=new
    V.dot_alpha_cluster.data.update(da=[M.state["dot_alpha"]])
    M.save_state_callback()
    
load_multimedia_callback_code="""
const aud = document.getElementById("context_audio")
aud.src="data:audio/wav;base64,"+%r

const vid = document.getElementById("context_video")
vid.src="data:video/mp4;base64,"+%r
"""

play_callback_code="""
const aud = document.getElementById("context_audio")
const vid = document.getElementById("context_video")
var x0 = p.x_range.start
var r0 = waveform_span_red.location
var v0 = vid.currentTime

aud.ontimeupdate = function() {
  waveform_span_red.location = x0+aud.currentTime
  spectrogram_span_red.location = x0+aud.currentTime
  probability_span_red.location = x0+aud.currentTime
};

aud.onended = function() {
  waveform_span_red.location = r0
  spectrogram_span_red.location = r0
  probability_span_red.location = r0
  vid.currentTime = v0
};

vid.currentTime = 0

aud.play();
vid.play();
"""

video_slider_callback_code="""
const vid = document.getElementById("context_video")
vid.currentTime = parseFloat(cb_obj.value - cb_obj.start);
waveform_span_red.location = parseFloat(cb_obj.value)
spectrogram_span_red.location = parseFloat(cb_obj.value)
probability_span_red.location = parseFloat(cb_obj.value)
"""

def isnippet2contextsound():
    if M.used_sounds and M.clustered_sounds[M.isnippet] in M.used_sounds:
        M.context_sound = M.clustered_sounds[M.isnippet]
    else:
        M.context_sound = None

def contextsound2isnippet():
    if M.clustered_sounds and M.context_sound in M.clustered_sounds:
        M.isnippet = M.clustered_sounds.index(M.context_sound)
    else:
        M.isnippet = -1

def _recordings_callback(n):
    if M.clustered_activations is None:
        M.context_sound = None if n=="" else \
                M.used_sounds[M.used_recording2firstsound[V.recordings.value]]
    else:
        if n=="":
            M.isnippet = -1
            M.context_sound = None
            M.xcluster = M.ycluster = M.zcluster = np.nan
        else:
            M.context_sound = M.used_sounds[M.used_recording2firstsound[V.recordings.value]]
            contextsound2isnippet()
            if M.isnippet>=0:
                coordinates = M.clustered_activations[M.ilayer][M.isnippet,:]
                M.xcluster, M.ycluster = coordinates[0], coordinates[1]
                if M.ndcluster==3:
                    M.zcluster = coordinates[2]
            else:
                M.xcluster = M.ycluster = M.zcluster = np.nan
        V.cluster_circle_fuchsia.data.update(cx=[M.xcluster],
                                             cy=[M.ycluster],
                                             cz=[M.zcluster],
                                             cr=[M.state["circle_radius"]],
                                             cc=[M.cluster_circle_color])
        M.xsnippet = M.ysnippet = 0
        V.snippets_update(True)
    V.context_update()
    V.recordings.disabled=False
    V.recordings.css_classes = []

def recordings_callback(a,o,n):
    if M.user_changed_recording:
        V.recordings.disabled=True
        V.recordings.css_classes = ['changed']
        if bokeh_document:
            bokeh_document.add_next_tick_callback(lambda: _recordings_callback(n))
        else:
            _recordings_callback(n)
    M.user_changed_recording=True

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
    M.context_sound = None
    M.user_changed_recording=False
    V.recordings.value = ''
    V.snippets_update(True)
    V.context_update()

def snippets_tap_callback(event):
    if np.isnan(M.xcluster) or np.isnan(M.ycluster):
        return
    M.xsnippet = min(M.snippets_nx-1,
                     max(0, int(np.rint(event.x/(M.snippets_gap_pix+M.snippets_pix)-0.5))))
    M.ysnippet = min(M.snippets_ny-1, max(0, int(np.floor(-(event.y-1)/V.snippets_dy))))
    M.isnippet = M.nearest_sounds[M.ysnippet*M.snippets_nx + M.xsnippet]
    isnippet2contextsound()
    V.snippets_update(False)
    V.context_update()

def get_shortest_tapped_sound(x_tic, currfile):
    ileft = np.searchsorted(M.used_starts_sorted, x_tic)
    sounds_righthere = set(range(0,ileft))
    iright = np.searchsorted(M.used_stops, x_tic,
                             sorter=M.iused_stops_sorted)
    sounds_righthere &= set([M.iused_stops_sorted[i] for i in \
            range(iright, len(M.iused_stops_sorted))])
    sounds_inthisfile = filter(lambda x: M.used_sounds[x]['file'] == currfile,
                                sounds_righthere)
    return sorted(sounds_inthisfile, key=lambda x: \
                  M.used_sounds[x]['ticks'][1]-M.used_sounds[x]['ticks'][0])

def context_doubletap_callback(event, midpoint):
    if not event.x:  return
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    currfile = V.recordings.value
    if event.y<midpoint:
        idouble_tapped_sound=-1
        if len(M.annotated_starts_sorted)>0:
            undo_proximity_tic = M.context_undo_proximity_pix * M.context_width_tic / M.gui_width_pix
            idouble_tapped_sound = np.searchsorted(M.annotated_starts_sorted,
                                                   x_tic + undo_proximity_tic,
                                                   side='right') - 1
            while (idouble_tapped_sound > 0) and \
                  (M.annotated_sounds[idouble_tapped_sound]['file'] != currfile):
                idouble_tapped_sound -= 1
            if (M.annotated_sounds[idouble_tapped_sound]['file'] != currfile) or \
               (x_tic - undo_proximity_tic > M.annotated_sounds[idouble_tapped_sound]['ticks'][1]):
                idouble_tapped_sound = -1
        if idouble_tapped_sound >= 0:
            M.delete_annotation(idouble_tapped_sound)
        elif M.state['labels'][M.ilabel] != '':
            ticks = M.doubleclick_annotation(M.context_data, M.context_data_istart,
                                             M.audio_tic_rate, V.doubleclick_parameters, x_tic)
            thissound = {'file':currfile, 'ticks':ticks, 'label':M.state['labels'][M.ilabel]}
            M.add_annotation(thissound)
    else:
        isounds_shortest = get_shortest_tapped_sound(x_tic, currfile)
        if len(isounds_shortest)>0:
            M.toggle_annotation(M.used_sounds[isounds_shortest[0]])

pan_start_x = pan_start_y = pan_start_sx = pan_start_sy = None

def waveform_pan_start_callback(event):
    if not event.x:  return
    global pan_start_x, pan_start_y, pan_start_sx, pan_start_sy
    pan_start_x, pan_start_y = event.x, event.y
    pan_start_sx, pan_start_sy = event.sx, event.sy
    V.waveform_quad_grey_pan.data.update(left=[event.x], right=[event.x],
                                            top=[event.y], bottom=[event.y])

def waveform_pan_callback(event):
    if not event.x:  return
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    left_limit_tic = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
    right_limit_tic = left_limit_tic + M.context_width_tic
    if x_tic < left_limit_tic or x_tic > right_limit_tic:
        return
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        ichannel_start = round(((pan_start_y-1) * len(M.context_waveform) + 1) / -2)
        ichannel_end = round(((event.y-1) * len(M.context_waveform) + 1) / -2)
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
    if not event.x:  return
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        ichannel = round(((pan_start_y-1) * len(M.context_waveform) + 1) / -2)
        event_wy_start = (pan_start_y-1) * len(M.context_waveform) + 1 + 2*ichannel
        event_wy_end = (event.y-1) * len(M.context_waveform) + 1 + 2*ichannel
        if event.y > pan_start_y:
            M.context_waveform_low[ichannel], M.context_waveform_high[ichannel] = \
                    _waveform_scale(event_wy_start, min(event_wy_end, 1), ichannel)
        else:
            M.context_waveform_low[ichannel], M.context_waveform_high[ichannel] = \
                    _waveform_scale(max(event_wy_end, -1), event_wy_start, ichannel)
    elif pan_start_y<0 and M.state['labels'][M.ilabel]!='':
        x_tic0 = int(np.rint(event.x*M.audio_tic_rate))
        x_tic1 = int(np.rint(pan_start_x*M.audio_tic_rate))
        M.add_annotation({'file':V.recordings.value,
                          'ticks':sorted([x_tic0,x_tic1]),
                          'label':M.state['labels'][M.ilabel]})
    V.waveform_quad_grey_pan.data.update(left=[], right=[], top=[], bottom=[])
    V.context_update()

def waveform_tap_callback(event):
    if not event.x:  return
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    isounds_shortest = get_shortest_tapped_sound(x_tic, V.recordings.value)
    if event.y<0 or len(isounds_shortest)==0 or M.clustered_activations is None:
        M.context_waveform_low = [-1]*M.audio_nchannels
        M.context_waveform_high = [1]*M.audio_nchannels
    elif M.used_sounds[isounds_shortest[0]] in M.clustered_sounds:
        M.context_sound = M.used_sounds[isounds_shortest[0]]
        contextsound2isnippet()
        coordinates = M.clustered_activations[M.ilayer][M.isnippet,:]
        M.xcluster, M.ycluster = coordinates[0], coordinates[1]
        if M.ndcluster==3:
            M.zcluster = coordinates[2]
        V.cluster_circle_fuchsia.data.update(cx=[M.xcluster],
                                             cy=[M.ycluster],
                                             cz=[M.zcluster],
                                             cr=[M.state["circle_radius"]],
                                             cc=[M.cluster_circle_color])
        M.xsnippet = M.ysnippet = 0
        V.snippets_update(True)
    V.context_update()

def _spectrogram_window_callback(any_changed):
    if any_changed:
        time.sleep(0.5)
        V.spectrogram_length.css_classes = []
    M.save_state_callback()
    V.context_update()

def spectrogram_window_callback(attr, old, new):
    M.spectrogram_length_ms = [float(x) for x in V.spectrogram_length.value.split(',')]
    any_changed = False
    for ilength, length_ms in enumerate(M.spectrogram_length_ms):
        changed, length_ms2 = M.next_pow2_ms(length_ms)
        if changed:
            any_changed = True
            V.spectrogram_length.css_classes = ['changed']
            M.spectrogram_length_ms[ilength] = length_ms2
    if any_changed:
        V.spectrogram_length.value = ','.join([str(x) for x in M.spectrogram_length_ms])
    if bokeh_document:
        bokeh_document.add_next_tick_callback(lambda x=any_changed: _spectrogram_window_callback(x))
    else:
        _spectrogram_window_callback(any_changed)
    
def zoom_width_callback(attr, old, new):
    M.context_width_ms = float(new)
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    M.save_state_callback()
    V.context_update()

def zoom_offset_callback(attr, old, new):
    M.context_offset_ms = float(new)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    M.save_state_callback()
    V.context_update()

def zoomin_callback():
    if M.context_width_tic>20:
        M.context_width_ms /= 2
        M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
        V.zoom_width.value = str(M.context_width_ms)
    
def zoomout_callback():
    limit = M.file_nframes/M.audio_tic_rate*1000
    M.context_width_ms = np.minimum(limit, 2*M.context_width_ms)
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    V.zoom_width.value = str(M.context_width_ms)
    limit_lo = (M.context_width_tic//2 - M.context_midpoint_tic) / M.audio_tic_rate*1000
    limit_hi = (M.file_nframes - M.context_width_tic//2 - M.context_midpoint_tic) / \
               M.audio_tic_rate*1000
    M.context_offset_ms = np.clip(M.context_offset_ms, limit_lo, limit_hi)
    M.context_offset_tic = int(np.rint(M.context_offset_ms/1000*M.audio_tic_rate))
    V.zoom_offset.value = str(M.context_offset_ms)
    
def zero_callback():
    M.context_width_ms = M.context_width_ms0
    M.context_width_tic = int(np.rint(M.context_width_ms/1000*M.audio_tic_rate))
    V.zoom_width.value = str(M.context_width_ms)
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
    V.zoom_width.value = str(M.context_width_ms)
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

def _label_callback(inequality_fun, idx, button):
    tapped_start = M.context_sound['ticks'][0]
    snippets = np.nonzero([inequality_fun(x['ticks'][0], tapped_start) and
                           V.recordings.value == x['file'] for x in M.used_sounds])[0]
    if len(snippets)>0:
        M.context_sound = M.used_sounds[snippets[idx]]
        contextsound2isnippet()
        if M.clustered_activations is not None:
            if M.isnippet>=0:
                M.xcluster = M.clustered_activations[M.ilayer][M.isnippet,0]
                M.ycluster = M.clustered_activations[M.ilayer][M.isnippet,1]
                if M.ndcluster==3:
                    M.zcluster = M.clustered_activations[M.ilayer][M.isnippet,2]
                V.cluster_circle_fuchsia.data.update(cx=[M.xcluster],
                                                     cy=[M.ycluster],
                                                     cz=[M.zcluster],
                                                     cr=[M.state["circle_radius"]],
                                                     cc=[M.cluster_circle_color])
                M.xsnippet = M.ysnippet = 0
            else:
                M.xcluster = M.ycluster = M.zcluster = np.nan
                V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
        V.snippets_update(True)
        V.context_update()
    button.button_type="default"
 
def firstlabel_callback():
    V.firstlabel.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: _label_callback(operator.lt, 0, V.firstlabel))

def prevlabel_callback():
    V.prevlabel.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: _label_callback(operator.lt, -1, V.prevlabel))

def nextlabel_callback():
    V.nextlabel.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: _label_callback(operator.gt, 0, V.nextlabel))

def lastlabel_callback():
    V.lastlabel.button_type="warning"
    bokeh_document.add_next_tick_callback(lambda: _label_callback(operator.gt, -1, V.lastlabel))

def _touse_callback(n,button):
    if ' ' in n:
        bokehlog.info('ERROR: textboxes should not contain spaces')
    V.recordings_update()
    M.save_state_callback()
    V.recordings.disabled=False
    V.recordings.css_classes = []
    button.disabled=False
    M.user_copied_parameters=0

def touse_callback(n,button):
    if M.user_copied_parameters<2:
        M.user_copied_parameters += 1
        V.recordings.disabled=True
        V.recordings.css_classes = ['changed']
        button.disabled=True
        if bokeh_document:
            bokeh_document.add_next_tick_callback(lambda: _touse_callback(n,button))
        else:
            _touse_callback(n,button)

spectrogram_mousewheel_last_change = time.time()

def spectrogram_mousewheel_callback(event):
    global spectrogram_mousewheel_last_change
    this_change = time.time()
    if this_change-spectrogram_mousewheel_last_change < 1:
        return
    if not event.y:
        return
    ichannel = len(M.context_spectrogram) - 1 - int(np.floor(event.y))
    if ichannel < 0 or ichannel >= len(M.context_spectrogram):
        return
    spectrogram_mousewheel_last_change = this_change
    if event.delta<0:
        if 2*M.spectrogram_length_ms[ichannel] <= float(V.zoom_width.value):
            M.spectrogram_length_ms[ichannel] = 2*M.spectrogram_length_ms[ichannel]
        else:
            return
    else:
        if M.spectrogram_length_ms[ichannel]/2 > 1000/M.audio_tic_rate*2:
            M.spectrogram_length_ms[ichannel] = M.spectrogram_length_ms[ichannel]/2
        else:
            return
    V.spectrogram_length.value = ','.join([str(x) for x in M.spectrogram_length_ms])
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
                                                top=[len(M.context_spectrogram)/2])
    
def spectrogram_pan_end_callback(event):
    if abs(event.sx - pan_start_sx) < abs(event.sy - pan_start_sy):
        idx = len(M.context_spectrogram) - 1 - int(np.floor(pan_start_y))
        ichannel = M.context_spectrogram[idx] - 1
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
      M.add_annotation({'file':V.recordings.value,
                        'ticks':sorted([x_tic0,x_tic1]),
                        'label':M.state['labels'][M.ilabel]})
    V.spectrogram_quad_grey_pan.data.update(left=[], right=[], top=[], bottom=[])
    V.context_update()
    
def spectrogram_tap_callback(event):
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    isounds_shortest = get_shortest_tapped_sound(x_tic, V.recordings.value)
    if np.modf(event.y)[0]<0.5 or len(isounds_shortest)==0 or M.clustered_activations is None:
        idx = len(M.context_spectrogram) - 1 - int(np.floor(event.y))
        ichannel = M.context_spectrogram[idx] - 1
        M.spectrogram_low_hz[ichannel] = 0
        M.spectrogram_high_hz[ichannel] = M.audio_tic_rate/2
    elif M.used_sounds[isounds_shortest[0]] in M.clustered_sounds:
        M.context_sound = M.used_sounds[isounds_shortest[0]]
        contextsound2isnippet()
        coordinates = M.clustered_activations[M.ilayer][M.isnippet,:]
        M.xcluster, M.ycluster = coordinates[0], coordinates[1]
        if M.ndcluster==3:
            M.zcluster = coordinates[2]
        V.cluster_circle_fuchsia.data.update(cx=[M.xcluster],
                                             cy=[M.ycluster],
                                             cz=[M.zcluster],
                                             cr=[M.state["circle_radius"]],
                                             cc=[M.cluster_circle_color])
        V.snippets_update(True)
    V.context_update()
    
def snippets_doubletap_callback(event):
    x_tic = int(np.rint(event.x/(M.snippets_gap_pix+M.snippets_pix)-0.5))
    y_tic = int(np.floor(-(event.y-1)/V.snippets_dy))
    idouble_tapped_sound = M.nearest_sounds[y_tic*M.snippets_nx + x_tic]
    if idouble_tapped_sound!=-1:
        M.toggle_annotation(M.clustered_sounds[idouble_tapped_sound])

def _find_nearest_used_sound(history_idx):
    icontext_sound = np.searchsorted(M.used_starts_sorted, \
                                 M.history_stack[history_idx][1]['ticks'][0])
    delta=0
    while True:
        if icontext_sound+delta < len(M.used_sounds) and \
                  M.used_sounds[icontext_sound+delta]['file'] == \
                  M.history_stack[history_idx][1]['file']:
            icontext_sound += delta
            break
        elif icontext_sound-delta >= 0 and \
                  M.used_sounds[icontext_sound-delta]['file'] == \
                  M.history_stack[history_idx][1]['file']:
            icontext_sound -= delta
            break
        if icontext_sound+delta >= len(M.used_sounds) and icontext_sound-delta < 0:
            break
        delta += 1
    M.context_sound = M.used_sounds[icontext_sound]
    contextsound2isnippet()
    if V.recordings.value != M.history_stack[history_idx][1]['file']:
        bokehlog.info("WARNING: can't jump to undone annotation")

def _undo_callback():
    time.sleep(0.5)
    if M.history_stack[M.history_idx][0]=='add':
        iannotated = np.searchsorted(M.annotated_starts_sorted, \
                                     M.history_stack[M.history_idx][1]['ticks'][0])
        while M.annotated_sounds[iannotated]!=M.history_stack[M.history_idx][1]:
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
        _find_nearest_used_sound(M.history_idx)
        M.context_offset_tic = M.history_stack[M.history_idx][1]['ticks'][0] - \
                               M.context_sound['ticks'][0]
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
        while M.annotated_sounds[iannotated]!=M.history_stack[M.history_idx-1][1]:
            iannotated += 1
        M.delete_annotation(iannotated, addto_history=False)

def redo_callback():
    if M.history_idx<len(M.history_stack):
        M.history_idx+=1
        V.cluster_circle_fuchsia.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
        M.xcluster = M.ycluster = M.zcluster = np.nan
        M.isnippet = -1
        V.snippets_update(True)
        _find_nearest_used_sound(M.history_idx-1)
        M.context_offset_tic = M.history_stack[M.history_idx-1][1]['ticks'][0] - \
                             M.context_sound['ticks'][0]
        M.context_offset_ms = M.context_offset_tic/M.audio_tic_rate*1000
        V.zoom_offset.value = str(M.context_offset_ms)
        V.context_update()
        if bokeh_document:
            bokeh_document.add_next_tick_callback(_redo_callback)
        else:
            _redo_callback()

def remaining_callback():
    for isound in M.remaining_isounds:
        thissound = M.clustered_sounds[isound].copy()
        thissound.pop('kind', None)
        idx = np.searchsorted(M.annotated_starts_sorted, thissound['ticks'][0])
        M.annotated_sounds.insert(idx, thissound.copy())
        M.annotated_starts_sorted = [x['ticks'][0] for x in M.annotated_sounds]
        if thissound['label'] in M.state['labels']:
            thislabel = M.state['labels'].index(thissound['label'])
            count = int(V.nsounds_per_label_buttons[thislabel].label)
            V.nsounds_per_label_buttons[thislabel].label = str(count+1)

    M.annotated_stops = [x['ticks'][1] for x in M.annotated_sounds]
    M.iannotated_stops_sorted = np.argsort(M.annotated_stops)
    M.nrecent_annotations+=1
    V.save_update(M.nrecent_annotations)
    V.context_update()

def action_callback(thisaction, thisactuate):
    M.action=None if M.action is thisaction else thisaction
    M.function=thisactuate
    V.buttons_update()

def classify_callback():
    labels_file = os.path.dirname(V.model_file.value)
    labels_touse_update(os.path.join(labels_file, "labels.txt"))
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
    for sec in [3,10,30,100,300]:
        if displaystring in M.status_ticker_queue:
            M.status_ticker_queue[displaystring] = "succeeded" if succeededfun() else "failed"
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(V.status_ticker_update)
        if displaystring not in M.status_ticker_queue:
            if results:
                results[idx]=False
            return
        elif M.status_ticker_queue[displaystring] == "succeeded":
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
            if "start time" in line or "finish time" in line:
                try:
                    datetime.strptime(line[:line.rindex(':')], "%Y-%m-%d %H:%M:%S.%f")
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
    wavfiles = V.wavcsv_files.value.split(',')
    threads = [None] * len(wavfiles)
    results = [None] * len(wavfiles)
    await _detect_actuate(0, wavfiles, threads, results)

async def _detect_actuate(i, wavfiles, threads, results):
    wavfile = wavfiles.pop(0)
    currtime = time.time()
    logfile = os.path.splitext(wavfile)[0]+'-detect.log'
    jobid = generic_actuate(M.detect_plugin+".py", logfile, \
                            M.detect_where,
                            M.detect_ncpu_cores,
                            M.detect_ngpu_cards,
                            M.detect_ngigabytes_memory,
                            M.detect_cluster_flags,
                            "--filename="+wavfile, \
                            "--parameters="+json.dumps({k:v.value for k,v in V.detect_parameters.items()}), \
                            "--audio_tic_rate="+str(M.audio_tic_rate), \
                            "--audio_nchannels="+str(M.audio_nchannels),
                            "--audio_read_plugin="+str(M.audio_read_plugin),
                            "--audio_read_plugin_kwargs='"+json.dumps(M.audio_read_plugin_kwargs)+"'")
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
    csvfile1 = V.wavcsv_files.value.split(',')[0]
    basepath = os.path.dirname(csvfile1)
    with open(csvfile1) as fid:
        csvreader = csv.reader(fid)
        row1 = next(csvreader)
    wavfile = row1[0]
    noext = os.path.join(basepath, os.path.splitext(wavfile)[0])
    logfile = noext+'-misses.log'
    jobid = generic_actuate("misses", logfile, \
                            M.misses_where,
                            M.misses_ncpu_cores,
                            M.misses_ngpu_cards,
                            M.misses_ngigabytes_memory,
                            M.misses_cluster_flags, \
                            V.wavcsv_files.value)
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
        np.any([x.startswith(b+'-') and x.endswith('.wav') for b in basewavs]) or \
        x.endswith('-classify.log') or \
        '-predicted' in x or \
        x.endswith('-ethogram.log') or \
        '-missed' in x or \
        x.endswith('-misses.log') or \
        x == subdir+'.csv'

def _validation_test_files(files_string, comma=True):
    if files_string.rstrip(os.sep) == V.groundtruth_folder.value.rstrip(os.sep):
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
            wavfiles = sorted(list(set(df.loc[df[3]=="annotated"][0])))
            return [','.join(wavfiles)] if comma else list(wavfiles)
    elif os.path.dirname(files_string.rstrip(os.sep)) == V.groundtruth_folder.value.rstrip(os.sep):
        dfs = []
        for csvfile in filter(lambda x: x.endswith('.csv'), os.listdir(files_string)):
            filepath = os.path.join(files_string, csvfile)
            if os.path.getsize(filepath) > 0:
                dfs.append(pd.read_csv(filepath, header=None, index_col=False))
        if dfs:
            df = pd.concat(dfs)
            wavfiles = sorted(list(set(df.loc[df[3]=="annotated"][0])))
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
        bokehlog.info("ERROR: summaries_"+model+os.sep+" does not exist.")
        return False
    if not os.path.isdir(train_dir):
        bokehlog.info("ERROR: "+train_dir+os.sep+" does not exist.")
        return False
    train_files = os.listdir(train_dir)
    if "labels.txt" not in train_files:
        bokehlog.info("ERROR: "+train_dir+os.sep+"labels.txt does not exist.")
        return False
    validate_step_period = save_step_period = how_many_training_steps = None
    with open(train_dir+".log") as fid:
        for line in fid:
            if "validate_step_period = " in line:
                m=re.search('validate_step_period = (\d+)',line)
                validate_step_period = int(m.group(1))
            if "save_step_period = " in line:
                m=re.search('save_step_period = (\d+)',line)
                save_step_period = int(m.group(1))
            if "how_many_training_steps = " in line:
                m=re.search('how_many_training_steps = (\d+)',line)
                how_many_training_steps = int(m.group(1))
    if validate_step_period is None or save_step_period is None or how_many_training_steps is None:
        bokehlog.info("ERROR: "+train_dir+".log should contain `validate_step_period`, `save_step_period`, and `how_many_training_steps`")
        return False
    if save_step_period>0:
        nckpts = how_many_training_steps // save_step_period + 1
        if len(list(filter(lambda x: x.startswith("ckpt-"), \
                           train_files))) != 2*nckpts:
            bokehlog.info("ERROR: "+train_dir+os.sep+" should contain "+ \
                          str(2*nckpts)+" ckpt-* files.")
            return False
    if validate_step_period>0:
        nevals = how_many_training_steps // validate_step_period 
        if len(list(filter(lambda x: x.startswith("logits.validation.ckpt-"), \
                           train_files))) != nevals:
            bokehlog.info("ERROR: "+train_dir+os.sep+" should contain "+str(nevals)+\
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
    M.annotated_sounds=[]
    M.annotated_starts_sorted=[]
    M.annotated_stops=[]
    M.iannotated_stops_sorted=[]
    M.annotated_csvfiles_all=set([])
    for button in V.nsounds_per_label_buttons:
        button.label = str(0)
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
    test_files = _validation_test_files(V.test_files.value)[0]
    currtime = time.time()
    jobids = []
    os.makedirs(V.logs_folder.value, exist_ok=True)
    nreplicates = int(V.nreplicates.value)
    for ireplicate in range(1, 1+nreplicates, M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "train"+str(ireplicate)+".log")
        args = ["--context_ms="+V.context_ms.value, \
                "--shiftby_ms="+V.shiftby_ms.value, \
                "--optimizer="+V.optimizer.value, \
                "--learning_rate="+V.learning_rate.value, \
                "--audio_read_plugin="+str(M.audio_read_plugin), \
                "--audio_read_plugin_kwargs="+json.dumps(M.audio_read_plugin_kwargs), \
                "--video_read_plugin="+str(M.video_read_plugin), \
                "--video_read_plugin_kwargs="+json.dumps(M.video_read_plugin_kwargs), \
                "--video_findfile="+M.video_findfile_plugin, \
                "--video_bkg_frames="+str(M.video_bkg_frames), \
                "--data_loader_queuesize="+str(M.data_loader_queuesize), \
                "--data_loader_maxprocs="+str(M.data_loader_maxprocs), \
                "--model_architecture="+M.architecture_plugin, \
                "--model_parameters="+json.dumps({k:v.value for k,v in V.model_parameters.items()}), \
                "--logdir="+V.logs_folder.value, \
                "--data_dir="+V.groundtruth_folder.value, \
                "--labels_touse="+V.labels_touse.value, \
                "--kinds_touse="+V.kinds_touse.value, \
                "--nsteps="+V.nsteps.value, \
                "--restore_from="+V.restore_from.value, \
                "--save_and_validate_period="+V.save_and_validate_period.value, \
                "--validation_percentage="+V.validate_percentage.value, \
                "--mini_batch="+V.mini_batch.value, \
                "--testing_files="+test_files, \
                "--audio_tic_rate="+str(M.audio_tic_rate), \
                "--audio_nchannels="+str(M.audio_nchannels), \
                "--video_frame_rate="+str(M.video_frame_rate), \
                "--video_frame_width="+str(M.video_frame_width), \
                "--video_frame_height="+str(M.video_frame_height), \
                "--video_channels="+str(M.video_channels), \
                "--batch_seed="+V.batch_seed.value, \
                "--weights_seed="+V.weights_seed.value, \
                "--deterministic="+M.deterministic, \
                "--igpu=QUEUE1", \
                "--ireplicates="+','.join([str(x) for x in range(ireplicate, min(1+nreplicates, \
                                                                ireplicate+M.models_per_job))])]
        jobid = generic_actuate("train", logfile, M.train_where,
                                M.train_ncpu_cores,
                                M.train_ngpu_cards,
                                M.train_ngigabytes_memory,
                                M.train_cluster_flags,
                                *args)
        jobids.append(jobid)
    displaystring = "TRAIN "+os.path.basename(V.logs_folder.value.rstrip(os.sep))+ \
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
    test_files = _validation_test_files(V.test_files.value)[0]
    validation_files = list(filter(
            lambda x: not any([y!='' and y in x for y in test_files.split(',')]),
            _validation_test_files(V.validation_files.value, comma)))
    currtime = time.time()
    jobids = []
    os.makedirs(V.logs_folder.value, exist_ok=True)
    for ivalidation_file in range(0, len(validation_files), M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "generalize"+str(1+ivalidation_file)+".log")
        args = ["--context_ms="+V.context_ms.value, \
                "--shiftby_ms="+V.shiftby_ms.value, \
                "--optimizer="+V.optimizer.value, \
                "--learning_rate="+V.learning_rate.value, \
                "--audio_read_plugin="+str(M.audio_read_plugin), \
                "--audio_read_plugin_kwargs="+json.dumps(M.audio_read_plugin_kwargs), \
                "--video_read_plugin="+str(M.video_read_plugin), \
                "--video_read_plugin_kwargs="+json.dumps(M.video_read_plugin_kwargs), \
                "--video_findfile_plugin="+M.video_findfile_plugin, \
                "--video_bkg_frames="+str(M.video_bkg_frames), \
                "--data_loader_queuesize="+str(M.data_loader_queuesize), \
                "--data_loader_maxprocs="+str(M.data_loader_maxprocs), \
                "--model_architecture="+M.architecture_plugin, \
                "--model_parameters="+json.dumps({k:v.value for k,v in V.model_parameters.items()}), \
                "--logdir="+V.logs_folder.value, \
                "--data_dir="+V.groundtruth_folder.value, \
                "--labels_touse="+V.labels_touse.value, \
                "--kinds_touse="+V.kinds_touse.value, \
                "--nsteps="+V.nsteps.value, \
                "--restore_from="+V.restore_from.value, \
                "--save_and_validate_period="+V.save_and_validate_period.value, \
                "--mini_batch="+V.mini_batch.value, \
                "--testing_files="+test_files, \
                "--audio_tic_rate="+str(M.audio_tic_rate), \
                "--audio_nchannels="+str(M.audio_nchannels), \
                "--video_frame_rate="+str(M.video_frame_rate), \
                "--video_frame_width="+str(M.video_frame_width), \
                "--video_frame_height="+str(M.video_frame_height), \
                "--video_channels="+str(M.video_channels), \
                "--batch_seed="+V.batch_seed.value, \
                "--weights_seed="+V.weights_seed.value, \
                "--deterministic="+M.deterministic, \
                "--ioffset="+str(ivalidation_file),
                "--igpu=QUEUE1", \
                "--subsets",
                *validation_files[ivalidation_file:ivalidation_file+M.models_per_job]]
        jobid = generic_actuate("generalize", logfile, M.generalize_where,
                                M.generalize_ncpu_cores,
                                M.generalize_ngpu_cards,
                                M.generalize_ngigabytes_memory,
                                M.generalize_cluster_flags, \
                                *args)
        jobids.append(jobid)
    displaystring = "GENERALIZE "+os.path.basename(V.logs_folder.value.rstrip(os.sep))+ \
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
    test_files = _validation_test_files(V.test_files.value)[0]
    currtime = time.time()
    jobids = []
    os.makedirs(V.logs_folder.value, exist_ok=True)
    kfolds = int(V.kfold.value)
    for ifold in range(1, 1+kfolds, M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "xvalidate"+str(ifold)+".log")
        args = ["--context_ms="+V.context_ms.value, \
                "--shiftby_ms="+V.shiftby_ms.value, \
                "--optimizer="+V.optimizer.value, \
                "--learning_rate="+V.learning_rate.value, \
                "--audio_read_plugin="+str(M.audio_read_plugin), \
                "--audio_read_plugin_kwargs="+json.dumps(M.audio_read_plugin_kwargs), \
                "--video_read_plugin="+str(M.video_read_plugin), \
                "--video_read_plugin_kwargs="+json.dumps(M.video_read_plugin_kwargs), \
                "--video_findfile_plugin="+M.video_findfile_plugin, \
                "--video_bkg_frames="+str(M.video_bkg_frames), \
                "--data_loader_queuesize="+str(M.data_loader_queuesize), \
                "--data_loader_maxprocs="+str(M.data_loader_maxprocs), \
                "--model_architecture="+M.architecture_plugin, \
                "--model_parameters="+json.dumps({k:v.value for k,v in V.model_parameters.items()}), \
                "--logdir="+V.logs_folder.value, \
                "--data_dir="+V.groundtruth_folder.value, \
                "--labels_touse="+V.labels_touse.value, \
                "--kinds_touse="+V.kinds_touse.value, \
                "--nsteps="+V.nsteps.value, \
                "--restore_from="+V.restore_from.value, \
                "--save_and_validate_period="+V.save_and_validate_period.value, \
                "--mini_batch="+V.mini_batch.value, \
                "--testing_files="+test_files, \
                "--audio_tic_rate="+str(M.audio_tic_rate), \
                "--audio_nchannels="+str(M.audio_nchannels), \
                "--video_frame_rate="+str(M.video_frame_rate), \
                "--video_frame_width="+str(M.video_frame_width), \
                "--video_frame_height="+str(M.video_frame_height), \
                "--video_channels="+str(M.video_channels), \
                "--batch_seed="+V.batch_seed.value, \
                "--weights_seed="+V.weights_seed.value, \
                "--deterministic="+M.deterministic, \
                "--igpu=QUEUE1", \
                "--kfold="+V.kfold.value, \
                "--ifolds="+','.join([str(x) for x in range(ifold, min(1+kfolds, ifold+M.models_per_job))])]
        jobid = generic_actuate("xvalidate", logfile, M.xvalidate_where,
                                M.xvalidate_ncpu_cores,
                                M.xvalidate_ngpu_cards,
                                M.xvalidate_ngigabytes_memory,
                                M.xvalidate_cluster_flags, \
                                *args)
        jobids.append(jobid)

    displaystring = "XVALIDATE "+os.path.basename(V.logs_folder.value.rstrip(os.sep))+ \
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
    jobid = generic_actuate("mistakes", logfile,
                            M.mistakes_where,
                            M.mistakes_ncpu_cores,
                            M.mistakes_ngpu_cards,
                            M.mistakes_ngigabytes_memory,
                            M.mistakes_cluster_flags,
                            V.groundtruth_folder.value)
    displaystring = "MISTAKES "+os.path.basename(V.groundtruth_folder.value.rstrip(os.sep))+ \
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
    args = ["--context_ms="+V.context_ms.value, \
            "--shiftby_ms="+V.shiftby_ms.value, \
            "--video_findfile="+M.video_findfile_plugin, \
            "--video_bkg_frames="+str(M.video_bkg_frames), \
            "--data_loader_queuesize="+str(M.data_loader_queuesize), \
            "--data_loader_maxprocs="+str(M.data_loader_maxprocs), \
            "--model_architecture="+M.architecture_plugin, \
            "--model_parameters="+json.dumps({k:v.value for k,v in V.model_parameters.items()}), \
            "--start_checkpoint="+os.path.join(logdir,model,"ckpt-"+check_point),
            "--data_dir="+V.groundtruth_folder.value, \
            "--labels_touse="+V.labels_touse.value,
            "--kinds_touse="+V.kinds_touse.value, \
            "--testing_equalize_ratio="+V.activations_equalize_ratio.value, \
            "--testing_max_sounds="+V.activations_max_sounds.value, \
            "--batch_size="+V.mini_batch.value, \
            "--audio_tic_rate="+str(M.audio_tic_rate),
            "--audio_nchannels="+str(M.audio_nchannels),
            "--audio_read_plugin="+M.audio_read_plugin,
            "--audio_read_plugin_kwargs="+json.dumps(M.audio_read_plugin_kwargs),
            "--video_frame_rate="+str(M.video_frame_rate),
            "--video_frame_height="+str(M.video_frame_height),
            "--video_frame_width="+str(M.video_frame_width),
            "--video_read_plugin="+M.video_read_plugin,
            "--video_read_plugin_kwargs="+json.dumps(M.video_read_plugin_kwargs),
            "--video_channels="+str(M.video_channels),
            "--validation_percentage=0.0",
            "--validation_offset_percentage=0.0",
            "--random_seed_batch="+V.batch_seed.value,
            "--deterministic="+str(M.deterministic),
            "--igpu=QUEUE1", \
            "--save_activations=True"]
    jobid = generic_actuate("activations", logfile, M.activations_where,
                            M.activations_ncpu_cores,
                            M.activations_ngpu_cards,
                            M.activations_ngigabytes_memory,
                            M.activations_cluster_flags,
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
    algorithm, ndims = V.cluster_algorithm.value[:-1].split(' ')
    these_layers = ','.join([x for x in V.cluster_these_layers.value])
    logfile = os.path.join(V.groundtruth_folder.value, "cluster.log")
    args = ["--data_dir="+V.groundtruth_folder.value, \
            "--layers="+these_layers, \
            "--pca_fraction_variance_to_retain="+V.pca_fraction_variance_to_retain.value, \
            "--pca_batch_size="+str(M.pca_batch_size), \
            "--algorithm="+algorithm, 
            "--ndims="+ndims, 
            "--parallelize="+str(M.cluster_parallelize)]
    if algorithm == "tSNE":
        args.append('--kwargs={"perplexity":'+V.tsne_perplexity.value+
                    ',"exaggeration":'+V.tsne_exaggeration.value+'}')
    elif algorithm == "UMAP":
        args.append('--kwargs={"n_neighbors":'+V.umap_neighbors.value+
                    ',"min_distance":'+V.umap_distance.value+'}')
    jobid = generic_actuate("cluster", logfile,
                            M.cluster_where,
                            M.cluster_ncpu_cores,
                            M.cluster_ngpu_cards,
                            M.cluster_ngigabytes_memory,
                            M.cluster_cluster_flags,
                            *args)

    displaystring = "CLUSTER "+os.path.basename(V.groundtruth_folder.value.rstrip(os.sep))+ \
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
    M.context_sound = None
    V.snippets_update(True)
    V.context_update()

def accuracy_succeeded(logdir, reftime):
    logfile = os.path.join(logdir, 'accuracy.log')
    if not logfile_succeeded(logfile, reftime):
        return False
    traindirs = list(filter(lambda x: os.path.isdir(os.path.join(logdir,x)) and \
                            not x.startswith('summaries_'), os.listdir(logdir)))
    toplevelfiles = ["accuracy.pdf", "train-validation-loss.pdf", "validation-F1.pdf"]
    if len(traindirs)>1:
        toplevelfiles.append("validation-overlay.pdf")
        toplevelfiles.append("confusion-matrices.pdf")
    with open(os.path.join(logdir, traindirs[0], 'labels.txt'), 'r') as fid:
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
    args = ["--logdir="+V.logs_folder.value, \
            "--error_ratios="+V.precision_recall_ratios.value, \
            "--nprobabilities="+str(M.nprobabilities), \
            "--accuracy_parallelize="+str(M.accuracy_parallelize)]
    jobid = generic_actuate("accuracy", logfile,
                            M.accuracy_where,
                            M.accuracy_ncpu_cores,
                            M.accuracy_ngpu_cards,
                            M.accuracy_ngigabytes_memory,
                            M.accuracy_cluster_flags,
                            *args)
    displaystring = "ACCURACY "+os.path.basename(V.logs_folder.value.rstrip(os.sep))+ \
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
    pbfile = os.path.join(modeldir, "frozen-graph.ckpt-"+str(ckpt)+".pb", "saved_model.pb")
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
    with open(os.path.join(logdir,model,'labels.txt'), 'r') as fid:
        labels = fid.read().splitlines()
    jobid = generic_actuate("freeze", logfile,
                            M.freeze_where,
                            M.freeze_ncpu_cores,
                            M.freeze_ngpu_cards,
                            M.freeze_ngigabytes_memory,
                            M.freeze_cluster_flags,
                            "--start_checkpoint="+os.path.join(logdir,model,"ckpt-"+check_point), \
                            "--output_file="+os.path.join(logdir,model,"frozen-graph.ckpt-"+check_point+".pb"), \
                            "--labels_touse="+','.join(labels),
                            "--context_ms="+V.context_ms.value,
                            "--model_architecture="+M.architecture_plugin,
                            "--model_parameters="+json.dumps({k:v.value for k,v in V.model_parameters.items()}),
                            "--parallelize="+str(M.classify_parallelize),
                            "--audio_tic_rate="+str(M.audio_tic_rate),
                            "--audio_nchannels="+str(M.audio_nchannels),
                            "--video_frame_rate="+str(M.video_frame_rate),
                            "--video_frame_height="+str(M.video_frame_height),
                            "--video_frame_width="+str(M.video_frame_width),
                            "--video_channels="+str(M.video_channels),
                            "--igpu=QUEUE1")
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

def ensemble_succeeded(modeldir, reftime, ckpts):
    logfile = os.path.join(modeldir, "ensemble.log")
    if not logfile_succeeded(logfile, reftime):
        return False
    pbfile = os.path.join(modeldir, "frozen-graph.ckpt-"+ckpts+".pb", "saved_model.pb")
    if not pbfile_succeeded(pbfile, reftime):
        return False
    return True

async def ensemble_actuate():
    M.waitfor_job = []
    currtime = time.time()
    logdir = V.logs_folder.value
    if len(V.model_file.value.split(','))>1:
        V_model_file_value = []
        for model in V.model_file.value.split(','):
            thislogdir, thismodel, _, thisckpt = M.parse_model_file(model)
            V_model_file_value.append(os.path.join(thislogdir, thismodel, "ckpt-"+thisckpt))
    else:
        thislogdir, thismodel, _, thisckpt = M.parse_model_file(V.model_file.value)
        V_model_file_value = []
        for model in filter(lambda x: x.startswith(thismodel.split('_')[0]+'_') and
                                      ',' not in x and
                                      os.path.isdir(os.path.join(thislogdir,x)),
                            os.listdir(thislogdir)):
            V_model_file_value.append(os.path.join(thislogdir, model, "ckpt-"+thisckpt))
    V_model_file_value = ','.join(V_model_file_value)
    models = []
    ckpts = []
    for ckpt in V_model_file_value.split(','):
        _, thismodel, _, thisckpt = M.parse_model_file(ckpt)
        root, rkw = thismodel.split('_')
        model = root
        models.append(rkw)
        ckpts.append(thisckpt)
    model += '_'+','.join(models)
    os.makedirs(os.path.join(logdir, model), exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(V_model_file_value.split(',')[0].rstrip(os.path.sep)),
                             "labels.txt"),
                os.path.join(logdir, model))
    logfile = os.path.join(logdir, model, "ensemble.log")
    with open(os.path.join(logdir,model,'labels.txt'), 'r') as fid:
        labels = fid.read().splitlines()
    jobid = generic_actuate("ensemble", logfile,
                            M.ensemble_where,
                            M.ensemble_ncpu_cores,
                            M.ensemble_ngpu_cards,
                            M.ensemble_ngigabytes_memory,
                            M.ensemble_cluster_flags,
                            "--start_checkpoints="+V_model_file_value, \
                            "--output_file="+os.path.join(logdir, model,
                                                          "frozen-graph.ckpt-"+','.join(ckpts)+".pb"), \
                            "--labels_touse="+','.join(labels),
                            "--context_ms="+V.context_ms.value,
                            "--model_architecture="+M.architecture_plugin,
                            "--model_parameters="+json.dumps({k:v.value for k,v in V.model_parameters.items()}),
                            "--parallelize="+str(M.classify_parallelize),
                            "--audio_tic_rate="+str(M.audio_tic_rate),
                            "--audio_nchannels="+str(M.audio_nchannels),
                            "--video_frame_rate="+str(M.video_frame_rate),
                            "--video_frame_height="+str(M.video_frame_height),
                            "--video_frame_width="+str(M.video_frame_width),
                            "--video_channels="+str(M.video_channels))
    displaystring = "ENSEMBLE " + \
                    os.path.join(os.path.basename(logdir), model) + \
                    " ("+jobid+")"
    M.waitfor_job.append(jobid)
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                        lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                        lambda l=logfile: contains_two_timestamps(l), \
                        lambda m=os.path.join(logdir, model), t=currtime, c=','.join(ckpts): \
                               ensemble_succeeded(m, t, c)))
    V.waitfor_update()

def classify_isdone(wavlogfile, reftime):
    return recent_file_exists(wavlogfile, reftime, False) and \
           contains_two_timestamps(wavlogfile)

def classify_succeeded(modeldir, wavfile, reftime):
    with open(os.path.join(modeldir, 'labels.txt'), 'r') as fid:
        labels = fid.read().splitlines()
    for x in labels:
        if not recent_file_exists(wavfile[:-4]+'-'+x+'.wav', reftime, True):
            return False
    return True

async def classify_actuate():
    M.waitfor_job = []
    await _classify_actuate(V.wavcsv_files.value.split(','))

async def _classify_actuate(wavfiles):
    wavfile = wavfiles.pop(0)
    currtime = time.time()
    logdir, model, _, check_point = M.parse_model_file(V.model_file.value)
    logfile = os.path.splitext(wavfile)[0]+'-classify.log'
    args = ["--context_ms="+V.context_ms.value,
            "--shiftby_ms="+V.shiftby_ms.value,
            "--model="+os.path.join(logdir,model,"frozen-graph.ckpt-"+check_point+".pb"),
            "--model_labels="+os.path.join(logdir,model,"labels.txt"),
            "--wav="+wavfile,
            "--parallelize="+str(M.classify_parallelize),
            "--audio_tic_rate="+str(M.audio_tic_rate),
            "--audio_nchannels="+str(M.audio_nchannels),
            "--video_findfile="+M.video_findfile_plugin,
            "--video_bkg_frames="+str(M.video_bkg_frames),
            "--video_frame_rate="+str(M.video_frame_rate),
            "--video_frame_height="+str(M.video_frame_height),
            "--video_frame_width="+str(M.video_frame_width),
            "--video_channels="+str(M.video_channels),
            "--audio_read_plugin="+M.audio_read_plugin,
            "--audio_read_plugin_kwargs="+json.dumps(M.audio_read_plugin_kwargs),
            "--video_read_plugin="+M.video_read_plugin,
            "--video_read_plugin_kwargs="+json.dumps(M.video_read_plugin_kwargs),
            "--igpu=QUEUE1", \
            "--deterministic="+str(M.deterministic)]
    if V.prevalences.value!='':
        args += ["--labels="+V.labels_touse.value,
                 "--prevalences="+V.prevalences.value]
    else:
        args += ["--labels=", "--prevalences="]
    jobid = generic_actuate("classify", logfile, M.classify_where,
                              M.classify_ncpu_cores,
                              M.classify_ngpu_cards,
                              M.classify_ngigabytes_memory,
                              M.classify_cluster_flags,
                              *args)
    displaystring = "CLASSIFY "+os.path.basename(wavfile)+" ("+jobid+")"
    M.waitfor_job.append(jobid)
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda w=logfile, t=currtime: classify_isdone(w, t), \
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

def ethogram_succeeded(modeldir, ckpt, wavfile, reftime):
    thresholds_file = os.path.join(modeldir, 'thresholds.ckpt-'+str(ckpt)+'.csv')
    if not os.path.exists(thresholds_file):
        return False
    with open(thresholds_file) as fid:
        csvreader = csv.reader(fid)
        row1 = next(csvreader)
    precision_recalls = row1[1:]
    for x in precision_recalls:
        if not recent_file_exists(wavfile[:-4]+'-predicted-'+x+'pr.csv', reftime, True):
            return False
    return True

async def ethogram_actuate():
    wavfiles = V.wavcsv_files.value.split(',')
    threads = [None] * len(wavfiles)
    results = [None] * len(wavfiles)
    await _ethogram_actuate(0, wavfiles, threads, results)

async def _ethogram_actuate(i, wavfiles, threads, results):
    wavfile = wavfiles.pop(0)
    currtime = time.time()
    logdir, model, prefix, check_point = M.parse_model_file(V.model_file.value)
    if 'thresholds' in prefix:
        thresholds_file = os.path.basename(V.model_file.value)
    else:
        thresholds_file = 'thresholds.ckpt-'+check_point+'.csv'
    logfile = os.path.splitext(wavfile)[0]+'-ethogram.log'
    jobid = generic_actuate("ethogram", logfile, M.ethogram_where,
                            M.ethogram_ncpu_cores,
                            M.ethogram_ngpu_cards,
                            M.ethogram_ngigabytes_memory,
                            M.ethogram_cluster_flags,
                            logdir, model, thresholds_file, wavfile,
                            str(M.audio_tic_rate))
    displaystring = "ETHOGRAM "+os.path.basename(wavfile)+" ("+jobid+")"
    M.waitfor_job.append(jobid)
    threads[i] = asyncio.create_task(actuate_monitor(displaystring, results, i, \
                                     lambda l=logfile, t=currtime: \
                                            recent_file_exists(l, t, False), \
                                     lambda l=logfile: contains_two_timestamps(l), \
                                     lambda m=os.path.join(logdir, model), \
                                            c=check_point, l=wavfile, t=currtime: \
                                            ethogram_succeeded(m, c, l, t)))
    await asyncio.sleep(0.001)
    if wavfiles:
        if bokeh_document: 
            bokeh_document.add_next_tick_callback(lambda i=i+1, wav=wavfiles, th=threads, \
                                                  r=results: _ethogram_actuate(i,wav,th,r))
        else:
            _ethogram_actuate(i+1, wavfiles, threads, results)
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
    jobid = generic_actuate("compare", logfile,
                            M.compare_where,
                            M.compare_ncpu_cores,
                            M.compare_ngpu_cards,
                            M.compare_ngigabytes_memory,
                            M.compare_cluster_flags,
                            V.logs_folder.value)
    displaystring = "COMPARE "+os.path.basename(V.logs_folder.value.rstrip(os.sep))+ \
                    " ("+jobid+")"
    M.waitfor_job = jobid
    V.waitfor_update()
    asyncio.create_task(actuate_monitor(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda l=V.logs_folder.value, t=currtime: compare_succeeded(l, t)))

def congruence_succeeded(groundtruth_folder, reftime, regex_files, measure):
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
        nlabel = len(list(filter(lambda x: x.startswith("congruence.label") and reftime <=
                                          os.path.getmtime(os.path.join(groundtruth_folder,x)),
                                allfiles)))
        if (measure=="tic" and ntic==0) or \
           (measure=="label" and nlabel==0) or \
           (measure=="both" and (ntic != nlabel or ntic==0)):
            bokehlog.info("ERROR: missing congruence-{tic,label} "+suffix+" files.")
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
                            "disjoint-label-only", "disjoint-label-not"]:
            if (measure=="tic" and "label" in disjointstr) or \
               (measure=="label" and "tic" in disjointstr):
                continue 
            n = len(list(filter(lambda x: disjointstr in x, csvfiles)))
            if n % ndisjoint_everyone != 0:
                bokehlog.info("ERROR: # of "+k+ \
                              " CSV files does not match # of disjoint-everyone CSV files.")
                return False
    return True

async def congruence_actuate():
    currtime = time.time()
    validation_files = _validation_test_files(V.validation_files.value, False)
    test_files = _validation_test_files(V.test_files.value, False)
    all_files = validation_files + test_files
    if '' in all_files:
        all_files.remove('')
    logfile = os.path.join(V.groundtruth_folder.value,'congruence.log')
    jobid = generic_actuate("congruence", logfile,
                            M.congruence_where,
                            M.congruence_ncpu_cores,
                            M.congruence_ngpu_cards,
                            M.congruence_ngigabytes_memory,
                            M.congruence_cluster_flags,
                            "--basepath="+V.groundtruth_folder.value,
                            "--wavfiles="+','.join(all_files),
                            "--portion="+V.congruence_portion.value,
                            "--convolve_ms="+V.congruence_convolve.value,
                            "--measure="+V.congruence_measure.value,
                            "--nprobabilities="+str(M.nprobabilities),
                            "--audio_tic_rate="+str(M.audio_tic_rate),
                            "--parallelize="+str(M.congruence_parallelize))
    displaystring = "CONGRUENCE "+os.path.basename(all_files[0])+" ("+jobid+")"
    M.waitfor_job = [jobid]
    V.waitfor_update()
    regex_files = '('+'|'.join([os.path.splitext(x)[0] for x in all_files])+')*csv'
    threads, results = [None], [None]
    threads[0] = asyncio.create_task(actuate_monitor(displaystring, results, 0, \
                                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                                     lambda l=logfile: contains_two_timestamps(l), \
                                     lambda l=V.groundtruth_folder.value, t=currtime, r=regex_files,
                                            m=V.congruence_measure.value: congruence_succeeded(l, t, r, m)))
    asyncio.create_task(actuate_finalize(threads, results, V.groundtruth_update))

def deletefailures_callback(arg):
    M.status_ticker_queue = {k:v for k,v in M.status_ticker_queue.items() if v!="failed"}
    if bokeh_document: 
        bokeh_document.add_next_tick_callback(V.status_ticker_update)

def waitfor_callback(arg):
    if V.waitfor.active:
        V.waitfor.button_type="warning"
    else:
        V.waitfor.button_type="default"

def logs_callback():
    if len(V.file_dialog_source.selected.indices)>=2:
        bokehlog.info('ERROR: a directory must be selected in the file browser')
        return
    if len(V.file_dialog_source.selected.indices)==1 and \
            V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] != '.':
        bokehlog.info('ERROR: a directory must be selected in the file browser')
        return
    V.logs_folder.value = M.file_dialog_root

def model_callback():
    if len(V.file_dialog_source.selected.indices)==0:
        V.model_file.value = M.file_dialog_root
    elif len(V.file_dialog_source.selected.indices)==1 and \
            V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] != '.':
        filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]]
        V.model_file.value = os.path.join(M.file_dialog_root, filename)
    else:
        bokehlog.info('ERROR: a directory or file must be selected in the file browser')

def wavcsv_files_callback():
    if len(V.file_dialog_source.selected.indices)==0:
        bokehlog.info('ERROR: a file(s) must be selected in the file browser')
        return
    filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]]
    files = os.path.join(M.file_dialog_root, filename)
    for i in range(1, len(V.file_dialog_source.selected.indices)):
        filename = V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[i]]
        files += ','+os.path.join(M.file_dialog_root, filename)
    V.wavcsv_files.value = files

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
  V.validation_files.value = _validation_test_files_callback()

def test_files_callback():
  V.test_files.value = _validation_test_files_callback()

def labels_touse_update(labels_file):
    if os.path.isfile(labels_file):
        with open(labels_file, "r") as fid:
            labels = fid.readlines()
        V.labels_touse.value = str.join(',',[x.strip() for x in labels])

def labels_touse_callback():
    if len(V.file_dialog_source.selected.indices)!=1:
        bokehlog.info('ERROR: a file must be selected in the file browser')
        return
    idx = V.file_dialog_source.selected.indices[0]
    labels_file = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    labels_touse_update(labels_file)

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
    V.prevalences.value = prevalences.replace(' ',',')

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
            if "batch_size = " in line:
                m=re.search('batch_size = (\d+)', line)
                V.mini_batch.value = m.group(1)
            elif "context_ms = " in line:
                m=re.search('context_ms = (.*)', line)
                V.context_ms.value = m.group(1)
            elif "time_shift_ms = " in line:
                m=re.search('time_shift_ms = (.*)', line)
                V.shiftby_ms.value = m.group(1)
            elif "data_dir = " in line:
                m=re.search('data_dir = (.*)', line)
                V.groundtruth_folder.value = m.group(1)
            elif "random_seed_batch = " in line:
                m=re.search('random_seed_batch = (.*)', line)
                V.batch_seed.value = m.group(1)
            elif "random_seed_weights = " in line:
                m=re.search('random_seed_weights = (.*)', line)
                V.weights_seed.value = m.group(1)
            elif "validate_step_period = " in line:
                m=re.search('validate_step_period = (\d+)', line)
                V.save_and_validate_period.value = m.group(1)
            elif "how_many_training_steps = " in line:
                m=re.search('how_many_training_steps = (\d+)', line)
                V.nsteps.value = m.group(1)
            elif "kinds_touse = " in line:
                m=re.search('kinds_touse = (.*)', line)
                V.kinds_touse.value = m.group(1)
            elif "learning_rate = " in line:
                m=re.search('learning_rate = (.*)', line)
                V.learning_rate.value = m.group(1)
            elif "optimizer = " in line:
                m=re.search('optimizer = (.*)', line)
                V.optimizer.value = m.group(1)
            elif "start_checkpoint = " in line:
                m=re.search('start_checkpoint = .*ckpt-([0-9]+)', line)
                if m:
                    V.restore_from.value = m.group(1)
                else:
                    V.restore_from.value = ''
            elif "testing_files = " in line:
                m=re.search('testing_files = (.*)', line)
                V.test_files.value = m.group(1)
            elif "validation_files = " in line:
                m=re.search('validation_files = (.*)', line)
                V.validation_files.value = m.group(1)
            elif "validation_percentage = " in line:
                m=re.search('validation_percentage = (.*)', line)
                V.validate_percentage.value = m.group(1)
                validate_perentage_float = float(V.validate_percentage.value)
                if validate_perentage_float!=0:
                    V.kfold.value = str(int(round(100/validate_perentage_float)))
                else:
                    V.kfold.value = "0"
            elif "labels_touse = " in line:
                m=re.search('labels_touse = (.*)', line)
                V.labels_touse.value = m.group(1)
            elif "model_parameters = " in line:
                m=re.search('model_parameters = ({.*})', line)
                params = json.loads(m.group(1).replace("'",'"'))
                for k,v in params.items():
                  V.model_parameters[k].value = v
    _copy_callback_finalize()
    
def copy_callback():
    V.copy.button_type="warning"
    V.copy.disabled=True
    M.user_copied_parameters=1
    bokeh_document.add_next_tick_callback(_copy_callback)

def wizard_callback(wizard):
    M.wizard=None if M.wizard is wizard else wizard
    M.action=None
    if M.wizard==V.labelsounds:
        V.labels_touse.value=','.join(M.detect_labels)
        V.kinds_touse.value="detected"
        V.nsteps.value="0"
        V.save_and_validate_period.value="0"
        V.validate_percentage.value="0"
    elif M.wizard==V.makepredictions:
        V.labels_touse_update()
        V.kinds_touse.value="annotated"
    elif M.wizard==V.fixfalsepositives:
        V.labels_touse_update()
        V.kinds_touse.value="annotated,predicted"
    elif M.wizard==V.fixfalsenegatives:
        V.labels_touse_update(other=True)
        V.kinds_touse.value="annotated,missed"
    elif M.wizard==V.generalize:
        V.labels_touse_update()
        V.kinds_touse.value="annotated"
    elif M.wizard==V.tunehyperparameters:
        V.labels_touse_update()
        V.kinds_touse.value="annotated"
    elif M.wizard==V.examineerrors:
        V.kinds_touse.value="detected,mistaken"
    elif M.wizard==V.testdensely:
        V.kinds_touse.value="annotated"
    elif M.wizard==V.findnovellabels:
        V.labels_touse_update(detect=True)
        if M.action==V.train:
            V.kinds_touse.value="annotated"
        elif M.action==V.activations:
            V.kinds_touse.value="annotated,detected"
    V.buttons_update()
  
async def _doit_callback():
    await M.function()
    V.doit.button_type="default"
    V.doit.disabled=True
  
def doit_callback():
    V.doit.button_type="warning"
    V.doit.disabled=True
    bokeh_document.add_next_tick_callback(_doit_callback)

def file_dialog_callback(attr, old, new):
    if len(V.file_dialog_source.selected.indices)==1:
        iselected = V.file_dialog_source.selected.indices[0]
        selected_file = V.file_dialog_source.data['names'][iselected]
        if selected_file=='.'+os.sep:
            V.file_dialog_source.selected.indices = []
            V.file_dialog_update()
        elif selected_file=='..'+os.sep:
            if M.file_dialog_root.endswith(os.sep):
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

def nsounds_per_label_callback(i):
    M.ilabel=i
    for button in V.nsounds_per_label_buttons:
        button.button_type="default"
    V.nsounds_per_label_buttons[i].button_type="primary"

def label_callback(new, i):
    if M.state['labels'][i]!='':
        V.nsounds_per_label_buttons[i].label='0'
        M.history_stack=[]
        M.history_idx=0
        V.undo.disabled=True
        V.redo.disabled=True
    nsounds_per_label_callback(i)
    M.state['labels'][i]=new
    M.nrecent_annotations+=1
    V.save_update(M.nrecent_annotations)
    M.save_state_callback()

def init(_bokeh_document):
    global bokeh_document
    bokeh_document = _bokeh_document
