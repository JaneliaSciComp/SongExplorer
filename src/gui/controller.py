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

bokehlog = logging.getLogger("deepsong") 
#class Object(object):
#  pass
#bokehlog=Object()
#bokehlog.info=print

import model as M
import view as V

def generic_it(cmd, logfile, where, localargs, localdeps, clusterflags, *args):
    args = ["\'\'" if x=="" else x for x in args]
    if where == "local":
        p = run(["hetero", "submit",
                 "{ export CUDA_VISIBLE_DEVICES=$QUEUE1; "+cmd+" "+' '.join(args)+"; } &> "+logfile,
                 *localargs.split(' '), localdeps],
                stdout=PIPE, stderr=STDOUT)
        jobinfo = p.stdout.decode('ascii').rstrip()
        bokehlog.info(jobinfo)
    elif where == "server":
        p = run(["ssh", M.server_ipaddr, "export SINGULARITYENV_PREPEND_PATH="+M.source_path+";",
                 "$DEEPSONG_BIN", "hetero", "submit",
                 "\"{ export CUDA_VISIBLE_DEVICES=\$QUEUE1; "+cmd+" "+' '.join(args)+"; } &> "+logfile+"\"",
                 *localargs.split(' '), localdeps],
                stdout=PIPE, stderr=STDOUT)
        jobinfo=p.stdout.decode('ascii').rstrip()
        bokehlog.info(jobinfo)
    elif where == "cluster":
        pe = Popen(["echo",
                    "export SINGULARITYENV_PREPEND_PATH="+M.source_path+";",
                    os.environ["DEEPSONG_BIN"]+" "+cmd+" "+' '.join(args)],
                   stdout=PIPE)
        ps = Popen(["ssh", "login1", "bsub",
                    "-Ne",
                    "-P stern",
                    #"-J ${logfile//,/}.job",
                    clusterflags,
                    "-oo "+logfile],
                   stdin=pe.stdout, stdout=PIPE, stderr=STDOUT)
        pe.stdout.close()
        jobinfo=ps.communicate()[0].decode('ascii').rstrip()
        bokehlog.info(jobinfo)
    return jobinfo

def detect_it(logfile, *args):
    generic_it("detect.sh", logfile, M.detect_where,
               M.detect_local_resources, "", M.detect_cluster_flags, *args)

def misses_it(logfile, *args):
    generic_it("misses.sh", logfile, M.misses_where,
               M.misses_local_resources, "", M.misses_cluster_flags, *args)

def train_it(logfile, *args):
    if M.train_gpu == 1:
        generic_it("train.sh", logfile, M.train_where,
                   M.train_local_resources_gpu, "", M.train_cluster_flags_gpu, *args)
    else:
        generic_it("train.sh", logfile, M.train_where,
                   M.train_local_resources_cpu, "", M.train_cluster_flags_cpu, *args)

def generalize_it(logfile, *args):
    if M.generalize_gpu == 1:
        generic_it("generalize.sh", logfile, M.generalize_where,
                   M.generalize_local_resources_gpu, "", M.generalize_cluster_flags_gpu, *args)
    else:
        generic_it("generalize.sh", logfile, M.generalize_where,
                   M.generalize_local_resources_cpu, "", M.generalize_cluster_flags_cpu, *args)

def xvalidate_it(logfile, *args):
    if M.xvalidate_gpu == 1:
        generic_it("xvalidate.sh", logfile, M.xvalidate_where,
                   M.xvalidate_local_resources_gpu, "", M.xvalidate_cluster_flags_gpu, *args)
    else:
        generic_it("xvalidate.sh", logfile, M.xvalidate_where,
                   M.xvalidate_local_resources_cpu, "", M.xvalidate_cluster_flags_cpu, *args)

def mistakes_it(logfile, *args):
    generic_it("mistakes.sh", logfile, M.mistakes_where,
               M.mistakes_local_resources, "", M.mistakes_cluster_flags, *args)

def activations_it(logfile, *args):
    if M.activations_gpu:
        generic_it("activations.sh", logfile, M.activations_where,
                   M.activations_local_resources_gpu, "", M.activations_cluster_flags_gpu, *args)
    else:
        generic_it("activations.sh", logfile, M.activations_where,
                   M.activations_local_resources_cpu, "", M.activations_cluster_flags_cpu, *args)

def cluster_it(logfile, *args):
    generic_it("cluster.sh", logfile, M.cluster_where,
               M.cluster_local_resources, "", M.cluster_cluster_flags, *args)

def accuracy_it(logfile, *args):
    generic_it("accuracy.sh", logfile, M.accuracy_where,
               M.accuracy_local_resources, "", M.accuracy_cluster_flags, *args)

def freeze_it(logfile, *args):
    generic_it("freeze.sh", logfile, M.freeze_where,
               M.freeze_local_resources, "", M.freeze_cluster_flags, *args)

def classify_it(logfile1, logfile2, *args):
    p = run(["date", "+%s"], stdout=PIPE, stderr=STDOUT)
    currtime = p.stdout.decode('ascii').rstrip()
    if M.classify_gpu:
        jobinfo = generic_it("classify1.sh", logfile1, M.classify_where,
                             M.classify1_local_resources_gpu, "", M.classify1_cluster_flags_gpu,
                             *args)
    else:
        jobinfo = generic_it("classify1.sh", logfile1, M.classify_where,
                             M.classify1_local_resources_gpu, "", M.classify1_cluster_flags_gpu,
                             *args)
    jobid = re.search('([0-9]+)',jobinfo).group(1) if M.classify_where=="cluster" else jobinfo
    generic_it("classify2.sh", logfile2, M.classify_where,
               M.classify2_local_resources, "'hetero job "+jobid+"'",
               M.classify2_cluster_flags+" -w \"done("+jobid+")\"", *args)

def ethogram_it(logfile, *args):
    generic_it("ethogram.sh", logfile, M.ethogram_where,
               M.ethogram_local_resources, "", M.ethogram_cluster_flags, *args)

def compare_it(logfile, *args):
    generic_it("compare.sh", logfile, M.compare_where,
               M.compare_local_resources, "", M.compare_cluster_flags, *args)

def congruence_it(logfile, *args):
    generic_it("congruence.sh", logfile, M.congruence_where,
               M.congruence_local_resources, "", M.congruence_cluster_flags, *args)

def generic_parameters_callback():
    M.save_state_callback()
    V.buttons_update()

def layer_callback(new):
    M.ilayer=M.layers.index(new)
    V.circle_fuchsia_cluster.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
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
    V.cluster_initialize(False)
    V.cluster_update()
    
def circle_radius_callback(attr, old, new):
    M.state["circle_radius"]=new
    if len(V.circle_fuchsia_cluster.data['cx'])==1:
        V.circle_fuchsia_cluster.data.update(cr=[M.state["circle_radius"]])
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

var x0 = line_red_context.data.x[0];

aud.ontimeupdate = function() {
  line_red_context.data = { 'x': [x0+aud.currentTime, x0+aud.currentTime], 'y':[-1,0] };
};

aud.onended = function() {
  line_red_context.data = { 'x': [x0, x0], 'y':[-1,0] };
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
    V.circle_fuchsia_cluster.data.update(cx=[M.xcluster],
                                         cy=[M.ycluster],
                                         cz=[M.zcluster],
                                         cr=[M.state["circle_radius"]],
                                         cc=[M.cluster_circle_color])
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
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
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
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    currfile = M.clustered_samples[M.isnippet]['file']
    M.panned_sample = {'file':currfile, 'ticks':[x_tic,x_tic],
                       'label':M.state['labels'][M.ilabel]}
    V.quad_grey_context_pan.data.update(left=[x_tic/M.audio_tic_rate],
                                        right=[x_tic/M.audio_tic_rate], top=[0],
                                        bottom=[V.p_context.y_range.start +
                                                V.p_context.y_range.range_padding])

def context_pan_callback(event):
    if M.state['labels'][M.ilabel]=='':
        return
    x_tic = int(np.rint(event.x*M.audio_tic_rate))
    left_limit_tic = M.context_midpoint_tic-M.context_width_tic//2 + M.context_offset_tic
    right_limit_tic = left_limit_tic + M.context_width_tic
    if x_tic < left_limit_tic or x_tic > right_limit_tic:
        return
    M.panned_sample['ticks'][1]=x_tic
    V.quad_grey_context_pan.data.update(right=[x_tic/M.audio_tic_rate])

def context_pan_end_callback(event):
    if M.state['labels'][M.ilabel]=='':
        return
    M.panned_sample['ticks'] = sorted(M.panned_sample['ticks'])
    V.quad_grey_context_pan.data.update(left=[], right=[], top=[], bottom=[])
    M.add_annotation(M.panned_sample)

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
    
def toggle_annotation(idouble_tapped_sample):
    iannotated = M.isannotated(M.clustered_samples[idouble_tapped_sample])
    if len(iannotated)>0 and M.annotated_samples[iannotated[0]]['label'] != \
                             M.clustered_samples[idouble_tapped_sample]['label']:
        M.delete_annotation(iannotated[0])
    else:
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
        V.circle_fuchsia_cluster.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
        M.xcluster = M.ycluster = M.zcluster = np.nan
        M.isnippet = -1
        V.snippets_update(True)
        M.isnippet = np.searchsorted(M.clustered_starts_sorted, \
                                   M.history_stack[M.history_idx][1]['ticks'][0])
        M.context_offset_tic = M.history_stack[M.history_idx][1]['ticks'][0] - \
                             M.clustered_starts_sorted[M.isnippet]
        M.context_offset_ms = M.context_offset_tic/M.audio_tic_rate*1000
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
        V.circle_fuchsia_cluster.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
        M.xcluster = M.ycluster = M.zcluster = np.nan
        M.isnippet = -1
        V.snippets_update(True)
        M.isnippet = np.searchsorted(M.clustered_starts_sorted, \
                                   M.history_stack[M.history_idx-1][1]['ticks'][0])
        M.context_offset_tic = M.history_stack[M.history_idx-1][1]['ticks'][0] - \
                             M.clustered_starts_sorted[M.isnippet]
        M.context_offset_ms = M.context_offset_tic/M.audio_tic_rate*1000
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

def classify_callback():
    labels_file = os.path.dirname(V.model_file.value)
    wantedwords_update(os.path.join(labels_file, "vgg_labels.txt"))
    action_callback(V.classify, classify_actuate)

def actuate_monitor(displaystring, results, idx, isrunningfun, isdonefun, succeededfun):
    M.status_ticker_queue = {k:v for k,v in M.status_ticker_queue.items() if v!="succeeded"}
    M.status_ticker_queue[displaystring] = "pending"
    bokeh_document.add_next_tick_callback(V.status_ticker_update)
    while not isrunningfun():
        time.sleep(3)
    M.status_ticker_queue[displaystring] = "running"
    bokeh_document.add_next_tick_callback(V.status_ticker_update)
    while not isdonefun():
        time.sleep(3)
    for sec in [3,10,30,100,300,1000]:
        M.status_ticker_queue[displaystring] = "succeeded" if succeededfun() else "failed"
        bokeh_document.add_next_tick_callback(V.status_ticker_update)
        if M.status_ticker_queue[displaystring] == "succeeded":
            if results:
                results[idx]=True
            return
        time.sleep(sec)
    if results:
        results[idx]=False

def actuate_finalize(threads, results, finalizefun):
    for i in range(len(threads)):
        threads[i].join()
    if any(results):
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
                                   not "failed call to cuInit" in line):
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

def detect_actuate():
    wavfiles = V.wavtfcsvfiles_string.value.split(',')
    threads = [None] * len(wavfiles)
    results = [None] * len(wavfiles)
    for (i,wavfile) in enumerate(wavfiles):
        currtime = time.time()
        logfile = os.path.splitext(wavfile)[0]+'-detect.log'
        detect_it(logfile,
                  wavfile, \
                  V.time_sigma_string.value, V.time_smooth_ms_string.value, \
                  V.frequency_n_ms_string.value, V.frequency_nw_string.value, \
                  V.frequency_p_string.value, V.frequency_smooth_ms_string.value, \
                  str(M.audio_tic_rate), str(M.audio_nchannels))
        displaystring = "detect "+os.path.basename(wavfile)
        threads[i] = threading.Thread(target=actuate_monitor, args=( \
                         displaystring, \
                         results, i, \
                         lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                         lambda l=logfile: contains_two_timestamps(l), \
                         lambda w=wavfile, t=currtime: detect_succeeded(w, t)))
        threads[i].start()
    threading.Thread(target=actuate_finalize, \
                     args=(threads, results, V.wordcounts_update)).start()

def misses_succeeded(wavfile, reftime):
    logfile = wavfile[:-4]+'-misses.log'
    if not logfile_succeeded(logfile, reftime):
        return False
    csvfile = wavfile[:-4]+'-missed.csv'
    if not csvfile_succeeded(csvfile, reftime):
        return False
    return True

def misses_actuate():
    currtime = time.time()
    csvfile1 = V.wavtfcsvfiles_string.value.split(',')[0]
    basepath = os.path.dirname(csvfile1)
    with open(csvfile1) as fid:
      csvreader = csv.reader(fid)
      row1 = next(csvreader)
    wavfile = row1[0]
    displaystring = "misses "+wavfile
    noext = os.path.join(basepath, os.path.splitext(wavfile)[0])
    logfile = noext+'-misses.log'
    misses_it(logfile, V.wavtfcsvfiles_string.value)
    threads = [None]
    results = [None]
    threads[0] = threading.Thread(target=actuate_monitor, args=( \
                     displaystring, \
                     results, 0, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda w=os.path.join(basepath, wavfile), t=currtime: misses_succeeded(w, t)))
    threads[0].start()
    threading.Thread(target=actuate_finalize, \
                     args=(threads, results, V.wordcounts_update)).start()

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
    logfile = os.path.join(logdir, kind+"_"+model+".log")
    if not logfile_succeeded(logfile, reftime):
        return False
    if not os.path.isdir(os.path.join(logdir, "summaries_"+model)):
        bokehlog.info("ERROR: summaries_"+model+"/ does not exist.")
        return False
    train_dir = os.path.join(logdir, kind+"_"+model)
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
            if 'Error' in line:
                bokehlog.info("ERROR: "+train_dir+".log contains Errors.")
                return False
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

def train_succeeded(logdir, reftime):
    return _train_succeeded(logdir, "train", "1", reftime)

def sequester_stalefiles():
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

def train_actuate():
    M.save_annotations()
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    currtime = time.time()
    logfile = os.path.join(V.logs_folder.value, "train1.log")
    train_it(logfile,
         V.context_ms_string.value, V.shiftby_ms_string.value, \
         V.representation.value, V.window_ms_string.value, \
         *V.mel_dct_string.value.split(','), V.stride_ms_string.value, \
         V.dropout_string.value, V.optimizer.value, V.learning_rate_string.value, \
         V.kernel_sizes_string.value, V.last_conv_width_string.value, \
         V.nfeatures_string.value, V.dilate_after_layer_string.value, \
         V.stride_after_layer_string.value, \
         V.connection_type.value, V.logs_folder.value, '1', \
         V.groundtruth_folder.value, V.wantedwords_string.value, \
         V.labeltypes_string.value, V.nsteps_string.value, V.restore_from_string.value, \
         V.save_and_validate_period_string.value, \
         V.validate_percentage_string.value, V.mini_batch_string.value, test_files, \
         str(M.audio_tic_rate), str(M.audio_nchannels))
    displaystring = "train "+os.path.basename(V.logs_folder.value.rstrip('/'))
    threads = [None]
    results = [None]
    threads[0] = threading.Thread(target=actuate_monitor, args=( \
                     displaystring, \
                     results, 0, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda l=V.logs_folder.value, t=currtime: train_succeeded(l, t)))
    threads[0].start()
    threading.Thread(target=actuate_finalize, \
                     args=(threads, results, sequester_stalefiles)).start()

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

def generalize_xvalidate_finished(lastlogfile, reftime):
    if recent_file_exists(lastlogfile, reftime, False):
        return contains_two_timestamps(lastlogfile)
    return False

def leaveout_actuate(comma):
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    validation_files = list(filter(
            lambda x: not any([y!='' and y in x for y in test_files.split(',')]),
            _validation_test_files(V.validationfiles_string.value, comma)))
    currtime = time.time()
    for ivalidation_file in range(0, len(validation_files), M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "generalize"+str(1+ivalidation_file)+".log")
        generalize_it(logfile,
                      V.context_ms_string.value, \
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
                      str(ivalidation_file),
                      *validation_files[ivalidation_file:ivalidation_file+M.models_per_job])
    displaystring = "generalize "+os.path.basename(V.logs_folder.value.rstrip('/'))
    logfile1 = os.path.join(V.logs_folder.value, "generalize1.log")
    logfileN = os.path.join(V.logs_folder.value,
                            "generalize"+str(len(validation_files) % M.models_per_job)+".log")
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile1, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfileN, t=currtime: generalize_xvalidate_finished(l, t), \
                     lambda l=V.logs_folder.value, t=currtime: \
                            generalize_xvalidate_succeeded("generalize", l, t))).start()

def xvalidate_actuate():
    test_files = _validation_test_files(V.testfiles_string.value)[0]
    currtime = time.time()
    for ifold in range(1, 1+int(V.kfold_string.value), M.models_per_job):
        logfile = os.path.join(V.logs_folder.value, "xvalidate"+str(ifold)+".log")
        xvalidate_it(logfile,
                     V.context_ms_string.value, \
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
                     str(M.audio_tic_rate), str(M.audio_nchannels), V.kfold_string.value, \
                     ','.join([str(x) for x in range(ifold,ifold+M.models_per_job)]))
    displaystring = "xvalidate "+os.path.basename(V.logs_folder.value.rstrip('/'))
    logfile1 = os.path.join(V.logs_folder.value, "xvalidate1.log")
    logfileN = os.path.join(V.logs_folder.value,
                            "xvalidate"+str(int(V.kfold_string.value) % M.models_per_job)+".log")
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile1, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfileN, t=currtime: generalize_xvalidate_finished(l, t), \
                     lambda l=V.logs_folder.value, t=currtime: \
                            generalize_xvalidate_succeeded("xvalidate", l, t))).start()

def mistakes_succeeded(groundtruthdir, reftime):
    pass

def mistakes_actuate():
    currtime = time.time()
    logfile = os.path.join(V.groundtruth_folder.value, "mistakes.log")
    mistakes_it(logfile, V.groundtruth_folder.value)
    displaystring = "mistakes "+os.path.basename(V.groundtruth_folder.value.rstrip('/'))
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda g=V.groundtruth_folder.value, t=currtime: \
                            mistakes_succeeded(g, t))).start()

def activations_cluster_succeeded(kind, groundtruthdir, reftime):
    logfile = os.path.join(groundtruthdir, kind+".log")
    if not logfile_succeeded(logfile, reftime):
        return False
    npzfile = os.path.join(groundtruthdir, kind+".npz")
    if not npzfile_succeeded(npzfile, reftime):
        return False
    V.cluster_these_layers_update()
    return True

def activations_actuate():
    currtime = time.time()
    logdir, model, check_point = M.parse_model_file(V.model_file.value)
    logfile = os.path.join(V.groundtruth_folder.value, "activations.log")
    activations_it(logfile, \
                   V.context_ms_string.value, \
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
                   str(M.audio_tic_rate), str(M.audio_nchannels))
    displaystring = "activations "+os.path.join(os.path.basename(logdir), \
                                           model, "ckpt-"+check_point)
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda g=V.groundtruth_folder.value, t=currtime: \
                            activations_cluster_succeeded("activations", g, t))).start()

def cluster_actuate():
    currtime = time.time()
    algorithm, ndims = V.cluster_algorithm.value.split(' ')
    these_layers = ','.join([x for x in V.cluster_these_layers.value])
    logfile = os.path.join(V.groundtruth_folder.value, "cluster.log")
    if algorithm == "PCA":
        cluster_it(logfile,
                   V.groundtruth_folder.value, \
                   these_layers, \
                   V.pca_fraction_variance_to_retain_string.value, \
                   str(M.pca_batch_size), \
                   algorithm, ndims, str(M.cluster_parallelize))
    elif algorithm == "tSNE":
        cluster_it(logfile,
                   V.groundtruth_folder.value, \
                   these_layers, \
                   V.pca_fraction_variance_to_retain_string.value, \
                   str(M.pca_batch_size), \
                   algorithm, ndims, str(M.cluster_parallelize), \
                   V.tsne_perplexity_string.value, V.tsne_exaggeration_string.value)
    elif algorithm == "UMAP":
        cluster_it(logfile,
                   V.groundtruth_folder.value, \
                   these_layers, \
                   V.pca_fraction_variance_to_retain_string.value, \
                   str(M.pca_batch_size), \
                   algorithm, ndims, str(M.cluster_parallelize), \
                   V.umap_neighbors_string.value, V.umap_distance_string.value)
    displaystring = "cluster "+os.path.basename(V.groundtruth_folder.value.rstrip('/'))
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda g=V.groundtruth_folder.value, t=currtime: \
                            activations_cluster_succeeded("cluster", g, t))).start()

def visualize_actuate():
    V.cluster_initialize()
    M.ilayer = 0
    V.which_layer.value = M.layers[M.ilayer]
    M.ispecies = 0
    V.which_species.value = M.species[M.ispecies]
    M.iword = 0
    V.which_word.value = M.words[M.iword]
    M.inohyphen = 0
    V.which_nohyphen.value = M.nohyphens[M.inohyphen]
    M.ikind = 0
    V.which_kind.value = M.kinds[M.ikind]
    V.circle_fuchsia_cluster.data.update(cx=[], cy=[], cz=[], cr=[], cc=[])
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

def accuracy_actuate():
    currtime = time.time()
    logfile = os.path.join(V.logs_folder.value, "accuracy.log")
    accuracy_it(logfile,
                V.logs_folder.value, \
                V.precision_recall_ratios_string.value, \
                str(M.accuracy_nprobabilities), str(M.accuracy_parallelize))
    displaystring = "accuracy "+os.path.basename(V.logs_folder.value.rstrip('/'))
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda l=V.logs_folder.value, t=currtime: accuracy_succeeded(l, t))).start()

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

def freeze_actuate():
    for ckpt in V.model_file.value.split(','):
      currtime = time.time()
      logdir, model, check_point = M.parse_model_file(ckpt)
      logfile = os.path.join(logdir, model, "freeze.ckpt-"+str(check_point)+".log")
      freeze_it(logfile,
                V.context_ms_string.value, \
                V.representation.value, V.window_ms_string.value, \
                V.stride_ms_string.value, *V.mel_dct_string.value.split(','), \
                V.kernel_sizes_string.value, \
                V.last_conv_width_string.value, V.nfeatures_string.value, \
                V.dilate_after_layer_string.value, \
                V.stride_after_layer_string.value, \
                V.connection_type.value, \
                logdir, model, check_point, str(M.nstrides), \
                str(M.audio_tic_rate), str(M.audio_nchannels))
      displaystring = "freeze "+os.path.join(os.path.basename(logdir), \
                                             model, "ckpt-"+check_point)
      threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                       lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                       lambda l=logfile: contains_two_timestamps(l), \
                       lambda m=os.path.join(logdir, model), c=check_point, t=currtime: \
                              freeze_succeeded(m, c, t))).start()

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
        if not recent_file_exists(os.path.join(os.path.dirname(wavfile), \
                                               wavfile[:-4]+'-'+x+'.wav'), \
                                  reftime, True):
            return False
    return True

def classify_actuate():
    for wavfile in V.wavtfcsvfiles_string.value.split(','):
        currtime = time.time()
        logdir, model, check_point = M.parse_model_file(V.model_file.value)
        logfile0 = os.path.splitext(wavfile)[0]+'-classify'
        logfile1 = logfile0+'1.log'
        logfile2 = logfile0+'2.log'
        args = [logfile1, logfile2,
                V.context_ms_string.value, \
                V.shiftby_ms_string.value, V.representation.value, \
                V.stride_ms_string.value, \
                logdir, model, check_point, wavfile, str(M.audio_tic_rate), str(M.nstrides)]
        if V.prevalences_string.value!='':
            args += [V.wantedwords_string.value, V.prevalences_string.value]
        classify_it(*args)
        displaystring = "classify "+os.path.basename(wavfile)
        threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                         lambda l=logfile1, t=currtime: recent_file_exists(l, t, False), \
                         lambda w=logfile2, t=currtime: classify_isdone(w, t), \
                         lambda m=os.path.join(logdir, model), w=wavfile, t=currtime: \
                                classify_succeeded(m, w, t))).start()

def ethogram_succeeded(modeldir, ckpt, tffile, reftime):
    thresholds_file = os.path.join(modeldir, 'thresholds.ckpt-'+str(ckpt)+'.csv')
    if not os.path.exists(thresholds_file):
        return False
    with open(thresholds_file) as fid:
        csvreader = csv.reader(fid)
        row1 = next(csvreader)
    precision_recalls = row1[1:]
    for x in precision_recalls:
        if not recent_file_exists(os.path.join(os.path.dirname(tffile), \
                                               tffile[:-3]+'-predicted-'+x+'pr.csv'), \
                                  reftime, True):
            return False
    return True

def ethogram_actuate():
    tffiles = V.wavtfcsvfiles_string.value.split(',')
    threads = [None] * len(tffiles)
    results = [None] * len(tffiles)
    for (i,tffile) in enumerate(tffiles):
        currtime = time.time()
        logdir, model, check_point = M.parse_model_file(V.model_file.value)
        if tffile.lower().endswith('.wav'):
            tffile = os.path.splitext(tffile)[0]+'.tf'
        logfile = tffile[:-3]+'-ethogram.log'
        ethogram_it(logfile,
                    logdir, model, check_point, tffile, str(M.audio_tic_rate))
        displaystring = "ethogram "+os.path.basename(tffile)
        threads[i] = threading.Thread(target=actuate_monitor, args=( \
                         displaystring, \
                         results, i, \
                         lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                         lambda l=logfile: contains_two_timestamps(l), \
                         lambda m=os.path.join(logdir, model), c=check_point, \
                                l=tffile, t=currtime: ethogram_succeeded(m, c, l, t)))
        threads[i].start()
    threading.Thread(target=actuate_finalize, \
                     args=(threads, results, V.wordcounts_update)).start()

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

def compare_actuate():
    currtime = time.time()
    logfile = V.logs_folder.value+'-compare.log'
    compare_it(logfile, V.logs_folder.value)
    displaystring = "compare "+os.path.basename(V.logs_folder.value.rstrip('/'))
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda l=V.logs_folder.value, t=currtime: compare_succeeded(l, t))).start()

def congruence_succeeded(logdir, reftime, regex_files):
    logfile = os.path.join(logdir,'congruence.log')
    if not logfile_succeeded(logfile, reftime):
        return False
    listfiles = os.listdir(os.path.join(logdir))
    csvfiles = list(filter(lambda x: x.endswith(".csv"), listfiles))
    pdffiles = list(filter(lambda x: x.endswith(".pdf"), listfiles))
    for (suffix, allfiles) in zip(["CSV", csvfiles], ["PDF", pdffiles]):
        ntic = len(list(filter(lambda x: x.startswith("congruence-tic"), allfiles)))
        nword = len(list(filter(lambda x: x.startswith("congruence-word"), allfiles)))
        if ntic != nword:
            bokehlog.info("ERROR: # of congruence-tic "+suffix+ \
                          " files does not match # of congruence-word "+suffix+" files.")
            return False
    #pdffile = os.path.join(logdir, "congruence-vs-threshold.pdf")
    #if not pdffile_succeeded(pdffile, reftime):
    #    return False
    for subdir in filter(lambda x: os.path.isdir(os.path.join(logdir,x)), \
                         os.listdir(logdir)):
      listfiles = os.listdir(os.path.join(logdir,subdir))
      csvfiles = list(filter(lambda x: re.match(regex_files, x), listfiles))
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

def congruence_actuate():
    currtime = time.time()
    validation_files = _validation_test_files(V.validationfiles_string.value, False)
    test_files = _validation_test_files(V.testfiles_string.value, False)
    all_files = validation_files + test_files
    all_files.remove('')
    logfile = os.path.join(V.groundtruth_folder.value,'congruence.log')
    congruence_it(logfile,
                  V.groundtruth_folder.value, ','.join(all_files),
                  str(M.congruence_parallelize))
    displaystring = "congruence "+os.path.basename(all_files[0])
    regex_files = '('+'|'.join([os.path.splitext(x)[0] for x in all_files])+')*csv'
    threading.Thread(target=actuate_monitor, args=(displaystring, None, None, \
                     lambda l=logfile, t=currtime: recent_file_exists(l, t, False), \
                     lambda l=logfile: contains_two_timestamps(l), \
                     lambda l=V.groundtruth_folder.value, t=currtime,
                            r=regex_files: congruence_succeeded(l, t, r))).start()

def configuration_textarea_callback(attr, old, new):
    with open(M.configuration_file, 'w') as fid:
        fid.write(V.configuration_contents.value)
    V.editconfiguration.button_type="default"
    V.configuration_contents.disabled=True

def logs_callback():
    assert len(V.file_dialog_source.selected.indices)<2 
    if len(V.file_dialog_source.selected.indices)==1:
        assert V.file_dialog_source.data['names'][V.file_dialog_source.selected.indices[0]] == '.'
    V.logs_folder.value = M.file_dialog_root

def _dialog2list():
    assert len(V.file_dialog_source.selected.indices)>0 
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

def wantedwords_update(labels_file):
    with open(labels_file, "r") as fid:
        labels = fid.readlines()
    V.wantedwords_string.value = str.join(',',[x.strip() for x in labels])

def wantedwords_callback():
    assert len(V.file_dialog_source.selected.indices)==1
    idx = V.file_dialog_source.selected.indices[0]
    labels_file = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    wantedwords_update(labels_file)

def prevalences_callback():
    assert len(V.file_dialog_source.selected.indices)==1
    idx = V.file_dialog_source.selected.indices[0]
    classify_wav_log = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    with open(classify_wav_log,'r') as fid:
        for line in fid:
            if "prevalences: " in line:
                m=re.search('prevalences: \[(.*)\]', line)
                prevalences = m.group(1)
                break
    V.prevalences_string.value = prevalences.replace(' ',',')

def copy_callback():
    assert len(V.file_dialog_source.selected.indices)==1
    idx = V.file_dialog_source.selected.indices[0]
    logfile = os.path.join(M.file_dialog_root, V.file_dialog_source.data['names'][idx])
    with open(logfile, "r") as fid:
        for line in fid:
            if "batch_size" in line:
                m=re.search('batch_size = (\d+)', line)
                V.mini_batch_string.value = m.group(1)
            elif "clip_duration_ms" in line:
                m=re.search('clip_duration_ms = (.*)', line)
                V.context_ms_string.value = m.group(1)
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
                m=re.search('start_checkpoint = (.*)', line)
                V.restore_from_string.value = m.group(1)
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
                break
    
def wizard_callback(wizard):
    M.wizard=None if M.wizard is wizard else wizard
    M.action=None
    if M.wizard==V.labelsounds:
        wantedwords=[]
        for i in range(M.audio_nchannels):
            i_str = str(i) if M.audio_nchannels>1 else ''
            wantedwords.append("time"+i_str+",frequency"+i_str)
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
  
def _doit_callback():
    M.function()
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
