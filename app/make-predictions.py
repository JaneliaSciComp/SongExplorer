#!/usr/bin/env python3

# run classify and ethogram on the input recordings using the enclosed model

logdir = "trained-classifier2"
model = "train_1r"
ckpt = "ckpt-300.index"

import sys
import os
import psutil
import glob
from subprocess import run, PIPE, STDOUT
import asyncio
import tensorflow as tf
import re
import platform

use_aitch = False

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, "songexplorer", "bin", "songexplorer", "test"))
from libtest import wait_for_job

_, *wavpaths = sys.argv
  
sys.path.append(os.path.join(__dir__, "songexplorer", "bin", "songexplorer", "src", "gui"))
import model as M
import view as V
import controller as C

M.init(None, os.path.join(__dir__, "configuration.py"), use_aitch)
V.init(None)
C.init(None)

if use_aitch:
    local_ncpu_cores = os.cpu_count()
    local_ngpu_cards = len(tf.config.list_physical_devices("GPU"))
    local_ngigabytes_memory = int(psutil.virtual_memory().total/1024/1024/1024)

    print("detected "+str(local_ncpu_cores)+" local_ncpu_cores, "+
                      str(local_ngpu_cards)+" local_ngpu_cards, "+
                      str(local_ngigabytes_memory)+" local_ngigabytes_memory")

    run(["hstart", str(local_ncpu_cores)+','+str(local_ngpu_cards)+','+str(local_ngigabytes_memory)])

V.logs_folder.value = os.path.join(os.path.join(__dir__, logdir))
V.model_file.value = os.path.join(os.path.join(__dir__, logdir, model, ckpt))

with open(os.path.join(__dir__, logdir, model+".log"),'r') as fid:
    for line in fid:
        if "labels_touse = " in line:
            m=re.search('labels_touse = (.+)',line)
            V.labels_touse.value = m.group(1)
        if "context = " in line:
            m=re.search('context = (.+)',line)
            V.context.value = m.group(1)
        if "shiftby = " in line:
            m=re.search('shiftby = (.+)',line)
            V.shiftby.value = m.group(1)
        if "loss = " in line:
            m=re.search('loss = (.+)',line)
            V.loss.value = m.group(1)

V.prevalences.value = ""

def do_it(wavfile):
    V.wavcsv_files.value = os.path.join(wavfile)

    V.waitfor.active = False
    M.waitfor_job = []
    asyncio.run(C.classify_actuate())

    V.waitfor.active = True
    asyncio.run(C.ethogram_actuate())

    logfile = wavfile+"-post-process.log"
    ncpu_cores, ngpu_cards, ngigabyes_memory  = 1, 0, 8
    localdeps = ["-d "+M.waitfor_job.pop()]
    kwargs = {"process_group": 0} if sys.version_info.major == 3 and sys.version_info.minor >= 11 else {}
    cmd = os.path.join(__dir__, "post-process.py")
    arg = wavfile
    if platform.system()=='Windows':
        cmd = cmd.replace(os.path.sep, os.path.sep+os.path.sep)
        arg = arg.replace(os.path.sep, os.path.sep+os.path.sep)
    if use_aitch:
        run(["hsubmit",
             "-o", logfile, "-e", logfile, "-a",
             str(ncpu_cores)+','+str(ngpu_cards)+','+str(ngigabyes_memory),
             *localdeps,
             "python",
             cmd,
         arg],
            **kwargs)
    else:
        run(["python", cmd, arg], **kwargs)

wavfiles = []
for wavpath in wavpaths:
    if os.path.isdir(wavpath):
        for ext in M.audio_read_exts():
            wavfiles.extend(glob.glob("**/*"+ext, root_dir=wavpath, recursive=True))
    else:
        wavfiles.append(wavpath)
if len(M.audio_read_rec2ch()) > 1:
    wavfiles = [w+'-'+k for w in wavfiles for k in M.audio_read_rec2ch().keys()]
for wavfile in wavfiles:
    do_it(wavfile)

wait_for_job(M.status_ticker_queue)

run(["hstop"], stdout=PIPE, stderr=STDOUT)
