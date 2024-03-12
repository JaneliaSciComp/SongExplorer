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

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, "songexplorer", "bin", "songexplorer", "test"))
from libtest import wait_for_job

_, *wavfiles = sys.argv
  
sys.path.append(os.path.join(__dir__, "songexplorer", "bin", "songexplorer", "src", "gui"))
import model as M
import view as V
import controller as C

M.init(None, os.path.join(__dir__, "configuration.py"))
V.init(None)
C.init(None)

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
            V.labels_touse.value = labels_touse = m.group(1)
        if "context_ms = " in line:
            m=re.search('context_ms = (.+)',line)
            V.context_ms.value = m.group(1)
        if "shiftby_ms = " in line:
            m=re.search('shiftby_ms = (.+)',line)
            V.shiftby_ms.value = m.group(1)
        if "loss = " in line:
            m=re.search('loss = (.+)',line)
            V.loss.value = m.group(1)

V.prevalences.value = ""

for wavfile in wavfiles:
    V.wavcsv_files.value = os.path.join(wavfile)
    asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

for wavfile in wavfiles:
    V.wavcsv_files.value = os.path.join(wavfile)
    asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

run(["hstop"], stdout=PIPE, stderr=STDOUT)
