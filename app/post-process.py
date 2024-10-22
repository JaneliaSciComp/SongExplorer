import sys
import ast
import os
import re
import csv
import socket
from datetime import datetime

delete_wavs = False
delete_logs = False

print(str(datetime.now())+": start time")

__dir__ = os.path.dirname(__file__)

with open(os.path.join(__dir__, "songexplorer", "bin", "songexplorer", "VERSION.txt"), 'r') as fid:
    print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
print("hostname = "+socket.gethostname())

_, wavfile = sys.argv
print("wavefile = ", wavfile)

sys.path.append(os.path.join(__dir__, "songexplorer", "bin", "songexplorer", "src", "gui"))
import model as M

M.init(None, os.path.join(__dir__, "configuration.py"), False)

wavfile_noext = M.trim_ext(wavfile)

with open(wavfile_noext+"-classify.log",'r') as fid:
    for line in fid:
        if "labels: " in line:
            m=re.search('labels: (.+)',line)
            labels = ast.literal_eval(m.group(1))
        if "audio_tic_rate = " in line:
            m=re.search('audio_tic_rate = (.+)',line)
            audio_tic_rate = ast.literal_eval(m.group(1))
print("labels = ", labels)
print("audio_tic_rate  = ", audio_tic_rate)

if delete_wavs:
    for label in labels:
        fullpath = wavfile_noext+'-'+label+'.wav'
        os.remove(fullpath)
        print("deleting ", fullpath)
    
durations = {}
counts = {}
with open(wavfile_noext+'-predicted-1.0pr.csv') as fid:
    csvreader = csv.reader(fid)
    for row in csvreader:
        if row[4] not in durations:
            durations[row[4]] = 0
            counts[row[4]] = 0
        durations[row[4]] += (int(row[2]) - int(row[1]))
        counts[row[4]] += 1

with open(wavfile_noext+"-post-process.csv",'w') as fid:
    fid.write("wavfile,label,duration ("+M.context_time_units+"),num events\n")
    for label in durations.keys():
        fid.write(os.path.basename(wavfile)+','+
                  label+','+
                  str(durations[label] / audio_tic_rate / M.context_time_scale)+','+
                  str(counts[label])+'\n')

if delete_logs:
    os.remove(wavfile_noext+'-classify.log')
    os.remove(wavfile_noext+'-ethogram.log')
    os.remove(wavfile_noext+'-post-process.log')

print(str(datetime.now())+": finish time")
