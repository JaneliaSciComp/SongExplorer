#!/usr/bin/python3

# test congruence

# export SINGULARITYENV_SONGEXPLORER_STATE=/tmp
# ${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} test/congruence.py

import sys
import os
import shutil
from subprocess import run, PIPE, STDOUT
import asyncio
import tarfile
import pandas as pd

from lib import wait_for_job, check_file_exists, count_lines_with_word, count_lines

def check_value(dfrow, colname, shouldbe):
  if int(dfrow[colname]) != shouldbe:
    print("ERROR: "+colname+" is "+str(int(dfrow[colname]))+" but should be "+str(shouldbe))

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

with tarfile.open(os.path.join(repo_path, "test/congruence.tar.xz")) as fid:
  fid.extractall(os.path.join(repo_path, "test/scratch"))

dummydatadir = os.path.join(repo_path, "test/scratch/congruence/groundtruth-data/dummy-data")
for ethogramfile in filter(lambda x: x.endswith("ethogram.log"), os.listdir(dummydatadir)):
  with open(os.path.join(dummydatadir, ethogramfile), "r") as sources:
      lines = sources.readlines()
  with open(os.path.join(dummydatadir, ethogramfile), "w") as sources:
      for line in lines:
          sources.write(line.replace('REPO_PATH', repo_path))

shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/congruence"))

M.init(None, os.path.join(repo_path, "test/scratch/congruence/configuration.pysh"))
V.init(None)
C.init(None)

run(["hetero", "start", "1", "1", "1"])

V.groundtruth_folder.value = os.path.join(repo_path, "test/scratch/congruence/groundtruth-data")

V.testfiles_string.value = ""
V.validationfiles_string.value = "recording1.wav,recording2.wav,recording3.wav,recording4.wav"
asyncio.run(C.congruence_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.validationfiles_string.value[:-4]
check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence.log"))
for i in range(1,8):
  check_file_exists(os.path.join(V.groundtruth_folder.value,
                                 "congruence.tic.label"+str(i)+".csv"))
  check_file_exists(os.path.join(V.groundtruth_folder.value,
                                 "congruence.word.label"+str(i)+".csv"))

l1 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.word.label1.csv"))
pr1 = l1.loc[l1['Unnamed: 0'] == '1.0pr']

check_value(pr1, "everyone", 1)
check_value(pr1, "only person1", 2)
check_value(pr1, "only person2", 3)
check_value(pr1, "only songexplorer", 4)
check_value(pr1, "not person1", 5)
check_value(pr1, "not person2", 6)
check_value(pr1, "not songexplorer", 7)

l2 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.word.label2.csv"))
pr1 = l2.loc[l2['Unnamed: 0'] == '1.0pr']

check_value(pr1, "everyone", 1)
check_value(pr1, "only person1", 2)
check_value(pr1, "only person2", 3)
check_value(pr1, "only songexplorer", 0)
check_value(pr1, "not songexplorer", 0)

for i in range(3,8):
  li = pd.read_csv(os.path.join(V.groundtruth_folder.value,
                                "congruence.word.label"+str(i)+".csv"))
  pr1 = li.loc[li['Unnamed: 0'] == '1.0pr']
  check_value(pr1, "only person1", 1)
  check_value(pr1, "not person2", 2)

l1 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label1.csv"))
pr1 = l1.loc[l1['Unnamed: 0'] == '1.0pr']

check_value(pr1, "everyone", 1)
check_value(pr1, "only person1", 2)
check_value(pr1, "only person2", 3)
check_value(pr1, "only songexplorer", 4)
check_value(pr1, "not person1", 5)
check_value(pr1, "not person2", 6)
check_value(pr1, "not songexplorer", 7)

l2 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label2.csv"))
pr1 = l2.loc[l2['Unnamed: 0'] == '1.0pr']

check_value(pr1, "everyone", 1)
check_value(pr1, "only person1", 2)
check_value(pr1, "only person2", 3)
check_value(pr1, "only songexplorer", 22)
check_value(pr1, "not songexplorer", 0)

l3 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label3.csv"))
pr1 = l3.loc[l3['Unnamed: 0'] == '1.0pr']

check_value(pr1, "only songexplorer", 10)
check_value(pr1, "not songexplorer", 0)
check_value(pr1, "only person1", 5)
check_value(pr1, "not person1", 0)
check_value(pr1, "only person2", 0)
check_value(pr1, "not person2", 10)
check_value(pr1, "everyone", 0)

l4 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label4.csv"))
pr1 = l4.loc[l4['Unnamed: 0'] == '1.0pr']

check_value(pr1, "only songexplorer", 10)
check_value(pr1, "not songexplorer", 0)
check_value(pr1, "only person1", 6)
check_value(pr1, "not person1", 0)
check_value(pr1, "only person2", 0)
check_value(pr1, "not person2", 9)
check_value(pr1, "everyone", 0)

l5 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label5.csv"))
pr1 = l5.loc[l5['Unnamed: 0'] == '1.0pr']

check_value(pr1, "only songexplorer", 10)
check_value(pr1, "not songexplorer", 0)
check_value(pr1, "only person1", 7)
check_value(pr1, "not person1", 0)
check_value(pr1, "only person2", 0)
check_value(pr1, "not person2", 8)
check_value(pr1, "everyone", 0)

l6 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label6.csv"))
pr1 = l6.loc[l6['Unnamed: 0'] == '1.0pr']

check_value(pr1, "only songexplorer", 10)
check_value(pr1, "not songexplorer", 0)
check_value(pr1, "only person1", 5)
check_value(pr1, "not person1", 0)
check_value(pr1, "only person2", 0)
check_value(pr1, "not person2", 4)
check_value(pr1, "everyone", 0)

l7 = pd.read_csv(os.path.join(V.groundtruth_folder.value, "congruence.tic.label7.csv"))
pr1 = l7.loc[l7['Unnamed: 0'] == '1.0pr']

check_value(pr1, "only songexplorer", 10)
check_value(pr1, "not songexplorer", 0)
check_value(pr1, "only person1", 7)
check_value(pr1, "not person1", 0)
check_value(pr1, "only person2", 0)
check_value(pr1, "not person2", 12)
check_value(pr1, "everyone", 0)

correctvalues = [
  ["recording1-disjoint-everyone.csv", 1],
  ["recording1-disjoint-tic-not1.0pr.csv", 7],
  ["recording1-disjoint-tic-notperson1.csv", 5],
  ["recording1-disjoint-tic-notperson2.csv", 6],
  ["recording1-disjoint-tic-only1.0pr.csv", 4],
  ["recording1-disjoint-tic-onlyperson1.csv", 2],
  ["recording1-disjoint-tic-onlyperson2.csv", 3],
  ["recording1-disjoint-word-not1.0pr.csv", 7],
  ["recording1-disjoint-word-notperson1.csv", 5],
  ["recording1-disjoint-word-notperson2.csv", 6],
  ["recording1-disjoint-word-only1.0pr.csv", 4],
  ["recording1-disjoint-word-onlyperson1.csv", 2],
  ["recording1-disjoint-word-onlyperson2.csv", 3],
  ["recording2-disjoint-everyone.csv", 1],
  ["recording2-disjoint-tic-not1.0pr.csv", 0],
  #["recording2-disjoint-tic-notperson1.csv", ],
  #["recording2-disjoint-tic-notperson2.csv", ],
  ["recording2-disjoint-tic-only1.0pr.csv", 1],
  ["recording2-disjoint-tic-onlyperson1.csv", 2],
  ["recording2-disjoint-tic-onlyperson2.csv", 3],
  ["recording2-disjoint-word-not1.0pr.csv", 0],
  #["recording2-disjoint-word-notperson1.csv", ],
  #["recording2-disjoint-word-notperson2.csv", ],
  ["recording2-disjoint-word-only1.0pr.csv", 0],
  ["recording2-disjoint-word-onlyperson1.csv", 2],
  ["recording2-disjoint-word-onlyperson2.csv", 3],
  ["recording3-disjoint-everyone.csv", 0],
  ["recording3-disjoint-tic-not1.0pr.csv", 0],
  ["recording3-disjoint-tic-notperson1.csv", 0],
  ["recording3-disjoint-tic-notperson2.csv", 5],
  ["recording3-disjoint-tic-only1.0pr.csv", 0],
  ["recording3-disjoint-tic-onlyperson1.csv", 4],
  ["recording3-disjoint-tic-onlyperson2.csv", 0],
  ["recording3-disjoint-word-not1.0pr.csv", 0],
  ["recording3-disjoint-word-notperson1.csv", 0],
  ["recording3-disjoint-word-notperson2.csv", 5],
  ["recording3-disjoint-word-only1.0pr.csv", 0],
  ["recording3-disjoint-word-onlyperson1.csv", 0],
  ["recording3-disjoint-word-onlyperson2.csv", 0],
  ["recording4-disjoint-everyone.csv", 0],
  ["recording4-disjoint-tic-not1.0pr.csv", 0],
  ["recording4-disjoint-tic-notperson1.csv", 0],
  ["recording4-disjoint-tic-notperson2.csv", 5],
  ["recording4-disjoint-tic-only1.0pr.csv", 5],
  ["recording4-disjoint-tic-onlyperson1.csv", 5],
  ["recording4-disjoint-tic-onlyperson2.csv", 0],
  ["recording4-disjoint-word-not1.0pr.csv", 0],
  ["recording4-disjoint-word-notperson1.csv", 0],
  ["recording4-disjoint-word-notperson2.csv", 5],
  ["recording4-disjoint-word-only1.0pr.csv", 0],
  ["recording4-disjoint-word-onlyperson1.csv", 5],
  ["recording4-disjoint-word-onlyperson2.csv", 0],
  ]

for filename, correctvalue in correctvalues:
  filepath = os.path.join(V.groundtruth_folder.value, "dummy-data", filename)
  if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
    df = pd.read_csv(filepath, header=None, index_col=False)
  else:
    df = pd.DataFrame()
  if len(df.index)!=correctvalue:
    print("ERROR: "+filename+" has "+str(len(df.index))+" rows when it should have "+str(correctvalue))

run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)
