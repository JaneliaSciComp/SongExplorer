import os
from subprocess import run, PIPE, STDOUT
import time

def wait_for_job(status_ticker_queue):
  while True:
    p = run(["hetero", "jobs"], stdout=PIPE, stderr=STDOUT)
    if p.stdout.decode('ascii').rstrip() == "":
      for key in status_ticker_queue.keys():
        if status_ticker_queue[key] == "failed":
          print("ERROR: status is 'failed' for '"+key+"'")
      break
    time.sleep(1)

def check_file_exists(filename):
  if not os.path.isfile(filename):
    print("ERROR: "+filename+" is missing")
    return False
  return True

def count_lines_with_label(filename, label, rightanswer, kind):
  if not check_file_exists(filename): return
  count = 0
  with open(filename,'r') as fid:
    for line in fid:
      if label in line:
        count += 1
  if count != rightanswer:
    print(kind+": "+filename+" has "+str(count)+" "+label+" when it should have "+str(rightanswer))
    if kind=="WARNING":
      print(kind+": it is normal for this to be close but not exact")

def count_lines(filename, rightanswer):
  if not check_file_exists(filename): return
  count = 0
  with open(filename,'r') as fid:
    for line in fid:
      count += 1
  if count != rightanswer:
    print("ERROR: "+filename+" has "+str(count)+" lines when it should have "+str(rightanswer))
