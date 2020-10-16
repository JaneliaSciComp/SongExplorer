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

def count_lines_with_word(filename, word, rightanswer):
  if not check_file_exists(filename): return
  count = 0
  with open(filename,'r') as fid:
    for line in fid:
      if word in line:
        count += 1
  if count != rightanswer:
    print("ERROR: "+filename+" has "+str(count)+" "+word+" when it should have "+str(rightanswer))

def count_lines(filename, rightanswer):
  if not check_file_exists(filename): return
  count = 0
  with open(filename,'r') as fid:
    for line in fid:
      count += 1
  if count != rightanswer:
    print("ERROR: "+filename+" has "+str(count)+" lines when it should have "+str(rightanswer))
