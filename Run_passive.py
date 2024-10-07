"""
This python 3x code runs a user-defined set of workflows, each corresponding to a single type
of passive T and RH observation for the ParOSSE experiment. 

Derek Posselt
JPL
22 April 2024

"""
import subprocess, os, sys
import logging
from time import time

# Set timer
t1 = time()

# Set logging
logger = logging.getLogger(__name__)

# Set the root path
parosse_path = '/Users/derekp/parosse-noaa/'

# Set name and path to the pbl osse run script and the root name of the yaml config file
run_script = parosse_path+'Run_parosse.py'
yaml_root = parosse_path+'configs/parosse_'

# Define the set of passive measurement types to be run
# Each must correspond (for now) to a yaml file named parosse_<obs type>.yml
# inst = ['ATMS']
# inst = ['ATMS', 'MED_MW', 'LG_MW', 'HS_MW']
# inst = ['CRIS', 'CIRAS', 'GEOIR', 'MED_IR', 'LG_IR']
inst = ['CRIS', 'CIRAS', 'GEOIR', 'MED_IR', 'LG_IR', 'ATMS', 'MED_MW', 'LG_MW', 'HS_MW']
# inst = []

# Define the logging level
log_level=1
log_level_str = str(log_level)

for i in inst:

  # Define the log file
  log_file = parosse_path+i+'.log'

  # logger.info(f'Running workflow for instrument: {i}')
  print('-----------------------------------------------')
  print('Running workflow for instrument: ',i)
  print('-----------------------------------------------')

  yaml_file = yaml_root+i+'.yml'
  # run_pbl_osse = ['python', run_script, ' --file ', yaml_file, '--log-level',log_level]
  # print("Command line arguments: ",run_pbl_osse[:])
  # subprocess.check_call(run_pbl_osse[:])
  run_pbl_osse = 'python '+run_script+' --file '+yaml_file+' --log-level '+log_level_str+' --log-file '+log_file
  print("Command line arguments: ",run_pbl_osse)
  os.system(run_pbl_osse)

  print('-----------------------------------------------')
  print('Finished running workflow for instrument: ',i)
  print('-----------------------------------------------')

# Increment the timer
t2 = time()

# Print timing
print(f'Time to run passive experiment:           {t2-t1}')

