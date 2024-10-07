"""
This python 3x code runs the DAR retrieval on three radar reflectivity output files

The retrieval code requires at minimum the three file names as command line input.

Inputs:
1. files = (list of strings) Python list containing three strings, each consisting of one of  three radar reflectivity input files (required)
2. outname = (string) Path to and name of output file (optional, default = dar_retrievals.nc)
3. instfile = (string) Path to and name of the yaml file containing instrument / orbit parameters (optional, default = dar_orbit.yml)
4. Force = (logical, True/False), whether to overwrite the output file if it exists (optional, default = False)
5. rstep_inp = (float, units of km) Step size for the retrieval of water vapor. Should be > radar range resolution (optional, default = 0.2 km)

Retrieval code provided by Luis Millan, JPL

Derek Posselt
JPL
28 May 2024

"""

from dar_retrieval_driver import dar_retrieval_driver
import sys
from time import time
import yaml

# -------------------------------------------------------------------
# Main routine - drives the experiment
# -------------------------------------------------------------------
if __name__ == '__main__':

  # Obtain name of yaml file from command line, defaults to dar_retrieval.yml
  try:
    retr_config_file = sys.argv[1]
  except:
    retr_config_file = 'dar_retrieval.yml'

  # Start the timer
  t0 = time()

  # Read settings from yaml file
  with open(retr_config_file,'r') as f:
    retr_config = yaml.safe_load(f)

  # Populate variables
  files     = retr_config['files']
  outname   = retr_config['outname']
  instfile  = retr_config['instfile']
  force     = retr_config['force']
  rstep_inp = retr_config['rstep_inp']

  # Diagnostic print
  print('Input files:       ',files[:])
  print('Output file name:  ',outname)
  print('Instrument file:   ',instfile)
  print('Force logical:     ',force)
  print('rstep_inp:         ',rstep_inp)

  # sys.exit()

  # Run the retrieval
  result = dar_retrieval_driver(files, outname=outname, instfile=instfile, force=force, rstep_inp=rstep_inp)

  t1 = time()

  # Print timing
  print(f'Time to run DAR retrieval:           {t1-t0}')
