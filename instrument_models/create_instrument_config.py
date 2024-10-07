"""
This python function populates the experiment config dictionary with
instrument settings that the PBL OSSE will need to run one iteration of the workflow

Inputs:
* expt_config: dictionary containing "sat_type", which is: AIRS, CRIS, IASI, CIRAS, MEDSAT, LGSAT

Output:
* Dictionary containing instrument model settings

This is essentially just a series of if - elif - else with settings for each sat type

If the user would like to add a new satellite type, they need only provide:
1. Specific paths, grid specifications, and variable names

Derek Posselt
JPL
12 September 2023

Change log:
25 Sep 2023, DSM: Add error handling.
22 Apr 2024, DJP: Instrument settings are now contained in config files contained in the configs/
                  directory. Files are assumed to be named <sat_type>.yml with sat_type all upper case.
                  Instrument information is appended to the input expt_config dictionary

"""

# Import modules
import os
import yaml

CONFIG_FILES_PATH = os.path.dirname(os.path.abspath(__file__))

def create_instrument_config(expt_config):

  # Convert sat type to upper case
  SAT_Type = expt_config['sat_type'].upper()

  with open(os.path.join(CONFIG_FILES_PATH, 'configs', SAT_Type+'.yml'),'r') as f:
    expt_config.update(yaml.safe_load(f))

  # Return dictionary to calling function
  return expt_config

