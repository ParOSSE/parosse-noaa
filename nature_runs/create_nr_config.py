"""
This python function populates the nature run settings dictionary with
all that the PBL OSSE will need to run one iteration of the workflow

Inputs:
* NR type: GCE, RAMS, G5NR, WRF, CM1, LES

Output:
* Dictionary containing NR settings and also the appropriate read function

This is essentially just a series of if - elif - else with settings for each NR type

If the user would like to add a new NR type, they need only provide:
1. A reader following the form of the existing readers
2. Specific paths, grid specifications, and variable names

Derek Posselt
JPL
12 September 2023

Change log:
25 Sep 2023, DSM: Replace verbose with logger. Add error handling.
"""

import os
import sys
import logging
import yaml
import numpy as np
from utils.data_handling import copy_by_keys

# Set up logging
logger = logging.getLogger(__name__)

# Path to configuration files
CONFIG_FILES_PATH = os.path.dirname(os.path.abspath(__file__))


def load_configs(nr_type):
    """
    Loads configuration for a given nature run type from the corresponding YAML file.
    """
    config_path = os.path.join(CONFIG_FILES_PATH, 'configs', f'{nr_type}.yml')
    with open(config_path) as file:
        return yaml.safe_load(file)


def generate_config(nr_type, expt=None, config_file=None):
    """
    Generates the configuration dictionaries by combining defaults and specific experiment settings.
    """
    default_configs = load_configs('defaults')

    # If the configuration has already been loaded from CSV
    if isinstance(config_file, dict):
        return {**default_configs, **config_file}

    # Otherwise, load the configuration from YAML
    try:
        config_file = load_configs(nr_type)
    except FileNotFoundError:
        raise ValueError(f'Unknown NR type: {nr_type}. Stopping.')

    # Return the configuration based on whether an experiment is specified
    if expt is None:
        return {**default_configs, **config_file['config']}
    elif expt in config_file['expt']:
        return {**default_configs, **config_file['config'], **config_file['expt'][expt]} if 'config' in config_file else {**default_configs, **config_file['expt'][expt]}
    else:
        raise ValueError(f'Unrecognized {nr_type} experiment: {expt}. Stopping.')


def validate_tile_size(nr_config):
    """
    Ensures the tile size is a multiple of the coarsening factor.
    """
    for cfact in ['icoarse', 'jcoarse']:
        nr_config[cfact] = np.int32(np.round(nr_config[cfact]))

    if np.mod(nr_config['nx_tile'], nr_config['icoarse']) != 0:
        logger.error(f"x-tile size ({nr_config['nx_tile']}) is not a multiple of icoarse ({nr_config['icoarse']})")
        sys.exit('Check x-tile and icoarse compatibility.')

    if np.mod(nr_config['ny_tile'], nr_config['jcoarse']) != 0:
        logger.error(f"y-tile size ({nr_config['ny_tile']}) is not a multiple of jcoarse ({nr_config['jcoarse']})")
        sys.exit('Check y-tile and jcoarse compatibility.')


def create_nr_config(expt_config, nr_config=None):
    """
    Creates a Nature Run configuration based on experiment type and other inputs.
    """
    nr_type = expt_config['nr_type'].upper()
    expt = expt_config.get('nr_expt', '').upper()
    nr_date = expt_config.get('nr_date')
    nr_time = expt_config.get('nr_time')

    logger.info(f'Setting configuration for NR type {nr_type} and experiment {expt}')

    # Basic input validation for specific NR types
    if nr_type == 'RAMS' and (not nr_date or not nr_time or not expt):
        raise ValueError('RAMS requires date, time, and experiment name. Stopping.')

    if nr_type == 'GCE' and not expt:
        raise ValueError('GCE requires an experiment name. Stopping.')

    # Generate configuration based on NR type
    nr_config = generate_config(nr_type, expt, config_file=nr_config)

    # Process specific NR types
    if nr_type == 'GCE':
        from nature_runs.read_gce import GCEReader
        nr_config['read_nr'] = GCEReader
        nr_config['file_prefix'] = f"{nr_config['file_prefix']}_{nr_date}_{nr_time}"
        nr_config['nr_file1'] = os.path.join(nr_config['path'], nr_config['file_prefix'] + nr_config['file_suffix1'])
        nr_config['nr_file2'] = os.path.join(nr_config['path'], nr_config['file_prefix'] + nr_config['file_suffix2'])

    elif nr_type == 'RAMS':
        from nature_runs.read_rams import RAMSReader
        nr_config['read_nr'] = RAMSReader
        nr_config['path'] = os.path.join(nr_config['path'], expt_config['nr_expt'], '')
        nr_config['file_prefix'] = f"{nr_config['file_prefix']}-{nr_date}-{nr_time}"
        nr_config['nr_file1'] = os.path.join(nr_config['path'], nr_config['file_prefix'] + nr_config['file_suffix1'])
        if nr_config['nr_file2']:
            nr_config['nr_file2'] = os.path.join(nr_config['path'], nr_config['nr_file2'])

    elif nr_type == 'WRF':
        from nature_runs.read_wrf import WRFReader
        nr_config['read_nr'] = WRFReader
        if expt in ['HARVEY', 'MCS']:
            nr_config['file_prefix'] = f"{nr_config['nr_file1']}{nr_date}-{nr_time}"
            nr_config['nr_file1'] = os.path.join(nr_config['path'], nr_config['nr_file1'] + f'{nr_date}_{nr_time}')
            nr_config['nr_file2'] = nr_config['nr_file1']
            nr_config['dx'] = nr_config['dxm'] * 1e-3
            nr_config['dy'] = nr_config['dym'] * 1e-3

    elif nr_type == 'GEOS':
        from nature_runs.read_g5nr import G5NRReader
        nr_config['read_nr'] = G5NRReader
        nr_config['path'] = os.path.join(nr_config['path'], expt, '')
        nr_config['file_prefix'] = f"{nr_config['file_prefix']}{nr_date}-{nr_time}"
        nr_config['nr_file1'] = os.path.join(nr_config['path'], nr_config['nr_file1'])
        nr_config['nr_file2'] = f'{nr_date}_{nr_time}'

    elif nr_type == 'LES':
        from nature_runs.read_les import LESReader
        nr_config['read_nr'] = LESReader
        nr_config['path'] = os.path.join(nr_config['path'], expt)
        nr_config['file_prefix'] = f'LES_{expt}_{nr_time}'
        nr_config['nr_file1'] = os.path.join(nr_config['path'], f'{expt}.{nr_time}s.nc')
        nr_config['nr_file2'] = nr_config['nr_file1']

        # Specific settings for LES experiments
        if expt in ['RICO', 'VOCALS']:
            nr_config['dx'] = nr_config['dxm'] * 1e-3
            nr_config['dy'] = nr_config['dym'] * 1e-3

    # Additional validations
    elif nr_type == 'CM1':
        pass
    else:
        raise ValueError(f'Unknown NR type: {nr_type}. Stopping.')

    # Validate tile sizes and coarsening factors
    validate_tile_size(nr_config)

    # Copy relevant variables from nr_config to expt_config
    copy_by_keys(nr_config, expt_config, keys=[
        'file_prefix', 'h2o_var', 'temp_var', 'cloud_var', 'pcp_var', 'lat_var',
        'lon_var', 'z_idx', 'icoarse', 'jcoarse'
    ])

    return nr_config, expt_config
