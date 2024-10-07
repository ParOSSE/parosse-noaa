"""
This python function ingests the nature run and experiment configurations and populates the list of
nature run dictionaries for ingest into parmap for the radiative transfer model

It does the following:
1. Read in the nature run dimensions
2. Loop over all possible domain tiles, reading the NR data into sub-dictionaries
3. Return the run list, dimensions, and updated experiment dictionary

Derek Posselt
JPL
22 Sept 2023

Change log:
25 Sep 2023, DSM: Replace verbose with logger.
14 Nov 2023, DJP: There are now several incidences of "reader", which handles all of the file and grid operations

"""
import copy
import logging
import numpy as np
import pandas as pd

# Set logging
logger = logging.getLogger(__name__)

def adjust_domain_bounds(nr_config, nr_dims):
    """
    Adjusts the i1, i2, j1, j2 domain bounds if not set, using nature run dimensions.
    """
    nr_config['i1'] = 0 if pd.isnull(nr_config['i1']) else nr_config['i1']
    nr_config['j1'] = 0 if pd.isnull(nr_config['j1']) else nr_config['j1']
    nr_config['i2'] = nr_dims['nx'] if pd.isnull(nr_config['i2']) else nr_config['i2']
    nr_config['j2'] = nr_dims['ny'] if pd.isnull(nr_config['j2']) else nr_config['j2']

    logger.info(f"Start and end i and j: {nr_config['i1']} {nr_config['i2']} {nr_config['j1']} {nr_config['j2']}")

    # Reset the horizontal dimensions according to the adjusted bounds
    nr_dims['nx'] = nr_config['i2'] - nr_config['i1']
    nr_dims['ny'] = nr_config['j2'] - nr_config['j1']
    logger.info(f'Adjusted nx, ny: {nr_dims["nx"]} {nr_dims["ny"]}')


def adjust_dimensions_to_coarsening(nr_config, nr_dims):
    """
    Ensures that the nature run dimensions (nx, ny) are a multiple of the coarsening factors (icoarse, jcoarse).
    """
    nx_mod = np.mod(nr_dims['nx'], nr_config['icoarse'])
    ny_mod = np.mod(nr_dims['ny'], nr_config['jcoarse'])

    if nx_mod != 0:
        logger.info('Number of x-points is NOT a multiple of the coarsening factor')
        logger.info(f'Resetting number of x-points to {nr_dims["nx"] - nx_mod}')
        nr_dims['nx'] -= nx_mod

    if ny_mod != 0:
        logger.info('Number of y-points is NOT a multiple of the coarsening factor')
        logger.info(f'Resetting number of y-points to {nr_dims["ny"] - ny_mod}')
        nr_dims['ny'] -= ny_mod


def generate_fwd_runs(reader, expt_config, nr_config, nr_dims):
    """
    Generates a list of forward model runs if RTM is not pre-read.
    """
    nr_fwd_runs = []
    nx_tile = nr_config['nx_tile']
    ny_tile = nr_config['ny_tile']

    for nr_data in reader.read(nr_config['i1'], nr_config['j1'], nr_config['icoarse'], nr_config['jcoarse'], nr_dims['nx'], nx_tile, nr_dims['ny'], ny_tile):
        # Append information to be passed to the forward model
        nr_fwd_runs.append(
            [copy.deepcopy(expt_config), copy.deepcopy(nr_config), nr_data, len(nr_fwd_runs) - 1]
        )

    return nr_fwd_runs


def pre_process_nr(nr_config, expt_config):
    """
    Pre-processes the nature run data based on the configuration.

    Steps:
    1. Initialize the reader.
    2. Adjust domain bounds and dimensions.
    3. Adjust dimensions to coarsening factors.
    4. Generate the forward model run list if RTM is not pre-read.
    """
    # Initialize the reader object for the nature run type
    reader = nr_config['read_nr']()
    reader.open_data(nr_config['nr_file1'], nr_config['nr_file2'])

    # Obtain the dimensions from the NR file
    nr_dims = reader.get_dimensions()
    logger.info(f'Original nx, ny: {nr_dims["nx"]} {nr_dims["ny"]}')

    # Adjust domain bounds (i1, i2, j1, j2) based on nature run dimensions
    adjust_domain_bounds(nr_config, nr_dims)

    # Ensure dimensions are multiples of the coarsening factors
    adjust_dimensions_to_coarsening(nr_config, nr_dims)

    # If the user has not requested RTM data to be read from a file, generate the forward run list
    nr_fwd_runs = []
    if not expt_config['read_RTM']:
        nr_fwd_runs = generate_fwd_runs(reader, expt_config, nr_config, nr_dims)

    reader.close_data()

    # Return the list of nature run data and the adjusted dimensions
    return nr_fwd_runs, nr_dims