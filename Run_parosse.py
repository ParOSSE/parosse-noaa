"""
This is the driver for the ParOSSE experiment designed to evaluate the ability
of various measurement concepts to estimate temperature and moisture profiling

Procedure:
1. Define settings for the experiment
2. Read nature run data
3. Translate nature run data into retrieved temperature and water vapor fields
4. Run analysis and plotting routines

Derek Posselt
JPL
12 September 2023

Change log:
12 Sep 2023, DJP: Initial version
13 Sep 2023, DJP: Modified to parmap over the RTM (including pre-processing)
25 Sep 2023, DSM: Replace verbose with logger.
22 Apr 2023, DJP: Import workflow settings from a yaml configuration file,
                  the name and path to which are provided on the command line.
                  If none is provided, it defaults to parosse.yml in the current directory.
03 Jul 2024, DSM: Reduce code redundancy. Add support for csv configs and command line.

"""

import os
from time import time
import logging
import yaml
import argparse
import pathlib
import pandas as pd

# Nature run subsystem
from nature_runs.create_nr_config import create_nr_config
from nature_runs.pre_process_nr import pre_process_nr

# Forward / radiative transfer model subsystem
from forward_models.fwd_temp_rh import fwd_temp_rh
from forward_models.post_process_rtm import post_process_rtm

# Instrument model subsystem
from instrument_models.create_instrument_config import create_instrument_config
from instrument_models.gauss_filt_temp_rh import gauss_filt_temp_rh

# Retrieval subsystem
from retrievals.simple_sounder_temp_rh_retr import simple_sounder_temp_rh_retr

# Analysis and plotting subsystem
from analysis_functions.rmse_and_yield import rmse_and_yield
from analysis_functions.rmse_and_yield_latlon import rmse_and_yield_latlon
from analysis_functions.plot_temp_rh import plot_temp_rh

from utils.logger_config import configure_logger, crashed_log
from utils.data_handling import copy_by_keys, csv_get_nth_row, csv_get_number_rows

# Parmap - used here for vertical interpolation, and later for running forward models
from Parmap.parmap import parmap

# Set logging
logger = logging.getLogger(__name__)

exclude_keywords = ['readers', 'matplotlib']

OUTPUT_FILE_COLUMNS = ['nr_expt','nr_date','nr_time','sat_type','i1','i2','j1','j2',]
OUTPUT_COLUMNS = ['rmse_temp', 'rmse_h2o', 'rmse_rh', 'yield_temp', 'yield_h2o', 'yield_rh']

LOG_FILE = None

def experiment_config(experiment_config_file, csv_line=None):
    """
    Populates configuration dictionaries for the experiment.

    The configurations consist of:
    1. experiment_config: General settings for the entire workflow.
    2. nr_config: Settings related to the Nature Run (NR) subsystem.
    3. Instrument model settings (added to the experiment config).
    4. Retrieval settings (currently none).
    5. Plotting and analysis settings (added to the experiment config).
    """

    # -------------------------------------------------------------------
    # Initialize the NR config dictionary
    # -------------------------------------------------------------------
    nr_config = None

    # -------------------------------------------------------------------
    # Load general experiment settings from the config file (CSV or YAML)
    # -------------------------------------------------------------------
    if experiment_config_file.endswith('.csv'):
        assert csv_line is not None, "csv_line must be provided for CSV input."
        logger.info(f"Loading configuration from CSV: {experiment_config_file}, line {csv_line}")
        nr_config, experiment_config = csv_get_nth_row(experiment_config_file, csv_line)
    else:
        logger.info(f"Loading configuration from YAML: {experiment_config_file}")
        with open(experiment_config_file, 'r') as config_file:
            experiment_config = yaml.safe_load(config_file)

    # Log the contents of the experiment configuration
    log_config_keys(experiment_config, "experiment")

    # -------------------------------------------------------------------
    # Define settings for the Nature Run Subsystem (NR)
    # -------------------------------------------------------------------
    # Update experiment and NR config with Nature Run settings
    nr_config, experiment_config = create_nr_config(experiment_config, nr_config=nr_config)

    # Make sure that the output/plot directory exists.
    output_path = pathlib.Path(experiment_config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    plot_path = pathlib.Path(experiment_config['plot_path'])
    plot_path.mkdir(parents=True, exist_ok=True)

    # Log the contents of the NR configuration
    log_config_keys(nr_config, "Nature Run")

    # -------------------------------------------------------------------
    # Define settings for the Instrument Model Subsystem
    # -------------------------------------------------------------------
    # Add satellite instrument information to the experiment config
    experiment_config = create_instrument_config(experiment_config)

    # Log the contents of the updated experiment configuration
    log_config_keys(experiment_config, "experiment")

    return experiment_config, nr_config


def log_config_keys(config_dict, config_name):
    """
    Logs the keys and values of the given configuration dictionary.

    :param config_dict: The configuration dictionary to log.
    :param config_name: A label for the type of config (e.g., 'experiment', 'Nature Run').
    """
    logger.info(f'Keys in {config_name} configuration:')
    for key, value in config_dict.items():
        logger.info(f'  {key}: {value}')

# -----------------------------------
# Run the Forward Model Subsystem
# -----------------------------------
def run_fwd_model(nr_fwd_run):

  """
  This will run the Forward Model Subsystem
  For a single nx_tile x ny_tile section of the nature run
  1. Extract data from the input list
  2. Run the forward model
  3. Insert data into the output dictionary
  """
  # Extract data from the input list
  expt_config = nr_fwd_run[0]
  nr_config   = nr_fwd_run[1]
  nr_data     = nr_fwd_run[2]
  run_num     = nr_fwd_run[3]

  # Run the forward model.
  # In this case, this consists just of computing RH and interpolating in height
  rtm_data = fwd_temp_rh(nr_data, nr_config, expt_config, dz=nr_config['dz'], ztop=nr_config['ztop'])

  # Put the tile indices and dimensions into the rtm_data dictionary
  copy_by_keys(nr_data, rtm_data, ['ix', 'jy', 'nx_tile', 'ny_tile'])

  # Return the rtm data dictionary
  return rtm_data


def run_experiment(config_path, nth_row=None):
    # Start the timer
    t0 = time()

    # Configure the experiment
    expt_config, nr_config = experiment_config(config_path, nth_row)

    # Pre-process the nature run data for ingest into parmap
    # This needs doing, even if data is being read from file
    nr_fwd_runs, nr_dims = pre_process_nr(nr_config,expt_config)

    # If user has not requested RTM data be read from file, run RTM
    if ( not expt_config['read_RTM']):
      # Initialize parmap
      pmap = parmap.Parmap(mode=expt_config['parmode'], numWorkers=expt_config['num_workers'])

      # Increment the timer
      t1 = time()

      # Run the RTM, returning a list of output dictionaries
      rtm_data_list = pmap(run_fwd_model, nr_fwd_runs)

      # Increment the timer
      t2 = time()

    # User has requested RTM data be read
    else:
      # Create an empty data list and increment the timers
      rtm_data_list = []
      t1 = time()
      t2 = time()

    # If user has requested RTM be run, then reconstruct the RTM data and write it to file
    # Otherwise, read data from file
    rtm_data = post_process_rtm(rtm_data_list, expt_config, nr_config, nr_dims)

    # Increment the timer
    t3 = time()

    # -------------------------------------------------------------------
    # Run the Instrument Model Subsystem - No Parallelism
    # -------------------------------------------------------------------
    # The instrument model simply applies gaussian filtering to the "RTM" output
    inst_data = gauss_filt_temp_rh(rtm_data, nr_config, expt_config)

    # Increment the timer
    t4 = time()

    # -------------------------------------------------------------------
    # Run the Retrieval Subsystem
    # -------------------------------------------------------------------
    retr_data = simple_sounder_temp_rh_retr(expt_config, nr_config, rtm_data, inst_data)

    # Increment the timer
    t5 = time()

    # -------------------------------------------------------------------
    # Run the Plotting and Analysis Subsystem
    # -------------------------------------------------------------------
    # Compute RMSE
    if nr_config['grid_type'] == 'latlon':
      retr_data = rmse_and_yield_latlon(expt_config, rtm_data, retr_data)
    else:
      retr_data = rmse_and_yield(expt_config, rtm_data, retr_data)

    output_status = plot_temp_rh(nr_config, expt_config, rtm_data, retr_data, inst_data=inst_data)

    # Increment the timer
    t6 = time()

    # Print timing
    logger.info(f'Time to initialize:           {t1-t0}')
    logger.info(f'Time to run RTM:              {t2-t1}')
    logger.info(f'Time to reconstruct RTM:      {t3-t2}')
    logger.info(f'Time to run instrument model: {t4-t3}')
    logger.info(f'Time to run retrieval:        {t5-t4}')
    logger.info(f'Time to run analysis:         {t6-t5}')
    logger.info(f'Total run time:               {t6-t0}')

    return retr_data

def get_output_columns(retr_data):
    rmse_output = OUTPUT_COLUMNS
    return {key: retr_data[key] for key in rmse_output if key in retr_data}

# -------------------------------------------------------------------
# Main routine - drives the experiment
# -------------------------------------------------------------------
def main():

    """
    The workflow consists of 7 components:
    1. Fill the configuration dictionaries
    2. Pre-process the nature run data
    3. Run the forward model
    4. Post-process the forward model output
    5. Run the instrument model
    6. Run the retrieval
    7. Run the analysis and plotting routine(s)
    """

    # Parse command line arguments
    # Initialize parser
    parser = argparse.ArgumentParser(description="This is the driver for the ParOSSE experiment \
                                     designed to evaluate the ability of various measurement concepts \
                                     to estimate near-surface temperature and moisture contrasts associated \
                                     with convective cold pools"
                                     )
    # First argument is the config file name - defaults to parosse.yml
    parser.add_argument('--file', help="The file to parse (YAML or CSV)", default='parosse.yml')
    # Second argument is the line in the CSV file containing the settings to be used
    # Ignores the first row (containing headers), row 0 is the first row with data
    parser.add_argument('--n', type=int, help="The row number to extract from CSV (0-based index)", default=None)
    # Third argument is the log level, default is 2, warnings and errors.
    parser.add_argument('--log-level', type=int, help="0: all logs, 1: relevant messages, 2: only warnings", default=2)
    # Fourth argument is the log file, if none was provided log will be redirected to STDOUT (Terminal).
    parser.add_argument('--log-file', help="If None it will output the logs into the STDOUT, otherwise to the file",default=None)

    # Parse the input arguments
    args = parser.parse_args()

    LOG_FILE = args.log_file
    file_path = args.file

    configure_logger(exclude_keywords, args.log_level, LOG_FILE)

    # If the input file is not found, exit.
    if not os.path.exists(file_path):
        raise ValueError(f"Error: File {file_path} does not exist.")

    # Check if the file is a CSV
    if args.file.endswith('.csv'):

        retr_outputs = []

        # If no specific row is provided, iterate over all rows in the CSV
        if args.n is None:
            n_rows = csv_get_number_rows(file_path)  # Get the number of rows in the CSV
            logger.info(f"Iterating over all CSV lines, total rows: {n_rows}")
            # Loop over all rows and run the experiment for each
            for nth_row in range(n_rows):
                logger.info(f" -- Processing CSV: {file_path}, Line: {nth_row}")
                retr_data = run_experiment(file_path, nth_row)
                retr_outputs.append(get_output_columns(retr_data))
        else:
            # If a specific row is provided, process that row only
            logger.info(f"Processing CSV row {args.n}")
            retr_data = run_experiment(file_path, args.n)
            retr_outputs.append(get_output_columns(retr_data))


        retr_outputs = pd.DataFrame(retr_outputs)
        config_df = pd.read_csv(args.file)
        config_df = config_df[OUTPUT_FILE_COLUMNS]

        config_df.update(retr_outputs)
        config_df = config_df.combine_first(retr_outputs)
        config_df = config_df.reindex(columns=OUTPUT_FILE_COLUMNS+OUTPUT_COLUMNS)

        directory, file_name = os.path.split(args.file)
        file_name = os.path.splitext(file_name)[0]
        config_df.to_csv(os.path.join(directory, f"{file_name}_output.csv"))

    # Check if the file is YAML
    elif args.file.endswith('.yml') or args.file.endswith('.yaml'):
        logger.info(f"Processing YAML file: {file_path}")
        run_experiment(file_path)  # Directly run the experiment with the YAML file

    # Handle unsupported file types
    else:
        raise ValueError("File format not recognized. Supported formats are CSV and YAML.")


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    crashed_log(LOG_FILE)
    raise
