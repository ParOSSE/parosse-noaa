"""
This file contains a set of miscellaneous utility functions used to conduct repetetive
tasks in the OSSE code. Descriptions are included in each of the functions below.

"""

import copy
import logging
from typing import Any
from collections.abc import Mapping
import numpy as np
import pandas

# Set logging
logger = logging.getLogger(__name__)


def copy_by_keys(from_obj, to_dict=None, keys=()):
    """
    Copies specific keys from a source mapping object to a new dictionary.

    Args:
    - from_obj (Mapping): The source object from which to copy data.
    - to_dict (dict, optional): The destination dictionary to which data will be copied. Defaults to None, which means a new dictionary will be created.
    - keys (iterable): An iterable of keys to copy from the source to destination.

    Returns:
    - dict: A dictionary containing the copied key-value pairs.
    """
    assert isinstance(from_obj, Mapping)

    if to_dict is None:
        to_dict = {}

    for key in keys:
        to_dict[key] = copy.deepcopy(from_obj[key])

    return to_dict


def default_netcf_encoding_dict(keys, complevel=5):
    """
    Creates a dictionary specifying compression settings for NetCDF variables.

    Args:
    - keys (list): A list of variable names to which the settings will be applied.
    - complevel (int, optional): Compression level for zlib algorithm. Defaults to 5.

    Returns:
    - dict: A dictionary with keys as variable names and values specifying compression settings.
    """
    encode_dict = {}
    for key in keys:
      encode_dict[key] = {'zlib': True, 'complevel': complevel}
    return encode_dict


def get_coords(xdim, ydim, zdim):
    """
    Constructs a dictionary of coordinates based on dimensions provided.

    Args:
    - xdim, ydim, zdim (array-like): Arrays representing the x, y, and z dimensions respectively.

    Returns:
    - dict: A dictionary with keys 'nx', 'ny', and 'nz', each mapped to a tuple containing the dimension name and its corresponding array.
    """
    coords = {
        "nx": (["xdim"], xdim),
        "ny": (["ydim"], ydim),
        "nz": (["zdim"], zdim),
    }
    return coords

"""
The lists of keys below are used in the csv_get_nth_row function

The idea is to include the experiment configuration keys AND the nature run configuration keys
on a single row of the CSV file.
CONFIG_KEYS describes the experiment configuration, and comes first
NR_KEYS describes the nature run settings, and comes second

"""
CONFIG_KEYS = [
    'nr_type','nr_expt','nr_date','nr_time','sat_type','mask_cloud','mask_precip','cloud_thresh','precip_thresh',
    'parmode','num_workers','output_path','plot_path','read_RTM','read_INST','read_RETR',
]

NR_KEYS = [
    'path','nr_file1','nr_file2','file_prefix','file_suffix1','file_suffix2','h2o_var','temp_var','pres_var',
    'hgt_var','cloud_var','pcp_var','lat_var','lon_var','dlat','dlon','clat','clon','grid_type','y_idx','z_idx','i1','i2',
    'j1','j2','icoarse','jcoarse','nx_tile','ny_tile','dz','ztop','dxm','dx','dym','dy',
]

def csv_get_nth_row(csv_file, n):
    """
    Retrieves the n-th row from a CSV file and extracts specific configuration settings.

    Args:
    - csv_file (str): The file path to the CSV file.
    - n (int): The row index to retrieve.

    Returns:
    - tuple: A tuple containing two dictionaries with NR and experimental configuration data.
    """
    with open(csv_file, newline='') as f:
        df = pandas.read_csv(csv_file, converters={'nr_date': str, 'nr_time': str}, encoding='utf8')
        row = df.iloc[n].to_dict()

        expt_config = copy_by_keys(row, keys=CONFIG_KEYS)
        nr_config = copy_by_keys(row, keys=NR_KEYS)
        return nr_config, expt_config


def csv_get_number_rows(csv_file):
    """
    Get the number of rows in a CSV file.

    This function reads the CSV file into a pandas DataFrame and returns the total number of rows.
    It assumes that the 'nr_date' and 'nr_time' columns (if present) should be read as strings to avoid any automatic
    type conversion. The function uses UTF-8 encoding for reading the file.

    Args:
        csv_file (str): The file path to the CSV file.

    Returns:
        int: The number of rows in the CSV file.
    """
    with open(csv_file, newline='') as f:
        df = pandas.read_csv(csv_file, converters={'nr_date': str, 'nr_time': str}, encoding='utf8')
        return df.shape[0]

# This casts types - applies a specific type to a data value.
# Used to ensure we are writing float32 not double precision
def ensure_type(val: Any, dtype: Any):
    if type(val) != dtype:
        return dtype(val)
    else:
        return val


def zero_dict(variables, dimensions):
    data = {}
    for key in variables:
        data[key] = np.zeros(dimensions, dtype=np.float32)
    return data
