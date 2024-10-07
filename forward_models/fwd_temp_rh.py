"""
This python code serves as the only needed "forward model" for the simple case in which
we are running a "retrieval" on nature run temperature and water vapor fields

It ingests the NR dictionary and the dictionary of settings and:
1. Computes RH
2. Resets the water vapor variable name to RH from Qv
3. Interpolates in height to a constant grid for the user-specified dz value

It then returns a dictionary with only the variables needed for the instrument model and retrieval
In this case, this is the temperature, water vapor, and cloud on the same horizontal grid as is input, and interpolated to height

Derek Posselt
JPL
12 September 2023

Change log:
25 Sep 2023, DSM: Replace verbose with logger.
22 Apr 2024, DJP: Added option to read output of RTM from file.
"""
from copy import deepcopy
import logging

import numpy as np
import pandas as pd
from utils.column_interp import column_interp
from utils.data_handling import copy_by_keys

# Set logging
logger = logging.getLogger(__name__)
DEV_LOG = False

def fwd_temp_rh(nr_data, nr_config, expt_config, dz=None, ztop=None, Verbose=False):
  # If the vertical height spacing is not provided, set it to 100 meters
  if pd.isnull(dz):
    dz = 100.0 # Height spacing in meters

  # If the domain top height is not provided, set it to 20000 meters
  if pd.isnull(ztop):
    ztop = 20000.0 # top height in meters

  # For simplicity, extract the temperature, pressure, water vapor, and height variable names
  temp_var = nr_config['temp_var']
  pres_var = nr_config['pres_var']
  h2o_var = nr_config['h2o_var']
  hgt_var = nr_config['hgt_var']
  cloud_var = nr_config['cloud_var']
  pcp_var = nr_config['pcp_var']
  lat_var = nr_config['lat_var']
  lon_var = nr_config['lon_var']

  # Now, set up the output RTM_data dictionary
  rtm_data = {}

  # Compute relative humidity
  # First, saturation vapor pressure (hPa, temperature in K)
  esat = np.exp( 53.67957 - (6743.769/(273.15+nr_data[temp_var])) - 4.8451 * np.log((273.15+nr_data[temp_var])))

  # Now, saturation mixing ratio (vapor and air pressure in hPa), mixing ratio in g/kg
  qv_sat = 0.622 * esat / nr_data[pres_var] * 1.e3

  # Finally, fill the RTM dictionary with RH as Qv / Qv_sat
  nr_data['RH'] = 100.0 * nr_data[h2o_var] / qv_sat

  # Filter relative humidity - should not be identically zero, and should never be negative.

  nr_data['RH'][np.where(nr_data['RH'] < 1.e-5)] = 1.e-5

  # Set the RH variable name
  rh_var = 'RH'
  nr_config['rh_var'] = rh_var
  expt_config['rh_var'] = rh_var

  # Nature Run Subsystem, Process Data for Forward Model Input

  # Extract necessary variables from the NR dictionary
  relative_humidity = np.squeeze(np.array(nr_data[rh_var]))
  water_vapor = np.squeeze(np.array(nr_data[h2o_var]))
  temperature = np.array(nr_data[temp_var])
  hgt = np.array(nr_data[hgt_var])
  cloud = np.array(nr_data[cloud_var])

  nx = np.int32(nr_data['nx'])
  ny = np.int32(nr_data['ny'])
  nz = np.int32(nr_data['nz'])

  min_dz = np.min(np.diff(hgt,axis=0))
  max_dz = np.max(np.diff(hgt,axis=0))

  if DEV_LOG:
    logger.info(f'Min/max T: {np.min(nr_data[temp_var]+273.15)} {np.max(nr_data[temp_var]+273.15)}')
    logger.info(f'Min/max P: {np.min(nr_data[pres_var])} {np.max(nr_data[pres_var])}')
    logger.info(f'Min/max esat: {np.min(esat)} {np.max(esat)}')
    logger.info(f'Min/max qv_sat: {np.min(qv_sat)} {np.max(qv_sat)}')
    logger.info(f'{h2o_var} \n {nr_data.keys()}')
    logger.info(f'Min/max RH:  {np.min(nr_data["RH"])} {np.max(nr_data["RH"])}')
    logger.info(f'Shape of water vapor array: {np.shape(water_vapor)}')
    logger.info(f'Shape of temperature array: {np.shape(temperature)}')
    logger.info(f'Shape of height array:      {np.shape(hgt)}')
    logger.info(f'Shape of lat array:         {np.shape(nr_data[lat_var])}')
    logger.info(f'Shape of lon array:         {np.shape(nr_data[lon_var])}')
    logger.info(f'Min diff in input height array: {min_dz}')
    logger.info(f'Max diff in input height array: {max_dz}')

  # Now, interpolate to a regular grid, if necessary
  do_interp = False

  # If the input data is not on a regular vertical grid, or if it is not already on a 100 meter dz grid, then regrid
  if min_dz != dz or max_dz != dz:
    do_interp = True

  if do_interp:
    # Fill a vector with a uniform distribution of heights for given dz
    zvec = np.arange(np.int32(ztop / dz + 1)) * dz
    # Get the number of layers to interpolate to
    nz1 = len(zvec)

    if DEV_LOG:
      logger.info(f'Interpolating in height to constant grid of thickness {dz}')
      logger.info(f'Interpolating to number of layers: {nz1}')
      logger.info(f'New high res height vector: \n {zvec}')

    # Create 3D arrays to hold interpolated output
    rh3d          = np.zeros((nz1,ny,nx))
    water_vapor3d = np.zeros((nz1,ny,nx))
    temperature3d = np.zeros((nz1,ny,nx))
    cloud3d       = np.zeros((nz1,ny,nx))
    hgt3d         = np.zeros((nz1,ny,nx))

    # Loop over all x and y points, doing interp
    for i in range(nx):
      # logger.info(f'Processing column {i}')
      for j in range(ny):
        # Fill the 1d height for this column
        hgt1d = np.squeeze(hgt[:,j,i])
        # print('Length of height vector: ',len(hgt1d))
        # Make a copy of the zvec for this column
        zvec_scr = deepcopy(zvec)
        # Make sure we are only interpolating to layers above the lowest layer in the input data
        idx_z = 0 # Default to interpolating all layers
        if min(zvec) < min(hgt1d): # If the lowest layer in the interp height array is < lowest layer in data
            idx_z = np.where(zvec > np.min(hgt1d))[0][0] # Find the location where the high res array is > lowest layer in data
            zvec_scr = zvec_scr[idx_z:] # Subset the array

        # Interpolate RH
        run_list = (i, j, hgt1d, zvec_scr, relative_humidity[:,j,i])
        data_dict = column_interp(run_list)
        rh3d[idx_z::,j,i] = data_dict['data_out']
        # Fill water vapor up to and including idx_z with lowest layer values
        if idx_z > 0:
          rh3d[0:idx_z,j,i] = relative_humidity[0,j,i]
        else:
          rh3d[0,j,i] = relative_humidity[0,j,i]

        # Interpolate water vapor
        run_list = (i, j, hgt1d, zvec_scr, water_vapor[:,j,i])
        data_dict = column_interp(run_list)
        water_vapor3d[idx_z::,j,i] = data_dict['data_out']
        # Fill water vapor up to and including idx_z with lowest layer values
        if idx_z > 0:
          water_vapor3d[0:idx_z,j,i] = water_vapor[0,j,i]
        else:
          water_vapor3d[0,j,i] = water_vapor[0,j,i]

        # Interpolate temperature
        run_list = (i, j, hgt1d, zvec_scr, temperature[:,j,i])
        data_dict = column_interp(run_list)
        temperature3d[idx_z::,j,i] = data_dict['data_out']
        # Fill temperature up to idx_z with lowest layer values
        if idx_z > 0:
          temperature3d[0:idx_z,j,i] = temperature[0,j,i]
        else:
          temperature3d[0,j,i] = temperature[0,j,i]

        # Interpolate cloud
        run_list = (i, j, hgt1d, zvec_scr, cloud[:,j,i])
        data_dict = column_interp(run_list)
        cloud3d[idx_z::,j,i] = data_dict['data_out']
        # Fill cloud up to idx_z with lowest layer values
        if idx_z > 0:
          cloud3d[0:idx_z,j,i] = cloud[0,j,i]
        else:
          cloud3d[0,j,i] = cloud[0,j,i]

        # Fill 3d height array
        hgt3d[:,j,i] = zvec

    rtm_data.update({
      rh_var: rh3d,
      h2o_var: water_vapor3d,
      temp_var: temperature3d,
      cloud_var: cloud3d,
      'hgt3d': hgt3d,
      'hgt1d': hgt1d,
      'nz': np.int32(nz1),
    })

  # If not doing interpolation, fill temperature, water vapor, and cloud arrays with inputs and height vector with first column
  else:
    rtm_data.update({
      rh_var: relative_humidity,
      h2o_var: water_vapor,
      temp_var: temperature,
      cloud_var: cloud,
      'hgt1d': np.squeeze(hgt[:,0,0]),
      'hgt3d': hgt,
      'nz': np.int32(nr_data['nz']),
    })
    zvec = rtm_data['hgt1d']

  # Insert precip, lat, and lon, since these are not interpolated in the vertical
  copy_by_keys(nr_data, rtm_data, [pcp_var, lat_var, lon_var])

  rtm_data.update({
    # Fill output dictionary with dimensions and distance vectors
    'nx': np.int32(nr_data['nx']),
    'ny': np.int32(nr_data['ny']),
    'dz': dz,
    # Insert z axis vector into the rtm data dictionary
    'zvec': zvec,
    # Insert the variable names into the dictionary
    'rh_var': rh_var,
    'h2o_var': h2o_var,
    'temp_var': temp_var,
    'cloud_var': cloud_var,
    'pcp_var': pcp_var,
    'lat_var': lat_var,
    'lon_var': lon_var,
    # Update the name of the water vapor variable in the experiment config dictionary
    # 'h2o_var': h2o_var

  })
  return rtm_data
