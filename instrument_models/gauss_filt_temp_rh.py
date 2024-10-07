"""
This python code serves as the "instrument model" for the simple PBL example

It simply applies Gaussian filtering to the "forward model" output data (RH and T in this case)
according to the settings in the instrument model dictionary

Inputs:
* inst_dict - contains settings for the instrument of interest
* rtm_data - contains temperature and rh to be blurred to the instrument specs

Output:
* inst_data - dictionary containing the gaussian-blurred T and RH fields

Derek Posselt
JPL
12 Sept 2023

Change log:
25 Sep 2023, DSM: Replace verbose with logger.
15 Jan 2024, DJP: Account for non-uniform x-direction grid spacing on a lat-lon grid
08 Jul 2024, DSM: Update dicts and xr.dataset assignments.
"""

import logging
import os
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from utils.data_handling import copy_by_keys, default_netcf_encoding_dict, get_coords
from utils.grid import DIM

# Set logging
logger = logging.getLogger(__name__)


def gauss_filt_temp_rh(rtm_data, nr_config, expt_config, Verbose=False):

  # Set the zlib compression level. 0 = no compression, 9 = maximum (possibly lossy, and very slow)
  # A setting of 5 has been shown to strike a reasonable balance between speed and level of compression
  complevel = 5

  # Create output netcdf file name
  inst_output_file = os.path.join(expt_config['output_path'],
    expt_config['file_prefix']+'_Instrument_'+expt_config['sat_type']+'.nc'
  )
  # Create the output instrument data dictionary
  inst_data = {}

  # Get the variable names from the nr dictionary
  rh_var   = expt_config['rh_var']
  h2o_var  = expt_config['h2o_var']
  temp_var = expt_config['temp_var']
  cloud_var = expt_config['cloud_var']
  pcp_var = expt_config['pcp_var']
  lat_var = expt_config['lat_var']
  lon_var = expt_config['lon_var']

  # Insert the variable names into the dictionary
  copy_by_keys(
    expt_config, inst_data, [
      'rh_var', 'h2o_var', 'temp_var', 'cloud_var', 'pcp_var', 'lat_var', 'lon_var',
  ])

  copy_by_keys(rtm_data, inst_data, ['nx', 'ny', 'nz', lat_var, lon_var])

  # First, extract the resolutions and grid spacings - make sure we modify the grid spacing by the input coarsening factor
  x_res = expt_config['x_res']
  y_res = expt_config['y_res']
  z_res = expt_config['z_res']
  dxm = nr_config['dxm'] * expt_config['icoarse'] # Note that this will be a vector if the data is on a lat-lon grid
  dym = nr_config['dym'] * expt_config['jcoarse']
  dz = rtm_data['dz']

  # The instrument model simply applies gaussian filtering to the NR input
  # Obtain the coarsening factors for the output grid, and put them into the settings dictionary - note that these are now consistent with the coarsened grid spacings!
  expt_config['iout'] = np.int32(np.rint(np.float32(x_res)/dxm)) # this will be a vector if the input data is on a lat/lon grid
  expt_config['jout'] = np.int32(np.rint(np.float32(y_res)/dym))
  expt_config['kout'] = np.int32(np.rint(np.float32(z_res)/dz))

  # Make sure coarsening factors are > 0
  logger.info(
    f'Number of dimensions in expt_config["iout"]: {np.ndim(expt_config["iout"])}'
  )
  if np.ndim(expt_config['iout']) > 0: # If this is an array
    expt_config['iout'][np.where(expt_config['iout'] < 1)] = 1
  else:
    expt_config['iout'] = max(1, expt_config['iout'])
  expt_config['jout'] = max(1, expt_config['jout'])
  expt_config['kout'] = max(1, expt_config['kout'])

  logger.info(
    f'Satellite, x_res, y_res, z_res: {expt_config["sat_type"]} {x_res} {y_res} {z_res}'
  )
  logger.info(
    f'Output (retrieved) coarsening factors in x, y, z:  {expt_config["iout"]} \
      {expt_config["jout"]} {expt_config["kout"]}'
  )

  # If data is not being read from file, then
  if not expt_config['read_INST']:
    # blur the vapor and temperature fields according to the user specifications
    # First, compute the sigma of a Gaussian in y and z from full-width at half max for horizontal and vertical
    # Y-direction
    sigma_y = y_res / (2.0* (2.0 * np.log(2.0))**.5)  # meters
    logger.info(f'Y Sigma (meters):      {sigma_y}')

    sigma_y = sigma_y / dym
    logger.info(f'Y Sigma (grid points): {sigma_y}')

    # Vertical
    sigma_z = z_res / (2.0* (2.0 * np.log(2.0))**.5)  # meters
    logger.info(f'Z Sigma (meters):      {sigma_z}')

    sigma_z = sigma_z / dz
    logger.info(f'Z Sigma (grid points): {sigma_z}')

    # Case in which the model grid is equidistant
    if nr_config['grid_type'] == 'equidistant':

      # X-direction
      sigma_x = x_res / (2.0* (2.0 * np.log(2.0))**.5)  # meters
      logger.info(f'X Sigma (meters):      {sigma_x}')

      sigma_x = sigma_x / dxm
      logger.info(f'X Sigma (grid points): {sigma_x}')

      # Now, call the averaging function for water vapor and temperature
      logger.info('Filtering RH')
      inst_data[rh_var] = gaussian_filter(rtm_data[rh_var],[sigma_z,sigma_y,sigma_x],truncate=4)

      logger.info('Filtering water vapor')
      inst_data[h2o_var] = gaussian_filter(rtm_data[h2o_var],[sigma_z,sigma_y,sigma_x],truncate=4)

      logger.info('Filtering temperature')
      inst_data[temp_var] = gaussian_filter(rtm_data[temp_var],[sigma_z,sigma_y,sigma_x],truncate=4)

      logger.info('Filtering cloud')
      inst_data[cloud_var] = gaussian_filter(rtm_data[cloud_var],[sigma_z,sigma_y,sigma_x],truncate=4)

      logger.info('Filtering precip')
      inst_data[pcp_var] = gaussian_filter(rtm_data[pcp_var],[sigma_y,sigma_x],truncate=4)

      # Try filtering the surface temperature individually.
      logger.info('Filtering surface temperature')
      inst_data['Tsfc'] = gaussian_filter(np.squeeze(rtm_data[temp_var][0,:,:]), [sigma_y,sigma_x], truncate=4)

    # If the data is on a lat-lon grid, then filter the dimensions individually
    elif nr_config['grid_type'] == 'latlon':

      # First, do the y and z directions
      logger.info('Filtering RH')
      inst_data[rh_var] = gaussian_filter(rtm_data[rh_var],[sigma_z,sigma_y,0],truncate=4)

      logger.info('Filtering water vapor')
      inst_data[h2o_var] = gaussian_filter(rtm_data[h2o_var],[sigma_z,sigma_y,0],truncate=4)

      logger.info('Filtering temperature')
      inst_data[temp_var] = gaussian_filter(rtm_data[temp_var],[sigma_z,sigma_y,0],truncate=4)

      logger.info('Filtering cloud')
      inst_data[cloud_var] = gaussian_filter(rtm_data[cloud_var],[sigma_z,sigma_y,0],truncate=4)

      logger.info('Filtering precip')
      inst_data[pcp_var] = gaussian_filter(rtm_data[pcp_var],[sigma_y,0],truncate=4)

      # Try filtering the surface temperature individually.
      logger.info('Filtering surface temperature')
      inst_data['Tsfc'] = gaussian_filter(np.squeeze(rtm_data[temp_var][0,:,:]), [sigma_y,0], truncate=4)

      # Create a 1d vector to hold x direction sigma
      sigma_x_1d = np.zeros(inst_data['ny'])

      # Loop over all y-direction points
      for j in range(inst_data['ny']):

        sigma_x = x_res / (2.0* (2.0 * np.log(2.0))**.5)  # meters
        logger.info(f'X Sigma (meters):      {sigma_x}')

        sigma_x = sigma_x / nr_config['dxm'][j]
        sigma_x_1d[j] = sigma_x
        logger.info(f'X Sigma (grid points): {sigma_x}')

        # Now, call the averaging function for water vapor, temperature, and precip sequentially in 1D
        # logger.info('Filtering RH')
        data1d = np.squeeze(inst_data[rh_var][:,j,:])
        inst_data[rh_var][:,j,:] = gaussian_filter1d(data1d, sigma_x, truncate=4)

        # logger.info('Filtering water vapor')
        data1d = np.squeeze(inst_data[h2o_var][:,j,:])
        inst_data[h2o_var][:,j,:] = gaussian_filter1d(data1d, sigma_x, truncate=4)

        # logger.info('Filtering temperature')
        data1d = np.squeeze(inst_data[temp_var][:,j,:])
        inst_data[temp_var][:,j,:] = gaussian_filter1d(data1d, sigma_x, truncate=4)

        # logger.info('Filtering cloud')
        data1d = np.squeeze(inst_data[cloud_var][:,j,:])
        inst_data[cloud_var][:,j,:] = gaussian_filter1d(data1d, sigma_x, truncate=4)

        # logger.info('Filtering precip')
        data1d = np.squeeze(inst_data[pcp_var][j,:])
        inst_data[pcp_var][j,:] = gaussian_filter1d(data1d, sigma_x, truncate=4)

        # Try filtering the surface temperature individually.
        # logger.info('Filtering surface temperature')
        data1d = np.squeeze(inst_data[temp_var][0,j,:])
        inst_data['Tsfc'][j,:] = gaussian_filter1d(data1d, sigma_x, truncate=4)


    # Fill output dictionary
    if nr_config['grid_type'] == 'latlon':
      inst_data['sigma_x'] = sigma_x_1d
    else:
      inst_data['sigma_x'] = sigma_x
    inst_data['sigma_y'] = sigma_y
    inst_data['sigma_z'] = sigma_z

    logger.info(f'Dimensions of filtered water vapor field {np.shape(inst_data[h2o_var])}')
    logger.info(f'Dimensions of filtered T3d field:        {np.shape(inst_data[temp_var])}')
    logger.info(f'Dimensions of filtered Tsfc field:       {np.shape(inst_data["Tsfc"])}')
    logger.info(f'Dimensions of filtered cloud field:      {np.shape(inst_data[cloud_var])}')
    logger.info(f'Dimensions of filtered pcp field:        {np.shape(inst_data[pcp_var])}')

    # sys.exit()

    # Write instrument data to file

    # Set dimensions
    zdim = expt_config['zvec']
    # If lat/lon, fill x and y dimensions with lat and lon
    if nr_config['grid_type'] == 'latlon':
      ydim = np.squeeze(rtm_data[rtm_data['lat_var']][:,0])
      xdim = np.squeeze(rtm_data[rtm_data['lon_var']][0,:])
    else:
      ydim = expt_config['yvec']
      xdim = expt_config['xvec']

    # Create the stub of the xarray dataset
    if nr_config['grid_type'] == 'latlon':
      ds = xr.Dataset(
            coords=get_coords(xdim, ydim, zdim),
            attrs={
              "Sigma_y": inst_data['sigma_y'],
              "Sigma_z": inst_data['sigma_z']
            }
      )
    else:
      ds = xr.Dataset(
            coords=get_coords(xdim, ydim, zdim),
            attrs={
              "Sigma_x": inst_data['sigma_x'],
              "Sigma_y": inst_data['sigma_y'],
              "Sigma_z": inst_data['sigma_z']
            }
      )

    # Add attribute that defines which grid type we are using
    # ds.assign_attrs(grid_type = nr_config['grid_type'])
    ds = ds.assign_attrs({"grid_type": nr_config['grid_type']})

    # Insert the x and y grid spacings
    if nr_config['grid_type'] == 'latlon':
      ds['dxm'] = (['ydim'], nr_config["dxm"])
    else:
      ds['dxm'] = (nr_config["dxm"])
    ds['dym'] = (nr_config["dym"])

    ds.update({
      inst_data['rh_var']: (DIM.ZYX, inst_data[inst_data['rh_var']]),
      inst_data['h2o_var']: (DIM.ZYX, inst_data[inst_data['h2o_var']]),
      inst_data['temp_var']: (DIM.ZYX, inst_data[inst_data['temp_var']]),
      inst_data['cloud_var']: (DIM.ZYX, inst_data[inst_data['cloud_var']]),
      inst_data['pcp_var']: (DIM.YX, inst_data[inst_data['pcp_var']]),
      inst_data['lat_var']: (DIM.YX, inst_data[inst_data['lat_var']]),
      inst_data['lon_var']: (DIM.YX, inst_data[inst_data['lon_var']]),
      'Tsfc': (DIM.YX, inst_data['Tsfc']),
    })

    # If we are on a lat-lon grid, put the x-sigma in as a vector that varies as a function of latitude (y)
    if nr_config['grid_type'] == 'latlon':
      ds['Sigma_x'] = (["ydim"], inst_data['sigma_x'])

    # Set encoding to include compression
    netcdf_names = [inst_data[x] for x in['rh_var', 'h2o_var', 'temp_var', 'cloud_var', 'pcp_var', 'lat_var', 'lon_var']]

    # Write to file
    ds.to_netcdf(inst_output_file, encoding=default_netcf_encoding_dict(netcdf_names, complevel))

    # Close dataset
    ds.close()

  # If user has specified read of data from file, do so here
  else:

    ds = xr.open_dataset(inst_output_file)
    logger.info(f'Opening file: {inst_output_file}')

    # If the data is on a lat/lon grid, sigma_x will be a vector
    if nr_config['grid_type'] == 'latlon':
      inst_data['sigma_x'] = (ds['Sigma_x']).data
    # If not, it will be an attribute
    else:
      inst_data['sigma_x'] = ds.attrs['Sigma_x']

    # Sigma y and z are always attributes
    inst_data['sigma_y'] = ds.attrs['Sigma_y']
    inst_data['sigma_z'] = ds.attrs['Sigma_z']

    # Read in surface temperature
    inst_data['Tsfc'] = (ds['Tsfc']).data

    # Read in the remaining 2d and 3d variables
    for var in ['lat_var', 'lon_var', 'pcp_var', 'temp_var', 'h2o_var', 'rh_var', 'cloud_var']:
      logger.info(f'Loading variable: {inst_data[var]}')
      inst_data[inst_data[var]] = (ds[inst_data[var]]).data

  # Return the instrument data dictionary and experiment configuration dictionary
  return inst_data
