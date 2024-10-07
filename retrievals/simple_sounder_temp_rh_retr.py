"""
This python code conducts the simplest possible "retrieval" of temperature and water vapor
It simply applies Gaussian noise to the input temperature and water vapor fields from the instrument model

Inputs:
* inst_dict - dictionary containing instrument settings
* inst_data - dictionary containing geophysical variables on the native (interpolated in height) grid, smoothed to instrument footprints

Output:
retr_data - dictionary containing noisy temperature and water vapor fields on the native (interpolated in height) grid

Derek Posselt
JPL
12 Sept 2023

25 Sep 2023, DSM: Replace verbose with logger.

"""
import logging
import os
import numpy as np

import xarray as xr

from utils.data_handling import default_netcf_encoding_dict, get_coords
from utils.grid import DIM

# Set logging
logger = logging.getLogger(__name__)


def simple_sounder_temp_rh_retr(expt_config, nr_config, rtm_data, inst_data, Verbose=False):

  # Set the zlib compression level. 0 = no compression, 9 = maximum (possibly lossy, and very slow)
  # A setting of 5 has been shown to strike a reasonable balance between speed and level of compression
  complevel = 5

  # Set up dictionary for encoding (only used for compression for now)
  encode_dict = {}

  # Set the retrieval file name
  retr_output_file = os.path.join(expt_config['output_path'],
    expt_config['file_prefix']+'_Retrieval_'+expt_config['sat_type']+'.nc'
                              )

  # Obtain the temperature and water vapor variable names from the instrument data dictionary
  rh_var = inst_data['rh_var']
  h2o_var = inst_data['h2o_var']
  temp_var = inst_data['temp_var']
  cloud_var = inst_data['cloud_var']
  pcp_var = inst_data['pcp_var']
  lat_var = inst_data['lat_var']
  lon_var = inst_data['lon_var']
  hgt3d = rtm_data['hgt3d']

  # Extract the dimensions
  nx = inst_data['nx']; ny = inst_data['ny']; nz = inst_data['nz']

  # Create the output data dictionary
  retr_data = {}
  retr_data['rh_var'] = rh_var
  retr_data['h2o_var'] = h2o_var
  retr_data['temp_var'] = temp_var
  # retr_data['cloud_var'] = cloud_var
  # retr_data['pcp_var'] = pcp_var
  retr_data['lat_var'] = lat_var
  retr_data['lon_var'] = lon_var

  # Fill dimensions
  retr_data['nx'] = nx; retr_data['ny'] = ny; retr_data['nz'] = nz

  # Fill lat and lon
  retr_data[lat_var] = inst_data[lat_var]
  retr_data[lon_var] = inst_data[lon_var]

  # If we are not reading from an existing file, run the "retrieval"
  if not expt_config['read_RETR']:

    # Compute standard deviations, either as a constant, or as a function of the field
    # Temperature
    if expt_config['err_temp_pct']: # If a percentage
      stddev_temp = (expt_config['err_temp'] * 0.01) * inst_data[temp_var]
      stddev_temp2d = (expt_config['err_temp'] * 0.01) * inst_data['Tsfc']
      tsfc_err = np.random.normal(scale=stddev_temp2d)
      t3d_err = np.random.normal(scale=stddev_temp)
    else: # If static
      # create the array of constant stddev for writing into the output file
      stddev_temp = np.ones( (nz,ny,nx) ) * expt_config['err_temp']

      t3d_err = np.random.normal(scale=expt_config['err_temp'],size=(nz,ny,nx))
      tsfc_err = np.random.normal(scale=expt_config['err_temp'],size=(ny,nx))
    # Water vapor
    if expt_config['err_vapor_pct']: # If a percentage
      stddev_vapor = (expt_config['err_vapor'] * 0.01) * inst_data[h2o_var]
      vapor_err = np.random.normal(scale=stddev_vapor)
      stddev_rh = (expt_config['err_vapor'] * 0.01) * inst_data[rh_var]
      rh_err = np.random.normal(scale=stddev_rh)
    else: # If static
      stddev_vapor = np.ones( (nz,ny,nx) ) * expt_config['err_vapor']
      vapor_err = np.random.normal(scale=expt_config['err_vapor'],size=(nz,ny,nx))
      stddev_rh = np.ones( (nz,ny,nx) ) * expt_config['err_vapor']
      rh_err = np.random.normal(scale=expt_config['err_vapor'],size=(nz,ny,nx))

    logger.info(f'Shape of vapor error: {np.shape(vapor_err)}',)
    logger.info(f'Shape of T3d error:   {np.shape(t3d_err)}',)
    logger.info(f'Shape of Tsfc error:  {np.shape(tsfc_err)}',)
    logger.info(f'Min,max vapor error:  {np.min(vapor_err)} {np.max(vapor_err)}')
    logger.info(f'Min,max T3d error:    {np.min(t3d_err)} {np.max(t3d_err)}')
    logger.info(f'Min,max Tsfc error:   {np.min(tsfc_err)} {np.max(tsfc_err)}')

    # Apply noise to water vapor and temperature fields
    retr_data[rh_var]   = inst_data[rh_var]   + rh_err
    retr_data[h2o_var]  = inst_data[h2o_var]  + vapor_err
    retr_data[temp_var] = inst_data[temp_var] + t3d_err
    retr_data['Tsfc']   = inst_data['Tsfc']   + tsfc_err

    # Finally mask cloud or precipitation, if called for - use a dummy x dimension
    x_scr1d = np.arange(inst_data['nx'])
    # First, create an xarray dataset with the filtered (to instrument resolution) cloud condensate
    cloud_ds = xr.DataArray(inst_data[cloud_var],dims=['z','y','x'], coords=dict(z=expt_config['zvec'],y=expt_config['yvec'],x=x_scr1d))
    # Cloud top height is the max height where we find cloud condensate above the user-defined threshold
    cth = cloud_ds.z.where(cloud_ds > expt_config['cloud_thresh']).max(dim='z')
    # Place cth into the output dataset
    retr_data['cth'] = np.array(cth)

    # If called for, mask cloud using cloud top height
    if expt_config['mask_cloud']:
      logger.info(f'Masking cloud > {expt_config["cloud_thresh"]}')
      # Now, mask out RH and T below cloud top height
      # First, create a masked array where heights are < cth
      cf_hgt3d = np.ma.masked_less(hgt3d,cth)
      cf_mask = np.ma.getmask(cf_hgt3d)
      # Mask the water vapor and temperature
      retr_data[rh_var][np.where(cf_mask)] = np.nan
      retr_data[h2o_var][np.where(cf_mask)] = np.nan
      retr_data[temp_var][np.where(cf_mask)] = np.nan

    # Now, create a 3d mask from 2d precip - true where values are > threshold
    if expt_config['mask_precip']:
      pcp_mask_3d = np.broadcast_to(inst_data[pcp_var]>expt_config['precip_thresh'], retr_data[h2o_var].shape)
      # Now, mask out RH and T where precip > thresh cloud top height
      retr_data[temp_var][pcp_mask_3d] = np.nan
      retr_data[h2o_var][pcp_mask_3d] = np.nan
      retr_data[rh_var][pcp_mask_3d] = np.nan

    # Write retrieval data to file
    # Set dimensions
    zdim = expt_config['zvec']
    # If lat/lon, fill x and y dimensions with lat and lon
    if nr_config['grid_type'] == 'latlon':
      ydim = np.squeeze(retr_data[retr_data['lat_var']][:,0])
      xdim = np.squeeze(retr_data[retr_data['lon_var']][0,:])
    else:
      ydim = expt_config['yvec']
      xdim = expt_config['xvec']

    # Create the stub of the xarray dataset
    ds = xr.Dataset(coords=get_coords(xdim, ydim, zdim))

    # Add attribute that defines which grid type we are using
    ds = ds.assign_attrs({"grid_type": nr_config['grid_type']})

    # Insert the x and y grid spacings
    if nr_config['grid_type'] == 'latlon':
      ds['dxm'] = (['ydim'], nr_config["dxm"])
    else:
      ds['dxm'] = (nr_config["dxm"])
    ds['dym'] = (nr_config["dym"])

    # Populate variables in the dataset

    ds[retr_data['h2o_var']] = (DIM.ZYX, retr_data[retr_data['h2o_var']])
    ds[retr_data['rh_var']] = (DIM.ZYX, retr_data[retr_data['rh_var']])
    ds[retr_data['temp_var']] = (DIM.ZYX, retr_data[retr_data['temp_var']])
    ds['vapor_err'] = (DIM.ZYX, stddev_vapor)
    ds['rh_err'] = (DIM.ZYX, stddev_rh)
    ds['temp_err']  = (DIM.ZYX, stddev_temp)
    ds['cth'] = (DIM.YX, retr_data['cth'])
    ds[retr_data['lat_var']] = (DIM.YX, retr_data[retr_data['lat_var']])
    ds[retr_data['lon_var']] = (DIM.YX, retr_data[retr_data['lon_var']])

    # write out the error config parameters into the retrieval file

    ds[ 'err_rh' ] = expt_config['err_vapor']
    ds[ 'err_vapor' ] = expt_config['err_vapor']
    ds[ 'err_temp'] = expt_config['err_temp']
    ds[ 'err_rh_pct' ] = expt_config['err_vapor_pct']
    ds[ 'err_vapor_pct' ] = expt_config['err_vapor_pct']
    ds[ 'err_temp_pct' ] = expt_config['err_temp_pct']

    # Set encoding to include compression
    netcdf_names = [rtm_data[x] for x in['rh_var', 'h2o_var', 'temp_var', 'lat_var', 'lon_var']]
    netcdf_names += ['cth', 'vapor_err', 'temp_err']

    # Write to file
    ds.to_netcdf(retr_output_file, encoding=default_netcf_encoding_dict(netcdf_names, complevel))

    # Close dataset
    ds.close()

  # Otherwise, we are reading in retrieval output from file
  else:
    # Open the file
    ds = xr.open_dataset(retr_output_file)
    logger.info(f'Opening file: {retr_output_file}')

    # Read in the cloud top height
    retr_data['cth']       = (ds['cth']).data

    # Read in the errors
    retr_data['temp_err']  = (ds['temp_err']).data
    retr_data['vapor_err'] = (ds['vapor_err']).data

    # Read in the remaining 2d and 3d variables
    for var in ['temp_var', 'h2o_var', 'rh_var']:
      logger.info(f'Loading variable: {retr_data[var]}')
      retr_data[inst_data[var]] = (ds[retr_data[var]]).data

  # Return dictionary
  return retr_data

