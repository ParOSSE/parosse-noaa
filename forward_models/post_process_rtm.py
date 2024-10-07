"""
This python function reconstructs the full domain RTM data from a set of dictionaries after running parmap over
a set of nature run tiles

It does the following:
1. Copy the first dictionary to obtain scalars and 1d vectors in height
2. Populate dimensions and vectors
3. Set up empty arrays for output
4. Reconstruct domain
5. If the NR is on a lat/lon grid, compute the grid spacing in meters and km

Derek Posselt
JPL
22 Sept 2023

Change log:
25 Sep 2023, DSM: Replace verbose with logger.
15 Jan 2024, DJP: Now includes option to compute grid spacing from lat/lon
22 Apr 2024, DJP: Includes the option to read output from file
08 Jul 2023, DSM: Improve readability.
"""

import logging
import os
from copy import deepcopy
import numpy as np
import xarray as xr

from utils.data_handling import copy_by_keys, default_netcf_encoding_dict, get_coords
from utils.grid import DIM

# Set logging
logger = logging.getLogger(__name__)

def post_process_rtm(rtm_data_list, expt_config, nr_config, nr_dims):

  # Set the netcdf file name
  rtm_output_file = os.path.join(expt_config['output_path'],
    expt_config['file_prefix']+'_RTM_'+expt_config['sat_type']+'.nc'
                                )
  # ----------------------------------------------------------
  # IF THE RTM HAS JUST BEEN RUN, THEN RECONSTRUCT THE DOMAIN
  # ----------------------------------------------------------

  if not expt_config['read_RTM']:

    # Set the zlib compression level. 0 = no compression, 9 = maximum (possibly lossy, and very slow)
    # A setting of 5 has been shown to strike a reasonable balance between speed and level of compression
    complevel = 5

    # First, deep copy the first dictionary in the output list - this will populate all of the 1d vectors and scalars
    rtm_data = deepcopy(rtm_data_list[0])

    # Need to insert the full nx and ny into this dictionary, accounting for the coarsening factor applied to the model data upon input
    rtm_data['nx'] = np.int32(nr_dims['nx']/expt_config['icoarse'])
    rtm_data['ny'] = np.int32(nr_dims['ny']/expt_config['jcoarse'])

    # logger.info(f'Full grid dimensions (nx,ny): {rtm_data['nx']} {rtm_data['ny']}')

    # And, update the expt config dictionary with the temp and water vapor names
    # And, fill the vertical height vector from the rtm dictionary
    copy_by_keys(rtm_data, expt_config, ['rh_var', 'h2o_var', 'temp_var', 'cloud_var', 'zvec'])

    # Now, get a list of the 3d and 2d data to rebuild
    varlist3d = [rtm_data['rh_var'], rtm_data['h2o_var'], rtm_data['temp_var'], rtm_data['cloud_var'], 'hgt3d']
    varlist2d = [rtm_data['pcp_var'], rtm_data['lat_var'], rtm_data['lon_var']]

    # Set up empty arrays to hold the reconstructed 3d and 2d data
    for v3d in varlist3d:
      rtm_data[v3d] = np.empty((rtm_data['nz'],rtm_data['ny'],rtm_data['nx']))
    for v2d in varlist2d:
      rtm_data[v2d] = np.empty((rtm_data['ny'],rtm_data['nx']))

    # Next, loop over all dictionaries in the list, filling the 3d arrays
    for idx in range(len(rtm_data_list)):
      # Populate the indices
      ix = rtm_data_list[idx]['ix']
      jy = rtm_data_list[idx]['jy']
      # Populate the tile dimensions
      nx_tile = rtm_data_list[idx]['nx_tile']
      ny_tile = rtm_data_list[idx]['ny_tile']

      logger.info(f'Inserting tile with indices and dimensions: {ix} {jy} {nx_tile} {ny_tile}')
      # Reconstruct data
      for v3d in varlist3d:
        # logger.info(f'Var, dims: {v3d} {np.shape(rtm_data_list[idx][v3d])}')
        rtm_data[v3d][:,jy:jy+ny_tile,ix:ix+nx_tile] = rtm_data_list[idx][v3d]
      for v2d in varlist2d:
        # logger.info(f'Var, dims: {v2d} {np.shape(rtm_data_list[idx][v2d])}')
        rtm_data[v2d][jy:jy+ny_tile,ix:ix+nx_tile] = rtm_data_list[idx][v2d]

    # If the NR is on a lat/lon grid, compute grid spacings. In this case, the "xvec" will actually be a 2d array dimensioned (ny,nx)
    if nr_config['grid_type'] == 'latlon':
      # Set the Earth's radius in meters
      rad_earth = 6371008.8
      # Obtain angular distance in longitude (dlon) in radians, uncoarsened
      dlon = nr_config['dlon'] * np.pi / 180.0
      # Compute physical grid spacing in latitude - this will be on the native model grid (uncoarsened)
      nr_config['dym']  = rad_earth * nr_config['dlat'] * np.pi / 180.0
      nr_config['dy'] = nr_config['dym'] * 1.0e-3 # Convert to km

      # Compute the x-dimension grid spacing from the latitude vector
      # First, get the vector of latitudes in radians - this will include coarsening factor, if applied
      ylat = np.abs(rtm_data[rtm_data['lat_var']][:,0]) * np.pi / 180.0 # Latitude in radians

      # logger.info(f'N dim(ylat), len(ylat), ylat (deg), ylat (rad): {np.ndim(ylat)} {len(ylat)} {np.abs(rtm_data[rtm_data["lat_var"]][:,0])} {ylat}')

      # Compute grid spacing in the x direction at each latitude - arc length
      # radius of Earth at this lat * angular distance, a vector on the coarsened y-grid, but filled with uncoarsened x grid spacing
      nr_config['dxm'] = rad_earth * np.cos(ylat) * dlon # This is a 1-d vector dimensioned ny
      nr_config['dx']  = nr_config['dxm'] * 1.0e-3

      # Create 2d array of x-vectors - accounts for coarsening
      expt_config['xvec'] = np.zeros((rtm_data['ny'],rtm_data['nx']))
      for j in range(rtm_data['ny']):
        expt_config['xvec'][j,:] = np.arange(nr_dims['nx']/nr_config['icoarse']) * nr_config['dx'][j] * nr_config['icoarse']

    # Otherwise, if the input array is on an equidistant grid, x spacing is constant
    else:
      expt_config['xvec'] = np.arange(nr_dims['nx']/nr_config['icoarse']) * nr_config['dx'] * nr_config['icoarse']

    # logger.info(f'N dim(dxm), dxm: {np.ndim(nr_config["dxm"])} {nr_config["dxm"]}')

    # Get y axis vector (in km) and insert into expt_config dictionary
    # make sure these are consistent with coarsening
    expt_config['yvec'] = np.arange(nr_dims['ny']/nr_config['jcoarse']) * nr_config['dy'] * nr_config['jcoarse']

    # Compute cloud top height - uses xarray for masking, and uses a dummy x-dimension in case we have a 2D xvec
    x_scr1d = np.arange(rtm_data['nx'])
    cloud_ds = xr.DataArray(rtm_data[rtm_data['cloud_var']],dims=['z','y','x'], coords=dict(z=expt_config['zvec'],y=expt_config['yvec'],x=x_scr1d))
    # Cloud top height is the max height where we find cloud condensate above the user-defined threshold
    cth = cloud_ds.z.where(cloud_ds > expt_config['cloud_thresh']).max(dim='z')
    # Place cth into the output dataset
    rtm_data['cth'] = np.array(cth)

    # Write the RTM data to file

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
    ds = xr.Dataset(
          coords=get_coords(xdim, ydim, zdim),
          attrs={
            "Cloud_threshold": expt_config['cloud_thresh']
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

    ds.update({
      'dym': (nr_config["dym"]),
      'dz': rtm_data['dz'],
      'hgt3d': (DIM.ZYX, rtm_data['hgt3d']),
      rtm_data['rh_var']: (DIM.ZYX, rtm_data[rtm_data['rh_var']]),
      rtm_data['h2o_var']: (DIM.ZYX, rtm_data[rtm_data['h2o_var']]),
      rtm_data['temp_var']: (DIM.ZYX, rtm_data[rtm_data['temp_var']]),
      rtm_data['cloud_var']: (DIM.ZYX, rtm_data[rtm_data['cloud_var']]),
      'Cloud_Top_Height': (DIM.YX, rtm_data['cth']),
      rtm_data['pcp_var']: (DIM.YX, rtm_data[rtm_data['pcp_var']]),
      rtm_data['lat_var']: (DIM.YX, rtm_data[rtm_data['lat_var']]),
      rtm_data['lon_var']: (DIM.YX, rtm_data[rtm_data['lon_var']]),
    })

    if nr_config['grid_type'] == 'latlon':
      ds['xvec'] = (DIM.YX, expt_config['xvec'])
    else:
      ds['xvec'] = (DIM.X, expt_config['xvec'])

    # Set encoding to include compression
    netcdf_names = [rtm_data[x] for x in['rh_var', 'h2o_var', 'temp_var', 'cloud_var', 'pcp_var', 'lat_var', 'lon_var']]
    netcdf_names += ['hgt3d', 'Cloud_Top_Height']

    # Write to file
    ds.to_netcdf(rtm_output_file, encoding=default_netcf_encoding_dict(netcdf_names, complevel))

    # Close dataset
    ds.close()

  # ----------------------------------------------------------
  # IF THE RTM HAS BEEN PREVIOUSLY RUN, THEN READ FROM FILE
  # ----------------------------------------------------------
  else:

    # Create the empty dictionary
    rtm_data = {}

    # # Set the water vapor variable equal to RH instead of vapor mixing ratio
    # h2o_var = 'RH'

    # Fill variable names
    copy_by_keys(
      nr_config, rtm_data, [
        'temp_var', 'pres_var', 'h2o_var', 'rh_var', 'hgt_var', 'cloud_var', 'pcp_var', 'lat_var', 'lon_var'
    ])

    ds = xr.open_dataset(rtm_output_file)
    logger.info(f'Opened file: {rtm_output_file}')

    rtm_data.update({
      # Get the x and y dimensions and vectors from the dataset
      'nx': len((ds['nx']).data),
      'xvec': (ds['xvec']).data,
      'ny': len((ds['ny']).data),
      'yvec': (ds['ny']).data,
      # Fill the vertical height dimension and vector from the xarray dataset
      'nz': len((ds['nz']).data),
      'zvec': (ds['nz']).data,
      'dz': (ds['dz']).data,
      'cth': (ds['Cloud_Top_Height']).data,
      'hgt3d': (ds['hgt3d']).data,
      rtm_data['h2o_var']: (ds[rtm_data['h2o_var']]).data,
    })

    # Read in dx and dy in meters, into the nr_config dictionary
    nr_config['dxm'] = (ds['dxm']).data
    nr_config['dym'] = (ds['dym']).data

    # Update the variable names in the expt_config dictionary
    copy_by_keys(rtm_data, expt_config, ['rh_var','h2o_var','temp_var','cloud_var', 'xvec', 'yvec', 'zvec'])

    for var in ['lat_var', 'lon_var', 'temp_var', 'rh_var', 'cloud_var', 'pcp_var']:
      logger.info(f'Loading variable: {nr_config[var]}')
      rtm_data[nr_config[var]] = (ds[nr_config[var]]).data

  return rtm_data
