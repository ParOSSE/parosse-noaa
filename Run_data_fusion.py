"""
This python 3x script runs data fusion for one or more user-defined combinations of observations

The instructions for running data fusion can be found in the data_fusion/ directory

Procedure:
1. Read in the configuration file for data fusion
2. Read in the data to be fused
3. Fuse the data
4. Plot the results

Derek Posselt
JPL
22 April 2024

"""
# System
from time import time
import logging
import sys
import yaml
import numpy as np

# Import modules
from data_fusion.fusion import Fusion
from instrument_models.create_instrument_config import create_instrument_config

# -------------------------------------------------------------------
# Main routine - drives the experiment
# -------------------------------------------------------------------
if __name__ == '__main__':

  # Start the timer
  t0 = time()

  # Get the fusion config file name from the command line. Defaults to "data_fusion.yml"
  try:
    data_fusion_config = sys.argv[1]
  except:
    data_fusion_config = 'data_fusion.yml'

  # Open the yaml file containing the fusion settings
  with open(data_fusion_config,'r') as f:
    params = yaml.safe_load(f)

  # print('x, y, and z plotting indices: ',params['x_dim_idx'], params['y_dim_idx'], params['z_dim_idx'])

  # for x_idx in params['x_dim_idx']:
  #   print(x_idx)
  #   if x_idx > 0:
  #     print('Found valid x idx')
  #   else:
  #     print('Found invalid x idx')

  # sys.exit()

  # Concatenate the data directory to the names of all the files to be fused in a list. 
  # NOTE: These files are assumed to share a common spatial grid (e.g., multiple instrument runs from the same WRF region, etc.)
  # Loop over every element in the list
  for i in range(len(params['input_files'])):
    params['input_files'][i] = params['data_dir']+params['input_files'][i]

  # Output file prefix - append plane and index
  params['output_files_prefix'] = params['output_files_prefix'] \
                                + params['variable'] \
                                + '_'+params['cross_section_plane'] \
                                + '_'+str(params['remaining_dim_index'])

  # Output file name
  params['output_file_name'] = params['output_path']+params['output_files_prefix']+'_fused_output.nc'

  # Validation file name - append to directory
  params['validation_file'] = [params['data_dir']+params['validation_file'][0]]

  print('Keys, values in the parameter dictionary: ')
  for k in params.keys():
    print(k,':   ',params[k])

  print()

  # Create multiple experiment config dictionaries - one per instrument
  expt_config_dicts = []
  for i in range(len(params['input_names'])):
    expt_config = {}
    expt_config['sat_type'] = params['input_names'][i]
    expt_config = create_instrument_config (expt_config)
    expt_config_dicts.append(expt_config)

  # Update timer
  t1 = time()
  # Print timing
  print(f'Time to read and process input:         {t1-t0}')

  # Create the fusion object
  print('Creating fuseObj ')
  fuseObj = Fusion( params )

  # Update timer
  t2 = time()
  # Print timing
  print(f'Time to create fusion object:           {t2-t1}')

  # Create the fusion matrix
  print('Creating fused_mat')
  fused_mat = fuseObj.fuse( params['fuse_method'] )

  # Update timer
  t3 = time()
  # Print timing
  print(f'Time to create fusion matrix:           {t3-t2}')

  # Plot the results
  print('Plotting results')
  # Loop over all elements of each of the x, y, and z cross-section indices
  # fuseObj.plot_data( save_fig = True )
  for x_idx in params['x_dim_idx']:
    if x_idx >= 0:
      print('Plotting cross-section at x = ',x_idx)
      fuseObj.plot_crossSection_data( x = x_idx , dimension = (10 , 10 ), save_fig = True)
      fuseObj.plot_crossSection_validation_comparisons( x = x_idx, save_fig = True )
  for y_idx in params['y_dim_idx']:
    if y_idx >= 0:
      print('Plotting cross-section at y = ',y_idx)
      fuseObj.plot_crossSection_data( y = y_idx , dimension = (10 , 10 ), save_fig = True)
      fuseObj.plot_crossSection_validation_comparisons( y = y_idx, save_fig = True )
  for z_idx in params['z_dim_idx']:
    if z_idx >= 0:
      print('Plotting cross-section at z = ',z_idx)
      fuseObj.plot_crossSection_data( z = z_idx , dimension = (10 , 10 ), save_fig = True)
      fuseObj.plot_crossSection_validation_comparisons( z = z_idx, save_fig = True )

  # # Compute RMSE and yield, uses footprint information 
  # # Obtain the coarsening factors for the output grid, and put them into the settings dictionary - note that these are now consistent with the coarsened grid spacings!
  # expt_config['iout'] = np.int32(np.rint(np.float32(expt_config['x_res'])/dxm)) # this will be a vector if the input data is on a lat/lon grid
  # expt_config['jout'] = np.int32(np.rint(np.float32(expt_config['y_res'])/dym))
  # expt_config['kout'] = np.int32(np.rint(np.float32(expt_config['z_res'])/dz))

