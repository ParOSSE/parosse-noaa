"""
This python code conducts a simple UQ analysis for the temperature and water vapor retrieval

It ingests the experiment dictionary, as well as the dictionaries containing nature run data and retrieval data

It then computes the following diagnostics:
1. Global (total domain, all layers) RMSE for T and RH (or Qv)
2. Global yield fraction for all layers for T and RH (should be the same)
3. Profiles of RMSE for T and RH
4. Profiles of yield fraction for T and RH

Considerations:
* What is "truth"? Nature run coarsened to retrieval grid? Nature run smoothed to satellite footprint?
* Should we allow for footprint overlap?
* Both the nature run and retrieval need to be on the same vertical grid

Assumptions currently:
1. Input data and retrieved data are on the exact same spatial grid
2. Nature run data is on the native horizontal resolution, interpolated to the retrieval vertical grid, and noise free

Note that, if RMSE is computed with respect to the smoothed nature run grid, then:

This means that the RMSE returned will be identically the error standard deviation that is provided for the instrument.

This is because RMSE is 1/N * sqrt((x - x')^2)

And, the procedure used to map NR to observations was:
1. Smooth to instrument footprint - means that x (the reference "truth") is now on the retrieval grid
2. Add noise to the smoothed and remapped NR data - x' = x + Norm(0,sigma).

In this case, sigma is defined to be the mean of the deviations between truth and reference. Which is exactly the RMSE.

The question is whether RMSE should incorporate considerations of spatial resolution. In reality, it would.

Output:
* Diagnostics, written to file
* Plots of profiles of RMSE and yield overlaid with values of global RMSE and yield

Derek Posselt
JPL
13 October 2023

Change log:
15 Jan 2024, DJP: Modified this so that it accommodates non-uniform x-direction grid spacing (model on lat-lon grid)

"""

# Import modules
import logging
import numpy as np

# Set logging
logger = logging.getLogger(__name__)

# Define function
def rmse_and_yield (expt_config, rtm_data, retr_data):

  # Set status upon entry
  Status = False

  # Get the variable names from the expt_config dictionary
  rh_var   = expt_config['rh_var']
  h2o_var  = expt_config['h2o_var']
  temp_var = expt_config['temp_var']
  cloud_var = expt_config['cloud_var']
  pcp_var = expt_config['pcp_var']

  # Get coarsening factors for instrument horizontal and vertical grid
  iout = expt_config['iout'] # Will be a vector if the input data was on a lat-lon grid
  jout = expt_config['jout']
  kout = expt_config['kout']

  # Print to log the coarsening factors
  logger.info(
    f'In RMSE calc; coarsening factors in z, y, x:  {expt_config["kout"]} \
      {expt_config["jout"]} {expt_config["iout"]}'
  )

  # Print to log the array dimensions
  logger.info(
    f'In RMSE calc; nr_temp grid dimensions, kout, and jout: {np.shape(rtm_data[temp_var])} {expt_config["kout"]} \
      {expt_config["jout"]}'
  )

  # Extract the temperature and water vapor fields on the coarsened grid
  nr_temp = rtm_data[temp_var][::kout,::jout,::iout]
  nr_h2o  = rtm_data[h2o_var][::kout,::jout,::iout]
  nr_rh   = rtm_data[rh_var][::kout,::jout,::iout]
  retr_temp = retr_data[temp_var][::kout,::jout,::iout]
  retr_h2o  = retr_data[h2o_var][::kout,::jout,::iout]
  retr_rh   = retr_data[rh_var][::kout,::jout,::iout]

  # Get non-nan and non-inf indices in the retrieval data
  idx_temp = np.isfinite(retr_temp)
  idx_h2o  = np.isfinite(retr_h2o)
  idx_rh   = np.isfinite(retr_rh)

  # Diagnostic print
  logger.info(
    f'Number of finite values for temperature, water vapor, and RH: \
      {np.count_nonzero(idx_temp)} {np.count_nonzero(idx_h2o)} {np.count_nonzero(idx_rh)}'
  )
  logger.info(
    f'Shape of NR and retrieved temperature, h2o, and RH arrays: \
      {np.shape(nr_temp)} {np.shape(nr_h2o)} {np.shape(retr_temp)} {np.shape(retr_h2o)} {np.shape(retr_rh)}'
  )
  logger.info(
    f'Size of NR and retrieved temperature, h2o, and RH arrays: \
      {np.size(nr_temp)} {np.size(nr_h2o)} {np.size(retr_temp)} {np.size(retr_h2o)} {np.size(retr_rh)}'
  )

  # Compute global RMSE and yield
  rmse_temp = np.sqrt(np.square(np.subtract(nr_temp[idx_temp],retr_temp[idx_temp])).mean())
  rmse_h2o = np.sqrt(np.square(np.subtract(nr_h2o[idx_h2o],retr_h2o[idx_h2o])).mean())
  rmse_rh  = np.sqrt(np.square(np.subtract(nr_rh[idx_rh],retr_rh[idx_rh])).mean())
  yield_temp = np.count_nonzero(idx_temp)/np.size(retr_temp)
  yield_h2o  = np.count_nonzero(idx_h2o)/np.size(retr_h2o)
  yield_rh   = np.count_nonzero(idx_rh)/np.size(retr_rh)

  # Print to log
  logger.info(
    f'Global RMSE for temperature, water vapor, and RH:  {rmse_temp} {rmse_h2o} {rmse_rh}'
  )
  logger.info(
    f'Global yield for temperature, water vapor, and RH: \
      {yield_temp} \
      {yield_h2o} \
      {yield_rh}'
  )

  # Compute profiles of RMSE
  # Get the number of layers in the retrieval (assumed to be equal to the unfiltered NR data)
  nz = len(retr_temp[:,0,0])
  # Get the number of grid points in the retrieval field
  nxny = np.size(retr_temp[0,:,:])
  # Set up profiles to hold output
  rmse_temp_prof = np.zeros(nz)
  rmse_h2o_prof  = np.zeros(nz)
  rmse_rh_prof   = np.zeros(nz)
  yield_temp_prof = np.zeros(nz)
  yield_h2o_prof  = np.zeros(nz)
  yield_rh_prof  = np.zeros(nz)
  # Loop over layers
  for k in range(nz):
    # Find indices where temperature at this level is not NaN
    idx_temp2d = np.isfinite(retr_temp[k,:,:])
    # Compute temperature rmse
    rmse_temp_prof[k] = np.sqrt(np.square(np.subtract(nr_temp[k,idx_temp2d],retr_temp[k,idx_temp2d])).mean())
    # Find indices where water vapor at this level is not NaN
    idx_h2o2d = np.isfinite(retr_h2o[k,:,:])
    # Compute water vapor rmse
    rmse_h2o_prof[k] = np.sqrt(np.square(np.subtract(nr_h2o[k,idx_h2o2d],retr_h2o[k,idx_h2o2d])).mean())
    # Find indices where RH at this level is not NaN
    idx_rh2d = np.isfinite(retr_rh[k,:,:])
    # Compute water vapor rmse
    rmse_rh_prof[k] = np.sqrt(np.square(np.subtract(nr_rh[k,idx_rh2d],retr_rh[k,idx_rh2d])).mean())
    # Compute the yield at this layer
    yield_temp_prof[k] = np.count_nonzero(idx_temp2d)/np.float32(nxny)
    yield_h2o_prof[k]  = np.count_nonzero(idx_h2o2d)/np.float32(nxny)
    yield_rh_prof[k]   = np.count_nonzero(idx_rh2d)/np.float32(nxny)

  logger.info(f'Temperature RMSE Profile: {rmse_temp_prof}')
  logger.info(f'Water vapor RMSE Profile: {rmse_h2o_prof}')
  logger.info(f'RH RMSE Profile:          {rmse_rh_prof}')
  logger.info(f'Temperature Yield:   {yield_temp_prof}')
  logger.info(f'Water vapor Yield:   {yield_h2o_prof}')
  logger.info(f'RH Yield:            {yield_rh_prof}')

  # Insert the global and profile RMSE and yield into the retr_data dictionary
  retr_data['rmse_temp'] = rmse_temp
  retr_data['rmse_h2o']  = rmse_h2o
  retr_data['rmse_rh']   = rmse_rh
  retr_data['yield_temp'] = yield_temp
  retr_data['yield_h2o']  = yield_h2o
  retr_data['yield_rh']   = yield_rh
  retr_data['rmse_temp_prof'] = rmse_temp_prof
  retr_data['rmse_h2o_prof']  = rmse_h2o_prof
  retr_data['rmse_rh_prof']   = rmse_rh_prof
  retr_data['yield_temp_prof'] = yield_temp_prof
  retr_data['yield_h2o_prof']  = yield_h2o_prof
  retr_data['yield_rh_prof']   = yield_rh_prof

  return retr_data