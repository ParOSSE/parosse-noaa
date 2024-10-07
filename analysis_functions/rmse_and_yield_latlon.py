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
2. Input data has already been smoothed to the instrument footprint (but without noise)

Output:
* Diagnostics, written to file
* Plots of profiles of RMSE and yield overlaid with values of global RMSE and yield

Derek Posselt
JPL
13 October 2023

Change log:
15 Jan 2024, DJP: Modified this so that it accommodates non-uniform x-direction grid spacing (model on lat-lon grid)
01 Jan 2024, DSMG: Add rmse_rh and refactor code.

"""

# Import modules
import logging
import numpy as np
from typing import Dict, Any

# Set logging
logger = logging.getLogger(__name__)


def rmse_and_yield_latlon(expt_config: Dict[str, Any], rtm_data: Dict[str, np.ndarray], retr_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate RMSE and yield for temperature, water vapor, and relative humidity fields on a lat-lon grid.

    Args:
    - expt_config: Dictionary containing experiment configuration.
    - rtm_data: Dictionary containing the reference data.
    - retr_data: Dictionary containing the retrieval data.

    Returns:
    - Updated retr_data dictionary with RMSE and yield values.
    """
    # Get the variable names from the expt_config dictionary
    h2o_var = expt_config.get('h2o_var')
    temp_var = expt_config.get('temp_var')
    rh_var = expt_config.get('rh_var')
    cloud_var = expt_config.get('cloud_var')
    pcp_var = expt_config.get('pcp_var')

    # Get coarsening factors for instrument horizontal and vertical grid
    iout = expt_config.get('iout')
    jout = expt_config.get('jout')
    kout = expt_config.get('kout')

    # Print to log the coarsening factors
    logger.info(f'In RMSE calc; coarsening factors in z, y, x: {kout} {jout} {iout}')

    # Print to log the array dimensions
    logger.info(f'In RMSE calc; nr_temp grid dimensions, kout, and jout: {np.shape(rtm_data[temp_var])} {kout} {jout}')

    # Extract the coarsened fields - leave the x-direction un-coarsened
    nr_temp = rtm_data[temp_var][::kout, ::jout, :]
    nr_h2o = rtm_data[h2o_var][::kout, ::jout, :]
    nr_rh = rtm_data[rh_var][::kout, ::jout, :]
    retr_temp = retr_data[temp_var][::kout, ::jout, :]
    retr_h2o = retr_data[h2o_var][::kout, ::jout, :]
    retr_rh = retr_data[rh_var][::kout, ::jout, :]

    # Diagnostic print
    logger.info(f'Shape of NR and retrieved temperature, h2o, and RH arrays: {np.shape(nr_temp)} {np.shape(nr_h2o)} {np.shape(retr_temp)} {np.shape(retr_h2o)} {np.shape(retr_rh)}')
    logger.info(f'Size of NR and retrieved temperature, h2o, and RH arrays: {np.size(nr_temp)} {np.size(nr_h2o)} {np.size(retr_temp)} {np.size(retr_h2o)} {np.size(retr_rh)}')

    # Loop over the y-direction, computing RMSE and number of valid points (for later calc of yield)
    valid_temp, total_temp = 0, 0
    valid_h2o, total_h2o = 0, 0
    valid_rh, total_rh = 0, 0
    rmse_temp, rmse_h2o, rmse_rh = 0.0, 0.0, 0.0

    for j in range(retr_temp.shape[1]):
        # Extract x-z slice for this y point, staggered in x according to satellite footprint
        nr_temp2d = nr_temp[:, j, ::iout[j]]
        nr_h2o2d = nr_h2o[:, j, ::iout[j]]
        nr_rh2d = nr_rh[:, j, ::iout[j]]
        retr_temp2d = retr_temp[:, j, ::iout[j]]
        retr_h2o2d = retr_h2o[:, j, ::iout[j]]
        retr_rh2d = retr_rh[:, j, ::iout[j]]

        # Get non-nan and non-inf indices in the retrieval data
        idx_temp = np.isfinite(retr_temp2d)
        idx_h2o = np.isfinite(retr_h2o2d)
        idx_rh = np.isfinite(retr_rh2d)

        # Get the number of non-zero elements - for computation of yield later
        total_temp += np.size(retr_temp2d)
        valid_temp += np.count_nonzero(idx_temp)
        total_h2o += np.size(retr_h2o2d)
        valid_h2o += np.count_nonzero(idx_h2o)
        total_rh += np.size(retr_rh2d)
        valid_rh += np.count_nonzero(idx_rh)

        # Diagnostic print
        # logger.info(f'Index, # finite values for temperature, h2o, and RH: {j} {np.count_nonzero(idx_temp)} {np.size(retr_temp2d)} {valid_temp} {total_temp} {np.count_nonzero(idx_h2o)} {np.size(retr_h2o2d)} {valid_h2o} {total_h2o} {np.count_nonzero(idx_rh)} {np.size(retr_rh2d)} {valid_rh} {total_rh}')

        # Compute running RMSE
        rmse_temp += np.sqrt(np.mean(np.square(nr_temp2d[idx_temp] - retr_temp2d[idx_temp])))
        rmse_h2o += np.sqrt(np.mean(np.square(nr_h2o2d[idx_h2o] - retr_h2o2d[idx_h2o])))
        rmse_rh += np.sqrt(np.mean(np.square(nr_rh2d[idx_rh] - retr_rh2d[idx_rh])))

    # Compute global RMSE and yield by dividing through by the number of y-points
    rmse_temp /= retr_temp.shape[1]
    rmse_h2o /= retr_temp.shape[1]
    rmse_rh /= retr_temp.shape[1]

    logger.info(f'Valid points, total points for temperature, h2o, and RH: {valid_temp} {total_temp} {valid_h2o} {total_h2o} {valid_rh} {total_rh}')

    # Compute yield = valid / total
    yield_temp = valid_temp / total_temp
    yield_h2o = valid_h2o / total_h2o
    yield_rh = valid_rh / total_rh

    logger.info(f'Global RMSE for temperature, water vapor, and RH: {rmse_temp} {rmse_h2o} {rmse_rh}')
    logger.info(f'Global yield for temperature, water vapor, and RH: {yield_temp} {yield_h2o} {yield_rh}')

    # Compute profiles of RMSE and yield - note that these will be valid for the
    # ENTIRE grid, not the grid subset to footprints
    nz = retr_temp.shape[0]
    nxny = np.size(retr_temp[0, :, :])

    rmse_temp_prof = np.zeros(nz, dtype=np.float32)
    rmse_h2o_prof = np.zeros(nz, dtype=np.float32)
    rmse_rh_prof = np.zeros(nz, dtype=np.float32)
    yield_temp_prof = np.zeros(nz, dtype=np.float32)
    yield_h2o_prof = np.zeros(nz, dtype=np.float32)
    yield_rh_prof = np.zeros(nz, dtype=np.float32)

    # Loop over layers
    for k in range(nz):
        idx_temp2d = np.isfinite(retr_temp[k, :, :])
        idx_h2o2d = np.isfinite(retr_h2o[k, :, :])
        idx_rh2d = np.isfinite(retr_rh[k, :, :])

        rmse_temp_prof[k] = np.sqrt(np.mean(np.square(nr_temp[k, idx_temp2d] - retr_temp[k, idx_temp2d])))
        rmse_h2o_prof[k] = np.sqrt(np.mean(np.square(nr_h2o[k, idx_h2o2d] - retr_h2o[k, idx_h2o2d])))
        rmse_rh_prof[k] = np.sqrt(np.mean(np.square(nr_rh[k, idx_rh2d] - retr_rh[k, idx_rh2d])))

        yield_temp_prof[k] = np.count_nonzero(idx_temp2d) / nxny
        yield_h2o_prof[k] = np.count_nonzero(idx_h2o2d) / nxny
        yield_rh_prof[k] = np.count_nonzero(idx_rh2d) / nxny

    # Print to log
    logger.info(f'Temperature RMSE Profile: {rmse_temp_prof}')
    logger.info(f'Water vapor RMSE Profile: {rmse_h2o_prof}')
    logger.info(f'RH RMSE Profile: {rmse_rh_prof}')
    logger.info(f'Temperature Yield: {yield_temp_prof}')
    logger.info(f'Water vapor Yield: {yield_h2o_prof}')
    logger.info(f'RH Yield: {yield_rh_prof}')

    # Insert the global and profile RMSE and yield into the retr_data dictionary
    retr_data.update({
        'rmse_temp': rmse_temp,
        'rmse_h2o': rmse_h2o,
        'rmse_rh': rmse_rh,
        'yield_temp': yield_temp,
        'yield_h2o': yield_h2o,
        'yield_rh': yield_rh,
        'rmse_temp_prof': rmse_temp_prof,
        'rmse_h2o_prof': rmse_h2o_prof,
        'rmse_rh_prof': rmse_rh_prof,
        'yield_temp_prof': yield_temp_prof,
        'yield_h2o_prof': yield_h2o_prof,
        'yield_rh_prof': yield_rh_prof
    })

    return retr_data