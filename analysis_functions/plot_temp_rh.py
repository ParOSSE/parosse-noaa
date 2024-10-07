"""
This python code plots the temperature and Rh fields from nature run, filtered (non-noisy) retrieval,
and noisy retrieval.

It uses information from most of the dictionaries populated during the PBL OSSE workflow

Inputs:
* nr_config
* expt_config
* rtm_data
* retr_data
* inst_data (optional)

Outputs: None

Derek Posselt
JPL
12 Sept 2023

Change log:
25 Sep 2023, DSM: Replace verbose with logger.
16 Jan 2024, DJP: Modified to accommodate lat/lon grids.
30 Apr 2024, DJP: Made input of instrument data (smoothed, non-noisy) optional

"""
# Import modules
import os
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Tell matplotlib to use truetype fonts (better for Illustrator)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Set the font to something Illustrator will understand...
mpl.rcParams['font.sans-serif'] = "Arial"
# Force to always use sans-serif fonts
mpl.rcParams['font.family'] = "sans-serif"

# Set the axis tick labels larger
mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)

def get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, name):
  return os.path.join(
    plot_path, f"{model_prefix}_{sat_type}_{name}_dx{str(x_res).strip()}_dz{str(z_res).strip()}.png"
  )

def plot_temp_rh(nr_config, expt_config, rtm_data, retr_data, inst_data=None, Verbose=False):

  # Set status message
  output_status = 'Plotting failed - check plot_temp_rh'

  # Extract data from dictionaries
  model_prefix = f"{nr_config['file_prefix']}_{nr_config['i1']}i1_{nr_config['i2']}i2_{nr_config['j1']}j1_{nr_config['j2']}j2"
  sat_type = expt_config['sat_type']
  x_res = expt_config['x_res']
  z_res = expt_config['z_res']
  xvec = expt_config['xvec'] # This will be a 2D array in the case of lat/lon grids
  yvec = expt_config['yvec']
  zvec = expt_config['zvec']

  logger.info(f"Length of xvec: {len(xvec)} yvec: {len(yvec)} zvec: {len(zvec)}")

  y_idx = np.int32(np.floor(nr_config['y_idx'] / nr_config['icoarse']))
  z_idx = nr_config['z_idx']
  if pd.isnull(nr_config['ztop']):
    ztop = 20000
  else:
    ztop = nr_config['ztop']

  if nr_config['y_idx'] < 0:
    y_idx = np.int32(np.floor(rtm_data['ny']/2/nr_config['icoarse']))

  # Set additional y indices
  y_incices = [
    y_idx,
    np.int32(np.floor(rtm_data['ny']/4)), # 1/4 of the way through the domain
    np.int32(np.floor(rtm_data['ny']/4))+np.int32(np.floor(rtm_data['ny']/2)) # 3/4 of the way through the domain
  ]

  logger.info(f'Index of y-cross-section: {y_idx}')

  iout = expt_config['iout'] # This will be a vector in the case of lat/lon grid
  jout = expt_config['jout']
  kout = expt_config['kout']
  plot_path = expt_config['plot_path']
  h2o_var = expt_config['h2o_var']
  rh_var = expt_config['rh_var']
  temp_var = expt_config['temp_var']

  # Obtain lat and lon arrays
  xlat = retr_data[retr_data['lat_var']]
  xlon = retr_data[retr_data['lon_var']]

  #----------------------------------------------
  # SET PLOT FILE NAMES
  #----------------------------------------------
  rh_rmse_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "RH_rmse")
  vap_rmse_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "Qv_rmse")
  temp_rmse_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "T_rmse")
  yield_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "yield")

  # Set RH plot file names
  rh_unfilt_file = os.path.join(plot_path, model_prefix+'_RH_unfiltered.png')
  rh_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "RH_filtered")
  rh_noisy_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "RH_filtered_noisy")

  # Set water vapor plot file names
  vap_unfilt_file = os.path.join(plot_path, model_prefix+'_Qv_unfiltered.png')
  vap_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "Qv_filtered")
  vap_noisy_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "Qv_filtered_noisy")

  # Set temperature plot file names
  t_unfilt_file = os.path.join(plot_path, model_prefix+'_Tsfc_unfiltered.png')
  t_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "Tsfc_filtered")
  t_noisy_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, "Tsfc_filtered_noisy")

  # Set cloud top height name
  cth_unfilt_file = os.path.join(plot_path, model_prefix+'_cth_unfiltered.png')
  cth_filt_file = os.path.join(plot_path, model_prefix+'_'+sat_type+'_cth_filtered.png')

  logger.info(f'Vapor filtered file name: {vap_filt_file}')
  logger.info(f'Temp filtered file name:  {t_filt_file}')

  #----------------------------------------------
  # PRODUCE RMSE AND YIELD PLOTS
  #----------------------------------------------

  # Plot the RMSE profiles for temperature and water vapor
  rmse_temp_prof = retr_data['rmse_temp_prof']
  rmse_temp = retr_data['rmse_temp']
  rmse_h2o_prof  = retr_data['rmse_h2o_prof']
  rmse_h2o  = retr_data['rmse_h2o']
  rmse_rh_prof  = retr_data['rmse_rh_prof']
  rmse_rh  = retr_data['rmse_rh']
  yield_temp_prof = retr_data['yield_temp_prof'] * 100.0 # Convert to percent
  yield_temp = retr_data['yield_temp'] * 100.0 # Convert to percent

  # Plot temperature RMSE profile
  rmse_fig = plt.figure(figsize=(3,6))
  plt.plot(rmse_temp_prof,zvec[::kout])
  plt.xlim(0,7.0)
  plt.ylim(0,ztop)
  plt.ylabel('Height (m)', fontsize=24)
  plt.xlabel('T RMSE (K)', fontsize=24)
  plt.title(f'T RMSE: {rmse_temp:.2f}', fontsize=24)
  plt.savefig(temp_rmse_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot RH RMSE profile
  rmse_fig = plt.figure(figsize=(3,6))
  plt.plot(rmse_rh_prof,zvec[::kout])
  plt.xlim(0,15.0)
  plt.ylim(0,ztop)
  plt.ylabel('Height (m)', fontsize=24)
  plt.xlabel('RH RMSE (%)', fontsize=24)
  plt.title(f'RH RMSE: {rmse_rh:.2f}', fontsize=24)
  plt.savefig(rh_rmse_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot vapor RMSE profile
  rmse_fig = plt.figure(figsize=(3,6))
  plt.plot(rmse_h2o_prof,zvec[::kout])
  plt.xlim(0,5.0)
  plt.ylim(0,ztop)
  plt.ylabel('Height (m)', fontsize=24)
  plt.xlabel('Qv RMSE (%)', fontsize=24)
  plt.title(f'Qv RMSE: {rmse_h2o:.2f}', fontsize=24)
  plt.savefig(vap_rmse_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot the yield profile - should be the same for both temperature and water vapor
  yield_fig = plt.figure(figsize=(3,6))
  plt.plot(yield_temp_prof,zvec[::kout])
  plt.xlim(0,105.0)
  plt.ylim(0,ztop)
  plt.ylabel('Height (m)', fontsize=24)
  plt.xlabel('Yield (%)', fontsize=24)
  plt.title(f'Retrieval Yield: {yield_temp:.2f}', fontsize=24)
  plt.savefig(yield_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  #----------------------------------------------
  # PRODUCE PLAN VIEW AND CROSS-SECTION PLOTS
  #----------------------------------------------

  # Plot the nature run water vapor cross-section
  Qvplot = rtm_data[h2o_var][:,y_idx,:]

  Qvmin = np.floor(np.min(Qvplot))
  Qvmax = np.ceil(np.max(Qvplot))

  qfig = plt.figure(figsize=(20,6))
  # The x-vector is a 2d array in the case of a lat-lon grid
  if nr_config['grid_type'] == 'latlon':
    plt.pcolormesh(xvec[y_idx],zvec,Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
  else:
    plt.pcolormesh(xvec,zvec,Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
  plt.ylim(0,ztop)
  plt.ylabel('Height (m)', fontsize=24)
  plt.xlabel('km', fontsize=24)
  plt.title('Water Vapor Slice', fontsize=24)
  plt.colorbar().set_label(label='g/kg',size=20)
  plt.savefig(vap_unfilt_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot the nature run RH cross-section
  qfig = plt.figure(figsize=(20,6))
  # The x-vector is a 2d array in the case of a lat-lon grid
  if nr_config['grid_type'] == 'latlon':
    plt.pcolormesh(xvec[y_idx],zvec,rtm_data[rh_var][:,y_idx,:],vmin=0.0,vmax=100.0, cmap='viridis')
  else:
    plt.pcolormesh(xvec,zvec,rtm_data[rh_var][:,y_idx,:],vmin=0.0,vmax=100.0, cmap='viridis')
  plt.ylim(0,ztop)
  plt.ylabel('Height (m)', fontsize=24)
  plt.xlabel('km', fontsize=24)
  plt.title('Relative Humidity Slice', fontsize=24)
  plt.colorbar().set_label(label='%',size=20)
  plt.savefig(rh_unfilt_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot the nature run surface temperature
  Tplot = np.squeeze(rtm_data[temp_var][z_idx,:,:])
  Tmin = np.floor(np.min(Tplot))
  Tmax = np.ceil(np.max(Tplot))

  logger.info(f'Min,max T: {Tmin} {Tmax}')

  # Plot the temperature distribution
  tfig = plt.figure(figsize=(20,15))
  if nr_config['grid_type'] == 'latlon':
    plt.pcolormesh(xlon[0,:],xlat[:,0],Tplot,vmin=Tmin,vmax=Tmax, cmap='RdYlBu_r')
    plt.ylabel('Latitude  (deg)', fontsize=36)
    plt.xlabel('Longitude (deg)', fontsize=36)
  else:
    plt.pcolormesh(xvec,yvec,Tplot,vmin=Tmin,vmax=Tmax, cmap='RdYlBu_r')
    plt.ylabel('Distance (km)', fontsize=36)
    plt.xlabel('Distance (km)', fontsize=36)
  plt.yticks(fontsize=32)
  plt.xticks(fontsize=32)
  plt.title('Surface Temperature', fontsize=40)
  plt.colorbar().set_label(label='Temperature (K)',size=36)
  plt.savefig(t_unfilt_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot the unfiltered cloud top height on a range from 0 to 20 km
  CTmin = 0.0
  CTmax = 20000.0
  ctfig = plt.figure(figsize=(20,15))
  if nr_config['grid_type'] == 'latlon':
    plt.pcolormesh(xlon[0,:],xlat[:,0],rtm_data['cth'],vmin=CTmin,vmax=CTmax, cmap='RdYlBu_r')
    plt.ylabel('Latitude  (deg)', fontsize=36)
    plt.xlabel('Longitude (deg)', fontsize=36)
  else:
    plt.pcolormesh(xvec,yvec,rtm_data['cth'],vmin=CTmin,vmax=CTmax, cmap='RdYlBu_r')
    plt.ylabel('Distance (km)', fontsize=36)
    plt.xlabel('Distance (km)', fontsize=36)
  plt.yticks(fontsize=32)
  plt.xticks(fontsize=32)
  plt.title('Cloud Top Height', fontsize=40)
  plt.colorbar().set_label(label='Height (m)',size=36)
  plt.savefig(cth_unfilt_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot the filtered non-noisy surface temperature distribution, if available
  if inst_data is not None:
    Tplot = np.squeeze(inst_data[temp_var][z_idx,:,:])
    Tmin = np.floor(np.min(Tplot))
    Tmax = np.ceil(np.max(Tplot))

    # print('Z index: ',z_idx)
    logger.info(f'Min,max T: {Tmin} {Tmax}')
    tfig = plt.figure(figsize=(20,15))
    if nr_config['grid_type'] == 'latlon':
      jy = np.int32(np.floor(0.5 * len(Tplot[:,0])))
      logger.info(f'Dimensions of lat and lon, jy, jout, iout: {np.shape(xlat)} {np.shape(xlon)} {jy} {jout} {iout}')
      xlat1d = np.squeeze(xlat[0::jout,0])
      xlon1d = np.squeeze(xlon[jy,0::iout[jy]])
      logger.info(f'Dimensions of reduced size lat and lon {len(xlat1d)} {len(xlon1d)}')
      plt.pcolormesh(xlon1d,xlat1d,Tplot[0::jout,0::iout[jy]],vmin=Tmin,vmax=Tmax, cmap='RdYlBu_r')
      plt.ylabel('Latitude  (deg)', fontsize=36)
      plt.xlabel('Longitude (deg)', fontsize=36)
    else:
      plt.pcolormesh(xvec[0::iout],yvec[0::jout],Tplot[0::jout,0::iout],vmin=Tmin,vmax=Tmax, cmap='RdYlBu_r')
      plt.ylabel('Distance (km)', fontsize=36)
      plt.xlabel('Distance (km)', fontsize=36)
    plt.yticks(fontsize=32)
    plt.xticks(fontsize=32)
    plt.title('Surface Temperature, Filtered', fontsize=40)
    plt.colorbar().set_label(label='Temperature (K)',size=36)
    plt.savefig(t_filt_file, dpi=150, transparent=False, bbox_inches="tight")
    plt.close()

  # Plot the filtered cloud top height on a range from 0 to 20 km
  CTmin = 0.0
  CTmax = 20000.0
  ctfig = plt.figure(figsize=(20,15))
  if nr_config['grid_type'] == 'latlon':
    jy = np.int32(np.floor(0.5 * len(Tplot[:,0])))
    xlat1d = np.squeeze(xlat[0::jout,0])
    xlon1d = np.squeeze(xlon[jy,0::iout[jy]])
    plt.pcolormesh(xlon1d,xlat1d,retr_data['cth'][0::jout,0::iout[jy]],vmin=CTmin,vmax=CTmax, cmap='RdYlBu_r')
    plt.ylabel('Latitude  (deg)', fontsize=36)
    plt.xlabel('Longitude (deg)', fontsize=36)
  else:
    plt.pcolormesh(xvec[0::iout],yvec[0::iout],retr_data['cth'][0::jout,0::iout],vmin=CTmin,vmax=CTmax, cmap='RdYlBu_r')
    plt.ylabel('Distance (km)', fontsize=36)
    plt.xlabel('Distance (km)', fontsize=36)
  plt.yticks(fontsize=32)
  plt.xticks(fontsize=32)
  plt.title('Cloud Top Height, Filtered', fontsize=40)
  plt.colorbar().set_label(label='Height (m)',size=36)
  plt.savefig(cth_filt_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Plot the noisy surface temperature
  Tplot = np.squeeze(retr_data[temp_var][z_idx,:,:])

  # Plot the filtered noisy temperature distribution
  tfig = plt.figure(figsize=(20,15))
  if nr_config['grid_type'] == 'latlon':
    jy = np.int32(np.floor(0.5 * len(Tplot[:,0])))
    xlat1d = np.squeeze(xlat[0::jout,0])
    xlon1d = np.squeeze(xlon[jy,0::iout[jy]])
    plt.pcolormesh(xlon1d,xlat1d,Tplot[0::jout,0::iout[jy]],vmin=Tmin,vmax=Tmax, cmap='RdYlBu_r')
    plt.ylabel('Latitude  (deg)', fontsize=36)
    plt.xlabel('Longitude (deg)', fontsize=36)
  else:
    plt.pcolormesh(xvec[0::iout],yvec[0::jout],Tplot[0::jout,0::iout],vmin=Tmin,vmax=Tmax, cmap='RdYlBu_r')
    plt.ylabel('Distance (km)', fontsize=36)
    plt.xlabel('Distance (km)', fontsize=36)
  plt.yticks(fontsize=32)
  plt.xticks(fontsize=32)
  plt.title('Surface Temperature, Retrieved', fontsize=40)
  plt.colorbar().set_label(label='Temperature (K)',size=36)
  plt.savefig(t_noisy_filt_file, dpi=150, transparent=False, bbox_inches="tight")
  plt.close()

  # Loop over all y indices
  for y_idx in y_incices:

    # Set water vapor plot file names
    vap_unfilt_file =  os.path.join(plot_path, model_prefix+'_'+str(y_idx)+'y_Qv_unfiltered.png')
    vap_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, str(y_idx)+'y_Qv_filtered')
    vap_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, str(y_idx)+'y_Qv_filtered_noisy')

    # Plot the nature run water vapor cross-section
    qfig = plt.figure(figsize=(20,6))
    Qvplot = rtm_data[h2o_var][:,y_idx,:]
    Qvmin = np.floor(np.min(Qvplot))
    Qvmax = np.ceil(np.max(Qvplot))
    if nr_config['grid_type'] == 'latlon':
      plt.pcolormesh(xvec[y_idx],zvec,Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
    else:
      plt.pcolormesh(xvec,zvec,Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
    plt.ylim(0,ztop)
    plt.ylabel('Height (m)', fontsize=24)
    plt.xlabel('km', fontsize=24)
    plt.title('Water Vapor Slice', fontsize=24)
    plt.colorbar().set_label(label='%',size=20)
    plt.savefig(vap_unfilt_file, dpi=150, transparent=False, bbox_inches="tight")
    plt.close()

    # Plot the filtered non-noisy water vapor cross-section, if available - use NR min/max
    if inst_data is not None:
      qfig = plt.figure(figsize=(20,6))
      if nr_config['grid_type'] == 'latlon':
        Qvplot = inst_data[h2o_var][0::kout,y_idx,0::iout[y_idx]]
        plt.pcolormesh(xvec[y_idx][0::iout[y_idx]],zvec[0::kout],Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
      else:
        Qvplot = inst_data[h2o_var][0::kout,y_idx,0::iout]
        plt.pcolormesh(xvec[0::iout],zvec[0::kout],Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
      plt.ylim(0,ztop)
      plt.ylabel('Height (m)', fontsize=24)
      plt.xlabel('km', fontsize=24)
      plt.title('Water Vapor Slice, Filtered', fontsize=24)
      plt.colorbar().set_label(label='%',size=20)
      plt.savefig(vap_filt_file, dpi=150, transparent=False, bbox_inches="tight")
      plt.close()

    # Plot the noisy "retrieved" water vapor - use NR min/max
    qfig = plt.figure(figsize=(20,6))
    if nr_config['grid_type'] == 'latlon':
      Qvplot = retr_data[h2o_var][0::kout,y_idx,0::iout[y_idx]]
      plt.pcolormesh(xvec[y_idx][0::iout[y_idx]],zvec[0::kout],Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
    else:
      Qvplot = retr_data[h2o_var][0::kout,y_idx,0::iout]
      plt.pcolormesh(xvec[0::iout],zvec[0::kout],Qvplot,vmin=Qvmin,vmax=Qvmax, cmap='viridis')
    plt.ylim(0,ztop)
    plt.ylabel('Height (m)', fontsize=24)
    plt.xlabel('km', fontsize=24)
    plt.title('Water Vapor Retrieved', fontsize=24)
    plt.colorbar().set_label(label='%',size=20)
    plt.savefig(vap_noisy_filt_file, dpi=150, transparent=False, bbox_inches="tight")
    plt.close()

    # Set RH plot file names
    rh_unfilt_file =  os.path.join(plot_path, model_prefix+'_'+str(y_idx)+'y_RH_unfiltered.png')
    rh_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, str(y_idx)+'y_RH_filtered')
    rh_noisy_filt_file = get_file_path(plot_path, model_prefix, sat_type, x_res, z_res, str(y_idx)+'y_RH_filtered_noisy')

    # Plot the nature run RH cross-section
    qfig = plt.figure(figsize=(20,6))
    if nr_config['grid_type'] == 'latlon':
      plt.pcolormesh(xvec[y_idx],zvec,rtm_data[rh_var][:,y_idx,:],vmin=0.0,vmax=100.0, cmap='viridis')
    else:
      plt.pcolormesh(xvec,zvec,rtm_data[rh_var][:,y_idx,:],vmin=0.0,vmax=100.0, cmap='viridis')
    plt.ylim(0,ztop)
    plt.ylabel('Height (m)', fontsize=24)
    plt.xlabel('km', fontsize=24)
    plt.title('Relative Humidity Slice', fontsize=24)
    plt.colorbar().set_label(label='%',size=20)
    plt.savefig(rh_unfilt_file, dpi=150, transparent=False, bbox_inches="tight")
    plt.close()

    # Plot the filtered non-noisy RH cross-section, if available
    if inst_data is not None:
      qfig = plt.figure(figsize=(20,6))
      if nr_config['grid_type'] == 'latlon':
        plt.pcolormesh(xvec[y_idx][0::iout[y_idx]],zvec[0::kout],inst_data[rh_var][0::kout,y_idx,0::iout[y_idx]],vmin=0.0,vmax=100.0, cmap='viridis')
      else:
        plt.pcolormesh(xvec[0::iout],zvec[0::kout],inst_data[rh_var][0::kout,y_idx,0::iout],vmin=0.0,vmax=100.0, cmap='viridis')
      plt.ylim(0,ztop)
      plt.ylabel('Height (m)', fontsize=24)
      plt.xlabel('km', fontsize=24)
      plt.title('Relative Humidity Slice, Filtered', fontsize=24)
      plt.colorbar().set_label(label='%',size=20)
      plt.savefig(rh_filt_file, dpi=150, transparent=False, bbox_inches="tight")
      plt.close()

    # Plot the noisy "retrieved" RH
    qfig = plt.figure(figsize=(20,6))
    if nr_config['grid_type'] == 'latlon':
      plt.pcolormesh(xvec[y_idx][0::iout[y_idx]],zvec[0::kout],retr_data[rh_var][0::kout,y_idx,0::iout[y_idx]],vmin=0.0,vmax=100.0, cmap='viridis')
    else:
      plt.pcolormesh(xvec[0::iout],zvec[0::kout],retr_data[rh_var][0::kout,y_idx,0::iout],vmin=0.0,vmax=100.0, cmap='viridis')
    plt.ylim(0,ztop)
    plt.ylabel('Height (m)', fontsize=24)
    plt.xlabel('km', fontsize=24)
    plt.title('Relative Humidity Retrieved', fontsize=24)
    plt.colorbar().set_label(label='%',size=20)
    plt.savefig(rh_noisy_filt_file, dpi=150, transparent=False, bbox_inches="tight")
    plt.close()

  output_status = 'Plotting finished!'
  return output_status
