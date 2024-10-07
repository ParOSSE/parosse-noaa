"""
This python code contains the simple DAR retrieval

Inputs:
  nx, ny, nrh, nf:  (integers) number of x, y, z, and frequency points, respectively
  numsteps:         (integer) number of model levels in each radar retrieval bin
  rstep:            (float) retrieval vertical layer thickness (km)
  frqs:             (vector, floats, (nf)) vector of frequencies (GHz)
  alt:              (array, floats, (nz)) vector of reflectivity heights
  kwv:              (array, floats, (nz,nf)) water vapor microwave absorption per unit mixing ratio (1/km/(g/m3))
  Qv:               (array, floats, (nz,ny,nx)) Water vapor volumetric mixing ratio (g/m3)
  dBZ:              (array, floats, (nz,ny,nx,nf)) single scatter reflectivity (attenuated) at each frequency
  dBZm:             (array, floats, (nz,ny,nx,nf)) Idealized refelectivity (unattenuated)
  dBZe:             (array, floats, (nz,ny,nx,nf)) Reflectivity uncertainty (dB)

Returns:
  dar_dict:         (dictionary) containing retrieved water vapor, water vapor uncertainty, and cloud-masked Qv


To do; 
- remove the cloud edges bins to avoid artifacts
- iterate using the retrieved WV to see if we can get better retrievals

Luis Millan Apr 1, 2024

Retrieval references:
Roy et al., (2020): Validation of a G-Band Differential Absorption Cloud Radar for Humidity Remote Sensing, doi: 10.1175/JTECH-D-19-0122.1

Changes:
29 May 2024, DJP: Converted the core of the retrieval to standalone code. Main driver is dar_retrieval_driver.py
28 May 2024, DJP: Added comments to the code, added a name == __main__ statement, and reversed the 
                  order of the data upon write (was top down, and is now bottom up)

"""

# Import necessary modules
import numpy as np
import scipy as scipy
import scipy.linalg

import sys

def dar_retrieval(nx, ny, nrh, nf, numsteps, rstep, frqs, alt, kwv, Qv, dBZ, dBZm, dBZe):  

  # Set up arrays to save the water vapor retrievals from pairs of frequencies
  retwv1 = np.zeros((nrh, ny, nx)) # 1,2
  retwv2 = np.zeros((nrh, ny, nx)) # 1,3
  retwv3 = np.zeros((nrh, ny, nx)) # 2,3

  # "SLO" = slope (we are retrieving water vapor slope). WV = water vapor, WE = water vapor error
  # Idealized errors
  slowv  = np.zeros((nrh, ny, nx))
  slowe  = np.zeros((nrh, ny, nx))

  # Real errors
  slowv_m  = np.zeros((nrh, ny, nx))
  slowe_m  = np.zeros((nrh, ny, nx))

  # Loop over the x and y domain in the model simulations
  # Note that the retrievals are oversampling water vapor in height
  for x in np.arange(nx):
    # print('Running x-point ',x,' of ',nx)
    for y in np.arange(ny):
      # print(y,x)
      # 3-frequency reflectivity, masked reflectivity (passed SNR check), and reflectivity error profiles at this horizontal point
      tdBZ  = dBZ[:,y,x,:]
      tdBZm = dBZm[:,y,x,:]
      tdBZe = dBZe[:,y,x,:]

      # Column water vapor at this point
      tQv = Qv[:,y,x]
      # Compute the difference in water vapor absorption coefficient between frequencies 2-1
      dkappa = kwv[:,1]-kwv[:,0]

      ###simple 2 frequency retrievals for testing
      # DJP - need some explanation for this
      # Frequencies 1,2
      DFR = (tdBZ[:,1]-tdBZ[:,0])/(2*rstep)
      hum1= (DFR[:-numsteps]-DFR[numsteps:])/(dkappa[numsteps//2:-numsteps//2]*10*np.log10(np.e))

      # Frequencies 1,3
      DFR = (tdBZ[:,2]-tdBZ[:,0])/(2*rstep)
      dkappa = kwv[:,2]-kwv[:,0]
      hum2= (DFR[:-numsteps]-DFR[numsteps:])/(dkappa[numsteps//2:-numsteps//2]*10*np.log10(np.e))

      # Frequencies 2,3
      DFR = (tdBZ[:,2]-tdBZ[:,1])/(2*rstep)
      dkappa = kwv[:,2]-kwv[:,1]
      hum3= (DFR[:-numsteps]-DFR[numsteps:])/(dkappa[numsteps//2:-numsteps//2]*10*np.log10(np.e))

      # Fill profiles of retrieved water vapor
      retwv1[:,y,x] = hum1
      retwv2[:,y,x] = hum2
      retwv3[:,y,x] = hum3

      # 3 frequency  retreival 
      slopehum       = np.zeros(nrh)       #idealized dBZ
      # Initial field contains NaNs
      slopehum[:]    = np.nan
      slopehum_m     = np.copy(slopehum)   #realistic dBZ
      slopehum_err   = np.copy(slopehum)   #idealized dBZ precision 
      slopehum_err_m = np.copy(slopehum)	 #realistic dBZ precision 

      slopefit     = np.copy(slopehum)
      slopefit_err = np.copy(slopehum)
      slopeoff     = np.copy(slopehum)

      # Set the error equal to the noise-free reflectivity. This is an array dimensioned (nz,nfreq)
      # DJP - should use np.copy to avoid overwriting tdBZ?
      terr = np.copy(tdBZ)
      # Now, divide by 100 --> 1% error
      terr = terr / 100.0   #idealized dBZ error

      # Set the more "realistic" error equal to the dBZ error
      terr_m = np.copy(tdBZe)
      # Loop over the number of retrieval heights
      for j in range(nrh):
        # Set indices for layers that bound "numsteps" 
        indr1 = j
        indr2 = j+numsteps

        # Compute the index that is in the middle of the range
        midind = int(j+numsteps//2)
        # Compute the water vapor absorption coefficient for this layer
        kabs_wv = kwv[midind,:]*10*np.log10(np.e)

        # If we have found any NaN values at this layer (missing reflectivity), cycle the loop - go to the next layer
        if np.sum(np.isnan(tdBZ[indr1,:]))>0 or np.sum(np.isnan(tdBZ[indr2,:]))>0:
          continue

        # Retrieval layer depth (km)
        dr = np.abs(alt[indr2]-alt[indr1])

        # beta: from Roy paper, Z difference for idealized and realistic
        beta_obs  = (tdBZ[indr1,:]-tdBZ[indr2,:])/(2*dr)
        beta_obsm = (tdBZm[indr1,:]- tdBZm[indr2,:])/(2*dr)

        # beta_err: error in the quantity beta, again idealized and realistic
        beta_err    = np.sqrt(terr[indr1,:]**2+terr[indr2,:]**2)/(2*dr)
        beta_err_m  = np.sqrt(terr_m[indr1,:]**2 +terr_m[indr2,:]**2)/(2*dr)

        # Forming diagonal covariance matrices
        covar_mat   = np.diag(beta_err**2)
        covar_mat_m = np.diag(beta_err_m**2)

        # As above, if we have found NaNs in any of the covariance elements, cycle the loop
        if ( np.sum(np.isnan(covar_mat)) > 0 ):
          # print('Found NaN in covar_mat')
          # print('Shape covar_mat: ',np.shape(covar_mat))
          # Cycle the loop
          continue
        if ( np.sum(np.isnan(covar_mat_m)) > 0 ):
          # print('Found NaN in covar_mat_m')
          # print('nrh:               ',nrh)
          # print('Shape tdBZm:       ',np.shape(tdBZm))
          # print('Shape covar_mat_m: ',np.shape(covar_mat_m))
          # print('indr1,indr2:       ',indr1,indr2)
          # print('dr:                ',dr)
          # print('terr_m[indr1,:]    ', terr_m[indr1,:])
          # print('terr_m[indr2,:]    ', terr_m[indr2,:])
          # print('beta_err_m:        ',beta_err_m)
          # print('covar_mat_m:       ',covar_mat_m)
          # print('terr_m:            ',terr_m)
          # print('')
          # print('terr:              ',terr)
          # print('')
          # Cycle the loop
          continue
          
        # Matrix used in retrieval - not the Rogers A-matrix. See Roy et al. for details
        Amat = np.zeros((nf,3))
        Amat[:,0] = 1
        Amat[:,1] = frqs-frqs[0]
        Amat[:,2] = kabs_wv.T


        #estimating humidity using Idealized dBZ and almost negligible errors
        # x_est = (np.linalg.inv(Amat.T @ np.linalg.inv(covar_mat) @ Amat) @\
        #           Amat.T @ np.linalg.inv(covar_mat) @ beta_obs.T).T
        x_est = (scipy.linalg.inv(Amat.T @ scipy.linalg.inv(covar_mat) @ Amat) @\
                  Amat.T @ scipy.linalg.inv(covar_mat) @ beta_obs.T).T


        #estimating humidity using realisitc dBZ and realistic dBZ errors
        # x_est_m = (np.linalg.inv(Amat.T @ np.linalg.inv(covar_mat_m) @ Amat) @\
        #        	 Amat.T @ np.linalg.inv(covar_mat_m) @ beta_obsm.T).T
        x_est_m = (scipy.linalg.inv(Amat.T @ scipy.linalg.inv(covar_mat_m) @ Amat) @\
                  Amat.T @ scipy.linalg.inv(covar_mat_m) @ beta_obsm.T).T


        # Posterior error covariance matrix
        # S_x = np.linalg.inv(Amat.T @ np.linalg.inv(covar_mat) @ Amat)
        # S_x_m = np.linalg.inv(Amat.T @ np.linalg.inv(covar_mat_m) @ Amat)
        S_x = scipy.linalg.inv(Amat.T @ scipy.linalg.inv(covar_mat) @ Amat)
        S_x_m = scipy.linalg.inv(Amat.T @ scipy.linalg.inv(covar_mat_m) @ Amat)


        slopeoff[j] = x_est[0]
        slopefit[j] = x_est[1]

        slopehum[j] = x_est[2]   #g m-3
        slopehum_m[j] = x_est_m[2]

        slopehum_err[j]  = np.sqrt(S_x[2,2])
        slopehum_err_m[j] = np.sqrt(S_x_m[2,2])

        slopefit_err[j] = np.sqrt(S_x[1,1])

        # End of loop over vertical retrieval layers

      slowv[:,y,x]   = slopehum
      slowv_m[:,y,x] = slopehum_m

      slowe[:,y,x]   = slopehum_err
      slowe_m[:,y,x] = slopehum_err_m
      #for testing - plots the simple 2-frequency retrievals vs the 3-frequency retrieval
      # plt.plot(hum1, rethgt, color = 'blue')
      # plt.plot(hum2, rethgt, color = 'green')
      # plt.plot(hum3, rethgt, color = 'pink')
      # plt.plot(tQv, alt, color = 'black')
      # plt.plot(slopehum, rethgt, color = 'red')
      # plt.show()

      # breakpoint()

      # End of loop over all horizontal grid points

  #getting the cloudy model fields as seen as DAR (using the closest to the line frequency)
  Qvc = np.copy(Qv)
  Qvc[:] = np.nan
  Qvc[np.isfinite(dBZm[:,:,:,2]) ]  = Qv[np.isfinite(dBZm[:,:,:,2])]

  # Fill dictionary for export to calling program
  dar_dict = {}
  dar_dict['slowv_m'] = slowv_m
  dar_dict['slowe_m'] = slowe_m
  dar_dict['Qvc'] = Qvc

  # Return dictionary of outputs to calling function
  return dar_dict

# -------------------------------------------------------------------
# Main routine - here in case this file is called from the command line
# -------------------------------------------------------------------
if __name__ == '__main__':

  print('dar_retrieval.py is meant to be called as a function')
  sys.exit('Stopping.')


