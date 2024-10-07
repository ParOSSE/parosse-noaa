"""
Python 3x code that averages input data over a specified footprint region
defined by the full width at quarter max of the radar beam given an input 
orbital altitude and beam width (in degrees). 

The user must also enter the grid box length (in meters; assumed square)
and whether the averaging is done over reflectivity (need log transform)
or any other variable (no need for log transform)

The user can optionally average in the vertical as well. The user must input 
an array of heights matching the input data, as well as the desired 'resolution'
of the radar - defined as the full width at half max of a vertical pulse. The 
output data will be on a vertical grid with dz equal to this resolution, or the
user can specify the output dz to account for oversampling. 

NOTE: It is assumed that the input variable (x) is on a constant height grid, 
      and, if it is 3D, each x and y point is on the exact same set of heights

Inputs: 
  x:          the array that contains values to be averaged over
  dx:         the length of a grid box in meters
  beam_width: the radar beam width in degrees
  orbit_hgt:  the orbital altitude in meters
  vert_res:   (optional) desired vertical resolution in meters (full width at half max of the radar pulse)
                  default = 0.0, no vertical averaging
  height:     (optional) height vector corresponding to the vertical grid of the input variable in meters, necessary if vert_res is used
                  default = None
  dz_out:     (optional) the desired vertical spacing for the output grid in meters
                  default = 0.0, in this case vert_res will be used
  x_fac:      (optional) the factor by which to scale the footprint sigma in the x-direction
                  default = 1.0
  y_fac:      (optional) the factor by which to scale the footprint sigma in the y-direction
                  default = 1.0
  log_scale:  (optional) whether to apply log scaling to the input
                  default = False
  Verbose:    (optional) flag for verbose output
                  default = False

Output:
  x_filt:     the filtered (smoothed) version of the input field
  height:     the height vector of the output variable, or None if the variable is not 3d

Uses the gaussian filter from scipy ndimage

Derek Posselt
JPL
14 May 2021

"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
# from parosse.utils import hdfio
import sys


def fp_average( x, dx, beam_width, orbit_hgt, vert_res = 0.0, height = None, dz_out = 0.0, x_fac=1.0, y_fac=1.0, log_scale=False, Verbose=False):

    # If needed, convert from log to linear units
    if log_scale:
      x = 10.0**(x/10.0)

    # Obtain the full width at half max from the beamwidth and altitude
    #sigma = (xrad / (2*np.log(2))**.5) * 4.
    fwhm = np.tan(np.radians(beam_width)) * orbit_hgt   #beam width .35, 407km is GPM orbit  (x2 for double test)
    # Obtain the standard deviation of the gaussian filter
#     sigma = fwhm / (2.0* (2.0 * np.log(2.0))**.5)  # meters
    # Correction - we need to use full width at quarter max (M. Lebsock, pers comm. June 2021)
    sigma = fwhm / (2.0* (2.0 * np.log(4.0))**.5)  # meters

    # If called for, print out the footprint sigma in meters
    if Verbose:
      print('Averaging over horizontal footprint with sigma of ',sigma)
      print('Beam width and orbit height are        ',beam_width, orbit_hgt)

    sigma = sigma / dx  # convert from meters to gridpoints

    if Verbose:
       if height is not None:
          print('Min height in input: ',np.min(height))
          print('Max height in input: ',np.max(height))

    # Perform the filtering operation
    # If input is 3d, assume it is ordered z, y, x
    if x.ndim == 3:
      # If the user has entered a vertical resolution > 0, then apply vertical smoothing
      if vert_res > 0:
          # Ensure the user has passed in a height array - if not, exit.
          if height is None:
              sys.exit('Input height array needed for vertical averaging.')
          # Height array is found - proceed to averaging
          else:
            # If the user has not requested a different sampling height grid, set the output dz to the input resolution
            if dz_out <= 0.0:
              dz_out = vert_res
            if Verbose:
              print('Averaging vertical resolution to ',vert_res, 'm sampled at ',dz_out)
            # Obtain the maximum height, and reduce it to 20 km if necessary
            maxz = np.max(height)
            if maxz > 20000:
                maxz = 20000
            # Obtain a vector of height layer thicknesses
            dzin = np.diff(height)
            print('Max z and min/max delta-z: ',maxz,np.min(dzin),np.max(dzin))
            # Set the delta-z to the minimum layer thickness - ignore zero-thickness layers
            dz = np.min(dzin[np.where(dzin > 0.0)])
            # Compute the standard deviation of the vertical Gaussian averaging function in meters, then convert to grid points
            # Assumes the vertical resolution is the full width at half max of the radar pulse
            vertsigma = vert_res / (2.* (2*np.log(4.))**.5)  #m 
            vertsigma = vertsigma/dz #convert from meters to gridpoints
            
            # If the input height dz is not constant (max > min), then regrid to the highest vertical resolution prior to applying the vertical averaging
            if np.max(dzin) > dz:
                # Get the number of layers to interpolate to
                nz1 = np.floor(maxz/dz)
                print('Number of layers to interpolate to: ',nz1)
                # Populate a high resolution height grid
                z_highres = np.arange(nz1)*dz 
                print('High res grid: ',z_highres)
                # Make sure we are only interpolating to layers above the lowest layer in the input data
                if np.min(z_highres) < np.min(height): # If the lowest layer in the interp height array is < lowest layer in data
                  idx_z = np.where(z_highres > np.min(height))[0][0] # Find the location where the high res array is > lowest layer in data
                  z_highres = z_highres[idx_z:] # Subset the array
                # Interpolate
                xfunc = interp1d(height, x, axis=0)
                x = xfunc(z_highres)
                # x = hdfio.regrid(x, height, z_highres) #put variable on a uniform vertical grid
            # Otherwise, if the input height is constant, just set the high resolution height vector equal to the input height vector
            else:
                z_highres = height
            
            # Apply a Gaussian filter in all 3 dimensions using the scipy Gaussian filter routine
            avgvar = gaussian_filter(x,[vertsigma,sigma*y_fac,sigma*x_fac],truncate=2)
            
            # Create output gridded data sampled at "dz_out" meter intervals
            nzout = np.floor(maxz/dz_out)
            height = np.arange(nzout)*dz_out
            # Ensure the height array to be interpolated to is within the bounds of the data
            # bottom is >= to the lowest layer in the full height array
            if np.min(height) < np.min(z_highres):
              idx_z = np.where(height > np.min(z_highres))[0][0] # Find the location where the interp array is > lowest layer in data
              height = height[idx_z:] # Subset the array
            # top is <= to the highest layer in the full height array
            if np.max(height) < np.min(z_highres):
              idx_z = np.where(height < np.max(z_highres))[0][0] # Find the location where the interp array is > lowest layer in data
              height = height[:idx_z] # Subset the array
            if Verbose:
              print('nzout:        ',nzout)
              print('nz in averaged array: ',len(avgvar[:,0,0]))
              print('Min z in full array:   ',min(z_highres))
              print('Min in new height arr: ',min(height))
              print('nz,height:    ',len(height),height)
              print('nz,z_highres: ',len(z_highres),z_highres)
        # def regrid(varin, z, newz):
            func = interp1d(z_highres,avgvar,axis=0)
            x_filt = func(height)
            if Verbose:
              print('nz in regridded array: ',len(x_filt[:,0,0]))
              print('nz,height:    ',len(height),height)
            # x_filt = avgvar
            # We have replaced the input height array with z_highres
            # height = z_highres
      # If the user has not requested vertical filtering, spatially filter (horizontally), leaving the vertical dimension alone
      else:
        x_filt = gaussian_filter(x, [0,sigma*y_fac,sigma*x_fac])#, mode='nearest')
    # If the input is 2D, filter in both directions
    elif x.ndim == 2:
      x_filt = gaussian_filter(x, [sigma*y_fac,sigma*x_fac])#,   mode='nearest')
    # If the input is 1D, filter in a single direction
    elif x.ndim == 1:
      x_filt = gaussian_filter(x, sigma*x_fac, mode='nearest')
    # Dimensions beyond 3 are not supported
    else:
      sys.exit('Check number of dimensions in input to fp_average!')

    # Convert back to log units, making sure to set to missing where the linear variable = 0.0
    if log_scale:
      x_filt[x_filt<=1.e-10] = 1.e-10 # set all zero values in the linear variable to a very small positive number (-100 dBZ)
      x_filt = 10.0*np.log10(x_filt)
      x_filt[x_filt<=-100] = -999
    
    if Verbose and vert_res != 0.0:
      print('Just prior to return: nz,height:    ',len(height),height)

    return x_filt, height

