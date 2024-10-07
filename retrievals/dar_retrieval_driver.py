"""
This python code is the driver for the simple DAR retrieval

It can be run in standalone mode from the command line (for command line, see below)

When run as a function call, the inputs are:
files      Input    Radar reflectivities files (full path) for 158.6, 167, and 174.8 GHz (in that order!)
 					  for example files = ['/bigdata/lmillan/pams/a-L-2016-12-30-120000-g3_12_14_20_29_34_c1_VIPR_158.6_x300-1000y300-800.nc',\
 		              '/bigdata/lmillan/pams/a-L-2016-12-30-120000-g3_12_14_20_29_34_c1_VIPR_167.0_x300-1000y300-800.nc',\
 		              '/bigdata/lmillan/pams/a-L-2016-12-30-120000-g3_12_14_20_29_34_c1_VIPR_174.8_x300-1000y300-800.nc']

outname	    opt      output file path and name, 
                     if not provided the output file will be dar_retrievals.nc

rstep_inp   opt      the retrieval vertical step to be used in kilometers the default is 0.2 km (200 meters)
                     if this is not an integer of the original resulution the closest value will be used. 

force       opt      (def False) if the output file already exist the program will do nothing, unless force is set to True


insfile     opt       (def dar_orbit.yml) file that describes the instruemnt characteristics  to be used to compute the errors


Note that this function calls py_mw_absorb.py and dbzerrors these files need to be in the same folder

to run on the command line do:
python dar_retrieval.py  file1,file2,file3    (no spaces in between the filenames)
or 
python dar_retrieval.py  file1,file2,file3 outname 0.2 False 'instfile.yml'
the optional arguments need to be in order, i.e., outname, then rstep_inp and then force

Output:
The output is a nc file with the water vapor retrieved files 

To do; 
- remove the cloud edges bins to avoid artifacts
- iterate using the retrieved WV to see if we can get better retrievals

Luis Millan Apr 1, 2024

Changes:
29 May 2024, DJP: Converted the core of the retrieval to standalone code now contained in dar_retrieval.py
28 May 2024, DJP: Added comments to the code, added a name == __main__ statement, and reversed the 
                  order of the data upon write (was top down, and is now bottom up)


"""


import netCDF4
import numpy as np
import scipy as scipy
import scipy.linalg
import matplotlib.pyplot as plt
import os

import sys
from time import time

# Import reader
from getZH import getZH

# Import MW absorption, radar uncertainty calculations, and dar retrieval
from py_mw_absorb import py_mw_absorb as mwabs
from dbzerrors import getdBZ_errors
from dar_retrieval import dar_retrieval

from Parmap.parmap import parmap

"""
Set up a small function that can be parmapped to run the radar retrieval. 
Takes as input a dictionary of inputs to the retrieval function and returns the dictionary of outputs from the dar_retrieval

"""

def run_dar_retrieval(dar_retr_inputs):

  # Retrieval takes as input (nx, ny, nrh, nf, numsteps, rstep, frqs, alt, kwv, Qv, dBZ, dBZm, dBZe)
  # The only inputs that change in each call are the dBZ, dBZm, dBZe, and Qv
  dar_dict = dar_retrieval(*dar_retr_inputs)
  return dar_dict


def dar_retrieval_driver(files, outname = None, rstep_inp = 0.2, force = False, instfile = 'dar_orbit.yml'):  

  # sfrq = str(frqs[0])
  # pos = files[0].index(sfrq)
  # outname = files[0][0:pos] +'retrievals'+files[0][pos+len(sfrq):]

  # Set domain tile sizes - later this will be moved to a config file
  nx_tile = 50
  ny_tile = 50

  # Set the parmap mode and number of workers - later this will be moved to a config file
  # parmode = 'seq' # Set "seq" to run on a single processor
  parmode = 'par'
  num_workers = 8

  # If no output name is provided, set the default
  if outname == None: 
    outname = 'dar_retrievals.nc'
  print('The output file will be: '+outname)

  # Only continue with the retrieval if there is no output file of the name requested, or if the user asks for it to be overwritten
  if os.path.isfile(outname) == False or force == True:

    # Ensure at least 3 files are provided
    nf = len(files)
    if nf != 3: 
      print('###---- At least 3 files are needed')
      print('###---- exiting')
      exit()

    # Set the timer
    t0 = time()

    # Read data from each of the reflectivity files
    ii = 0
    # Loop over files
    for x in files:

      print(x)
      # getZH is the functiont that reads radar and state data and returns it in the form of a dictionary
      aa = getZH(x)

      # Compute the dBZ and dBZ errors based on the instrument file (instfile) characteristics
      dBZ2, dBZ_err = getdBZ_errors(x, instfile = instfile)

      # Get the dimensions of the data
      ss = aa['dBZ'].shape
      nh = ss[0]  # Number of layers
      ny = ss[1]  # Number of y-points
      nx = ss[2]  # Number of x-points

      # If this is the first file read, create full arrays and extract state variables
      if ii ==0:
        dBZ   = np.zeros((nh, ny, nx, nf))   # idealized dBZ
        dBZm  = np.zeros((nh, ny, nx, nf))   # dBZ masked due to power SNR >=1
        dBZe  = np.zeros((nh, ny, nx, nf))   # dB_error 

        frqs = np.zeros((nf))

        Qv = aa['Qv']

        alt = aa['alt']
        tem = aa['tem']
        pre = aa['pre']

      # Fill the full arrays for the current frequency
      dBZ[:,:,:,ii] = aa['dBZ']
      dBZm[:,:,:,ii] = dBZ2
      dBZe[:,:,:,ii] = dBZ_err

      frqs[ii] = aa['freq']

      # Increment frequency index
      ii = ii+1

    # Set the timer
    t1 = time()
    print(f'Time to read data and compute uncertainties:             {t1-t0}')

    # Compute the temperature and pressure initial guess profile
    # for simplicity, assume this is equal to the mean value over the entire domain
    tem1D = np.nanmean(tem, axis = (1,2))
    # Average the log of pressure
    pre1D = 10.**np.nanmean(np.log10(pre), axis = (1,2))

    # Reverse the vertical direction of the arrays so that they are ordered top down
    # State
    alt = np.flip(alt)
    tem1D = np.flip(tem1D)
    pre1D = np.flip(pre1D)
    Qv = np.flip(Qv, axis = 0)

    # Radar
    dBZ = np.flip(dBZ, axis = 0)
    dBZm = np.flip(dBZm, axis = 0)
    dBZe = np.flip(dBZe, axis = 0)
    
    # Not entirely sure what these two arrays are meant to be...
    ko2 = np.zeros((nh, nf))
    kwv = np.zeros((nh, nf))

    # wve is a linearization around water vapor absorption coefficient
    # Luis now implements a 2 stage retrieval. Start with small (0.1) then move to larger (retrieved) value based on the first step.
    # wve = 0.1 # Original value
    wve = 10.0 # Gave best result for RICO
    # Loop over the layers (top to bottom) computing absorption in each frequency
    for ii in np.arange(nh):
      # Loop over each frequency
      for jj in np.arange(nf):

        # Obtain vector of absorption coefficients from mean temperature, pressure, and a specified water vapor amount
        dmy  = mwabs(frqs[jj],tem1D[ii], pre1D[ii], wve)
        # Store the oxygen absorption coefficient
        ko2[ii,jj] = dmy[2]

        # Now we calculate the reference water vapor density profile, so that we can
        # divide the absorption coefficient to get the 
        # mass extinction coefficient (also known as the absorption cross section) with
        # units of 1/km/(g/m^3)

        # Water vapor absorption divided by a reference water vapor amount (DJP why is this not the domain average at the current layer?)
        kwv[ii,jj] = dmy[3] / wve


    # Compute the layer depth in km (this assumes it is constant)
    dr       = (alt[0]-alt[1])  
    print('dr:   ',dr)

    # Check to ensure the range resolution is <= the retrieval step
    # rstep allows for oversampling the radar - produces a smoother retrieval. 
    if dr > rstep_inp:
      print('###---- the radar range resolution (dr) is bigger than the desired retrieval step')
      print('###---- the code will use 2*dr as rstep_inp')
      rstep_inp = dr*2

    # Number of steps = number of reflectivity layers in each retrieval layer
    numsteps = int(rstep_inp//dr)+1
    print('numsteps: ',numsteps)

    # rstep is the layer thickness of the retrieval 
    rstep    = dr*numsteps
    # Rethgt is the retrieval vertical height grid
    rethgt = alt[numsteps//2:-numsteps//2]

    # Number of retrieval heights will always be <= number of input reflectivity layers
    nrh = len(rethgt)
    nmh = len(alt)
    print('nrh, nmh: ',nrh, nmh)

    # sys.exit()

    # Set the timer
    t2 = time()
    print(f'Time to compute absorption and set up retrieval inputs:  {t2-t1}')

    # ---------------------------
    # Calls to parallel retrieval
    # ---------------------------

    # Generate a run-list for input to the parmap function
    run_num = 0
    run_input  = []
    run_output = []
    indices = []
    for ix in np.arange(0,nx,nx_tile, dtype='int32'):
      for jy in np.arange(0,ny,ny_tile, dtype='int32'):
        # Make sure the tiles do not extend beyond the domain boundaries
        nx_tile_save = int(nx_tile)
        ny_tile_save = int(ny_tile)
        if ix+nx_tile >= nx:
            nx_tile = int(nx-ix)
        if jy+ny_tile >= ny:
            ny_tile = int(ny-jy)

        # Set bounds
        i1=ix
        i2=ix+nx_tile
        j1=jy
        j2=jy+ny_tile

        print('Extracting grid points: ',i1,i2,j1,j2,nx_tile,ny_tile)

        input_data = [nx_tile, ny_tile, nrh, nf, numsteps, rstep, frqs, alt, kwv, 
                      Qv[:,j1:j2,i1:i2],dBZ[:,j1:j2,i1:i2,:],dBZm[:,j1:j2,i1:i2,:],dBZe[:,j1:j2,i1:i2,:]]

        # Put the indices into a separate list of lists
        indices.append([run_num,i1,i2,j1,j2,nx_tile,ny_tile])

        # Add the input to the run_dar_retrieval function for this tile into the run list
        run_input.append(input_data)

        # Increment the run number
        run_num += 1

        # Restore the x and y tiles in case they needed modification
        nx_tile = nx_tile_save
        ny_tile = ny_tile_save

    # sys.exit("Check status")

    # Set the timer
    t3 = time()
    print(f'Time to extract data on sub-domain tiles:                {t3-t2}')


    # Run retrievals using parmap
    # First, set up the parmap instance
    pmap = parmap.Parmap(mode=parmode, numWorkers=num_workers)
#     pmap = parmap.Parmap(master=DASK_URL, mode=parmode, numWorkers=num_workers)

    # Now, run the retrieval in parallel using parmap
    run_output = pmap(run_dar_retrieval, run_input)

    # Set the timer
    t4 = time()
    print(f'Time to run parallel retrieval:                          {t4-t3}')

    # Reconstruct retrieval domain from tiles
    # Eventually, the indices will be part of the dictionary output from the retrieval
    print('nrh: ',nrh)
    Qvc = np.zeros((nmh, ny, nx))
    slowv_m = np.zeros((nrh, ny, nx))
    slowe_m = np.zeros((nrh, ny, nx))
    for irun in range(len(run_output)):
        i1 = np.int32(indices[irun][1])
        i2 = np.int32(indices[irun][2])
        j1 = np.int32(indices[irun][3])
        j2 = np.int32(indices[irun][4])
        nx_tile = indices[irun][5]
        ny_tile = indices[irun][6]
        dar_dict = run_output[irun]
        print('i1,i2,j1,j2,size of output: ',i1,i2,j1,j2,np.shape(dar_dict['slowv_m']))
        Qvc[:,j1:j2,i1:i2]      = dar_dict['Qvc'][:,:,:]
        slowv_m[:,j1:j2,i1:i2]  = dar_dict['slowv_m'][:,:,:]
        slowe_m[:,j1:j2,i1:i2]  = dar_dict['slowe_m'][:,:,:]

    # Place the output into a single dictionary
    dar_dict = {}
    dar_dict['Qvc'] = Qvc
    dar_dict['slowv_m'] = slowv_m
    dar_dict['slowe_m'] = slowe_m
    # sys.exit()

    # The code below will run the retrieval algorithm in serial - here for reference
    # Call the retrieval algorithm itself - returns a dictionary containing retrieved variables
    # dar_dict = run_dar_retrieval(dar_retrieval_list)
    # dar_dict = dar_retrieval(nx, ny, nrh, nf, numsteps, rstep, frqs, alt, kwv, Qv, dBZ, dBZm, dBZe)

    # Set the timer
    t5 = time()
    print(f'Time to reconstruct full domain:                         {t5-t4}')

    # Write results to a file 
    print('writing the results to: '+outname)

    # Create netcdf dataset
    dd = netCDF4.Dataset(outname, 'w', format='NETCDF4')

    # Fill netcdf dimensions
    xdim = dd.createDimension('xdim', nx)     # x axis
    ydim = dd.createDimension('ydim', ny)    # y axis
    hdim = dd.createDimension('hdim', nrh) # retrieval height
    mhdim = dd.createDimension('mhdim', nmh) # model height

    # Set up output variables and their ID's
    # DAR water vapor retrieval
    wvid = dd.createVariable('Qv_DAR',np.float64,('hdim','ydim','xdim'),zlib=True)
    wvid.units = 'g m-3'

    # DAR water vapor retrieval error
    wveid = dd.createVariable('Qv_DAR_precision',np.float64,('hdim','ydim','xdim'),zlib=True)
    wveid.units = 'g m-3'

    # wvmid = dd.createVariable('Qv_ret_masked',np.float64,('hdim','ydim','xdim'),zlib=True)
    # wvmid.units = 'gm-3'

    # Water vapor from the model
    modid = dd.createVariable('Qv_MOD',np.float64,('mhdim','ydim','xdim'),zlib=True)
    modid.units = 'g m-3'

    # Water vapor from the model, masked to cloudy regions
    cldid = dd.createVariable('Qv_MOD_cloudy',np.float64,('mhdim','ydim','xdim'),zlib=True)
    cldid.units = 'g m-3'
    cldid.description = 'model as sampled by DAR '

    altid = dd.createVariable('altitude_DAR',np.float64,('hdim'),zlib=True)
    altid.units = 'km'

    altmid = dd.createVariable('altitude_MOD',np.float64,('mhdim'),zlib=True)
    altmid.units = 'km'

    # Fill output variables
    # Flip the vertical dimension so that it reads bottom up
    wvid[:]  = np.flip(dar_dict['slowv_m'], axis = 0)
    wveid[:]  = np.flip(dar_dict['slowe_m'], axis = 0)

    altid[:] = np.flip(rethgt)

    modid[:]  = np.flip(Qv, axis = 0)
    cldid[:]  = np.flip(dar_dict['Qvc'], axis = 0)

    altmid[:] = np.flip(alt)

    # Close dataset
    dd.close()
    print('###--- Successfully completed DAR retrieval')

    # Set the timer
    t6 = time()
    print(f'Time to write data to output file:                       {t6-t5}')

  # If the file exists and the user has not asked for an overwrite, print an error message
  else:
    print('Output file exists and force is not set - DAR retrieval NOT performed')


# -------------------------------------------------------------------
# Main routine - drives the experiment
# -------------------------------------------------------------------
if __name__ == '__main__':

  # Get the number of command line arguments - must be at least one (a string containing the list of file paths+names)
  ninp = len(sys.argv) -1

  # If we have at least one command line input, the proceed
  if ninp >=1:

    # Obtain the list of files by splitting the first text string argument
    files  = sys.argv[1].split(',')

    # Set default values of optional input arguments
    outname   = None
    rstep_inp = 0.2
    force     = False 

    # If the user has provided additional arguments, then fill the respective fields
    if ninp >= 2: outname = sys.argv[2]
    if ninp >= 3: rstep_inp = float(sys.argv[3])
    if ninp >= 4: 
      force    = sys.argv[4]
      force=force == 'True'

    # Diagnostic print
    print(files)
    print(outname)
    print(rstep_inp)
    print(force)

    # Call the DAR retrieval routine
    result = dar_retrieval(files, outname = outname, rstep_inp  = rstep_inp, force = force )

  # If the user has provided no command line inputs, then exit
  else:
    print('User called dar_retrieval from the command line, but provided no command line inputs')
    sys.exit('Check inputs to dar_retrieval.py')


