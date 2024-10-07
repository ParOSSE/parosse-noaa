"""
read_g5nr.py

This python 3.x code reads all data necessary to run the radar forward model
and returns it to the calling routine.

It assumes the data is in a netcdf file written by the NASA GEOS5 model for the 7 km nature run (G5NR).

A file spec for the G5NR can be found here:

https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/docs/

All 2D variables are in a single file named:

c1440_NR.inst30mn_2d_met1_Nx.YYYYMMDD_hhmmz.nc4

Each 3D variable is stored in an individual file named:

c1440_NR.inst30mn_3d_<varname>_Nv.YYYYMMDD_hhmmz.nc4

Inputs:

Mandatory:
  Prefix of netcdf file (text up to "_2d_..." or "_3d_..."; e.g., <path>/c1440_NR.inst30mn)
  Suffix of netcdf file (date/time information: "YYYYMMDD_hhmm")

Optional:
  i1 and/or i2: start and end indices, x-direction (default None)
  j1 and/or j2: start and end indices, y-direction (default None)
  k1 and/or k2: start and end indices, z-direction (default None)
  mp_type: The microphysics type used in the GEOS model  (default 0 = Bacmeister)
    Options: 0 = Bacmeister, 1 = MG, 2 = MG2, 3 = MG3
  t_idx: the time index in the file (default 1)

Outputs: Dictionary containing the above variables

NOTE 1: the GEOS5 data is ordered top down. So, when it is inserted into the dictionary, the k-indices are reversed.
NOTE 2: While the forward models expect p0, z1, and z2 to be vertical columns, they are 3d variables in the GEOS model
        and are retained as 3d variables in the dictionary.

Requires:

netCDF4
numpy

25 Sep 2023, DSM: Replace verbose with logger. Add error handling.
02 Jan 2024, DSM: Adding G5NRReader, an instance of the BaseReader class. Replaces the old "read_g5nr"

"""

import logging
import numpy as np
from typing import Optional
from netCDF4 import Dataset

from nature_runs.readers import BaseReader
from utils.grid import Grid, slice_transform

# Set logging
logger = logging.getLogger(__name__)

# This class inherits from the BaseReader base class - each routine below expands on the routines in the base class

class G5NRReader(BaseReader):
    # Initialization is simple - just add the grid to the self object
    def __init__(self, grid=None) -> None:
        self.grid = grid

    # The open_data instance here is a little strange - it only serves the purpose of getting dimensions from the 3D temperature file
    def open_data(self, file_1: str, file_2: Optional[str] = None, **kwargs) -> None:
        # Fill the netcdf prefix and suffix - these will be used in all of the various G5NR files
        self.ncpre = file_1
        self.ncsuf = file_2

        # Construct the file name for the 3D temperature file, and attach it as a netCDF4 dataset
        file_name = self.ncpre+'_3d_T_Nv.'+self.ncsuf+'z.nc4'
        dataset = Dataset(file_name)

        # Get the 3D temperature as a numpy array, then close the dataset
        self.ds_3D_T = np.array(dataset['T'][:])
        # np.array(nc_variable[:])

        dataset.close()

        # Get time
        self.t_idx = kwargs.get('t_idx', 0)
        # Diagnostic print...
        # print(self.ds_3D_T)
        # Get the dimensions
        self.end_indexes = (
           len(self.ds_3D_T[self.t_idx,0,0,:]),
           len(self.ds_3D_T[self.t_idx,0,:,0]),
           len(self.ds_3D_T[self.t_idx,:,0,0])
        )

        # Open the netcdf file containing 2D variables
        file_name = self.ncpre+'_2d_met1_Nx.'+self.ncsuf+'z.nc4'
        self.ds_2D = Dataset(file_name)
        self.is_data_lodaded = True


    # Since the BaseReader class requires a close_data routine, define it and use it to close the 2D dataset
    def close_data(self) -> None:
        self.ds_2D.close()

    # Here, we get dimensions. Note that this assumes that we have read in 3D temperature as ds_3D_T and have put it in the self structure
    # It also assumes that the dimensions are ordered k, j, i
    def get_dimensions(self, icoarse: int = 1, jcoarse: int = 1, kcoarse: int = 1, t_idx: int = -1) -> dict:
        if t_idx == -1:
            t_idx = self.t_idx
        self.initialized()
        dimension = {
            "nx": len(self.ds_3D_T[t_idx,0,0,::icoarse]),
            "ny": len(self.ds_3D_T[t_idx,0,::jcoarse,0]),
            "nz": len(self.ds_3D_T[t_idx,::kcoarse,0,0])
        }
        return self.ensure_data(dimension, dims_only=True)


    # This is another required routine - here, we use it to read data from 2D and 3D files
    def read_slice(self, grid: Grid, **kwargs):
        self.initialized()

        mp_type = kwargs.get('mp_type', 0)
        t_idx = kwargs.get('t_idx', self.t_idx)

        # Call the routine that ensures that the lower index < higher index
        grid.increase_top_index()

        # Create the dictionary
        g5_data = {}

        # Advance to reading all data
        # logger.info(f'Bounds of output data are: {i1} {i2} {j1} {j2} {k1} {k2}')

        # Set lists of file name variables, variables in each 3D G5 file, and variable name needed in the output dictionary
        # Variable text in file name
        file_var  = ['T',   'QV', 'PL',     'H',  'U',  'V',  'W']
        # Name of variable inside GEOS file
        g5_var    = ['T',   'QV', 'PL',     'H',  'U',  'V',  'W']
        # Scale factor to be multiplied by the GEOS data
        g5_scale  = [1.0,  1.0e3, 1.e-2,    1.0,  1.0,  1.0,  1.0]
        # Offset to be added to the GEOS data
        g5_off    = [-273.15, 0.0,  0.0,    0.0,  0.0,  0.0,  0.0]
        out_var   = ['TT',   'Qv', 'Press', 'z1', 'uu', 'vv', 'ww']

        # Separate lists for microphysics
        # Variable text in file name
        file_var_micro  = ['QL', 'QI']
        # Name of variable inside GEOS file
        g5_var_micro    = ['QL', 'QI']
        # Scale factor to be multiplied by the GEOS data
        g5_scale_micro  = [1.0e3, 1.0e3]
        # Offset to be added to the GEOS data
        g5_off_micro    = [0.0,   0.0]
        out_var_micro   = ['Qc',  'Qi']

        # Additional variables for microphysics options
        if mp_type == 0:
            # Add microphysics variables
            file_var_micro.extend (['QR', 'QS'])
            g5_var_micro.extend   (['QR', 'QS'])
            g5_scale_micro.extend ([1.0e3, 1.0e3])
            g5_off_micro.extend   ([0.0,   0.0])
            out_var_micro.extend  (['Qr', 'Qs'])
            # Set fill variables (remaining microphysics species not used)
            out_fill_var   = ['Qc2', 'Qa', 'Qg', 'Qh', 'Nc', 'Nc2', 'Nr', 'Ni', 'Ns', 'Na', 'Ng', 'Nh']
            out_fill_value = [  0.0,  0.0,  0.0,  0.0,  1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]

        # slice_transform returns a dictionary containing the variables defined in the "variables" list of tuple pairs that map output name from input name
        # It takes as input the "data", which is the netcdf dataset, the grid object, and the dimension ordering
        # If "output_dict" is True, the result will be a dictionary, otherwise it will be a tuple
        data = slice_transform(data=self.ds_2D, variables=[('Precip', 'PRECTOT'), ('OLR', 'LWTUP')], grid=grid, order='ji', time=t_idx, output_dict=True)
        data['Precip'] = data['Precip'] * 3600.0 # Convert to mm/hour

        # Add the 2d variables that were just read to the dictionary
        g5_data.update(data)

        # Read in the lat and lon - these are vectors, and need to be made into 2D arrays
        xlat = np.float32(
          np.array(
            self.ds_2D['lat'][grid.j1:grid.j2:grid.jcoarse]
          )
        )
        xlon = np.float32(
          np.array(
            self.ds_2D['lon'][grid.i1:grid.i2:grid.icoarse]))
        nlat = len(xlat)
        nlon = len(xlon)
        xlat2d = np.transpose(np.tile(xlat,(nlon,1)))
        xlon2d = np.tile(xlon,(nlat,1))
        g5_data['Xlat']    = np.float32(xlat2d)
        g5_data['Xlon']    = np.float32(xlon2d)

        # For each 3D variable, read from the corresponding file (why are we not using the slice routines here?)
        for i in range(len(file_var)):
            # Construct the netcdf file name
            file_name = self.ncpre+'_3d_'+file_var[i]+'_Nv.'+self.ncsuf+'z.nc4'
            # Attach the file
            ds = Dataset(file_name)
            # Get the data, place in a numpy array, and flip the k-axis
            # Get data on full range of layers, including scale factor and offset
            out_data = np.float32(
                np.array(
                    ds[g5_var[i]][t_idx,:,grid.j1:grid.j2:grid.jcoarse,grid.i1:grid.i2:grid.icoarse] * g5_scale[i] + g5_off[i]
                )
            )
            # Flip the k-axis
            out_data = np.flip(out_data,axis=0)
            # Extract the range of layers
            out_data = out_data[grid.k1:grid.k2,:,:]
            # Place data into the dictionary
            g5_data[out_var[i]] = out_data
            logger.info(f'Variable: {out_var[i]} dimensions: {np.shape(g5_data[out_var[i]])}')

            # Close the dataset
            ds.close()

        # For each 3D microphysics variable, read from the corresponding file
        for i in range(len(file_var_micro)):
            # Construct the netcdf file name
            ncfile = self.ncpre+'_3d_'+file_var_micro[i]+'_Nv.'+self.ncsuf+'z.nc4'
            # Attach the file
            ds = Dataset(ncfile)
            # Get the data, place in a numpy array, and flip the k-axis
            # Get data on full range of layers, including scale factor and offset
            out_data = np.float32(
                np.array(
                    ds[g5_var_micro[i]][t_idx,:,grid.j1:grid.j2:grid.jcoarse,grid.i1:grid.i2:grid.icoarse] * g5_scale_micro[i] + g5_off_micro[i]
                )
            )
            # Flip the k-axis
            out_data = np.flip(out_data,axis=0)
            # Extract the range of layers
            out_data = out_data[grid.k1:grid.k2,:,:]
            # Place data into the dictionary
            g5_data[out_var_micro[i]] = out_data
            logger.info(f'Variable: {out_var_micro[i]} dimensions: {np.shape(g5_data[out_var_micro[i]])}')

            # Close the dataset
            ds.close()

        # Populate the data dimensions using the first 3d variable in the out_var list
        nz, ny, nx = g5_data[out_var[0]].shape
        g5_data.update({
            "nx": nx,
            "ny": ny,
            "nz": nz
        })

        # Fill total condensate
        g5_data["Qcond"] = np.zeros((nz,ny,nx))
        for i in range(len(out_var_micro)):
            g5_data["Qcond"] = g5_data["Qcond"] + g5_data[out_var_micro[i]]

        # Now, deal with height on layer boundaries
        # Layer midpoint heights are contained in z1
        z1 = g5_data['z1'][:,:,:]
        # z2 will hold the layer boundary heights
        z2 = np.zeros([nz,ny,nx])
        # Interpolate
        for k in np.arange(nz-1):
            z2[k,:,:] = 0.5 * (z1[k,:,:] + z1[k+1,:,:])
        #Extrapolate to top height - assume equal spacing as the layer below
        z2[nz-1,:,:] = z2[nz-2,:,:] + (z2[nz-2,:,:] - z1[nz-3,:,:])

        # Insert layer boundary data into the dictionary
        g5_data['z2'] = z2

        # p0 is legacy - meant to be the domain averaged base state pressure. Set it equal to Press at a single column at [0,0]
        g5_data['p0'] = g5_data['Press'][:,0,0]

        # Finally, fill the variables not available in the netcdf file
        for i in range(len(out_fill_var)):
            g5_data[out_fill_var[i]] = np.float32(np.full([nz,ny,nx], out_fill_value[i]))

        # Return data, ensuring that dimensions are correct and data has been read. if "soft" is false, any non-initialized variable will raise an exception
        return self.ensure_data(g5_data, soft=True)


