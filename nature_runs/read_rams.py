"""
read_rams.py

This python 3.x code reads all data necessary to run the radar forward model
and returns it to the calling routine.

It assumes the data is in an hdf5 file that contains output from the CSU RAMS model.
It also requires the user to provide both the hdf5 file and also the header (text) file
If the input is a "lite" file, then there may not be land use data. In this case,
we need a separate surface file

Necessary variables are:

Dimensions: x, y, z

1 'T' (deg C)
2 'Qc'
3 'Nc'
4 'Qi'
5 'Ni'
6 'Qr'
7 'Nr'
8 'Qs'
9 'Ns'
10 'Qa'
11 'Na'
12 'Qg'
13 'Ng'
14 'Qh'
15 'Nh'
16 'Qc2'
17 'Nc2'
18 'w'
19 'Qv'
20 'u'
21 'v'
22 'p0' (center column)
23 'z1'
24 'z2'
25 'Precip'

Input: Name of hdf5 and header files
Outputs: The above variables in a dictionary

Requires:

netCDF4
h5py
numpy

Change log:
12/2/2022, DJP: Added an option to return only the dimensions, no data, via optional input flag "dims_only"
                Note that this will return the full dimensions, not the ones the user has requested
12/6/2022, DJP: To reduce memory footprint, all data in the dictionary that is returned to the calling program
                is stored as 32 bit floating point
12/27/2022, DJP: Added an integer coarsening factor as input - one for each direction
05/03/2023, DJP: Converted z1 and z2 from columns to 3D
05/18/2023, DJP: Added lat, lon, terrain height, land mask, surface pressure, and surface temperature for CRTM
05/26/2023, DJP: Added routine to compute surface temperature
06/21/2023, DJP: Added read of latent heating due to vapor and freezing
25 Sep 2023, DSM: Replace verbose with logger. Add error handling.
08 Jan 2024, DSM: Adding RAMSReader, an instance of the BaseReader class. Replaces the old "read_rams"

"""
import logging
import numpy as np
from typing import Optional
from netCDF4 import Dataset

import h5py
import hdf5plugin

from nature_runs.readers import BaseReader
from utils.grid import Grid, slice_transform

# Set logging
logger = logging.getLogger(__name__)

class RAMSReader(BaseReader):

    # Adds the grid object and defines constants
    def __init__(self, grid=None) -> None:
        # Add the grid object to the self object
        self.grid = grid

        # Set constants
        self.cp = 1004.0
        self.rgas = 287.0
        self.cpor = self.cp / self.rgas
        self.p00 = 1.0e5

        # Initialize the header file contents structure
        self.header_contents = None

    # Open the dataset, depending on whether we are processing the HDF5 format analysis or "lite" file, or netcdf
    def open_data(self, file_1: str, file_2: Optional[str] = None, **kwargs) -> None:
        self.rams_format = kwargs.get('rams_format', 'HDF5')
        self.sfc_file = file_2

        # Open the file
        if self.rams_format == 'HDF5':
            # RAMS header file is always the same as the input file with the last 5
            # characters "gX.h5" replaced with "head.txt"
            self.header_file = file_1[:-5]+'head.txt'

            # Open the HDF5 dataset
            self.dataset = h5py.File(file_1)

            # Open and read the header file
            with open(self.header_file) as f:
                self.header_contents = f.readlines()

            # RAMS HDF5 processing requires a header file. Raise an exception if the contents are blank
            if self.header_contents is None:
                raise ValueError('HDF5 format RAMS files require a header file! Stopping in read_rams')

            # If the user has provided the name of the surface file, load it
            if self.sfc_file is not None:
              self.ds_sfc = h5py.File(self.sfc_file)

        elif self.rams_format == 'NC':
          self.dataset = Dataset(file_1)
        else:
          raise ValueError(f'Unrecognized RAMS file format: {self.rams_format} \n \
                          Exiting in read_rams')
        self.is_data_lodaded = True

    # Close the dataset(s)
    def close_data(self) -> None:
        self.dataset.close()
        # Close the surface dataset, if it has been provided
        if self.sfc_file is not None:
           self.ds_sfc.close()

    # Here, we get dimensions. Note that this assumes that "RV" is a 3d variable contained in the dataset file
    # And that the dimensions are ordered k, j, i
    def get_dimensions(self, icoarse: int = 1, jcoarse: int = 1, kcoarse: int = 1, t_idx: int = -1) -> dict:
        self.initialized(soft=True)

        rv = self.dataset['RV'][t_idx] if t_idx >= 0 else self.dataset['RV']
        self.end_indexes = (len(rv[0,0,:]), len(rv[0,:,0]), len(rv[:,0,0]))

        dimension = {
            "nx": len(rv[0,0,::icoarse]),
            "ny": len(rv[0,::jcoarse,0]),
            "nz": len(rv[::kcoarse,0,0])
        }
        return self.ensure_data(dimension, dims_only=True)

    # This is another required routine - here, we use it to read data from files
    def read_slice(self, grid: Grid, **kwargs):
        self.initialized()

        t_idx = kwargs.get('t_idx', -1)

        # These variables will be scaled / converted
        vars_to_convert = [
            ('Qv', 'RV'), ('Qc', 'RCP'), ('Qr', 'RRP'), ('Qc2', 'RDP'), ('Qi', 'RPP'),
            ('Qs', 'RSP'), ('Qa', 'RAP'), ('Qg', 'RGP'), ('Qh', 'RHP')
        ]

        # These variables need no scaling
        vars = [
          ('Nc', 'CCP'), ('Nr', 'CRP'), ('Nc2', 'CDP'), ('Ni', 'CPP'), ('Ns', 'CSP'),
          ('Na', 'CAP'), ('Ng', 'CGP'), ('Nh', 'CHP'), ('uu', 'UP'), ('vv', 'VP'), ('ww', 'WP')
        ]

        # List of surface precipitation rate variables
        vars_precip = ['PCPRR', 'PCPRP', 'PCPRS', 'PCPRA', 'PCPRG', 'PCPRH', 'PCPRD']

        # Dimensioning ordering upon read from the data file(s)
        order = 'kji'

        data_slice = {}
        if t_idx < 0:
           t_idx = None

        # slice_transform returns a dictionary containing the variables defined in the "variables" list of tuple pairs that map output name from input name
        # It takes as input the "data", which is the netcdf or hdf5 dataset, the grid object, and the dimension ordering
        # If "output_dict" is True, the result will be a dictionary, otherwise it will be a tuple
        # Read the variables that need to be scaled - uses a lambda function for this
        data = slice_transform(
            data=self.dataset, variables=vars_to_convert, grid=grid, order=order,
            output_dict=True, time=t_idx, function=lambda x: x * 1.0E3,
        )
        data_slice.update(data)
        logger.info('Read vapor, liquid, and ice mixing ratios')
        # Read the data that do not need scaling
        data = slice_transform(
            data=self.dataset, variables=vars, grid=grid, order=order,
            output_dict=True, time=t_idx,
        )
        data_slice.update(data)
        logger.info('Read winds and number concentrations')

        precip = slice_transform(
            data=self.dataset, variables=vars_precip, grid=grid, order='ji',
            output_dict=False, time=t_idx, function=lambda x: np.squeeze(x)
        )
        data_slice['Precip'] = sum(precip) * 3600.0
        data_slice['Precip'][np.where(data_slice['Precip'] < 0.0)] = 0.0

        logger.info('Computed precipitation rate')

        # Set any negative mixing ratio, number concentration, and precip values to zero
        for qq in ['Qv', 'Qc', 'Qc2', 'Qr', 'Qi', 'Qs', 'Qa', 'Qg', 'Qh', 'Precip']:
          data_slice[qq][data_slice[qq] < 0.0] = 0.0
        for nn in ['Nc', 'Nc2', 'Nr', 'Ni', 'Ns', 'Na', 'Ng', 'Nh']:
          data_slice[nn][data_slice[nn] < 0.0] = 0.0

        # Try reading OLR. If it is not there, then fill with zeros
        try:
            data_slice['OLR'] = np.array(self.dataset['RLONTOP'][grid.slice('ji', time=t_idx)])
        except:
            data_slice['OLR'] = np.zeros([ny,nx])

        # Obtain potential temperature and exner function
        theta = np.array(self.dataset['THETA'][grid.slice('kji', time=t_idx)])
        exner = np.array(self.dataset['PI'][grid.slice('kji', time=t_idx)]) / self.cp

        # Read terrain height
        hgt_topo = np.squeeze(np.array(self.dataset['TOPT'][grid.slice('ji', time=t_idx)]))

        # Read lat and lon
        xlat = np.squeeze(np.array(self.dataset['GLAT'][grid.slice('ji', time=t_idx)]))
        xlon = np.squeeze(np.array(self.dataset['GLON'][grid.slice('ji', time=t_idx)]))

        # Read vapor latent heating
        LHRvap = np.squeeze(np.array(self.dataset['LATHEATVAP'][grid.slice('kji', time=t_idx)]))
        # Read freeze/melt latent heating
        LHRfrz = np.squeeze(np.array(self.dataset['LATHEATFRZ'][grid.slice('kji', time=t_idx)]))

        # Obtain the number of grid points in this slice
        nz, ny, nx = data_slice['Qv'].shape

        # Read the patch area - contains the land mask.
        # Note that RAMS sets 0=land, 1=water, while for CRTM and PAMS we need 1=land and 0=water...
        # Subtracting 1 and taking the absolute value swaps the 1s and 0s...
        if self.sfc_file is not None:
            # Get the 2d landmask
            landmask = np.abs(
              np.squeeze(np.array(self.ds_sfc['PATCH_AREA'][grid.slice('ji', time=t_idx)])) - 1
              )
        else:
          try:
            # Get the 2d landmask from the HDF5 file
            landmask = np.abs(
              np.squeeze(np.array(self.ds['PATCH_AREA'][grid.slice('ji', time=t_idx)])) - 1
              )
          except:
            logger.info('Could not read PATCH_AREA from RAMS file - setting surface to water (0)!')
            landmask = np.zeros((ny,nx))

        # Compute temperature (deg C) from potential temperature and exner function
        TT = (exner[:,:,:]*theta[:,:,:])-273.15 # convert to deg C

        # Compute pressure (hPa) from the exner function
        press = self.p00 * 0.01 * (exner[:,:,:]**self.cpor)
        # Set p0 (column pressure) to the center column
        ix = 0
        jy = 0
        if nx > 1:
          ix = np.int_(nx/2)
        if ny > 1:
          jy = np.int_(ny/2)
        p0 = np.array(press[:,jy,ix])

        # Compute surface pressure - extrapolate from lowest layers, linear in log-p  (PLACEHOLDER)
        psfc = np.exp( 2.0 * np.log(press[0,:,:]) - np.log(press[1,:,:]) )

        # Add 0.1 degrees to lowest layer temperature to get skin T (PLACEHOLDER)
        tskin = np.squeeze(TT[0,:,:] + 273.15 + 0.1)

        logger.info('Computed temperature and pressure')

        if self.rams_format == 'HDF5':
          if self.header_contents is None:
              raise ValueError('HDF5 format RAMS files require a header file! Stopping in read_rams')

          # Obtain the layer boundary (edge) heights
          idx_zmn = self.header_contents.index('__zmn01\n') # find edge level line index
          nz_m = int(self.header_contents[idx_zmn+1]) # grab number of edge levels
          # Read in edge level heights into a float array
          z2 = np.zeros(nz_m)
          for i in np.arange(0,nz_m):
              z2[i] =  float(self.header_contents[idx_zmn+2+i])

          # Obtain the layer center heights
          idx_ztn = self.header_contents.index('__ztn01\n') # find center levels line index
          nz_t = int(self.header_contents[idx_ztn+1]) # Grab number of centers levels
          # Read in center level heights into a float array
          z1 = np.zeros(nz_t)
          for i in np.arange(0,nz_t):
              z1[i] =  float(self.header_contents[idx_ztn+2+i])

          # Subset
          z1 = z1[grid.slice('k')]
          z2 = z2[grid.slice('k')]

        # Otherwise, read in the heights from file
        elif self.rams_format == 'NC':
          z1 = self.dataset['ztn'][grid.slice('k')] # Layer mid-points
          z2 = self.dataset['zmn'][grid.slice('k')] # Layer edges

        hgt_lay = np.transpose(np.reshape(np.tile(z1,(nx,ny)),(nx,ny,len(z1)), order='C'),(2,1,0))
        hgt_lev = np.transpose(np.reshape(np.tile(z2,(nx,ny)),(nx,ny,len(z2)), order='C'),(2,1,0))

        data_slice.update({
            "nx": nx, "ny": ny, "nz": nz, "p0": p0, "z1": hgt_lay, "z2": hgt_lev, "Xlat": xlat,
            "Xlon": xlon, "Landmask": landmask, "Topo_hgt": hgt_topo, "Psfc": psfc, "Tskin": tskin,
            "Press": press, "TT":TT, "LHRvap": LHRvap, "LHRfrz": LHRfrz,
            # Compute the total condensate mixing ratio
            'Qcond': sum([data_slice[x] for x in ['Qc', 'Qc2', 'Qr', 'Qi', 'Qs', 'Qa', 'Qg', 'Qh']]),
        })

        return self.ensure_data(data_slice)
