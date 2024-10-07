"""
read_wrf.py

This python 3.x code reads all data necessary to run the CRTM
and returns it to the calling routine.

It assumes the data is in a netcdf file written by the WRF model.

Inputs:

Mandatory:
  Name of netcdf file

Optional:
  i1 and/or i2: start and end indices, x-direction (default None)
  j1 and/or j2: start and end indices, y-direction (default None)
  k1 and/or k2: start and end indices, z-direction (default None)
  mp_type: The microphysics type used in the WRF model  (default 10 = Morrison)
    Options: 6 = WSM6, 8 = Thompson, 10 = Morrison
  t_idx: the time index in the file (default 1)

Outputs: Dictionary containing the above variables

Requires:

netCDF4
numpy

Change log:
12/2/2022, DJP: Added an option to return only the dimensions, no data, via optional input flag "dims_only"
                Note that this will return the full dimensions, not the ones the user has requested
12/6/2022, DJP: To reduce memory footprint, all data in the dictionary that is returned to the calling program
                is stored as 32 bit floating point
12/27/2022, DJP: Added an integer coarsening factor as input - one for each direction
05/03/2023, DJP: Converted z1 and z2 from columns to 3D
25 Sep 2023, DSM: Replace verbose with logger. Add error handling.
08 Jan 2024, DSM: Adding WRFReader, an instance of the BaseReader class. Replaces the old "read_wrf"

"""

import logging
from typing import Optional
from netCDF4 import Dataset
import numpy as np

from nature_runs.readers import BaseReader
from utils.grid import Grid, slice_transform
from utils.data_handling import zero_dict

# Set logging
logger = logging.getLogger(__name__)

class WRFReader(BaseReader):

    # Adds the grid object for global access
    def __init__(self, grid=None) -> None:
        self.grid = grid

    # Attach the dataset
    def open_data(self, file_1: str, file_2: Optional[str] = None, **kwargs) -> None:

        self.dataset = Dataset(file_1)
        self.is_data_lodaded = True

    # Close the dataset
    def close_data(self) -> None:
        self.dataset.close()

    # Here, we get dimensions. Note that this assumes that "QVAPOR" is a 3d variable contained in the dataset file
    # And that the dimensions are ordered k, j, i
    def get_dimensions(self, icoarse: int = 1, jcoarse: int = 1, kcoarse: int = 1, t_idx: int = -1) -> dict:
        self.initialized(soft=True)

        self.t_idx = t_idx
        qvapor = self.dataset['QVAPOR'][t_idx]
        # self.end_indexes = (len(qvapor[0,0,:]), len(qvapor[0,:,0]), len(qvapor[:,0,0]))

        self.end_indexes = self.dataset['QVAPOR'][t_idx].shape
        dimension = {
            "nx": len(qvapor[0,0,::icoarse]),
            "ny": len(qvapor[0,::jcoarse,0]),
            "nz": len(qvapor[::kcoarse,0,0])
        }
        return self.ensure_data(dimension, dims_only=True)

    # Read_slice is the primary read routine
    def read_slice(self, grid: Grid, **kwargs):
        self.initialized()

        wrf_data = {}
        logger.info('Computing pressure')
        self.t_idx = kwargs.get('t_idx', self.t_idx)

        # State variables, winds, heights, and precip
        # Pressure requires combination of perturbation and base state
        wrf_data['Press'] = (
            np.array(self.dataset['P'][self.t_idx][grid.slice('kji')]) +
            np.array(self.dataset['PB'][self.t_idx][grid.slice('kji')])
        ) * 1.e-2  # Convert to hPa from Pa

        logger.info('Computing temperature and reading in vapor and winds')

        # Temperature requires conversion from perturbation potential temperature and pressure
        wrf_data['TT'] = (
          (np.array(self.dataset['T'][self.t_idx][grid.slice('kji')]) + 300.0) * # Convert from perturbation to full theta
          ( (wrf_data['Press'] / 1000.0) ** (287.04/1004.5) ) # Convert to temperature in K
        ) - 273.15 # Convert to deg C

        variables = [('Qv', 'QVAPOR'), ('uu', 'U'), ('vv', 'V'), ('ww', 'W')]

        data = slice_transform(
            data=self.dataset, variables=variables, grid=grid, order='kji',
            output_dict=True, time=self.t_idx,
        )

        data['Qv'] *= 1.e3 # Convert to g/kg
        wrf_data.update(data)

        # Now that we have read the data, get the new dimensions
        nz, ny, nx = data['Qv'].shape

        logger.info('Reading 2d variables')

        # 2D variables
        # Since there is only accumulated precip in WRF, zero this for now...
        wrf_data['Precip'] = np.zeros((ny,nx))

        variables = [('Topo_hgt', 'HGT'), ('Xlat', 'XLAT'), ('Xlon', 'XLONG'), ('Landmask', 'LANDMASK'), ('Tskin', 'TSK'), ('Psfc', 'PSFC'), ('OLR', 'LWUPT')]

        data = slice_transform(
            data=self.dataset, variables=variables, grid=grid, order='ji',
            output_dict=True, time=self.t_idx,
        )
        wrf_data.update(data)

        logger.info('Calculating heights')

        # Calculate the layer boundary heights from the geopotential
        phi_b = np.array(self.dataset['PHB'][self.t_idx][grid.slice('kji')])
        phi_p = np.array(self.dataset['PH'][self.t_idx][grid.slice('kji')])
        wrf_data['z2'] = (phi_b + phi_p) / 9.806 - wrf_data['Topo_hgt'] # Layer mid-point pressure is geopotential height - topography height
        # Now, interpolate / extrapolate to get the layer mid-point heights
        wrf_data['z1'] = np.zeros((nz,ny,nx))

        for k in range(nz-1):
          wrf_data['z1'][k,:,:] = 0.5 * (wrf_data['z2'][k,:,:] + wrf_data['z2'][k+1,:,:])

        logger.info('Reading cloud variables')

        # Microphysics - set variables common to all schemes
        variables = [('Qc', 'QCLOUD'), ('Qi', 'QICE'), ('Qr', 'QRAIN'), ('Qs', 'QSNOW'), ('Qg', 'QGRAUP')]
        data = slice_transform(
            data=self.dataset, variables=variables, grid=grid, order='kji',
            output_dict=True, time=self.t_idx, function=lambda x: x * 1.e3,
        )
        wrf_data.update(data)

        # If we have Thompson microphysics, set a few others
        mp_type = kwargs.get('mp_type', 6)
        if mp_type == 8:
            variables = [('Ni', 'QNICE'), ('Nr', 'QNRAIN')]
            data = slice_transform(
                data=self.dataset, variables=variables, grid=grid, order='kji',
                output_dict=True, time=self.t_idx,
            )
            wrf_data.update(data)
        # If we have Morrison microphysics, set a few others
        elif mp_type == 10:
            variables = [('Ni', 'QNICE'), ('Nr', 'QNRAIN'), ('Ns', 'QNSNOW'), ('Ng', 'QNGRAUPEL')]
            data = slice_transform(
                data=self.dataset, variables=variables, grid=grid, order='kji',
                output_dict=True, time=self.t_idx,
            )
            wrf_data.update(data)
        # Otherwise, zero number concentrations for rain, snow, and graupel for non-2-moment-schemes
        else:
            wrf_data.update(zero_dict(['Ni', 'Nr', 'Ns', 'Ng'], (nz,ny,nx)))

        # Now, zero RAMS variables that are not in WRF (and as such never used)
        wrf_data.update(zero_dict(['Nc', 'Qa', 'Na', 'Qh', 'Nh', 'Qc2', 'Nc2'], (nz,ny,nx)))

        # Set any negative mixing ratio and number concentration values to zero
        for qq in ['Qv', 'Qc', 'Qc2', 'Qr', 'Qi', 'Qs', 'Qa', 'Qg', 'Qh']:
          wrf_data[qq][wrf_data[qq] < 0.0] = 0.0
        for nn in ['Nc', 'Nc2', 'Nr', 'Ni', 'Ns', 'Na', 'Ng', 'Nh']:
          wrf_data[nn][wrf_data[nn] < 0.0] = 0.0

        # In future, we may add the latent heating variables to WRF output registry
        # but, for now zero them
        wrf_data.update(zero_dict(['LHRvap', 'LHRfrz', 'p0'], (nz,ny,nx)))

        wrf_data['Qcond'] = sum([wrf_data[x] for x in ['Qc', 'Qc2', 'Qr', 'Qi', 'Qs', 'Qa', 'Qg', 'Qh']])

        wrf_data.update({'nx': nx, 'ny': ny, 'nz': nz})
        return self.ensure_data(wrf_data)
