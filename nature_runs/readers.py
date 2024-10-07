"""
This file contains routines that perform operations on nature run input gridded datasets.

slice_transform extracts and transforms gridded information.
nr_utils contains routines from the utils.py file in the nature_runs directory

It expects a "grid" object, defined in the grid class below



"""

import abc # Creates base classes
import logging
from typing import Optional

import numpy as np
# import xarray as xr
import pandas as pd

from utils.grid import Grid
from utils.data_handling import ensure_type

# Initialize logger
logger = logging.getLogger(__name__)


"""
This is the base reader class.
Readers for specific nature runs inherit from this class

For the most part, users should not have to modify anything here - just add a new reader that inherits from this base class.

NOTE: the default set of variables needed by any future operation on the NR data is set here.
Users can override this variable list within their own specific NR reader.

"""
class BaseReader(abc.ABC):

    # Set the default data type for data variables
    dtype = np.float32
    # Set the default list of variables
    variables = [
      'nx', 'ny', 'nz', 'p0', 'z1', 'z2', 'Precip', 'Xlat', 'Xlon', 'Landmask', 'Topo_hgt', 'Psfc', 'Tskin', 'OLR',
      'Press', 'TT', 'Qv', 'uu', 'vv', 'ww', 'Qcond', 'Qc', 'Nc', 'Qc2', 'Nc2', 'Qr', 'Nr', 'Qi', 'Ni', 'Qs', 'Ns',
      'Qa', 'Na', 'Qg', 'Ng', 'Qh', 'Nh', 'LHRvap', 'LHRfrz'
    ]

    # Integer variables containing dimension lengths
    dim_variables = ['nx', 'ny', 'nz']

    # Initialize the data dictionary
    data = {}

    # Initialize flags
    is_data_lodaded = False
    dimensions = None
    end_indexes = None

    # Any time we implement the base class in a specific reader, these methods MUST be implemented
    # ie, must open dataset, close dataset, get dimensions, and read data
    @abc.abstractmethod
    def open_data(self, file_1: str, file_2: Optional[str] = None, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def close_data(self) -> None:
        pass

    @abc.abstractmethod
    def get_dimensions(self, icoarse: int = 1, jcoarse: int = 1, kcoarse: int = 1, t_idx: int = -1) -> dict:
        pass

    @abc.abstractmethod
    def read_slice(self, grid: Grid, **kwargs):
        pass

    # Checks to make sure 1) the data is loaded (a call has been made to open_data) and 2) we have end indices for each dimension
    def initialized(self, soft=False):
        if not self.is_data_lodaded:
            raise ValueError('Data has not been initiliazed')
        if self.end_indexes is None and not soft:
            raise ValueError('End indexes have not been set')

    # Returns a grid object - contains start, end, and stride for each dimension
    def _get_sliced_grid(
            self, i1: int = 0, i2 : Optional[int] = None, j1: int = 0, j2: Optional[int] = None,  k1: int = 0, k2: Optional[int] = None, icoarse: int = 1, jcoarse: int = 1, kcoarse: int = 1) -> Grid:

        self.initialized()

        if pd.isnull(i2):
            i2 = self.end_indexes[0]
        if pd.isnull(j2):
            j2 = self.end_indexes[1]
        if pd.isnull(k2):
            k2 = self.end_indexes[2]

        return Grid(i1=i1, i2=i2, j1=j1, j2=j2, k1=k1, k2=k2, icoarse=icoarse, jcoarse=jcoarse, kcoarse=kcoarse)

    # Does the tiling over x and y given grid dimensions (nx and ny) and tile length in each dimension
    # kwargs are passed through to read_slice to accommodate any additional information needed by a specific reader
    # for example, clat and clon where needed, and dx for multiple readers
    def read(self, i1, j1, icoarse, jcoarse, nx, nx_tile, ny, ny_tile, **kwargs):
        for ix in np.arange(0, nx,  nx_tile , dtype='int32'):
            for jy in np.arange(0, ny, ny_tile, dtype='int32'):

                # Make sure the tiles do not extend beyond the domain boundaries
                # Save the original tile dimensions
                nx_tile_save = int(nx_tile)
                ny_tile_save = int(ny_tile)
                if ix+nx_tile >= nx:
                    nx_tile = int(nx-ix)
                if jy+ny_tile >= ny:
                    ny_tile = int(ny-jy)

                logger.info(f'Extracting grid points: {ix} {jy} {nx_tile} {ny_tile}')

                # Sets up the subset grid
                grid = self._get_sliced_grid(
                    i1=ix+i1, i2=ix+i1+nx_tile,
                    j1=jy+j1, j2=jy+j1+ny_tile,
                    icoarse=icoarse, jcoarse=jcoarse
                )

                # Gets the slice for the particular variable from the specific reader (uses kwargs if necessary)
                nr_data = self.read_slice(grid, **kwargs)
                # Put in the indices as keys in the dictionary - these will flow into the output from the fwd model
                # Note that we must account for coarsening factors here.
                nr_data['ix'] = np.int32(ix/icoarse)
                nr_data['jy'] = np.int32(jy/jcoarse)
                nr_data['nx_tile'] = np.int32(nx_tile/icoarse)
                nr_data['ny_tile'] = np.int32(ny_tile/jcoarse)

                # Restore the x and y tile dimensions
                nx_tile = nx_tile_save
                ny_tile = ny_tile_save

                # Send back the dictionary containing the domain subset, use yield so that this function can be iterated
                yield nr_data

    # Checks to make sure all necessary dimensions are in the data, and that all variables are representedin the NR dictionary
    def ensure_data(self, data: dict, dims_only=False, soft=False) -> dict:
        for key in self.dim_variables:
            if key in data:
                data[key] = ensure_type(data[key], np.int32)

        if dims_only:
            set_difference = set(self.dim_variables) - set(data.keys())
            if set_difference:
                raise ValueError(f'Expected variables expected: {self.dim_variables}, got {set(data.keys())}')
            else:
                return data

        # check if there is a variable is not initialized
        set_difference = set(self.variables) - set(data.keys())
        if set_difference and not soft:
            raise ValueError(f'Variables {set_difference} have not been initialized')

        for key in data.keys():
            data[key] = ensure_type(data[key], self.dtype)

        logger.info(f'Initialzaing: {set_difference} with 0s')

        for key in set_difference:
            data[key] = np.zeros(data[self.dim_variables[0]].shape, dtype=self.dtype)
        return data

    # Experimental: Automatically transforms the datasets created by a reader inherited from this class.
    # Dimensions are inferred from ny, nx and nz and generally works.
    # def to_xarray(self) -> xr.Dataset:
    #     xr_data = {}
    #     nx, ny, nz = self.grid.get_dimensions()

    #     for key in self.data.keys():
    #         skip_keys = ['nx', 'ny', 'nz', 'Xlat', 'Xlon']

    #         if key not in skip_keys:
    #             if self.data[key].shape == (nz, ny, nx):
    #                 xr_data[key] = xr.DataArray(self.data[key], dims=['z', 'y', 'x'])
    #             elif self.data[key].shape == (ny, nx):
    #                 xr_data[key] = xr.DataArray(self.data[key], dims=['y', 'x'])
    #             elif self.data[key].shape == (nz,):
    #                 xr_data[key] = xr.DataArray(self.data[key], dims=['z'])
    #             else:
    #                 raise ValueError(f'Unknown shape for {key}: {self.data[key].shape}')

    #     xr_coords = {
    #         'x':self. data['Xlon'][:,0],
    #         'y': self.data['Xlat'][:,0],
    #         # Not indexing Z because it can irregular
    #         # 'z': self.data['z1'][:,0,0]
    #     }

    #     return xr.Dataset(xr_data, coords=xr_coords)
