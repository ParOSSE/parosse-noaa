from typing import Tuple, Optional, Any
import numpy as np


def str_array_slicer(array, str_slices):
    slices = []
    for str_slice in [x for x in str_slices.split(',')]:
        parts = [int(x) if x else None for x in str_slice.split(':')]

    # Adjust the parts list to make sure it always has 3 elements: start, stop, step
    while len(parts) < 3:
        parts.append(None)

    slices.append(slice(*parts))

    # Convert list to tuple and slice the array
    return array[tuple(slices)]


def slice_transform_array(array, slice_info, function=None):
    if isinstance(slice_info, str):
       value = np.array(str_array_slicer(array, slice_info))
    else:
       value = np.array(array[slice_info])

    if function is not None:
       value = function(value)

    return value


def slice_transform_dataset(dataset, variables, slice_info, function=None, output_dict=False):
  names = []
  values = []

  for var in variables:
    if isinstance(var, tuple) and len(var) == 2:
      name, var = var
    else:
      name = var

    names.append(name)
    values.append(slice_transform_array(dataset[var], slice_info, function))

  if output_dict:
    return dict(zip(names, values))
  else:
    return tuple(values)


"""
Very general function - used to extract and transform information from one of the nature run datasets

Ingests:
data - a dictionary containing
variables - a list of the variables (strings) to operate on - anything not in this list will be ignored
grid - a grid object, defined before this function is called. See the Grid class, defined below.

Returns:
sliced_transformed_data - either a dictionary or a tuple containing the transformed data
"""
def slice_transform(data, variables, grid, order='ijk', function=None, time=None, output_dict=False) -> Optional[Tuple]:
        # If the user has not provided a time to extract, operate on just the spatial dimensions
        if time is None:
            slice_info = grid.slice(order)
        # User has provided a time, extract the slice associated with the specified time
        else:
            slice_info = (time, *grid.slice(order))

        # Sliced_transformed_data is either a dictionary or tuple containing the processed data
        sliced_transformed_data = slice_transform_dataset(
                data, variables, slice_info, function=function, output_dict=output_dict
        )
        return sliced_transformed_data


"""
Grid

This is the class used to track the slicing information, and to set default values:
i1                # Start index in the i dimension
i2                # End index in the i dimension
j1                # Start index in the j dimension
j2                # End index in the j dimension
k1                # Start index in the k dimension
k2                # End index in the k dimension
icoarse           # Coarsening factor in the i direction
jcoarse           # Coarsening factor in the j direction
kcoarse           # Coarsening factor in the k direction

"""
class Grid:
    # Initialize - fill self structure with default values
    def __init__(
            self, i1: int = 0, i2: int = 0, j1: int = 0, j2: int = 0, k1: int = 0, k2: int = 0,
            icoarse: int = 1, jcoarse: int = 1, kcoarse: int = 1
        ) -> None:
        self.i1 = i1                # Start index in the i dimension
        self.i2 = i2                # End index in the i dimension
        self.j1 = j1                # Start index in the j dimension
        self.j2 = j2                # End index in the j dimension
        self.k1 = k1                # Start index in the k dimension
        self.k2 = k2                # End index in the k dimension
        self.icoarse = icoarse      # Coarsening factor in the i direction
        self.jcoarse = jcoarse      # Coarsening factor in the j direction
        self.kcoarse = kcoarse      # Coarsening factor in the k direction

    # If i1=i2, etc, make sure i2>i1 (same for j and k indices)
    def increase_top_index(self, increment: int = 1) -> None:
        if self.i1 == self.i2:
            self.i2 += increment

        if self.j1 == self.j2:
            self.j2 += increment

        if self.k1 == self.k2:
            self.k2 += increment

    # Slicing grids requires start:end:step. In this code, this is denoted (for example) as i1:i2:icoarse.
    # This code takes as input a text string containing the name of the dimension to operate on
    # For example, indexes = 'ji' translates to [j1:j2:jcoarse,i1:i2:icoarse]
    # It returns a tuple containing the slice information
    def slice(self, indexes, time=None) -> Tuple[slice, ...]:
        _slice = tuple([
            slice(getattr(self, f'{idx}1'), getattr(self, f'{idx}2'), getattr(self, f'{idx}coarse')) for idx in indexes
        ])
        if time is not None:
            _slice = (slice(time, time+1), *_slice),
        return _slice

    # Takes indexes, which (as above) might be "ji", and returns a string containing the values "j1:j2:jcoarse,i1:i2:icoarse"
    # For example, if j1=0, j2=10, jcoarse=2, and indexes='j', this returns "0:10:2"
    def str_slice(self, indexes) -> str:
        return ','.join([
           f'{getattr(self, f"{idx}1")}:{getattr(self, f"{idx}2")}:{getattr(self, f"{idx}coarse")}' for idx in indexes
        ])

    # Not currently used.
    # Was implemented for the case in which the user did not provide a top index for one or more dimensions (e.g., i2=None)
    def update_top_index(self, i: Optional[int] = None, j: Optional[int] = None, k: Optional[int] = None) -> None:
        if i is not None:
            self.i2 = i
        if j is not None:
            self.j2 = j
        if k is not None:
            self.k2 = k

    # Gets a tuple of dimensions lengths from the slice values for the current grid
    def get_dimensions(self) -> Tuple[int, int, int]:
        nx = int((self.i2 - self.i1)/self.icoarse)
        ny = int((self.j2 - self.j1)/self.jcoarse)
        nz = int((self.k2 - self.k1)/self.kcoarse)

        return nx, ny, nz


class DIM:
    # Constants for dimension keys used across the data handling functions.
    ZYX = ["zdim","ydim","xdim"]
    YX = ["ydim","xdim"]
    X = ["xdim"]