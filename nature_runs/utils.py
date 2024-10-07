import os
import abc
import yaml
import logging

import numpy as np

CONFIG_FILES_PATH = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


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
