"""
This function performs vertical interpolation for a single column and returns a dictionary

It takes data from a run list so that the function is parmap-able

Inputs:
i: column index in the x-direction
j: column index in the y-direction
hgt_in: height vector for the input data
hgt_out: vector of heights to be interpolated to
data_in: column of input data on heights hgt_in

Output:
data_out: vector of data interpolated to hgt_out

Returns "dict_out", a dictionary containing the indix and the output data

Derek Posselt
JPL
12 Sept 2023

"""
from scipy.interpolate import interp1d
import numpy as np

def column_interp(run_list):

  # Pull data from run list
  i, j, hgt_in, hgt_out, data_in = run_list

  data_in = np.squeeze(data_in)
  zfunc   = interp1d(hgt_in, data_in)
  data_out = zfunc(hgt_out)

  dict_out = {}
  dict_out['data_out'] = data_out
  dict_out['i'] = i
  dict_out['j'] = j

  return dict_out
