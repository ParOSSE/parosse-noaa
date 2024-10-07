"""
This function reads a single file as output from the PAMS radar instrument simulator
In PAMS output, Zss is the 2-way attenuated single scattering reflectivity
It is also assumed that the PAMS output contains the state variables - temperature, pressure, and water vapor content

Luis Millan
JPL
1 Apr 2024

"""
import netCDF4
import numpy as np

def getZH(file):
  # Connect the dataset
	f2r = netCDF4.Dataset(file,'r')
  # Read reflectivity
	dBZ = f2r.variables['Zss'][:].filled(np.nan)
  # In PAMS, missing data is set to -999. Replace these with NaN
	dBZ[dBZ<-998] = np.nan
  # Obtain the 1D height vector (PAMS output is on constant height layers)
	hgt1d = f2r.variables['hgt1d'][:].filled(np.nan) / 1000.0   # Convert from m to km

  # Obtain the state variables
  # Water vapor volumetric mixing ratio g/m3
	Qv = f2r.variables['Qv'][:].filled(np.nan)* 1000.0    # Convert from kg m-3 ---> g m-3
  # Temperature in K
	tem = f2r.variables['T'][:].filled(np.nan)
  # Pressure in hPa
	pre = f2r.variables['P'][:].filled(np.nan)/ 100.0 # Convert pressure from Pa to hPa
  # Frequency is an attribute in the radar output file
	freq = f2r.Frequency   #GHz
  # Close the netcdf file
	f2r.close()

	# pos = file.find('GHz')
	
  # Fill a dictionary with reflectivity, height, state variables, and frequency
	oo = {'dBZ':dBZ, 'alt':hgt1d, 'Qv':Qv, 'tem':tem, 'pre':pre, 'freq':freq}

  # Return the dictionary to the calling program
	return oo

