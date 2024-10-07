#The following routines are all based or directly copy from Ricky Roy's LES simulations radar model 
#See Roy, R. J., Lebsock, M., and Kurowski, M. J.: 
#Spaceborne differential absorption radar water vapor retrieval capabilities in tropical and subtropical boundary layer cloud regimes, 
#Atmos. Meas. Tech., 14, 6443–6468, https://doi.org/10.5194/amt-14-6443-2021, 2021. 



import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
from scipy import constants
import yaml



def gauss_beam_radius(wavelength,antenna_diam):
	# wavelength     I   in meters
	# antenna_diam   I   in meters
	# computes the 1/e Gaussian beam radius for a 10.9 dB taper according to Goldsmith eqn. (6.41)


	theta0 = 1.167*wavelength/(2*antenna_diam*np.sqrt(np.log(2)))

	return theta0

def gauss_beam_gain(wavelength,antenna_diam):
	# wavelength     I   in meters
	# antenna_diam   I   in meters
	# This function  assumes a 10.9 dB taper, the efficiency for a 10.9 dB taper is 0.815
	efficiency = 0.815
	area_aperture = np.pi*(antenna_diam/2)**2
	gain = 4*np.pi*efficiency*area_aperture/wavelength**2


	return gain

def num_independent_samples(numpulses,v_sat,theta0,wavelength,PRI):

	sigma_f = v_sat*theta0/wavelength
	tau_i = 1/(2*np.sqrt(np.pi)*sigma_f)

	tau_ratio = PRI/tau_i

	m_arr = np.arange(1,numpulses)

	sum_term = np.sum((1-m_arr/numpulses)*np.exp(-tau_ratio**2*m_arr**2))

	return numpulses/(1+2*sum_term)

def orbital_velocity(orbital_altitude):

	GMe = 3.986E14 # product of big G and mass of Earth in mks
	Re = 6.371E6	# radius of Earth in m

	vg = np.sqrt(GMe/(Re+orbital_altitude))

	return vg

def ground_velocity(orbital_altitude):

	GMe = 3.986E14 # product of big G and mass of Earth in mks
	Re = 6.371E6	# radius of Earth in mP

	vg = np.sqrt(GMe/Re)*(1+orbital_altitude/Re)**(-3/2)

	return vg

######-------------Main 
######-------------Main 
######-------------Main 
######-------------Main 
######-------------Main 
######-------------Main 



# file = '/bigdata/lmillan/pams/a-L-2016-12-30-120000-g3_12_14_20_29_34_c1_VIPR_158.6_x300-1000y300-800.nc'


def getdBZ_errors(file, instfile = 'dar_orbit.yml'):

	 


	#instrument parameters used for determining the errors 
	with open(instfile,'r') as f:
		oo = yaml.safe_load(f)


	antenna_diam = oo['antenna_diam']   ##meters
	pow_tx       = oo['pow_tx']			##Watts
	orbital_alt  = oo['orbital_alt']          ## m /s
	v_sat        = oo['v_sat']          ## m /s
	tau_pulse    = oo['tau_pulse']		##seconds
	dutycycle    = oo['dutycycle']		#fraction  = 0.25
	noisefig     = oo['noisefig']		##dB
	Tscene       = oo['Tscene']			##Kelvins
	T_alongtrack = oo['T_alongtrack']		##seconds
	numfreqs     = oo['numfreqs']		##This will most likely always be 3
	minsnr       = oo['minsnr']			# 1



	vel_ground = v_sat

	#checking to see if the instrument is on orbitt or not. 100 km (or even 30) seems like a reasonable threshold. 
	if orbital_alt > 100:
		vel_ground = ground_velocity(orbital_alt)

	doppler_decorr_time = antenna_diam/(2*vel_ground)

	if tau_pulse == 0:
		tau_pulse = doppler_decorr_time
	elif tau_pulse > doppler_decorr_time:
		tau_pulse = doppler_decorr_time
		warningmsg = "\n\nThe user specified pulse time of %.04e seconds is longer than the Doppler"\
		" decorrelation time of D/(2*v_ground) = %.04e seconds. The pulse time has been redefined to"\
		" equal the Doppler decorrelation time."
		print(warningmsg)


	f2r = netCDF4.Dataset(file,'r')


	dBZ = f2r.variables['Zss'][:].filled(np.nan)
	dBZ[dBZ<-998] = np.nan
	alt = f2r.variables['hgt1d'][:].filled(np.nan) / 1000   ###km


	freq = f2r.Frequency   #GHz
	Ksqrd = f2r.Ksq   #GHz




	wavelength = constants.c/(freq*1e9)


	gain = gauss_beam_gain(wavelength,antenna_diam)

	theta_0 = gauss_beam_radius(wavelength,antenna_diam)
	gain = gauss_beam_gain(wavelength,antenna_diam)

	# the two way solid angle is a simple function of the 1/e gaussian beam radius theta_0
	omega = np.pi*theta_0**2/2

	rangeres = np.abs((alt[1] - alt[0]))*1000.  #in meters

	C_f = pow_tx * np.pi**2 * Ksqrd * gain**2 * omega * rangeres/(64 * wavelength**2)


	radarrange = (orbital_alt-alt)*1000   ##in meters


	pow_rx = C_f * 1E-18 * 10**(dBZ/10) / radarrange.reshape(-1,1,1)**2


	kb = constants.k
	noisefloor = kb*Tscene/tau_pulse

	Fn = 10**(noisefig/10)	# noise figure in dB to linear units


	snr = pow_rx/(Fn*noisefloor)




	 #  repetition interval for a single frequency assuming constant cycling through the frequencies
	pulse_rep_int = tau_pulse/dutycycle   
	freq_rep_int = pulse_rep_int*numfreqs


	# number of pules pulses at each frequency.
	numpulses = int(dutycycle*T_alongtrack/(tau_pulse*numfreqs))

	numind = num_independent_samples(numpulses,v_sat,theta_0,wavelength,freq_rep_int)

	# the true minimum SNR is given by snr_min = 1 divided by numind 
	minsnr_true = minsnr/np.sqrt(numind)


	mindbz = 10*np.log10(Fn*noisefloor*orbital_alt**2 / C_f / 1E-18)-5*np.log10(numind)



	pow_rx[snr<minsnr_true] = np.nan
	snr[   snr<minsnr_true] = np.nan



	relerr = np.sqrt(numpulses/numind+2/snr+1/snr**2)/np.sqrt(numpulses)

	dBZ2 = 10*np.log10(pow_rx * radarrange.reshape(-1,1,1)**2 * 1E18 / C_f)

	dBZ_err = relerr*10/np.log(10)


	return dBZ2, dBZ_err



# file = '/bigdata/lmillan/pams/a-L-2016-12-30-120000-g3_12_14_20_29_34_c1_VIPR_158.6_x300-1000y300-800.nc'


# dBZ2, dBZ_err =  getdBZ_errors(file)



# breakpoint()

# cmap = 'turbo'
# ss = dBZ.shape

# ix = 50


# vmin = -100
# vmax = 20

# x = np.arange(0, ss[1])
# x = np.arange(0, ss[2])

# # fig, ax = plt.subplots(3,1, sharex = True, figsize = (12,12))

# fig = plt.figure(figsize = (8,8))

# gs = fig.add_gridspec(2, 1, hspace=0.05, wspace=0.02, left = 0.08, 
# 	right = 0.98, top = 0.95, bottom = 0.15)

# ax= gs.subplots(sharex=True)



# cbar = ax[0].pcolormesh(x, alt, dBZ[:,ix,:], cmap = cmap, vmin = vmin, vmax = vmax)

# ax[1].pcolormesh(x, alt, dBZ2[:,ix,:], cmap = cmap, vmin = vmin, vmax = vmax)

# # ax[2].pcolormesh(x, alt, dBZ_err[:,ix,:], cmap = cmap, vmin = vmin, vmax = vmax)

# # ax[3].pcolormesh(x, alt, mindbz[:,ix,:], cmap = cmap, vmin = vmin, vmax = vmax)


# ax[0].grid()
# ax[1].grid()

# p0 = ax[0].get_position()
# p8 = ax[1].get_position()


# fig.text(0.01,  p8.y0+(p0.y1 - p8.y0)/2, 'altitude [km]', 
# 	transform=plt.gcf().transFigure, rotation = 90, va = 'center')


# cbar_ax = fig.add_axes([0.1, 0.08, 0.86, 0.015])
# fig.colorbar(cbar, cax=cbar_ax,label='dBZ ', orientation ='horizontal')

# fpng = 'tstnoise_ix_'+str(ix)+'.png'

# plt.savefig(fpng, dpi = 380)

# plt.show()

# breakpoint()