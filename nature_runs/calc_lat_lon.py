"""
This Python 3 code adds lat and lon variables to a GCE file.

It requires as input the grid center lat and lon, as well as the x and y grid spacing

It then works from center out computing the lat and lon of every grid point

Procedure
1. Read input from user
2. Compute latitude by row.
3. Compute lontitude by row.


"""
import numpy as np
# import sys

# Set the mean WGS84 Earth radius in meters
rad_earth = 6378137.0

# Set the value of pi
xpi = 3.1415926535

deg_to_rad = xpi / 180.0
rad_to_deg = 1.0 / deg_to_rad

def calc_lat_lon (clat, clon, dx, dy, nx, ny):

    # # Get the center lat and lon (deg) and grid spacing in x and y
    # clat = np.float32(sys.argv[1]) * deg_to_rad # Convert to radians
    # clon = np.float32(sys.argv[2]) * deg_to_rad # Convert to radians
    # dx = np.float32(sys.argv[3]) # assume these are in meters
    # dy = np.float32(sys.argv[4]) # assume these are in meters
    # nx = np.int32(sys.argv[5])
    # ny = np.int32(sys.argv[6])

    # Convert center lat and lon to radians
    clat = clat * deg_to_rad
    clon = clon * deg_to_rad

    # Set the lat and lon arrays
    xlat = np.zeros((ny,nx))
    xlon = np.zeros((ny,nx))

    # Get the index of the center point in the domain
    ixc = int(nx/2)
    jyc = int(ny/2)

    # Set the lat and lon at the center point
    xlat[jyc,:] = clat
    xlon[:,ixc] = clon

    # print(ixc, jyc)
    # print(xlat[jyc,ixc], xlon[jyc,ixc])

    # sys.exit()

    # Compute the latitude distance in radians from the dx
    dlat = (dy / rad_earth)

    # Compute the latitude for each row in the dataset
    # Center southwards
    for j in range(jyc-1,-1,-1):
        # print(j+1,xlat[0,j+1])
        xlat[j,:] = xlat[j+1,:] - dlat
    # Center northwards
    for j in range(jyc+1,ny):
        xlat[j,:] = xlat[j-1,:] + dlat

    # Now, compute the longitude for each row in the dataset
    # First, compute the Earth radius for every latitude point
    rad_earth_phi = rad_earth * np.cos(np.abs(xlat[:,ixc]))
    # print('Earth radii: ',rad_earth_phi)

    # Now, compute the delta-longitude (radians) for each latitude
    dlon = dx / rad_earth_phi[:]
    # print('dlon, deg: ', dlon * rad_to_deg)

    # sys.exit()

    # Now, iterate over all x points, computing longitudes
    # Center westwards
    for i in range(ixc-1,-1,-1):
        xlon[:,i] = xlon[:,i+1] - dlon[:]
    # Center eastwards
    for i in range(ixc+1,nx):
        xlon[:,i] = xlon[:,i-1] + dlon[:]

    # sys.exit()

    # Convert latitudes and longitudes from radians to degrees
    xlat = xlat * rad_to_deg
    xlon = xlon * rad_to_deg

    # for i in range(nx):
    #     print('Longitudes: ',xlon[0,i])
    # for j in range(ny):
    #     print('Latitudes:  ',xlat[j,0])

    # Convert center lat and lon back to degrees
    clat = clat * rad_to_deg
    clon = clon * rad_to_deg


    return xlat, xlon