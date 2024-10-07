import numpy as np

##routines copied from mw_absorb.f90



def abs_n2(T, p, f):

# T: temperature (K)
# p: pressure (mb)
# f: frequency (GHz) (valid 0-1000 GHz)


    TH = 300 / T
    FDEPEN = .5 + .5 / (1 + (f / 450)**2)  
    BF = 6.5e-14 * FDEPEN * p * p * f * f * TH**3.6  
    alpha = 1.29 * BF

    return alpha




def abs_o2(TEMP, PRES, VAPDEN, FREQ):

# T: temperature (K)
# p: pressure (mb)
# e: water vapor density [g/m^3]
# f: frequency (GHz) (valid 0-1000 GHz)

    # dtype = np.float32
    dtype = np.float32
    # dtype = 'f'
# Lines are arranged 1-,1+,3-,3+,etc. in spin-rotation spectrum
    F = np.array([   118.7503,  56.2648,  62.4863,  58.4466,  60.3061,  59.5910, 
        59.1642,  60.4348,  58.3239,  61.1506,  57.6125,  61.8002, 
        56.9682,  62.4112,  56.3634,  62.9980,  55.7838,  63.5685, 
        55.2214,  64.1278,  54.6712,  64.6789,  54.1300,  65.2241, 
        53.5957,  65.7648,  53.0669,  66.3021,  52.5424,  66.8368, 
        52.0214,  67.3696,  51.5034,  67.9009, 368.4984, 424.7632, 
       487.2494, 715.3931, 773.8397, 834.1458 ], dtype = dtype)

    S300 = np.array([   .2936E-14, .8079E-15, .2480E-14, .2228E-14, 
           .3351E-14, .3292E-14, .3721E-14, .3891E-14, 
           .3640E-14, .4005E-14, .3227E-14, .3715E-14, 
           .2627E-14, .3156E-14, .1982E-14, .2477E-14, 
           .1391E-14, .1808E-14, .9124E-15, .1230E-14, 
           .5603E-15, .7842E-15, .3228E-15, .4689E-15, 
           .1748E-15, .2632E-15, .8898E-16, .1389E-15, 
           .4264E-16, .6899E-16, .1924E-16, .3229E-16, 
           .8191E-17, .1423E-16, .6494E-15, .7083E-14, 
           .3025E-14, .1835E-14, .1158E-13, .3993E-14 ], dtype = dtype)

    BE = np.array([ .009, .015, .083, .084, .212, .212, .391, .391, .626, .626,         
       .915, .915,  1.260, 1.260, 1.660, 1.665, 2.119, 2.115, 2.624, 2.625,    
       3.194, 3.194, 3.814, 3.814, 4.484, 4.484, 5.224,  5.224, 6.004,  6.004, 6.844, 6.844, 
       7.744, 7.744, .048, .044, .049, .145, .141, .145 ], dtype = 'f')

# Widths in MHz/mb
    W300 = np.array([ 1.63,  1.646, 1.468, 1.449, 1.382, 1.360,        
       1.319, 1.297, 1.266, 1.248, 1.221, 1.207, 1.181, 1.171, 
       1.144, 1.139, 1.110, 1.108, 1.079, 1.078, 1.05, 1.05,     
       1.02, 1.02, 1.00, 1.00, .97, .97, .94, .94, .92, .92, .89, .89, 
       1.64, 1.64, 1.64, 1.81, 1.81, 1.81], dtype = dtype)

    Y300 = np.array([   -0.0233,  0.2408, -0.3486,  0.5227,          
       -0.5430,  0.5877, -0.3970,  0.3237, -0.1348,  0.0311, 
        0.0725, -0.1663,  0.2832, -0.3629,  0.3970, -0.4599, 
        0.4695, -0.5199,  0.5187, -0.5597,  0.5903, -0.6246, 
        0.6656, -0.6942,  0.7086, -0.7325,  0.7348, -0.7546, 
        0.7702, -0.7864,  0.8083, -0.8210,  0.8439, -0.8529, 
        0., 0, 0, 0, 0, 0 ], dtype = dtype)

    V = np.array([ 0.0079, -0.0978, 0.0844, -0.1273,                   
           0.0699, -0.0776, 0.2309, -0.2825, 0.0436, -0.0584, 
           0.6056, -0.6619, 0.6451, -0.6759, 0.6547, -0.6675, 
           0.6135, -0.6139, 0.2952, -0.2895, 0.2654, -0.2590, 
           0.3750, -0.3680, 0.5085, -0.5002, 0.6206, -0.6091, 
           0.6526, -0.6393, 0.6640, -0.6475, 0.6729, -0.6545, 
           0., 0, 0, 0, 0, 0 ], dtype = dtype)


    WB300 = 0.56 
    X = 0.8


    TH = 300 / TEMP  
    TH1 = TH - 1  
    B = TH**X  
    PRESWV = VAPDEN * TEMP / 217  
    PRESDA = PRES - PRESWV  
    DEN = .001 * (PRESDA * B + 1.1 * PRESWV * TH)  
    DENS = .001 * (PRESDA + 1.1 * PRESWV) * TH  
    DFNR = WB300 * DEN  
    SUM = 1.6E-17 * FREQ*FREQ*DFNR / (TH * (FREQ*FREQ + DFNR*DFNR))
    for K in range(0, 40): 
    #    exception for 1- line
        if (K==0):  
            DF = W300[0] * DENS  
            FCEN = F[0] - .14 * DENS  
        else:   
            DF = W300[K] * DEN  
            FCEN = F[K]  
    
        Y = .001 * PRES * B * (Y300[K] + V[K] * TH1)  
        STR = S300[K] * np.exp( - BE[K] * TH1)  
        SF1 = (DF + (FREQ - FCEN) * Y) / ((FREQ - FCEN)**2 + DF * DF)  
        SF2 = (DF - (FREQ + FCEN) * Y) / ((FREQ + FCEN)**2 + DF * DF)  
        SUM = SUM + STR * (SF1 + SF2) * (FREQ / F[K] ) **2


    alpha = 0.5034e12 * SUM * PRESDA * TH**3 / 3.14159  


    return alpha



def abs_h2o(T, P, RHO, F):

    NLINES = 15
    
    # Line Frequencies
    FL = np.array([ 22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508,
           443.0183, 448.0011, 470.8890, 474.6891, 488.4911, 556.9360, 
           620.7008, 752.0332, 916.1712 ])

    # Line intensities at 300K
    S1 = np.array([ .1314E-13, .2279E-11, .8058E-13, .2701E-11, .2444E-10, 
       .2185E-11, .4637E-12, .2568E-10, .8392E-12, .3272E-11, 
       .6676E-12, .1535E-08, .1711E-10, .1014E-08, .4238E-10 ])

    # T coeff. of intensities
    B2 = np.array([  2.144, .668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405, 
                     3.597, 2.379, 2.852, .159, 2.391, .396, 1.441 ])

    # air-broadened width parameters at 300K:
    W3 = np.array([ .00281, .00281, .0023,  .00278, .00287, 
       .0021,  .00186, .00263, .00215, .00236, 
       .0026,  .00321, .00244, .00306, .00267 ])

    # T-exponent of air-broadening:
    X = np.array([ .69, .64, .67, .68, .54, .63, .60, 
       .66, .66, .65, .69, .69, .71, .68, .70 ])

    # self-broadened width parameters at 300K
    WS = np.array([ .01349, .01491, .0108,  .0135,  .01541, 
       .0090,  .00788, .01275, .00983, .01095, 
       .01313, .01320, .01140, .01253, .01275 ])

    # T-exponent of self-broadening:
    XS = np.array([ .61, .85, .54, .74, .89, .52, .50, 
       .67, .65, .64, .72, 1.0, .68, .84, .78 ])


    alpha = 0 

    if RHO>0:
        PVAP = RHO * T / 217
        PDA = P - PVAP
        # const includes isotopic abundance
        DEN = 3.335E16 * RHO
        TI = 300. / T
        TI2 = TI**2.5

        # continuum terms
        CON = (5.43E-10 * PDA * TI**3 + 1.8E-8 * PVAP * TI**7.5) * PVAP*F*F

        # add resonances
        SUM = 0
        DF = np.zeros(2)
        for I in range(0, NLINES):
            WIDTH = W3[I] * PDA * TI**X[I] + WS[I] * PVAP * TI**XS[I]
            WSQ = WIDTH * WIDTH
            S = S1[I] * TI2 * np.exp(B2[I] * (1 - TI))
            DF[0] = F - FL[I]
            DF[1] = F + FL[I]

            #    use clough's definition of local line contribution
            BASE = WIDTH / (562500. + WSQ)
            #    do for positive and negative resonance
            RES = 0
            for J in range(0, 2):
                if np.abs(DF[J]) < 750:
                    RES = RES + WIDTH/(DF[J]**2 + WSQ) - BASE
 
            SUM = SUM + S * RES * (F / FL[I])**2

        alpha = .3183E-4 * DEN * SUM + CON

    return alpha






# f   Input    frequency [GHz]
# T   Input    temperature [K]
# p   Input    pressure {mb]
# wv  Input    water vapor density [g/m^3]


def py_mw_absorb(f, T, p, wv):

  

    K_N2 = abs_n2(T, p, f)
    K_O2 = abs_o2(T, p, wv, f)
    K_WV = abs_h2o(T, p, wv, f)


    Kabs_gas = (K_N2 + K_O2 + K_WV) /1000.

    
    return Kabs_gas, K_N2, K_O2, K_WV
