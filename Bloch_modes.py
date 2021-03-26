import numpy as np
import j_constants
from input_frequencies import *
import time

# Changing constants to new units

ell = j_constants.L0_eff
QQ = 100*j_constants.omega_d*(ell/j_constants.vcpw)  # Unit-less drive freq print(sidebauency , see Units notes
epsilon = ell/j_constants.L0_eff
j = np.complex(0,1)

# Creating grid of input frequencies
N_sidebands = 1  # Number of sidebands used for calculation
N_bins = 10  # Number of bins dividing (0, 2*N_Omega + 1)
freq_range_low, freq_range_high = (0, 1)  # Bounds for the input frequencies array, divided by drive frequency.

# To make q_input with unitless freq, pass QQ as drive freq to the input_frequencies function

q_input, input_shape = input_frequencies(False, freq_range_low, freq_range_high, N_bins, N_sidebands, QQ)

kk = np.abs(q_input)


# From Mathematica "periodic_array_082920", the description of the k-loop goes
# 1) Find M s.t. q_input is in [M* QQ, (M+1)*QQ] ====> here M is the index s.t. kk[M,k]
# 2) Parametrize q values ====> already taken care of in defining input frequencies
# 3) For given loop value k restrict the number of side bands s.t. abs( Cos[q(k,n)] + (epsilon/(2*q(k,n)*Sin[q(k,n)])) < 1
#   for all n (main band corresponds to n = N_sidebands + 1, here N_sidebands since indexes start with zero)/

# ---------------------------------------------------------------------------
# USE CONDITIONALS TO MINIMIZE CHECKING FOR CONDITIONS
# ---------------------------------------------------------------------------

condition = np.abs((np.cos(kk)) + ((epsilon / (2 * kk)) * np.sin(kk)))
allowed_bands = np.empty(np.shape(condition))

for k in range(0, input_shape[0]):

    # Check center band
    # Note: the index for the main band is the same as N_sidebands since indices run from zero
    if condition[k, N_sidebands] < 1:
        # If center band IS allowed, make center value True
        allowed_bands[k, N_sidebands] = True

        # Do forward and backward checks:

        # Check forward:
        for band in range(N_sidebands + 1, 2 * N_sidebands + 1):
            if condition[k, band] < 1:
                # If band is allowed, make test True
                allowed_bands[k, band] = True
            else:
                # If band is not allowed, make all FORWARD bands False, and don't check following bands (break loop)
                allowed_bands[k, band:] = False
                # print("Forbiden band:", band - N_sidebands, " for bin:", k)
                break

        # Check backward:
        # Go from center_band-1 to 0 (doesn't include end) in -1 increments
        for band in range(N_sidebands - 1, -1, -1):
            if condition[k, band] < 1:
                # If band IS allowed, make test True
                allowed_bands[k, band] = True
            else:
                # If band IS NOT allowed, make all BACKWARD bands False, and don't check previous bands (break loop)
                allowed_bands[k, band::-1] = False
                # print("Forbiden band:", band - N_sidebands, " for bin:", k)
                break
    else:
        # If center band IS NOT allowed, make test False for every band
        allowed_bands[k, :] = False
    # print("Forbiden band: 0  for bin:", k)



#print(allowed_bands)

#  We now move to calculating the bloch modes

#  Shorthand notations

def bloch_modes(q,epsilon):

    qqq = np.cos(q) + (epsilon / (2*q)) * np.sin(q)

    #print("qqq ", qqq)
    www = -np.cos(q) + ((2*q) / epsilon) * np.sin(q)
    #print("www ", www)

    vv1 = np.array([np.exp(j * q) * (www + ((2 * q)/epsilon)*np.sqrt(1-qqq**2)), 1])
    vv2 = np.array([np.exp(j * q) * (www - ((2 * q)/epsilon)*np.sqrt(1-qqq**2)), 1])

    v1norm = vv1/np.linalg.norm(vv1)
    #print("v1norm ", v1norm)
    #v2norm = np.exp(-j * q) * (vv2/np.linalg.norm(vv2))

    kkk = np.arccos(qqq)
    x = 0
    u1raw = ((v1norm[0] * np.exp(j*(q-kkk)*(x-1))) + (v1norm[1] * np.exp(-j*(q+kkk)*(x-1))))/(v1norm[0] + v1norm[1])
    #print("u1raw", u1raw)

    norm1 = (1/np.abs(v1norm[0] + v1norm[1])) * np.sqrt(
                    np.abs(v1norm[0])**2 + np.abs(v1norm[1])**2 + \
                    (v1norm[0]*np.conj(v1norm[1])*(1/(2*j*q))*(1 - np.exp(-j*2*q))) + \
                     (v1norm[1]*np.conj(v1norm[0])*(1/(-2*j*q))*(1 - np.exp(j*2*q))))

    u1 = u1raw/norm1
    #print("u1", u1)
    #print("u1 ", u1)
    
    ##up to u1 the results are the same as in Mathematica file, u1prime is still not the same, need to check for inconsistencies.
    u1prime = ((v1norm[0]*np.exp(-j*(q-kkk))*(j*(q-kkk))) + (v1norm[1]*np.exp(j*(q+kkk))*((-j)*(q+kkk))))/(np.abs(v1norm[0]+v1norm[1])*norm1)
    print("u1prime ", u1prime)
    C_k = u1
    D_k = u1prime

    return(C_k, D_k)

print(kk[5,:], allowed_bands[0,:])
#available = q_input[111, 3] * allowed_bands[111, 3]

for band in range(0,3):

    C_k, D_k = bloch_modes(kk[5,band], epsilon)
    #print("C_k ", C_k, " D_k ", D_k)

#print(bloch_test)