# ----------------------------------------------------------------------------------------------------------------------
# BLOCH TEST
# ----------------------------------------------------------------------------------------------------------------------

# Performs test to determine allowed and forbidden frequencies for bloch modes in the KP model.
# Returns allowed_frequencies array, which is bool and hold information about whether corresponding element in
# input frequency kk array is allowed or forbidden

# Function takes 3 arguments:
# input_freq = input frequencies kk (unitless frequency, np.abs of the output of input_frequencies function)
# N_sidebands = Number of sidebands used in numerical calculation.
# N_bins = Number of bins to break up the frequency range into.


def bloch_test(input_freq, N_bins, N_sidebands):

    kk = input_freq
    condition = np.abs((np.cos(kk)) + ((epsilon / (2 * kk)) * np.sin(kk)))
    allowed_bands = np.empty(np.shape(condition))

    for k in range(0, N_bins-1):

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
                    # print("Forbidden band:", band - N_sidebands, " for bin:", k)
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

    return allowed_bands


# ----------------------------------------------------------------------------------------------------------------------
# BLOCH COEFFICIENTS
# ----------------------------------------------------------------------------------------------------------------------

# Calculates C_k and D_k coefficient for input frequency q = kk[bin,band]
# Returns coefficients C_k and D_k for a given frequency.

# Function takes 2 arguments:
# individual_input_freq =  q = kk[bin,band]
# epsilon = parameter epsilon (see Units notes), contains information about potential.


def bloch_modes(individual_input_freq, epsilon):

    q = individual_input_freq
    qqq = np.cos(q) + (epsilon / (2*q)) * np.sin(q)
    www = -np.cos(q) + ((2*q) / epsilon) * np.sin(q)

    vv1 = np.array([np.exp(j * q) * (www + ((2 * q)/epsilon)*np.sqrt(1-qqq**2)), 1])
    #vv2 = np.array([np.exp(j * q) * (www - ((2 * q)/epsilon)*np.sqrt(1-qqq**2)), 1])

    v1norm = vv1/np.linalg.norm(vv1)
    #v2norm = np.exp(-j * q) * (vv2/np.linalg.norm(vv2))

    kkk = np.arccos(qqq)
    x = 0
    u1raw = ((v1norm[0] * np.exp(j*(q-kkk)*(x-1))) + (v1norm[1] * np.exp(-j*(q+kkk)*(x-1))))/(v1norm[0] + v1norm[1])
    norm1 = (1/np.abs(v1norm[0] + v1norm[1])) * np.sqrt(
                    np.abs(v1norm[0])**2 + np.abs(v1norm[1])**2 + \
                    (v1norm[0]*np.conj(v1norm[1])*(1/(2*j*q))*(1 - np.exp(-j*2*q))) + \
                     (v1norm[1]*np.conj(v1norm[0])*(1/(-2*j*q))*(1 - np.exp(j*2*q))))
    u1 = u1raw/norm1
    u1prime = (1/(v1norm[0] + v1norm[1]))*(v1norm[0]*np.exp(-j*(q-kkk))*(j*(q-kkk)) + v1norm[1]*np.exp(j*(q+kkk))*((-j)*(q+kkk)))/norm1
    C_k = u1
    D_k = u1prime

    return(C_k, D_k)


import numpy as np
import j_constants
from input_frequencies import input_frequencies

# Changing constants to new units
ell = j_constants.L0_eff
QQ = 100 * j_constants.omega_d * (ell / j_constants.vcpw)  # Unit-less drive freq print(sideband frequency , see Units notes
epsilon = ell / j_constants.L0_eff
j = np.complex(0, 1)

# Creating grid of input frequencies
N_sidebands = 1  # Number of sidebands used for calculation
N_bins = 10  # Number of bins dividing (0, 2*N_Omega + 1)
freq_range_low, freq_range_high = (0, 1)  # Bounds for the input frequencies array, divided by drive frequency.

# To make q_input with unitless freq, pass QQ as drive freq to the input_frequencies function

q_input, input_shape = input_frequencies(False, freq_range_low, freq_range_high, N_bins, N_sidebands, QQ)
kk = np.abs(q_input)

allowed_bands = bloch_test(q_input, N_bins,N_sidebands)


for band in range(0, 2*N_sidebands + 1):
    C_k, D_k = bloch_modes(kk[4,band], epsilon)
    print("C_k ", C_k, " D_k ", D_k)

