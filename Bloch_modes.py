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


def bloch_test(q_input_freq, N_bins, N_sidebands):

    kk = q_input_freq
    condition = np.abs((np.cos(kk)) + ((epsilon / (2 * kk)) * np.sin(kk)))
    allowed_freqs = np.empty(np.shape(condition))

    for k in range(0, N_bins-1):

        # Check center band
        # Note: the index for the main band is the same as N_sidebands since indices run from zero
        if condition[k, N_sidebands] < 1:
            # If center band IS allowed, make center value True
            allowed_freqs[k, N_sidebands] = True

            # Do forward and backward checks:

            # Check forward:
            for band in range(N_sidebands + 1, 2 * N_sidebands + 1):
                if condition[k, band] < 1:
                    # If band is allowed, make test True
                    allowed_freqs[k, band] = True
                else:
                    # If band is not allowed, make all FORWARD bands False, and don't check following bands (break loop)
                    allowed_freqs[k, band:] = False
                    # print("Forbidden band:", band - N_sidebands, " for bin:", k)
                    break

            # Check backward:
            # Go from center_band-1 to 0 (doesn't include end) in -1 increments
            for band in range(N_sidebands - 1, -1, -1):
                if condition[k, band] < 1:
                    # If band IS allowed, make test True
                    allowed_freqs[k, band] = True
                else:
                    # If band IS NOT allowed, make all BACKWARD bands False, and don't check previous bands (break loop)
                    allowed_freqs[k, band::-1] = False
                    # print("Forbiden band:", band - N_sidebands, " for bin:", k)
                    break
        else:
            # If center band IS NOT allowed, make test False for every band
            allowed_freqs[k, :] = False
        # print("Forbiden band: 0  for bin:", k)

    return allowed_freqs


# ----------------------------------------------------------------------------------------------------------------------
# DISPERSION RELATION
# ----------------------------------------------------------------------------------------------------------------------

# Returns omega(k) given k(omega) in first approximation. q_input = omega (ell/vcpw) //unitless freq.

# Function takes 2 arguments:
# q_raw = grid of frequencies, no regard for gaps
# allowed_freqs = output from bloch_test


def dispersion_relation_1st(q_k, allowed_freqs, N_bins, N_sidebands):

    # Step 1, find number of gaps (forbidden regions)
    N_gaps = 0
    for bin in range(1, N_bins-1):  # look for endpoints where 0 goes to 1
        if not allowed_freqs[bin-1, N_sidebands] and allowed_freqs[bin, N_sidebands]:
            N_gaps += 1
    if not allowed_freqs[N_bins-2, N_sidebands]:  # check if last value is in a gap
        N_gaps += 1

    print(N_gaps)

    limits = np.zeros((N_gaps, 2), dtype='int')

    # Step 2, find limits for each gap
    gap = 0
    while gap < N_gaps:
        if not allowed_freqs[0, N_sidebands]:  # check if first value is in gap
            gap += 1
            limits[gap-1, 0] = 0

        for bin in range(1, N_bins-1):
            # find where gap begins (1 to 0)
            if allowed_freqs[bin-1, N_sidebands] and not allowed_freqs[bin, N_sidebands]:
                gap += 1
                limits[gap-1, 0] = bin  # lower lim
            # find where gap ends (0 to 1)
            if not allowed_freqs[bin-1, N_sidebands] and allowed_freqs[bin, N_sidebands]:
                limits[gap-1, 1] = bin-1  # upper lim
            # check if last bin is in gap
            if bin == N_bins-2:
                if not allowed_freqs[bin, N_sidebands]:
                    limits[gap-1, 1] = bin
                gap += 1

    # Step 3, find gap width for every gap
    gap_width = np.zeros((N_gaps))
    for gap in range(0, N_gaps):
        gap_width[gap] = q_k[limits[gap, 1], N_sidebands] - q_k[limits[gap, 0], N_sidebands]


    # Step 4, shift frequencies at bandgaps
    for gap in range(0, N_gaps):
        if not gap_width[gap] == 0:
            for bin in range(limits[gap, 0], N_bins-1):
                #print('gap', gap, 'bin', bin)
                #print('before', q_input[bin, N_sidebands])
                q_k[bin, N_sidebands] = q_k[bin, N_sidebands] + gap_width[gap]
                #print('after', q_input[bin, N_sidebands],q_input[bin, N_sidebands] - gap_width[gap])

    return q_k, limits


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

# ----------------------------------------------------------------------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------------------------------------------------------------------



import numpy as np
import j_constants
from input_frequencies import input_frequencies
import matplotlib.pyplot as plt


# Changing constants to new units
ell = j_constants.L0_eff
QQ = 100 * j_constants.omega_d * (ell / j_constants.vcpw)  # Unit-less drive freq print(sideband frequency ,
# see Units notes
epsilon = ell / j_constants.L0_eff
j = np.complex(0, 1)

# Creating grid of input frequencies
N_sidebands = 0  # Number of sidebands used for calculation
N_bins = 100000  # Number of bins dividing (0, 2*N_Omega + 1)
freq_range_low, freq_range_high = (0, 1)  # Bounds for the input frequencies array, divided by drive frequency.

# To make q_input with unitless freq, pass QQ as drive freq to the input_frequencies function.

q_input, input_shape = input_frequencies(False, freq_range_low, freq_range_high, N_bins, N_sidebands, QQ)
# q = unitless omega
kk = np.abs(q_input) / j_constants.c  # Dispersion relation, here simplified expression.

allowed_freqs = bloch_test(q_input, N_bins, N_sidebands)

q_k, limits = dispersion_relation_1st(q_input, allowed_freqs, N_bins, N_sidebands)

if q_input[N_bins-2,N_sidebands] == q_k[N_bins-2, N_sidebands]:
    print('help')
else:
    print('you did it, the absolute madman')
    print(q_k-q_input)

#A_k = np.zeros((N_bins-1, 2*N_sidebands + 1), dtype='complex')

#for bin in range(0, N_bins-1):
#    for band in range(0, 2*N_sidebands + 1):
#        if allowed_bands[bin, band]:
#            C_k, D_k = bloch_modes(kk[bin, band], epsilon)
#            A_k[bin, band] = j*kk[bin, band] *C_k + D_k
#        elif not allowed_bands[bin, band]:
#            A_k[bin,band] = np.nan

#main_band = np.zeros((N_bins-1), dtype='complex')

#for bin in range(0, N_bins-1):
#    main_band[bin] = allowed_bands[bin, N_sidebands]*kk[bin, N_sidebands]
#    if main_band[bin] == 0:
#       main_band[bin] = np.nan

fig = plt.figure(figsize=(10, 4.5))

plt.subplot()
plt.plot(kk[:,N_sidebands], q_k[:,N_sidebands], '*')
#plt.plot(-kk[:,N_sidebands], q_k[:,N_sidebands], '*')
plt.vlines(kk[limits[1:,0], N_sidebands], ymin=0, ymax=q_k[-1,N_sidebands], linestyles='dashed')
#plt.vlines(-kk[limits[1:,0], N_sidebands], ymin=0, ymax=q_k[-1,N_sidebands], linestyles='dashed')

plt.ylabel("$\omega$")



plt.show()
