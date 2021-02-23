import numpy as np
import j_constants
from input_frequencies import *
import time

# Changing constants to new units

ell = j_constants.L0_eff
QQ = 100*j_constants.omega_d*(ell/j_constants.vcpw)  # Unit-less drive freq print(sidebauency , see Units notes
epsilon = ell/j_constants.L0_eff


# Creating grid of input frequencies
N_sidebands = 3  # Number of sidebands used fro calculation
N_bins = 100  # Number of bins dividing (0, 2*N_Omega + 1)
freq_range_low, freq_range_high = (0, 1)  # Bounds for the input frequencies array, divided by drive frequency.

w_input, input_shape = input_frequencies(False, freq_range_low, freq_range_high, N_bins, N_sidebands, j_constants.omega_d)

q_input = w_input*100*(ell/j_constants.vcpw)  # Make input

# frequencies unitless, see Units notes

kk = np.abs(q_input)


# From Mathematica "periodic_array_082920", the description of the k-loop goes
# 1) Find M s.t. q_input is in [M* QQ, (M+1)*QQ] ====> here M is the index s.t. kk[M,k]
# 2) Parametrize q values ====> already taken care of in defining input frequencies
# 3) For given loop value k restrict the number of side bands s.t. abs( Cos[q(k,n)] + (epsilon/(2*q(k,n)*Sin[q(k,n)])) < 1
#   for all n (main band corresponds to n = N_sidebands + 1, here N_sidebands since indexes start with zero)/

# ---------------------------------------------------------------------------
# USE CONDITIONALS TO MINIMIZE CHECKING FOR CONDITIONS, WRITES TEST ARRAY WITHOUT CHECKING UNNECESSARILY
# ---------------------------------------------------------------------------
t1 = time.perf_counter()
condition = np.abs((np.cos(kk)) + ((epsilon / (2 * kk)) * np.sin(kk)))
test2 = np.empty(np.shape(condition))

for k in range(0, input_shape[0]):

    # Check center band
    # Note: the index for the main band is the same as N_sidebands since indices run from zero
    if condition[k, N_sidebands] < 1:
        # If center band IS allowed, make center value True
        test2[k, N_sidebands] = True

        # Do forward and backward checks:

        # Check forward:
        for band in range(N_sidebands + 1, 2 * N_sidebands + 1):
            if condition[k, band] < 1:
                # If band is allowed, make test True
                test2[k, band] = True
            else:
                # If band is not allowed, make all FORWARD bands False, and don't check following bands (break loop)
                test2[k, band:] = False
                # print("Forbiden band:", band - N_sidebands, " for bin:", k)
                break

        # Check backward:
        # Go from center_band-1 to 0 (doesn't include end) in -1 increments
        for band in range(N_sidebands - 1, -1, -1):
            if condition[k, band] < 1:
                # If band IS allowed, make test True
                test2[k, band] = True
            else:
                # If band IS NOT allowed, make all BACKWARD bands False, and don't check previous bands (break loop)
                test2[k, band::-1] = False
                # print("Forbiden band:", band - N_sidebands, " for bin:", k)
                break
    else:
        # If center band IS NOT allowed, make test False for every band
        test2[k, :] = False
    # print("Forbiden band: 0  for bin:", k)
t2 = time.perf_counter()

lesschecks = t2 - t1



# ------------------------------------------------------------------------
# PERFORM CHECK FOR EVERY ALLOWED FREQUENCY, THEN MODIFY TEST ARRAY TO TAKE OUT FORBIDDEN FREQUENCIES OUTWARDS
# ------------------------------------------------------------------------
t3 = time.perf_counter()
condition = np.abs((np.cos(kk)) + ((epsilon / (2 * kk)) * np.sin(kk)))
test = np.empty(np.shape(condition))
# Note: the index for the main band is the same as N_sidebands since indices run from zero
for k in range(0, input_shape[0]):
    for band in range(0, 2 * N_sidebands + 1):
        if condition[k, band] < 1:
            # print("bin", k,"band:", band, "test passed")
            test[k, band] = True
        else:
            # print("bin", k,"band:", band, "test failed")
            test[k, band] = False

for k in range(0, input_shape[0]):
    # Check center band
    if test[k, N_sidebands] == 0:
        # If center band is not allowed, don't consider band at all
        test[k, :] = 0
    else:
        # Check forward
        for band in range(N_sidebands + 1, 2 * N_sidebands + 1):
            if test[k, band] == False:
                test[k, band:] = False
                # print(k,band, test[k,:])
                break
        # Check backward, go from center-1 to 0 (doesn't include end) in -1 increments
        for band in range(N_sidebands - 1, -1, -1):
            if test[k, band] == False:
                test[k, band::-1] = False
                break

t4 = time.perf_counter()

morechecks = t4-t3

# ---------------------------------------------------------------------------
# CHECK THAT BOTH KINDS TESTING ARE EQUIVALENT
# ---------------------------------------------------------------------------

print("ratio more_checks/less_checks =", morechecks/lesschecks)

if test.all() == test2.all():
    print("TESTING EQUIVALENT")
else:
    print("TESTING NOT EQUIVALENT")






