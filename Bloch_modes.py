import numpy as np
import j_constants
from input_frequencies import *

# Changing constants to new units

ell = j_constants.L0_eff
QQ = 100*j_constants.omega_d*(ell/j_constants.vcpw)  # Unit-less drive frequency , see Units notes
epsilon = ell/j_constants.L0_eff

# Creating grid of input frequencies
N_sidebands = 1  # Number of sidebands used fro calculation
N_bins = 10  # Number of bins dividing (0, 2*N_Omega + 1)
freq_range_low, freq_range_high = (0, 1)  # Bounds for the input frequencies array, divided by drive frequency.

w_input, input_size = input_frequencies(freq_range_low, freq_range_high, N_bins, N_sidebands, j_constants.omega_d)

q_input = w_input*100*(ell/j_constants.vcpw)  # Make inpu frequencies unitless, see Units notes

kk = np.abs(q_input)
print(kk/(QQ))


# From Mathematica "periodic_array_082920", the description of the k-loop goes
# 1) Find M s.t. q_input is in [M* QQ, (M+1)*QQ] ====> here M is the index s.t. kk[M,k]
# 2) Parametrize q values ====> already taken care of in defining input frequencies
# 3) For given loop value k restrict the number of side bands s.t. abs( Cos[q(k,n)] + (epsilon/(2*q(k,n)*Sin[q(k,n)])) < 1
#   for all n (main band corresponds to n = N_sidebands + 1, here N_sidebands since indexes start with zero)/

