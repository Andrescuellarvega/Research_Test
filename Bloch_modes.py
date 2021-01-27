import numpy as np
import j_constants
from input_frequencies import *

# Changing constants to new units

ell = j_constants.L0_eff
QQ = 100*j_constants.omega_d*(ell/j_constants.vcpw)  # Unitless drive frequency , see Units notes
epsilon = ell/j_constants.L0_eff

# creating grid of input frequencies
N_sidebands = 2  # Number of sidebands used fro calculation
N_bins = 10 # Number of bins dividing (0, 2*N_Omega + 1)
freq_range_low, freq_range_high = (0.4802, 0.5194)

frequency_grid, freq_grid_size = input_frequencies(freq_range_low, freq_range_high, N_bins, N_sidebands, j_constants.omega_d)
print(freq_grid_size)


#for (0 < omega_k < omega_d), (omega_d < omega_k < 2 * omega_d)
for k in range(0, freq_grid_size):
    ww = frequency_grid[current_k, :]
for m in range(1, 2*N_sidebands+2):
    ww = frequency_grid[current_k, :]


def ww(k, n):
    ww = k*QQ/n_d + (n_omega + 1 - n) * QQ
    return ww


