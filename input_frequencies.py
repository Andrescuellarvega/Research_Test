# ----------------------------------------------------------------------------------------------------------------------
# INPUT FREQUENCIES
# ----------------------------------------------------------------------------------------------------------------------

# Function takes six arguments:
# endpoint = bool, whether or not to include endpoint in array (omega_d)
# freq_range_low = lower bound of the frequency range examined (from zero to omega_d),
# freq_range_high = upper bound of the frequency range examined (from zero to omega_d),
# N_bins = Number of bins to break up the frequency range into.
# N_sidebands = Number of sidebands used in numerical calculation.
# omega_d = drive frequency (To make q_input unitles, pass QQ as input.
#
# Function returns frequency grid (ww in mathematica).
# frequency grid has columns for each sideband (N_sidebands to - N_sidebands) and rows for frequencies


def input_frequencies(endpoint, freq_range_low, freq_range_high, N_bins, N_sidebands, omega_d):
    import numpy as np

    low_index = int(N_bins * freq_range_low) - 1
    high_index = int(N_bins * freq_range_high) - 1

    freq_grid_size = high_index - low_index + 1
    freq_grid_shape = (freq_grid_size, 2*N_sidebands + 1)

    frequency_grid = np.empty(freq_grid_shape)
    k_space = np.array(range(low_index, high_index + 1))

    for k in range(0, freq_grid_size):
        for m in range(1, 2*N_sidebands + 2):
            frequency_grid[k, m-1] = (k_space[k]+1) * (omega_d/N_bins) + (N_sidebands + 1 - m) * omega_d

    if not endpoints:
        frequency_grid = frequency_grid[1:-1, :]
        freq_grid_shape = np.shape(frequency_grid)

    return frequency_grid, freq_grid_shape
