# ----------------------------------------------------------------------------------------------------------------------
# FREQUENCIES ARRAY
# ----------------------------------------------------------------------------------------------------------------------

# Function takes four arguments:
# freq_range_low = lower bound of the frequency range examined,
# freq_range_high = upper bound of the frequency range examined,
# N_SQUIDs = Number of SQUIDs connected in series,
# N_sidebands = Number of sidebands used in numerical calculation.
#
# Function returns frequency grid and its dimensions (www in mathematica)

def input_frequencies(freq_range_low, freq_range_high, N_bins, N_sidebands, omega_d):
    import numpy as np

    low_index = int(N_bins * freq_range_low) - 1
    high_index = int(N_bins * freq_range_high) - 1

    freq_grid_size = high_index - low_index + 1

    frequency_grid = np.empty((freq_grid_size, 2 * N_sidebands + 1))
    k_space = np.array(range(low_index, high_index + 1))

    for k in range(0, freq_grid_size):
        for m in range(1, 2 * N_sidebands + 2):
            frequency_grid[k, m-1] = (k_space[k]+1) * (omega_d/N_bins) + (N_sidebands + 1 - m) * omega_d

    return frequency_grid, freq_grid_size
