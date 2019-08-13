import scipy.constants as constants
import numpy as np

#  ---------------------------------------------------------------------------------------------------------------------
#  CONSTANT DEFINITIONS
#  ---------------------------------------------------------------------------------------------------------------------

hp = constants.h  # Planck constant, (Js)
hbar = constants.hbar  # Reduced Planck constant hp/2*pi (Js)
kb = constants.Boltzmann  # Boltzmann constant (J/K)
ec = constants.elementary_charge  # Elementary charge, (Coulombs)
phi0 = hp/(2 * ec)  # Magnetic flux quantum (Js/C=J/A)

# System Constants
ic = 1.25 * 10 ** (-6)  # Critical current of Josephson junctions in SQUID (A)
vcpw = 1.2*10**8  # Propagation velocity in CPW  (m/s), fig4
z0 = 55  # Characteristic impedance of CPW (Ohms), fig4
E_j = (ic * phi0) / (2 * constants.pi)  # Josephson energy of SQUID, fig4
E0_j = 1.3 * E_j  # Tunable Josephson energy for static applied magnetic flux, fig4
dE_j = E0_j / 4  # Amplitude of tunable Josephson energy for weak harmonic drive, fig4
L0 = z0 / vcpw  # Inductance of CPW per unit length,  (henry/m =V*s/A*m)
L0_eff = ((phi0 / (2 * constants.pi)) ** 2) * (1 / (E0_j * L0))  # Effective length (m), Eq.(18)
dL_eff = L0_eff / 4  # Modulation amplitude of effective length (m)
omega_d = (2*constants.pi*18.6)*10**9  # Drive angular frequency, where omega_d = 2*pi*f, f= 18.6 GHz, fig4

#  ---------------------------------------------------------------------------------------------------------------------
#  FUNCTION DEFINITIONS
#  ---------------------------------------------------------------------------------------------------------------------


def delta(x, y):
    if x == y:
        d = 1
    else:
        d = 0
    return d

def g(ww, n, m):
    g = delta(n, m) + 0.5 * (dE_j / E0_j) * np.sqrt(np.abs(ww[m - 1] / ww[n - 1])) * \
        (delta(n, m + 1) + delta(n, m - 1))
    return g

def n_in(frequency, T):
    n_in = 1 / (np.exp(hbar * np.abs(frequency) / (kb * T)) - 1)
    return n_in

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

def frequencies_array(freq_range_low, freq_range_high, N_bins, N_sidebands):

    low_index = int(N_bins * freq_range_low) - 1
    high_index = int(N_bins * freq_range_high) - 1

    freq_grid_size = high_index - low_index + 1

    frequency_grid = np.empty((freq_grid_size, 2 * N_sidebands + 1))
    k_space = np.array(range(low_index, high_index + 1))

    for k in range(0, freq_grid_size):
        for m in range(1, 2 * N_sidebands + 2):
            frequency_grid[k, m-1] = (k_space[k]+1) * (omega_d/N_bins) + (N_sidebands + 1 - m) * omega_d

    return frequency_grid, freq_grid_size

# ----------------------------------------------------------------------------------------------------------------------
#   PERIODIC COEFFICIENT MATRICES
# ----------------------------------------------------------------------------------------------------------------------

# Function takes three arguments:
# frequency_grid = Array with range of frequencies in different bands obtained from frequencies_array,
# current_k = Current bin index for which matrices are calculated,
# N_SQUIDs = Number of SQUIDs connected in series.
#
# Function returns coefficient matrices A, B, C, D from TR notes

def Periodic_Energy_Bands(frequency_grid, current_k, N_SQUIDs):

    ww = frequency_grid[current_k, :]

    # Create and Populate G (bigg) which are parts of the S_hat matrix (transfer matrix for semi-transparent SQUID)
    G_matrix = np.zeros((2 * N_sidebands + 1, 2 * N_sidebands + 1), dtype='complex')

    for m in range(1, 2*N_sidebands+2):
        for n in range(1, 2*N_sidebands+2):
            G_matrix[m-1, n-1] = (1/2) * np.complex(0,1) * (vcpw/(ww[m - 1]))*(1/L0_eff) * g(ww, n, m)

    # Identity matrix, (needed to define S_hat matrix)
    one_matrix = np.identity(2*N_sidebands+1, dtype='complex')

    # Transfer matrix for a semi-transparent SQUID
    S_hat = np.concatenate((np.concatenate((one_matrix - G_matrix, -G_matrix), axis=1),
                            np.concatenate((G_matrix, one_matrix + G_matrix), axis=1)), axis=0)

    # Wavelength associated with drive frequency omega_d in meters
    lambda_omega = 2 * constants.pi * vcpw / omega_d

    # lattice constant (distance between SQUIDs); with this choice, main and the side bands are in allowed energy
    # band for omega in [0.48, 0.52] omega_d
    ell = 0.96*lambda_omega

    # P matrix, parts of propagation matrices between SQUIDs (l = ell)
    P_matrix =  np.zeros((2 * N_sidebands + 1, 2 * N_sidebands + 1), dtype='complex')
    for n in range(1, 2 * N_sidebands + 2):
        for m in range(1, 2 * N_sidebands + 2):
            P_matrix[n-1, m-1] = np.exp(np.complex(0,1)*(ww[n-1]/vcpw)*ell)*delta(n, m)

    # Propagation matrices between SQUIDs
    zero_mat = np.zeros((2 * N_sidebands + 1, 2 * N_sidebands + 1), dtype='complex')
    P_hat = np.concatenate((np.concatenate((P_matrix, zero_mat), axis=1),
                np.concatenate((zero_mat, np.conj(P_matrix)), axis=1)), axis=0)

    T_hat = np.matmul(P_hat, S_hat)
    C_hat = np.matmul(S_hat, np.linalg.matrix_power(T_hat, N_SQUIDs))

    eigenvals = np.around(np.conj(np.linalg.eig(C_hat)[0])*np.linalg.eig(C_hat)[0])

    return np.array_equal(eigenvals, np.ones(np.shape(eigenvals)))
# ----------------------------------------------------------------------------------------------------------------------
# SETTING PARAMETERS AND OUTPUT CALCULATION
# ----------------------------------------------------------------------------------------------------------------------

N_sidebands = 4
N_bins = 10000
N_SQUIDs = 100

freq_range_low, freq_range_high = (1/N_bins, (1-(1/N_bins)))

frequency_grid, freq_grid_size = frequencies_array(freq_range_low, freq_range_high, N_bins, N_sidebands)

allowed_bins = np.zeros(freq_grid_size)


for k in range(0, freq_grid_size):

    allowed_bins[k] = Periodic_Energy_Bands(frequency_grid, k, N_SQUIDs)

non_zeros = np.flatnonzero(allowed_bins)
if non_zeros.size == 0:
    print("No allowed energies")
else:
    bin_limits = (non_zeros[0], non_zeros[-1])
    freq_limits = ((bin_limits[0]+1)/N_bins, (bin_limits[1]+1)/N_bins)
    bins_within_bounds = (bin_limits[1] - bin_limits[0])
    unallowed_within_bounds = bins_within_bounds - non_zeros.size
    print("N_SQUIDs = " + str(N_SQUIDs) + "\nfrequency limits = (" + str(freq_limits[0]) + ", " + str(freq_limits[1])\
          + ")\n# of bins within bounds = " + str(bins_within_bounds) + "\nunallowed frequencies within bounds = "\
          + str(unallowed_within_bounds))



