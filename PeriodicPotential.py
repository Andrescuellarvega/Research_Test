import scipy.constants as constants
import matplotlib.pyplot as plt
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

def Periodic_Coefficient_Matrices(frequency_grid, current_k, N_SQUIDs):

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
    ell = 0.96 * lambda_omega

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

    asub = C_hat[0:2*N_sidebands+1, 0:2*N_sidebands+1]
    bsub = C_hat[0:2*N_sidebands+1, 2*N_sidebands+1:]
    csub = C_hat[2*N_sidebands+1:, 0:2*N_sidebands+1]
    dsub = C_hat[2 * N_sidebands + 1:, 2*N_sidebands+1:]
    dsub_inverse = np.linalg.inv(dsub)

    A_matrix = asub - np.matmul(bsub, np.matmul(dsub_inverse, csub))
    B_matrix = np.matmul(bsub, dsub_inverse)
    C_matrix = - np.matmul(dsub_inverse, csub)
    D_matrix = dsub_inverse

    return A_matrix, B_matrix, C_matrix, D_matrix

# ----------------------------------------------------------------------------------------------------------------------
# LOWER/HIGHER COEFFICIENT CALCULATION, U, V IN MATHEMATICA
# ----------------------------------------------------------------------------------------------------------------------

# Function takes one arguments:
# matrix = coefficient matrix obtained from Periodic_Coefficient_Matrices function (A,B,C,D)
#
# Function calculates squared coefficients Anm for given matrix A, u_aa, v_aa in mathematica

def lower_higher_coefficients(matrix):

    u = np.zeros((2 * N_sidebands + 1))
    v = np.zeros((2 * N_sidebands + 1))

    for m in range(1, 2 * N_sidebands + 2):
        # lower portion, N_omega + 1 in mathematica
        u[m - 1] = np.real(np.conj(matrix[N_sidebands, m - 1]) * matrix[N_sidebands, m - 1])

        # upper portion, N_omega in mathematica
        v[m - 1] = np.real(np.conj(matrix[N_sidebands - 1, m - 1]) * matrix[N_sidebands - 1, m - 1])

    return u, v

# ----------------------------------------------------------------------------------------------------------------------
# SETTING PARAMETERS AND OUTPUT CALCULATION
# ----------------------------------------------------------------------------------------------------------------------

N_sidebands = 4
N_bins = 1000
N_SQUIDs = 100
temperature = 0.025  # In Kelvin
freq_range_low, freq_range_high = (0.48, 0.52)

# ----------------------------------------------------------------------------------------------------------------------
# OUTPUT RADIATION CALCULATION
# ----------------------------------------------------------------------------------------------------------------------


frequency_grid, freq_grid_size = frequencies_array(freq_range_low, freq_range_high, N_bins, N_sidebands)

nout_therm_right_lower = np.zeros(freq_grid_size)
nout_dce_right_lower = np.zeros(freq_grid_size)
nout_therm_right_higher = np.zeros(freq_grid_size)
nout_dce_right_higher = np.zeros(freq_grid_size)
nout_therm_left_lower = np.zeros(freq_grid_size)
nout_dce_left_lower = np.zeros(freq_grid_size)
nout_therm_left_higher = np.zeros(freq_grid_size)
nout_dce_left_higher = np.zeros(freq_grid_size)

sum_right_u1 = np.zeros(2 * N_sidebands + 1)
sum_right_u2 = np.zeros(2 * N_sidebands + 1)
sum_right_v1 = np.zeros(2 * N_sidebands + 1)
sum_right_v2 = np.zeros(2 * N_sidebands + 1)
sum_left_u1 = np.zeros(2 * N_sidebands + 1)
sum_left_u2 = np.zeros(2 * N_sidebands + 1)
sum_left_v1 = np.zeros(2 * N_sidebands + 1)
sum_left_v2 = np.zeros(2 * N_sidebands + 1)


for k in range(0, freq_grid_size):

    A_matrix, B_matrix, C_matrix, D_matrix = Periodic_Coefficient_Matrices(frequency_grid, k, N_SQUIDs)  # Populate coefficients for (0 < omega_k < omega_d),(omega_d < omega_k < 2*omega_d)
    u_A, v_A = lower_higher_coefficients(A_matrix)
    u_B, v_B = lower_higher_coefficients(B_matrix)
    u_C, v_C = lower_higher_coefficients(C_matrix)
    u_D, v_D = lower_higher_coefficients(D_matrix)

    # Populate thermal part of the output
    for m in range(1, 2 * N_sidebands + 2):
        sum_right_u1[m - 1] = n_in(frequency_grid[k, m - 1], temperature) * (u_A[m - 1] + u_B[m - 1])  # adding c and d in mathematica
        sum_right_v1[m - 1] = n_in(frequency_grid[k, m - 1], temperature) * (v_A[m - 1] + v_B[m - 1])  # adding e and f in mathematica

        sum_left_u1[m - 1] = n_in(frequency_grid[k, m - 1], temperature) * (u_C[m - 1] + u_D[m - 1])  # adding c and d in mathematica
        sum_left_v1[m - 1] = n_in(frequency_grid[k, m - 1], temperature) * (v_C[m - 1] + v_D[m - 1])  # adding e and f in mathematica


    nout_therm_right_lower[k] = np.sum(sum_right_u1)
    nout_therm_right_higher[k] = np.sum(sum_right_v1)

    nout_therm_left_lower[k] = np.sum(sum_left_u1)
    nout_therm_left_higher[k] = np.sum(sum_left_v1)

    # Populate DCE part of output
    for m in range(N_sidebands + 2, 2 * N_sidebands + 2):
        sum_right_u2[m - 1] = u_A[m - 1] + u_B[m - 1]  # Runs 5<m<8 which is effectively 6<m<9
        sum_right_v2[m - 1] = v_A[m - 1] + u_B[m - 1]  # Runs 5<m<8 which is effectively 6<m<9

        sum_left_u2[m - 1] = u_A[m - 1] + u_B[m - 1]  # Runs 5<m<8 which is effectively 6<m<9
        sum_left_v2[m - 1] = v_A[m - 1] + u_B[m - 1]  # Runs 5<m<8 which is effectively 6<m<9

    nout_dce_right_lower[k - 1] = np.sum(sum_right_u2)
    nout_dce_right_higher[k - 1] = np.sum(sum_right_v2)

    nout_dce_left_lower[k - 1] = np.sum(sum_left_u2)
    nout_dce_left_higher[k - 1] = np.sum(sum_left_v2)

# ----------------------------------------------------------------------------------------------------------------------
# SYMMETRY TEST
# ----------------------------------------------------------------------------------------------------------------------

zero_reference_array = np.zeros(freq_grid_size)

if (nout_dce_right_lower - nout_dce_left_lower).all() == zero_reference_array.all() and \
        (nout_dce_right_higher - nout_dce_left_higher).all() == zero_reference_array.all():
    print('DCE Radiation is symmetrical')
else:
    print('DCE Radiation is NOT symmetrical')

if (nout_therm_right_lower - nout_therm_left_lower).all() == zero_reference_array.all() and \
        (nout_therm_right_higher - nout_therm_left_higher).all() == zero_reference_array.all():
    print('Thermal Radiation is symmetrical')
else:
    print('Thermal Radiation is NOT symmetrical')
    print('Maximum Difference, lower band: ' + str((nout_therm_right_lower - nout_therm_left_lower).max()))
    print('Maximum Difference, higher band: ' + str((nout_therm_right_higher - nout_therm_left_higher).max()))


# ----------------------------------------------------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------------------------------------------------
freq_grid = frequency_grid[:, N_sidebands] / omega_d


fig = plt.figure(figsize=(10, 4.5))
fig1_name = 'DCE Radiation: ' + str(N_SQUIDs) + ' SQUIDs, '+ str(N_sidebands) + ' sidebands, ' + str(N_bins) + ' bins.'
plt.suptitle(fig1_name, fontsize=16)

plt.subplot(1, 2, 1)
dce_lower, = plt.plot(freq_grid, nout_dce_right_lower, 'b--')
plt.ylabel("$ n_{pout}(\omega)$")
plt.ylim(nout_dce_right_lower.min() - (0.1)*nout_dce_right_lower.max(), (1.1)*nout_dce_right_lower.max())
plt.title('DCE Radiation, Lower Band')

plt.subplot(1, 2, 2)
dce_higher, = plt.plot(freq_grid + 1, nout_dce_right_higher, 'b--')
plt.ylim(nout_dce_right_higher.min() - (0.1)*nout_dce_right_higher.max(), (1.1)*nout_dce_right_higher.max())
plt.title('DCE Radiation, Higher Band')

#plt.savefig(fig1_name)
plt.show()

# These are here just for graph aesthetics
high_exp = -(int(str(nout_therm_right_higher.max())[-2:]))
low_exp = -(int(str(nout_therm_right_lower.max())[-2:]))
low_weight = 10**(int(-low_exp))
high_weight = 10**(int(-high_exp))

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
therm_lower, = plt.plot(freq_grid, low_weight*nout_therm_left_lower, 'r--')
plt.xlabel("$ \omega / \omega_d $")
plt.ylabel("$ n_{pout}(\omega)$ * (e" + str(low_exp) + ')')
plt.ylim(low_weight*(nout_therm_left_lower.min() - 0.1*nout_therm_left_lower.max()), low_weight*1.1*nout_therm_left_lower.max())
plt.title('Left Thermal Radiation, Lower Band')

plt.subplot(2, 2, 2)
therm_right_lower, = plt.plot(freq_grid, low_weight*nout_therm_right_lower, 'r--')
plt.xlabel("$ \omega / \omega_d $")
plt.ylabel("$ n_{pout}(\omega)$ * (e" + str(low_exp) + ')')
plt.ylim(low_weight*(nout_therm_right_lower.min() - (0.1)*nout_therm_right_lower.max()), low_weight*1.1*nout_therm_right_lower.max())
plt.title('Right Thermal Radiation, Lower Band')

plt.subplot(2, 2, 3)
therm_higher, = plt.plot(freq_grid+1, high_weight*nout_therm_left_higher, 'r--')
plt.xlabel("$ \omega / \omega_d $")
plt.ylabel("$ n_{pout}(\omega)$ * (e" + str(high_exp) + ')')
plt.ylim(high_weight*(nout_therm_left_higher.min() - 0.1*nout_therm_left_higher.max()), high_weight*1.1*nout_therm_left_higher.max())
plt.title('Left Thermal Radiation, Higher Band')

plt.subplot(2, 2, 4)
therm_right_higher, = plt.plot(freq_grid+1, high_weight*nout_therm_right_higher, 'r--')
plt.ylabel("$ n_{pout}(\omega)$ * (e" + str(high_exp) + ')')
plt.ylim(high_weight*(nout_therm_right_higher.min() - 0.1*nout_therm_right_higher.max()), high_weight*1.1*nout_therm_right_higher.max())
plt.title('Right Thermal Radiation, Higher Band')


plt.tight_layout()
#plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# NOTES:
#
#   CHECK HOW TO CALCULATE ENERGY BANDS SO I CAN FIGURE OUT WHY THE LIMIT (0.48,0.52), CHECK IF IT DEPENDS ON N_SQUIDS
#       THIS LIMITS ARE WHERE THE EIGENVALUES OF T_hat, C_hat
#
#
#   CHECK HOW TO CALCULATE THAT LATTICE CONSTANT, WHY 0.96 * WAVELENGTH OF SQUID DRIVE?
#   WHY IS THERMAL RADIATION NOT COMPLETELY SYMMETRICAL?
#
#   AFTER RUNNING EMPIRICALLY:
#       * OUTPUT KEEPS FLUCTUATING WITH 1,2,3 SIDEBANDS. STABILIZES AFTER 4.
#       * RADIATION OUTPUT PATTERN SHOWS AND IS REFINED UNTIL ABOUT 10,000 BINS.
#
# IDEAS:
#
#      * HOW WOULD SPECTRUM LOOK IF SQUIDS DIDN'T HAVE THE SAME FREQUENCY BUT RATHER SOME DISTRIBUTION?
#           THIS CAN BE DONE EASILY BY CHANGING THE DRIVE FREQUENCY INTO A FUNCTION OF THE SQUID POSITIONS:
#               DRIVE_FREQ(SQUID_ENUMERATION), HOW DO I ACCOUNT FOR THE CHANGE IN THE LATTICE CONSTANT?
#
#      * ARE WE LOOKING TO PRODUCE A SPECIFIC SHAPE OF RADIATION? JUST MAXIMIZE OUTPUT?
#           TO HAVE A SHARP FREQUENCY DISTRIBUTION OR A BROAD ONE?
#
# ----------------------------------------------------------------------------------------------------------------------

