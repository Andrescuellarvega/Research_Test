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
# ANALYTICAL FUNCTION DEFINITIONS
#  ---------------------------------------------------------------------------------------------------------------------

# Function takes two arguments:
# N_bins = number of bins to partition the frequency range into,
# temp = temperature for the thermal output

# Function returns three arrays: (dce, thermal, total)


def n_out_analytic(N_bins, temp):

    omega = np.linspace(omega_d/N_bins, 2 * omega_d, N_bins, endpoint=False)

    def n_in(omega, temp):
        if temp == 0:
            n_in = 0  # Zero temperature state
        else:
            with np.errstate(divide='ignore'):  # Ignore the error when dividing by zero
                n_in = np.ma.masked_invalid \
                    (1 / (np.exp((hbar * omega) / (kb * temp)) - 1))  # Thermal initial state
            return n_in

    n_out_dce = ((dL_eff / vcpw) ** 2) * omega * (omega_d - omega) * np.heaviside(omega_d - omega, 1)

    n_out_thermal = n_in(omega, temp) + ((dL_eff / vcpw) ** 2) \
                    * (omega * np.abs(omega - omega_d)) * n_in(np.abs(omega - omega_d), temp)

    n_out_total = n_out_dce + n_out_thermal

    return n_out_dce, n_out_thermal, n_out_total


#  ---------------------------------------------------------------------------------------------------------------------
# NUMERICAL FUNCTION DEFINITIONS
#  ---------------------------------------------------------------------------------------------------------------------

# Function takes three arguments:
# N_sidebands = number of sidebands to consider for calculation,
# N_bins = number of bins to partition the frequency range into,
# temp = temperature for the thermal output

# Function returns two arrays: (

def n_out_numerical(N_sidebands, N_bins, temp):

    def delta(x, y):
        if x == y:
            d = 1
        else:
            d = 0
        return d

    def g(ww, n, m):
        g = delta(n, m) + 0.5 * (dE_j / E0_j) * np.sqrt(np.abs(ww[m - 1] / ww[n - 1])) * (
                    delta(n, m + 1) + delta(n, m - 1))
        return g

    def coefficients(frequency_grid, k):

        ww = frequency_grid[k - 1, :]  # Index runs 0 < k < N_bins-2, effective 1 < k < N_bins-1
        m_out = np.empty((2 * N_sidebands + 1, 2 * N_sidebands + 1), dtype=complex)
        m_in = np.empty((2 * N_sidebands + 1, 2 * N_sidebands + 1), dtype=complex)

        # Populate M_out, M_in matrices
        for m in range(1, 2 * N_sidebands + 2):  # Indexes run 0 < m,n < 8, effectively 1 < m,n < 9
            for n in range(1, 2 * N_sidebands + 2):
                m_out[m - 1, n - 1] = -g(ww, n, m) + np.complex(0, 1) * (ww[m - 1] / vcpw) * L0_eff * delta(n, m)
                m_in[m - 1, n - 1] = g(ww, n, m) + np.complex(0, 1) * (ww[m - 1] / vcpw) * L0_eff * delta(n, m)

        a_matrix = np.matmul(np.linalg.inv(m_out), m_in)

        u = np.zeros((2 * N_sidebands + 1))
        v = np.zeros((2 * N_sidebands + 1))

        for m in range(1, 2 * N_sidebands + 2):
            # upper portion, N_omega in mathematica
            v[m - 1] = np.real(np.conj(a_matrix[N_sidebands - 1, m - 1]) * a_matrix[N_sidebands - 1, m - 1])

            # lower portion, N_omega + 1 in mathematica
            u[m - 1] = np.real(np.conj(a_matrix[N_sidebands, m - 1]) * a_matrix[N_sidebands, m - 1])

        return u, v

    def n_in(frequency, T):
        n_in = 1 / (np.exp(hbar * np.abs(frequency) / (kb * T)) - 1)
        return n_in

    #  -----------------------------------------------------------------------------------------------------------------
    # ACTUAL CALCULATION
    #  -----------------------------------------------------------------------------------------------------------------

    www = np.empty((N_bins - 1, 2 * N_sidebands + 1))       # Populate www, N_bins - 1 to avoid singularities

    for k in range(1, N_bins):
        for m in range(1, 2 * N_sidebands + 2):             # m = 1,...,2*N_omega+1,  9 elements for 4 sidebands
            www[k-1, m-1] = k * (omega_d/N_bins) + (N_sidebands + 1 - m) * omega_d


    noutthermnum_lower = np.zeros(N_bins - 1)
    noutdcenum_lower = np.zeros(N_bins - 1)
    noutthermnum_upper = np.zeros(N_bins - 1)
    noutdcenum_upper = np.zeros(N_bins - 1)

    sum_array_u1 = np.zeros(2 * N_sidebands + 1)
    sum_array_u2 = np.zeros(2 * N_sidebands + 1)
    sum_array_v1 = np.zeros(2 * N_sidebands + 1)
    sum_array_v2 = np.zeros(2 * N_sidebands + 1)

    for k in range(1, N_bins):  # 1 < k < 99

        u, v = coefficients(www, k)  # Populate coefficients for (0 < omega_k < omega_d),(omega_d < omega_k < 2*omega_d)

        for m in range(1, 2 * N_sidebands + 2):                               # Populate thermal part of the output
            sum_array_u1[m-1] = n_in(www[k - 1, m-1], temp) * u[m-1]    # Runs 0<m<8, effectively 1<m<9
            sum_array_v1[m-1] = n_in(www[k - 1, m-1], temp) * v[m-1]    # Runs 0<m<8, effectively 1<m<9

        noutthermnum_lower[k - 1] = np.sum(sum_array_u1)
        noutthermnum_upper[k - 1] = np.sum(sum_array_v1)

        for m in range(N_sidebands + 2, 2 * N_sidebands + 2):     # Populate DCE part of output
            sum_array_u2[m-1] = u[m-1]              # Runs 5<m<8 which is effectively 6<m<9
            sum_array_v2[m-1] = v[m-1]              # Runs 5<m<8 which is effectively 6<m<9

        noutdcenum_lower[k-1] = np.sum(sum_array_u2)
        noutdcenum_upper[k-1] = np.sum(sum_array_v2)

    # Joining upper and lower arrays into single
    nouttotalnum = np.concatenate((noutdcenum_lower + noutthermnum_lower, noutdcenum_upper + noutthermnum_upper))
    frequency_grid_num = np.concatenate((www[:, N_sidebands] / omega_d, www[:, N_sidebands - 1] / omega_d))

    return frequency_grid_num, nouttotalnum


#  ---------------------------------------------------------------------------------------------------------------------
# ANALYTICAL RUN
#  ---------------------------------------------------------------------------------------------------------------------

N_bins = 500
temperature1 = 0.025 # In Kelvin
temperature2 = 0.05

# For 25 mK
(n_an_dce, n_an_thermal25, n_an_total25) = n_out_analytic(N_bins, temperature1)
# For 50 mK
(n_an_dce, n_an_thermal50, n_an_total50) = n_out_analytic(N_bins, temperature2)

frequency_grid_an = np.linspace(omega_d / N_bins, 2 * omega_d, N_bins, endpoint=False)/omega_d  # For plotting


#  ---------------------------------------------------------------------------------------------------------------------
# NUMERICAL RUN
#  ---------------------------------------------------------------------------------------------------------------------

N_sidebands = 4

# For 25 mK
frequency_grid_num, n_num_total25 = n_out_numerical(N_sidebands, N_bins, temperature1)
# For 50 mK
frequency_grid_num, n_num_total50 = n_out_numerical(N_sidebands, N_bins, temperature2)


#  ---------------------------------------------------------------------------------------------------------------------
# PLOTS
#  ---------------------------------------------------------------------------------------------------------------------


fig, ax = plt.subplots()
plt.xlim(0, 2)
plt.ylim(0, 4*10**(-3))
plt.xlabel("$ \omega / \omega_d $")
plt.yticks(np.arange(0, 5*10**(-3), step=10**(-3)))
plt.ylabel("$ n_{pout}(\omega)$")
plt.title("Figure 8")

an_dce, = ax.plot(frequency_grid_an, n_an_dce, 'b:')
an_thermal25, = ax.plot(frequency_grid_an, n_an_thermal25, 'r--')
an_total25, = ax.plot(frequency_grid_an, n_an_total25, 'b--')
an_thermal50, = ax.plot(frequency_grid_an, n_an_thermal50, 'r')
an_total50, = ax.plot(frequency_grid_an, n_an_total50, 'b')
num_total25 = ax.plot(frequency_grid_num, n_num_total25, 'g--')
num_total50 = ax.plot(frequency_grid_num, n_num_total50, 'g-')

plt.show()

