import j_constants

ell = j_constants.L0_eff
QQ = 100*j_constants.omega_d*(ell/j_constants.vcpw)  # Unitless drive frequency , see Units notes
epsilon = ell/j_constants.L0_eff

n_omega = 1  # Number of sidebands used fro calculation
n_d = 10  # Number of bins dividing (0, 2*N_Omega + 1)

def ww(k, n):
    ww = k*QQ/n_d + (n_omega + 1 - n) * QQ
    return ww

print (ww(2,3))
