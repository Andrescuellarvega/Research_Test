import numpy as np
import matplotlib.pyplot as plt

# Transfer Matrix
j = np.complex(0, 1)
def TransferMatrix(q, epsilon):
    ell = np.array(((1, 1), (j*q, -j*q)))
    v = np.array(((1, 0), (epsilon, 1)))
    p = np.array(((np.exp(j*q), 0), (0, np.exp(-j*q))))
    ell_inv = np.linalg.inv(ell)
    t = np.linalg.multi_dot([p, ell_inv, v, ell])

    t_evals, t_evects = np.linalg.eig(t)

    t_evects_normed = np.zeros(np.shape(t_evects), dtype=complex)
    t_evects_normed[0] = t_evects[0]/np.linalg.norm(t_evects[0])
    t_evects_normed[1] = t_evects[1]/np.linalg.norm(t_evects[1])

    return t_evects_normed

q = 1
epsilon = 1
bins = 1000
x = np.linspace(1/20, 20, bins)
n = np.ceil(x)

v1_normed, v2_normed = TransferMatrix(q, epsilon)
#v1_normed = np.array((v1_normed[1], v1_normed[0]))
qq = np.cos(q) + (epsilon/(2*q))*np.sin(q)
kk = np.arccos(qq)

exp1, exp2 = (np.zeros(bins, dtype=complex), np.zeros(bins, dtype=complex))
if np.abs(qq) < 1:
    exp1 = np.exp(j*kk*n)
    exp2 = np.exp(-j*kk*n)
else:
    exp1[:], exp2[:] = (0, 0)

phi1, phi2 = (np.zeros(bins, dtype=complex), np.zeros(bins, dtype=complex))


for bin in range(0, bins):
    phi1[bin] = exp1[bin] * (v1_normed[0] * np.exp(j*q*(x[bin]-n[bin]))
                       + v1_normed[1] * np.exp(-j*q*(x[bin]-n[bin])))/(v1_normed[0] + v1_normed[1])
    phi2[bin] = exp2[bin] * (v2_normed[0] * np.exp(j*q*(x[bin]-n[bin]))
                       + v2_normed[1] * np.exp(-j*q*(x[bin]-n[bin])))/\
                        (v2_normed[0] + v2_normed[1])

#phi1 = exp1 * (v1_normed[0] * np.exp(j*q*(7.9-8))
#                       + v1_normed[1] * np.exp(-j*q*(7.9-8)))/\
#                        (v1_normed[0] + v1_normed[1])

plt.plot(x,np.real(phi1))
plt.show()

# Eigenvectors are not coming out to what they are supposed to be when implementing np.linalg.ein(t),
# once I use the analytic form them, the program should work out. I should find out why that isn't working out though
# I think same thing happens in the mathematica file when imputting actual values and then calculating eigenvects