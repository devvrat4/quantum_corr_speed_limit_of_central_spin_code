import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

f_off = "inter_off"

y_off = np.loadtxt(f_off, dtype=complex)

f_on = "inter_on"

y_on = np.loadtxt(f_on, dtype=complex)

t = np.linspace(0, 10, 500)
dim = 4

tr_dist_int_on_off = np.zeros((len(t)))

for i in range(len(t)):
    k = 0 
    rho_off = np.zeros((dim, dim), dtype = complex)
    for u in range(dim):
        for j in range(dim):
            rho_off[u, j] = y_off[i, k]
            k = k+1
    
    k = 0 
    rho_on = np.zeros((dim, dim), dtype = complex)
    for u in range(dim):
        for j in range(dim):
            rho_on[u, j] = y_on[i, k]
            k = k+1
            
    rho_diff = rho_on - rho_off

    tr_dist_int_on_off[i] = 0.5*np.trace(sqrtm((np.conjugate(rho_diff.T)@rho_diff).real))

    # End for the loop i 

data = np.zeros((len(t), 2))
data[:, 0] = t
data[:, 1] = tr_dist_int_on_off
np.savetxt("tr_dist_2_qubit", data)

plt.plot(t, tr_dist_int_on_off, label = "Trace distance 2 qubit interaction on or off")
plt.legend()
plt.show()
