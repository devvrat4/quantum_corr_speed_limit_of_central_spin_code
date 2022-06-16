import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# plt.rcParams["mathtext.cal"] = cal


data_int_off = np.loadtxt("von_ent_int_off")

data_int_on = np.loadtxt("von_ent_int_on")


plt.plot(data_int_off[:, 0], data_int_off[:, 1],linewidth = 2.5, label = r"$-\mathrm{Tr}(\rho'_S)\log(\rho'_S)$", color = 'blue')
plt.plot(data_int_on[:, 0], data_int_on[:, 1],"-.",linewidth = 2.5,  label = r"$-\mathrm{Tr}(\rho_S)\log(\rho_S)$", color = 'red')
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel("von Neumann entropy", fontsize=18)
# plt.grid()
plt.legend(prop={"size":15})

plt.savefig("von_ent_single.pdf")

plt.show()

