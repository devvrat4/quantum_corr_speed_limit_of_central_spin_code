import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# plt.rcParams["mathtext.cal"] = cal


data_int_off_local = np.loadtxt("von_ent_int_off_local")

data_int_on_local = np.loadtxt("von_ent_int_on_local")

data_int_off_global = np.loadtxt("von_ent_int_off_global")

data_int_on_global = np.loadtxt("von_ent_int_on_global")

fig, ax = plt.subplots(2, figsize=[6,7], sharex= True)

ax[0].plot(data_int_off_local[:, 0], data_int_off_local[:, 1],linewidth = 2.5, label = r"$-\mathrm{Tr}(\rho'_{AB})\log(\rho'_{AB})$", color = 'blue')
ax[0].plot(data_int_on_local[:, 0], data_int_on_local[:, 1],"-.", linewidth = 2.5,  label = r"$-\mathrm{Tr}(\rho_{AB})\log(\rho_{AB})$", color = 'red')
# plt.xlabel(r"$t$", fontsize=18)
# plt.ylabel("von Neumann entropy", fontsize=18)
ax[0].set_title("a. Local bath")
# plt.grid()
ax[0].legend(prop={"size":12})


ax[1].plot(data_int_off_global[:, 0], data_int_off_global[:, 1],linewidth = 2.5, label = r"$-\mathrm{Tr}(\rho'_{S_1S_2})\log(\rho'_{S_1S_2})$", color = 'blue')
ax[1].plot(data_int_on_global[:, 0], data_int_on_global[:, 1],"-.", linewidth = 2.5,  label = r"$-\mathrm{Tr}(\rho_{S_1S_2})\log(\rho_{S_1S_2})$", color = 'red')
# plt.xlabel(r"$t$", fontsize=18)
# plt.ylabel("von Neumann entropy", fontsize=18)
ax[1].set_title("b. Global bath")
# plt.grid()
ax[1].legend(prop={"size":12})

fig.add_subplot(111, frameon=False)
plt.ylabel("von Neumann entropy", fontsize=18)
plt.xlabel(r"$t$", fontsize=18)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.savefig("von_ent_2_qubit_local_global.pdf")
plt.show()

