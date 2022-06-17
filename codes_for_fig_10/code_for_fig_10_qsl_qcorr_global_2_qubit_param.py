import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# plt.rcParams["mathtext.cal"] = cal


data_concur_discord_int_on = np.loadtxt("concur_discord_global_int_on_e")
data_concur_discord_int_off = np.loadtxt("concur_discord_global_int_off_e")

data_qsl_int_on = np.loadtxt("qsl_global_int_on_e")
data_qsl_int_off = np.loadtxt("qsl_global_int_off_e")
 

fig, ax = plt.subplots(2, 2, figsize = [7, 7], sharex=True)


ax[0, 0].plot(data_concur_discord_int_on[:, 0], data_concur_discord_int_on[:, 1],"--",linewidth = 2.5, label = r"$\mathcal{C}(\rho_{S_1S_2})$", color = 'blue')
ax[0, 0].plot(data_concur_discord_int_on[:, 0], data_concur_discord_int_on[:, 2],"-.", linewidth = 2.5, label = r"$\mathcal{D}(\rho_{S_1S_2})$", color = 'black')

ax[0, 1].plot(data_concur_discord_int_off[:, 0], data_concur_discord_int_off[:, 1],"--",linewidth = 2.5, label = r"$\mathcal{C}(\rho'_{S_1S_2})$", color = 'blue')
ax[0, 1].plot(data_concur_discord_int_off[:, 0], data_concur_discord_int_off[:, 2],"-.", linewidth = 2.5, label = r"$\mathcal{D}(\rho'_{S_1S_2})$", color = 'black')

ax[1, 0].plot(data_qsl_int_on[:, 0], data_qsl_int_on[:, 1], '-', linewidth = 2.5, color = 'red')
ax[1, 1].plot(data_qsl_int_off[:, 0], data_qsl_int_off[:, 1], '-', linewidth = 2.5, color = 'red')


fig.supxlabel(r"$\delta$", fontsize=18)

ax[0, 0].set_ylabel(r"Quantum Correlations", fontsize=14)

ax[1, 0].set_ylabel(r"$\tau_{QSL}$", fontsize=18)
# plt.grid()
ax[0, 0].legend(prop={"size":12})
ax[0, 1].legend(prop={"size":12})
# ax[1, 0].legend(prop={"size":12})
# ax[1, 1].legend(prop={"size":12})

ax[0, 0].set_title("a", fontsize = 18)
ax[0, 1].set_title("b", fontsize = 18)
ax[1, 0].set_title("c", fontsize = 18)
ax[1, 1].set_title("d", fontsize = 18)

plt.savefig("global_2_qubit_qcorr_qsl.pdf")

plt.show()

