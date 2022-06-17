import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# plt.rcParams["mathtext.cal"] = cal


data_50_int_off = np.loadtxt("data_qsl_50_e_int_off")
data_50_int_on = np.loadtxt("data_qsl_50_e_int_on")

data_100_int_off = np.loadtxt("data_qsl_100_e_int_off")
data_100_int_on = np.loadtxt("data_qsl_100_e_int_on")

data_200_int_off = np.loadtxt("data_qsl_200_e_int_off")
data_200_int_on = np.loadtxt("data_qsl_200_e_int_on")

data_500_int_off = np.loadtxt("data_qsl_500_e_int_off")
data_500_int_on = np.loadtxt("data_qsl_500_e_int_on")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)

ax1.plot(data_50_int_off[:, 0], data_50_int_off[:, 1], linewidth = 2, label = "a",  color = "blue")
ax1.plot(data_50_int_on[:, 0], data_50_int_on[:, 1],"-.", linewidth = 2, label = "b",  color = "violet")
ax1.set_title("N=50")

ax2.plot(data_100_int_off[:, 0], data_100_int_off[:, 1], linewidth = 2, label = "a", color = 'red')
ax2.plot(data_100_int_on[:, 0], data_100_int_on[:, 1],"-.", linewidth = 2, label = "b", color = 'navy')
ax2.set_title("N=100")

ax3.plot(data_200_int_off[:, 0], data_200_int_off[:, 1], linewidth = 2, label = "a", color = 'black')
ax3.plot(data_200_int_on[:, 0], data_200_int_on[:, 1],"-.", linewidth = 2, label = "b", color = 'grey')
ax3.set_title("N=200")

ax4.plot(data_500_int_off[:, 0], data_500_int_off[:, 1], linewidth = 2, label = "a", color = 'green')
ax4.plot(data_500_int_on[:, 0], data_500_int_on[:, 1],"-.", linewidth = 2, label = "b", color = 'magenta')
ax4.set_title("N=500")

fig.supxlabel(r"$\epsilon$", fontsize=18)
fig.supylabel(r"$\tau_{QSL}$", fontsize=18)
# plt.grid()
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.savefig("qsl_1_qubit_with_e.pdf")
plt.show()

