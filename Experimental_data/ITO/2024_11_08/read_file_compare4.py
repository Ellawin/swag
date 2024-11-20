import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv
from PyMoosh.models import ExpData
import PyMoosh.modes as modes

data = np.load("data/compare.npz")
nGP_1 = data['nGP_1']
nGP_2 = data['nGP_2']
nGP_3 = data['nGP_3']
nGP_4 = data['nGP_4']
list_gap = data['list_gap']

data2 = np.load("data/compare2.npz")
nGP_5 = data2['nGP_3']
start_index_eff = data2["start_index_eff"]

data3 = np.load("data/compare4_gold1.npz")
nGP_6 = data3['nGP_6']

plt.figure(6)
plt.plot(list_gap, np.real(nGP_1), label = "Ag / Air / Au / SiO2")
plt.plot(list_gap, np.real(nGP_2), label = " Ag / 1.45 / Au / SiO2")
plt.plot(list_gap, np.real(nGP_3), label = "Ag / Air / Au / ITO / SiO2")
plt.plot(list_gap, np.real(nGP_5), label = "Ag / 1.45 / Au / ITO / SiO2")
#plt.plot(list_gap, np.real(nGP_4), label = "Air / Ag / Air / Au / ITO / SiO2")
plt.plot(list_gap, np.real(nGP_6), label = "Ag / 1.45 / Au / Al / Glass")
#plt.plot(list_gap, np.real(start_index_eff),"x", label = "start")
plt.legend()
plt.xlabel("Gap dielectric thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("data/compare4.pdf")
plt.savefig("data/compare4.jpg")