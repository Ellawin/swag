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

plt.figure(2)
plt.plot(list_gap, np.real(nGP_1), label = "Ag / Air / Au / SiO2")
plt.plot(list_gap, np.real(nGP_2), label = " Ag / 1.45 / Au / SiO2")
plt.plot(list_gap, np.real(nGP_3), label = "Ag / Air / Au / ITO / SiO2")
plt.plot(list_gap, np.real(nGP_4), label = "Air / Ag / Air / Au / ITO / SiO2")
plt.legend()
plt.xlabel("Gap dielectric thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("data/compare_withoutStart.pdf")
plt.savefig("data/compare_withoutStart.jpg")

plt.figure(3)
plt.plot(list_gap, np.real(nGP_1), label = "Ag / Air / Au / SiO2")
plt.plot(list_gap, np.real(nGP_2), label = " Ag / 1.45 / Au / SiO2")
plt.plot(list_gap, np.real(nGP_3), label = "Ag / Air / Au / ITO / SiO2")
plt.legend()
plt.xlabel("Gap dielectric thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("data/compare_withoutStart_withoutAirIncident.pdf")
plt.savefig("data/compare_withoutStart_withoutAirIncident.jpg")