import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm 
from PyMoosh.modes import steepest
from scipy.special import erf
from scipy.linalg import toeplitz, inv
import matplotlib

font = {"family" : "DejaVu Serif", "weight" : "normal", "size": 15}
matplotlib.rc("font", **font)

i = complex(0,1)

material_list = [1., 1.54, 'Silver', 'Gold']
stack = [2,0,3,1]

wavelength = 600
polarization = 1

#Find the mode (it's unique) which is able to propagate in the GP gap
start_index_eff = 15
tol = 1e-12
step_max = 1000


list_gap = np.linspace(1,10,40)[::-1]
list_or = np.linspace(1.5,10,40)[::-1]
print(list_gap)
idx = 0

GP_effective_index = np.zeros((len(list_or), len(list_gap)), dtype=complex)

for i, thick_or in enumerate(list_or):
    print("\n", i)
    thicknesses = [0,list_gap[0],thick_or, 0]
    Layers = pm.Structure(material_list, stack, thicknesses, verbose=False)
    GP_effective_index[i, 0] = steepest(start_index_eff, tol, step_max, Layers, wavelength, polarization)

    for j, thick_gap in enumerate(list_gap[1:]):
        thicknesses = [0,thick_gap,thick_or, 0]
        Layers = pm.Structure(material_list, stack, thicknesses, verbose=False)
        GP_effective_index[i, j+1] = steepest(GP_effective_index[i, j], tol, step_max, Layers, wavelength, polarization)

GP_effective_index = np.real(GP_effective_index[::-1,::-1])
print(GP_effective_index)
list_or = list_or[::-1]
list_gap = list_gap[::-1]

step = 8
plt.figure(1)
for i, thick_or in enumerate(list_or[::step]):
    plt.plot(list_gap, GP_effective_index[i*step,:], label=f"gold = {np.round(thick_or, 1)} nm")

plt.xlabel("Gap thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.ylim([0, np.max(GP_effective_index)])
plt.legend()
plt.tight_layout()
plt.savefig("nanop25/effective_index_PM_gap.pdf")

plt.figure(2)
for j, thick_gap in enumerate(list_gap[::step]):
    plt.plot(list_or, GP_effective_index[:,j*step], label=f"gap = {np.round(thick_gap, 1)} nm")
plt.xlabel("Gold thickness (nm)")
plt.ylabel("$n_{eff}$")
#plt.legend()
plt.title("Effective index of Gap-Plasmon")
plt.ylim([0, np.max(GP_effective_index)])
plt.legend()
plt.tight_layout()
plt.savefig("nanop25/effective_index_PM_gold.pdf")

plt.figure(3)
plt.contourf(list_gap, list_or, GP_effective_index, levels=100)
plt.colorbar()
plt.xlabel("Gap thickness (nm)")
plt.ylabel("Gold thickness (nm)")
plt.title("$n_{eff}$")
plt.tight_layout()
plt.savefig("nanop25/effective_index_PM_map.pdf")

plt.show()
np.savez("nanop25/data_effective_index_PyMoosh_metal_gap_or_verre_startgrad15.npz", list_gap=list_gap, list_or=list_or, GP_effective_index=GP_effective_index)
