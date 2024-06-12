import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

material_list = [1., 1.54, 'Silver', 'Gold']
stack = [2,0,3,1]

wavelength = 600
polarization = 1

#Find the mode (it's unique) which is able to propagate in the GP gap
start_index_eff = 15
tol = 1e-12
step_max = 1000000

list_gap = np.linspace(1,10,50)
list_or = np.linspace(1,10,50)
idx = 0

GP_effective_index = np.zeros((list_or.size, list_gap.size))

for i, thick_or in enumerate(list_or):
    print(i)
    for j, thick_gap in enumerate(list_gap):
        #print(j)
        thicknesses = [0,thick_gap,thick_or, 0]
        Layers = pm.Structure(material_list, stack, thicknesses)
        GP_effective_index[i, j] = pm.steepest(start_index_eff, tol, step_max, Layers, wavelength, polarization)

plt.figure(1)
for i, thick_or in enumerate(list_or):
    plt.plot(list_gap, GP_effective_index[i,:], label=f"or = {thick_or} nm")

plt.xlabel("Gap thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("effective_index_PM_gap.pdf")

plt.figure(2)
for j, thick_gap in enumerate(list_gap):
    plt.plot(list_or, GP_effective_index[:,j])
plt.xlabel("Gold thickness (nm)")
plt.ylabel("$n_{eff}$")
#plt.legend()
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("effective_index_PM_gold.pdf")

plt.figure(3)
plt.contourf(list_gap, list_or, GP_effective_index, levels=100)
plt.colorbar()
plt.xlabel("Gap thickness (nm)")
plt.ylabel("Gold thickness (nm)")
plt.title("$n_{eff}$")
plt.show(block=False)
plt.savefig("effective_index_PM_map.pdf")

np.savez("data_effective_index_PyMoosh_metal_gap_or_verre_startgrad15.npz", list_gap=list_gap, list_or=list_or, GP_effective_index=GP_effective_index)
