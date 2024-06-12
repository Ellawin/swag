import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

material_list = [1., 'Gold', 'Silver']
stack = [2,0,1]

wavelength = 600
polarization = 1

#Find the mode (it's unique) which is able to propagate in the GP gap
start_index_eff = 4
tol = 1e-12
step_max = 10000

list_gap = np.linspace(1,25,100)
idx = 0

GP_effective_index = np.empty(list_gap.size)

for thick_gap in list_gap:
    thicknesses = [70,thick_gap,200]
    Layers = pm.Structure(material_list, stack, thicknesses)
    GP_effective_index[idx] = pm.steepest(start_index_eff, tol, step_max, Layers, wavelength, polarization)
    idx +=1

plt.figure(1)
plt.plot(list_gap, GP_effective_index)
plt.xlabel("Gap thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("effective_index_PM.pdf")


# thicknesses = [500,10,10, 500]
# Layers = pm.Structure(material_list, stack, thicknesses)
# neff_GP = pm.steepest(start_index_eff, tol, step_max, Layers, wavelength, polarization)

# print(neff_GP)