import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv
from PyMoosh.models import ExpData
import PyMoosh.modes as modes

#SiO2
shelf = "main"
book = "SiO2"
page = "Franta" #latest to date

SiO2 = pm.Material([shelf, book, page], specialType="RII", verbose=False)
# Important: The RII database will give off errors if the wavelength is not within
# the stored range (no extrapolation is allowed)

#Cube, metal, env and dielec
metallic_layer = pm.Material("Gold", verbose=False)
perm_cube = pm.Material("Silver", verbose=False)
perm_env = 1.0
perm_gap = 1.45**2

# Stack
material_list = [perm_env, perm_cube, SiO2, metallic_layer, perm_gap]
stack_ito = [1, 0, 3, 2] # Ag / dielec / Au / Glass

#Thicknesses
thick_cube = 30.001
#thick_gap = 10
thick_metal = 1.0213
#hick_ito = 100.0054
#list_ito = np.linspace(1,200,100)
thick_sio2 = 200.02357

#Wave
wavelength = 700.0584
k0 = 2*np.pi/wavelength
polarization = 1

data = np.load(f"data/nGP_PM_Cube-Thick{thick_cube}-MatAg_Gap-Thick-Var-1-10-10-Mat1.0_Metal-Thick{thick_metal}-MatAu_Sub-Thick{thick_sio2}-MatSiO2_wav700.npz")
list_gap = data['list_gap']
GP_effective_index = data['ngp']

plt.figure(1)
plt.plot(list_gap, np.real(GP_effective_index))#, label = "$n_{GP}$")
#plt.plot(list_gap, start_index_eff, label = "start $n_{GP}$")
#plt.legend()
plt.xlabel("Gap dielectric thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)