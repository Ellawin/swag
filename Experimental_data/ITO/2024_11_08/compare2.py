import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv
from PyMoosh.models import ExpData
import PyMoosh.modes as modes


#ITO
def exp_perm(filename):
    # the experimental data should provide wavelength in microns
    tableau3D = []

    file = open(filename, "r")    
    lines = file.readlines()
    file.close()

    nb_lines = len(lines)
    for idx in range (2,nb_lines-2):
        values = lines[idx].split("\t")
        values = [float(val) for val in values]
        tableau3D.append(values)
    
    tableau3D = np.array(tableau3D)
    wl = []
    wl = tableau3D[:,0] *1e3
    n = []
    n = tableau3D[:,1]
    k = []
    k = tableau3D[:,2]

    perm = (n + 1.0j * k)**2
    return(perm, wl)

perm_ito, wl_ito = exp_perm("ITO.txt")
ITO = pm.Material([ExpData, wl_ito, perm_ito], specialType="Model", verbose=False)

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
material_list = [perm_env, 'Aluminium', perm_cube, SiO2, ITO, metallic_layer, perm_gap]
stack_3 = [2, 6, 5, 4, 3] # Ag / 1.45 / Au / ITO / Glass


#Thicknesses
thick_cube = 30.001
list_gap = np.linspace(1,10,50)
thick_metal = 1.0213
thick_ito = 100.0054
thick_sio2 = 200.02357
thick_incident_air = 100.0012

#Wave
wavelength = 700.0584
k0 = 2*np.pi/wavelength
polarization = 1

#Find the mode (it's unique) which is able to propagate in the GP gap from the quasi_stat_milloupe_model
perm_metal = metallic_layer.get_permittivity(wavelength)
tol = 1e-12
step_max = 100000

idx_gap = 0
nGP_3 = np.empty(list_gap.size, dtype = complex)
start_index_eff = np.empty(list_gap.size, dtype = complex)

for thick_gap in list_gap:
    print(idx_gap)
    thicknesses_3 = [thick_cube, thick_gap, thick_metal, thick_ito, thick_sio2]
    Layers_1 = pm.Structure(material_list, stack_3, thicknesses_3, verbose=False)
    start_index_eff[idx_gap] = np.sqrt(perm_gap) * np.sqrt((4)/(k0**2 * thick_metal * thick_gap * abs(perm_metal)))
    nGP_3[idx_gap] = modes.steepest(start_index_eff[idx_gap], tol, step_max, Layers_1, wavelength, polarization)
    idx_gap+=1

np.savez("data/compare2.npz", nGP_3 = nGP_3, start_index_eff = start_index_eff, list_gap = list_gap)

plt.figure(1)
plt.plot(list_gap, np.real(nGP_3), label = "Ag / 1.45 / Au / ITO / SiO2")
plt.plot(list_gap, start_index_eff, "-", label = "start $n_{GP}$")
plt.legend()
plt.xlabel("Gap dielectric thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig("data/compare2.pdf")
plt.savefig("data/compare2.jpg")