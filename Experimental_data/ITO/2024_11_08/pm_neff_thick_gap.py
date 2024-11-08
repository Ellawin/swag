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
stack_ito = [0, 2, 6, 5, 4, 3]

#Thicknesses
thick_cube = 30
#thick_gap = 10
list_gap = np.linspace(1,15,100)
thick_metal = 5
thick_ito = 100
#list_ito = np.linspace(1,200,100)
thick_sio2 = 200

#Wave
wavelength = 700
k0 = 2*np.pi/wavelength
polarization = 1

#Find the mode (it's unique) which is able to propagate in the GP gap from the quasi_stat_milloupe_model
perm_metal = metallic_layer.get_permittivity(wavelength)
tol = 1e-12
step_max = 100000

idx_gap = 0
GP_effective_index = np.empty(list_gap.size)

for thick_gap in list_gap:
    print(idx_gap)
    neff_quasi_stat_milloupe = np.sqrt(perm_gap) * np.sqrt((4)/(k0**2 * thick_metal * thick_gap * abs(perm_metal)))
    start_index_eff = neff_quasi_stat_milloupe
    thicknesses = [0, thick_cube, thick_gap, thick_metal, thick_ito, thick_sio2]
    Layers = pm.Structure(material_list, stack_ito, thicknesses, verbose=False)
    GP_effective_index[idx_gap] = modes.steepest(start_index_eff, tol, step_max, Layers, wavelength, polarization)
    idx_gap+=1

plt.figure(1)
plt.plot(list_gap, GP_effective_index)
plt.xlabel("Gap dielectric thickness (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of Gap-Plasmon")
plt.show(block=False)
plt.savefig(f"nGP_PM_Cube-Thick{thick_cube}-MatAg_Gap-Thick-Var-1-15-100-Mat1.45_Metal-Thick{thick_metal}-MatAu_Sub-Thick{thick_sio2}-MatSiO2.pdf")

np.savez(f"nGP_PM_Cube-Thick{thick_cube}-MatAg_Gap-Thick-Var-1-15-100-Mat1.45_Metal-Thick{thick_metal}-MatAu_Sub-Thick{thick_sio2}-MatSiO2.pdf", ngp =GP_effective_index, list_gap = list_gap)