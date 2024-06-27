import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt

def nk_TiN(lam):
    tableau3D = []

    file = open('../TiN_nk.txt', "r")    
    lines = file.readlines()
    file.close()

    nb_lines = len(lines)
    for idx in range (2,nb_lines-2):
        values = lines[idx].split("\t")
        values = [float(val.replace(',','.')) for val in values]
        tableau3D.append(values)
    
    tableau3D = np.array(tableau3D)
    wl = []
    wl = tableau3D[:,0]
    n_thermal = []
    n_thermal = tableau3D[:,1]
    k_thermal = []
    k_thermal = tableau3D[:,2]
    n_nh3 = []
    n_nh3 = tableau3D[:,4]
    k_nh3 = []
    k_nh3 = tableau3D[:,5]
    n_n2 = []
    n_n2 = tableau3D[:,7]
    k_n2 = []
    k_n2 = tableau3D[:,8]

    n_thermal_int = np.interp(lam, wl, n_thermal)
    k_thermal_int = np.interp(lam, wl, k_thermal)
    n_nh3_int = np.interp(lam, wl, n_nh3)
    k_nh3_int = np.interp(lam, wl, k_nh3)
    n_n2_int = np.interp(lam, wl, n_n2)
    k_n2_int = np.interp(lam, wl, k_n2)

    ri_thermal = n_thermal_int + 1.0j * k_thermal_int
    ri_nh3 = n_nh3_int + 1.0j * k_nh3_int
    ri_n2 = n_n2_int + 1.0j * k_n2_int
    #return(ri_thermal, ri_nh3, ri_n2)
    return(ri_thermal ** 2)

def nk_Tibb(lam):
    tableau3D = []

    file = open('Ti_Rakic-BB.txt', "r")    
    lines = file.readlines()
    file.close()

    nb_lines = len(lines)
    for idx in range (2,nb_lines-2):
        values = lines[idx].split("\t")
        values = [float(val) for val in values]
        tableau3D.append(values)
    
    tableau3D = np.array(tableau3D)
    wl = []
    wl = tableau3D[:,0]
    n = []
    n = tableau3D[:,1]
    k = []
    k = tableau3D[:,2]

    n_int = np.interp(lam, wl, n)
    k_int = np.interp(lam, wl, k)

    ri = n_int + 1.0j * k_int
    return(ri ** 2)

material_list = [1.45, 'Silver', nk_TiN, 1.5 ** 2]
stack = [1,0,2,3]
    
start_index_eff = 4
tol = 1e-12
step_max = 10000

#thick_metal = 5
list_metal = np.linspace(1,50,100)
#thick_gap = 3
list_gap = np.linspace(3,10,5)

n_gp = np.empty((list_gap.size, list_metal.size), dtype = complex)

polarization = 1

#list_wavelength = np.linspace(450,1000,200)
wavelength = 600

# for idx_gap, thick_gap in enumerate(list_gap):
#     for idx_metal, thick_metal in enumerate(list_metal):
#         thicknesses = [200,thick_gap,thick_metal,200]
#         Layers = pm.Structure(material_list, stack, thicknesses)
#         n_gp[idx_gap, idx_metal] = pm.steepest(4,1e-12,10000,Layers,wavelength,polarization)

# plt.figure(2)
# for idx_gap in np.arange(list_gap.size):
#     plt.plot(list_metal, np.abs(n_gp[idx_gap])**2, label = f"thickness gap {int(list_gap[idx_gap])} nm")

# plt.legend()
# plt.ylabel("$n_{gp}$")
# plt.xlabel("Thickness TiN (nm)")
# plt.title("TiN only / Effective index / PyMoosh")

# plt.show(block=False)
# plt.savefig("ngp_TiN_gap_metal_lam600.pdf")
# plt.savefig("ngp_TiN_gap_metal_lam600.jpg")

#np.savez("data_ngp_TiN_gap_metal_lam600.npz", list_gap = list_gap, list_metal = list_metal, n_gp = n_gp)

thick_gap = 5
thick_metal = 10
thicknesses = [200,thick_gap,thick_metal,200]
Layers = pm.Structure(material_list, stack, thicknesses)
n_gp = pm.steepest(4,1e-12,10000,Layers,wavelength,polarization)
x,profil = pm.profile(Layers,n_gp,wavelength,polarization,pixel_size = 0.1)

plt.figure(3)
plt.plot(x,np.real(profil),linewidth = 2)
plt.ylabel("Amplitude du champ")
plt.xlabel("Position in the structure(nm)")
plt.title("TiN only / Effective index profile / PyMoosh")

plt.show(block=False)
plt.savefig("Profile_ngp_TiN10_gap5_lam600.pdf")
plt.savefig("Profile_ngp_TiN10_gap5_lam600.jpg")

np.savez("data_Profile_ngp_TiN10_gap5_lam600.npz", x = x, profil = profil)