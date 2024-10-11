import RCWA_2D.base2D as base2D
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt

i = complex(0,1)

np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

def nk_TiN(lam):
    tableau3D = []

    file = open('materials/TiN_nk.txt', "r")    
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

    ri_thermal = n_thermal_int + i*k_thermal_int
    ri_nh3 = n_nh3_int + i*k_nh3_int
    ri_n2 = n_n2_int + i*k_n2_int
    return(ri_thermal) #, ri_nh3, ri_n2)


def reflectance_2D(geometry, wave, materials, n_mod):  
    period = geometry["period"]
    #thick_super = geometry["thick_super"] / period
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period
    thick_gap = geometry["thick_gap"] / period
    thick_func = geometry["thick_func"] / period
    thick_mol = geometry["thick_mol"] / period
    thick_metalliclayer = geometry["thick_metalliclayer"] / period
    thick_sub = geometry["thick_sub"] / period
    thick_accroche = geometry["thick_accroche"] / period 

    wavelength = wave["wavelength"] / period
    angle = wave["angle"] 
    polarization = wave["polarization"]

    perm_env = materials["perm_env"]
    perm_dielec = materials["perm_dielec"]
    perm_sub = materials["perm_sub"]
    perm_reso = materials["perm_reso"]
    perm_metalliclayer =  materials["perm_metalliclayer"]
    perm_accroche = materials["perm_accroche"]

    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])

    n = 2 * n_mod + 1

    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    Pup, Vup = base2D.homogene(k0, a0,polarization, perm_env, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])

    if thick_mol < (thick_gap - thick_func):
        P1, V1 = base2D.grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S = base2D.cascade(S, base2D.interface(Pup, P1))
        S = base2D.c_bas(S, V1, thick_reso)
    
        P2, V2 = base2D.grating(k0, a0, polarization, perm_env, perm_dielec, n, pos_reso)
        S = base2D.cascade(S, base2D.interface(P1, P2))
        S = base2D.c_bas(S, V2, thick_gap - (thick_mol + thick_func))

        P3, V3 = base2D.homogene(k0, a0, polarization, perm_dielec, n)
        S = base2D.cascade(S, base2D.interface(P2, P3))
        S = base2D.c_bas(S, V3, thick_mol + thick_func)

    else:
        P1, V1 = base2D.grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S = base2D.cascade(S, base2D.interface(Pup, P1))
        S = base2D.c_bas(S, V1, thick_reso - (thick_mol - (thick_gap - thick_func)))

        P2, V2 = base2D.grating(k0, a0, polarization, perm_dielec, perm_reso, n, pos_reso)
        S = base2D.cascade(S, base2D.interface(P1, P2))
        S = base2D.c_bas(S, V2, thick_mol - (thick_gap - thick_func))

        P3, V3 = base2D.homogene(k0, a0, polarization, perm_dielec, n)
        S = base2D.cascade(S, base2D.interface(P2, P3))
        S = base2D.c_bas(S, V3, thick_gap)

    Pmetalliclayer, Vmetalliclayer = base2D.homogene(k0, a0, polarization, perm_metalliclayer, n)
    S = base2D.cascade(S, base2D.interface(P3, Pmetalliclayer))
    S = base2D.c_bas(S, Vmetalliclayer, thick_metalliclayer)

    Pacc, Vacc = base2D.homogene(k0, a0, polarization, perm_accroche, n)
    S = base2D.cascade(S, base2D.interface(Pmetalliclayer, Pacc))
    S = base2D.c_bas(S, Vacc, thick_accroche)

    Pdown, Vdown = base2D.homogene(k0, a0, polarization, perm_sub, n)
    S = base2D.cascade(S, base2D.interface(Pacc, Pdown))
    S = base2D.c_bas(S, Vdown, thick_sub)

    # reflexion quand on eclaire par le dessus
    Rup = abs(S[n_mod, n_mod]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    #Tup = abs(S[n + n_mod, n_mod]) ** 2 * np.real(Vdown[n_mod]) / (k0 * np.cos(angle)) / perm_sub * perm_env
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    #Rdown = abs(S[n + n_mod, n + n_mod]) ** 2 
    # transmission quand on éclaire par le dessous
    #Tdown = abs(S[n_mod, n + n_mod]) ** 2 / np.real(Vdown[n_mod]) * perm_sub * k0 * np.cos(angle) / perm_env

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[n_mod, n_mod])
    #phase_R_down = np.angle(S[n + n_mod, n + n_mod])
    #phase_T_up = np.angle(S[n + n_mod, n_mod])
    #phase_T_down = np.angle(S[n_mod, n + n_mod])
    return Rup


### 2D
thick_super = 200
width_reso = 30 # largeur du cube
thick_reso = width_reso # width_reso #hauteur du cube
thick_gap = 3 # hauteur de diéléctrique en dessous du cube
thick_func = thick_gap
thick_mol = 0
thick_metalliclayer = 10 # hauteur de l'or au dessus du substrat
period = 300.2153 # periode
thick_sub = 200
thick_acc = 0

# A modifier selon le point de fonctionnement
angle = 0
polarization = 1

## Paramètres des matériaux
perm_env = 1 ** 2
perm_dielec = 1.45 ** 2 # spacer
perm_sub = 1.5 ** 2 # substrat
#perm_Ag = epsAgbb(wavelength) # argent
#perm_Au = epsAubb(wavelength) # or
#perm_Cr = epsCrbb(wavelength)

n_mod = 100 
n_mod_total = 2 * n_mod + 1

geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_metalliclayer": thick_metalliclayer, "thick_sub": thick_sub, "period": period, "thick_accroche": thick_acc, "thick_func" : thick_func, "thick_mol" : thick_mol}

### Figure
lambdas = np.linspace(450,1000,200)
#r_3D = np.empty(lambdas.size, dtype = complex)
R_2D = np.empty(lambdas.size)

for idx, lambd in enumerate(lambdas):
    print(idx)
    #r_3D[idx] = reflectance_3D(lambd)
    perm_Ag = base2D.epsAgbb(lambd) # argent
    perm_Au = base2D.epsAubb(lambd)
    ri_TiN = nk_TiN(lambd)
    materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_sub": perm_sub, "perm_reso": perm_Ag, "perm_metalliclayer": perm_Au, "perm_accroche": ri_TiN ** 2}
    wave = {"wavelength": lambd, "angle": angle, "polarization": polarization}
    R_2D[idx] = reflectance_2D(geometry, wave, materials, n_mod)


plt.figure(1)
#plt.plot(lambdas, np.abs(r_3D)**2, label = '3D')
plt.plot(lambdas, R_2D, label = '2D')
#plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("R")
plt.savefig("Cube30gold10gap3_air_2D_comp_FDTD_period300.png")
plt.show()

np.savez("data_Cube30gold10gap3_air_2D_comp_FDTD_period300.npz", list_wavelength = lambdas, R_2D = R_2D)
