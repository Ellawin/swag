import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
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


def reflectance_3D(lambd):
    e_au = mat.epsAubb(lambd)
    e_ag = mat.epsAgbb(lambd)
    e_TiN = nk_TiN(lambd) ** 2 

    k0 = 2*np.pi/lambd
    super.k0 = k0
    gap.k0 = k0
    sub.k0 = k0
    cube.k0 = k0
    metal.k0 = k0
    accroche.k0 = k0

    a = -k0 * np.sin(theta) * np.cos(phi)
    super.a0 = a
    sub.a0 = a
    gap.a0 = a
    cube.a0 = a
    metal.a0 = a 
    accroche.a0 = a

    b = -k0 * np.sin(theta) * np.sin(phi)
    super.b0 = b
    sub.b0 = b
    gap.b0 = b
    cube.b0 = b
    metal.b0 = b 
    accroche.b0 = b 

    cube.eps =  np.array([[e_ag,1.],
                         [1.,1.]])
    metal.eps = np.array([[e_au,e_au],
                       [e_au,e_au]])

    accroche.eps = np.array([[e_TiN, e_TiN], [e_TiN, e_TiN]])
    
    [Pair,Vair], ext = base.homogene(super, ext=1)
    [Pgp,Vgp] = base.reseau(cube)
    [Pspa,Vspa] = base.homogene(gap)
    [Pml, Vml] = base.homogene(metal)
    [Pacc, Vacc] = base.homogene(accroche)
    [Psub,Vsub], ext2 = base.homogene(sub, ext=1)

    S = base.c_bas(base.interface(Pair, Pgp), Vgp, hcube)
    S = base.cascade(S, base.c_bas(base.interface(Pgp, Pspa), Vspa, hspacer))
    S = base.cascade(S, base.c_bas(base.interface(Pspa, Pml), Vml, hlayer))
    S = base.cascade(S, base.c_bas(base.interface(Pml, Pacc), Vacc, hacc))
    S = base.cascade(S, base.c_bas(base.interface(Pacc, Psub), Vsub, 0))

    # Creating the entry vector
    a = np.cos(pol) * np.cos(theta) * np.cos(phi) - np.sin(pol) * np.sin(phi)
    b = np.cos(pol) * np.cos(theta) * np.sin(phi) + np.sin(pol) * np.cos(phi)
    c = super.eps[0,0] * super.mu[0,0] * super.k0**2
    d = np.sqrt(c - super.a0**2 - super.b0**2)
    e = ((c-super.b0**2)*np.abs(a)**2 + (c-super.a0**2)*np.abs(b)**2 + 2*super.a0*super.b0*np.real(a*b)) / (super.mu[0,0]*d)
    
    V = np.zeros(4 * (2*Nm+1) *(2*Mm+1))
    V[int(np.real(ext[3,0]))] = a/np.sqrt(e)
    # ! Petite bizarrerie, ext[0,0] contient la moitié du nombre de 
    # ! modes propagatifs en espace libre. Normalement. A cause de la polarisation :
    # ! pour chaque mode, il y a deux polarisations !
    V[int(np.real(ext[3,int(np.real(ext[0,0]))]))] = b/np.sqrt(e)

    V = S @ V

    reflechi = base.efficace(super, ext, V[:2 * (2*Nm+1) *(2*Mm+1)])
    # print(reflechi)
    r = reflechi[3,0]
    print(lambd)
    return r


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


### 3D
Mm = 10
Nm = 10
eta = 0.999 # stretching

hcube = 30.0
hspacer = 3.0
hlayer = 10.0
hacc = 1.0
l_cubex = 30.0
l_cubey = 30.0
space_x = 301-l_cubex
space_y = 302-l_cubey
eps_env = 1.33 **2
eps_dielec = 1.45 **2
eps_glass = 1.5 **2
# fin_pml = 500.01
# deb_cube = 100.01

theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
pol = 90*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola

pi = np.pi

accroche = bunch.Bunch()

accroche.ox = [0,l_cubex,l_cubex+space_x]
accroche.nx = [0,l_cubex,l_cubex+space_x]
accroche.oy = [0,l_cubey,l_cubey+space_y]
accroche.ny = [0,l_cubey,l_cubey+space_y]
accroche.Mm=Mm
accroche.Nm=Nm
accroche.mu =  np.array([[1.,1.],
                  [1.,1.]])

accroche.eta=eta
accroche.pmlx=[0, 0]
accroche.pmly=[0, 0]

super = bunch.Bunch()

super.ox = [0,l_cubex,l_cubex+space_x]
super.nx = [0,l_cubex,l_cubex+space_x]
super.oy = [0,l_cubey,l_cubey+space_y]
super.ny = [0,l_cubey,l_cubey+space_y]
super.Mm=Mm
super.Nm=Nm
super.mu =  np.array([[1.,1.],
                  [1.,1.]])
super.eps =  np.array([[1.,1.],
                  [1.,1.]])

super.eta=eta
super.pmlx=[0, 0]
super.pmly=[0, 0]

sub = bunch.Bunch()

sub.ox = [0,l_cubex,l_cubex+space_x]
sub.nx = [0,l_cubex,l_cubex+space_x]
sub.oy = [0,l_cubey,l_cubey+space_y]
sub.ny = [0,l_cubey,l_cubey+space_y]
sub.Mm=Mm
sub.Nm=Nm
sub.mu = np.array([[1.,1],
                  [1.,1.]])

sub.eps = np.array([[eps_glass, eps_glass], 
                   [eps_glass, eps_glass]])

sub.eta=eta
sub.pmlx=[0, 0]
sub.pmly=[0, 0]

gap = bunch.Bunch()

gap.ox = [0,l_cubex,l_cubex+space_x]
gap.nx = [0,l_cubex,l_cubex+space_x]
gap.oy = [0,l_cubey,l_cubey+space_y]
gap.ny = [0,l_cubey,l_cubey+space_y]
gap.Mm=Mm
gap.Nm=Nm
gap.mu =  np.array([[1.,1],
                  [1.,1.]])
gap.eps =  np.array([[eps_dielec,eps_dielec],
                  [eps_dielec,eps_dielec]])
gap.eta=eta
gap.pmlx=[0, 0]
gap.pmly=[0, 0]

cube = bunch.Bunch()

cube.ox = [0,l_cubex,l_cubex+space_x]
cube.nx = [0,l_cubex,l_cubex+space_x]
cube.oy = [0,l_cubey,l_cubey+space_y]
cube.ny = [0,l_cubey,l_cubey+space_y]
cube.Mm = Mm
cube.Nm = Nm
cube.mu = np.array([[1.,1],
                  [1.,1.]])
cube.eta = eta
cube.pmlx=[0, 0]
cube.pmly=[0, 0]

metal = bunch.Bunch()

metal.ox = [0,l_cubex,l_cubex+space_x]
metal.nx = [0,l_cubex,l_cubex+space_x]
metal.oy = [0,l_cubey,l_cubey+space_y]
metal.ny = [0,l_cubey,l_cubey+space_y]
metal.Mm=Mm
metal.Nm=Nm
metal.mu =  np.array([[1.,1],
                  [1.,1.]])

metal.eta=eta
metal.pmlx=[0, 0]
metal.pmly=[0, 0]


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
thick_acc = 1

# A modifier selon le point de fonctionnement
angle = 0
polarization = 1

## Paramètres des matériaux
perm_env = 1.33 ** 2
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
    r_3D[idx] = reflectance_3D(lambd)
    perm_Ag = base2D.epsAgbb(lambd) # argent
    perm_Au = base2D.epsAubb(lambd)
    ri_TiN = nk_TiN(lambd)
    materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_sub": perm_sub, "perm_reso": perm_Ag, "perm_metalliclayer": perm_Au, "perm_accroche": ri_TiN ** 2}
    wave = {"wavelength": lambd, "angle": angle, "polarization": polarization}
    R_2D[idx] = reflectance_2D(geometry, wave, materials, n_mod)


plt.figure(1)
plt.plot(lambdas, np.abs(r_3D)**2, label = '3D')
plt.plot(lambdas, R_2D, label = '2D')
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("R")
plt.savefig("Cube30gold10gap3acc1TiN_2D_3Dcomp.png")
plt.show(block=False)

np.savez("data_Cube30gold10gap3acc1TiN_2D_3Dcomp.npz", list_wavelength = lambdas, r_3D = r_3D, R_2D = R_2D)

# plt.figure(2)
# #plt.plot(lambdas, np.abs(r_3D)**2, label = '3D')
# plt.plot(lambdas, R_2D, label = '2D')
# plt.legend()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("R")
# plt.savefig("Cube30gold10gap3acc1TiN_2D.png")
# plt.show(block=False)