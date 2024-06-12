import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

Mm = 0
Nm = 10
eta = 0.999 # stretching


l_cube = 30.0
l_gap = 3.0
l_metal = 10.0
l_cubex = 30.0
l_verre = 30
l_PML = 10
period = l_cube + l_gap + l_metal + l_cubex + l_verre + l_PML
space_x = period-l_cubex
eps_env = 1.0 **2
eps_dielec = 1.41 **2
eps_glass = 1.5 **2
# fin_pml = 500.01
# deb_cube = 100.01

l_cubey = period
space_y = period-l_cubey


# nb_lamb = 75
#lambdas = np.linspace(400,1800,200)
#r = np.zeros(len(lambdas), dtype=complex)
wavelength = 700
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
pol = 90*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola

pi = np.pi

e_au = mat.epsAubb(wavelength)
e_ag = mat.epsAgbb(wavelength)

top = bunch.Bunch()

#top.ox = [0,l_cubex,l_cubex+l_gap, l_cubex+l_gap+l_metal, l_cubex+l_gap+l_metal+l_verre, l_cubex+l_gap+l_metal+l_verre+l_PML]
top.ox = [0, l_cubex, l_cubex+l_gap, period, period+l_PML]
top.nx = top.ox
top.oy = [0,l_cubey]
top.ny = [0,l_cubey]
top.Mm=Mm
top.Nm=Nm
top.mu =  np.array([[1.,1., 1., 1.]])#,
#                  [1.]])
top.eps =  np.array([[e_ag,1., e_au, e_au]])#,
#                  [e_ag]])

top.eta=eta
top.pmlx=[0, 0,0,1]
top.pmly=[0]

bottom = bunch.Bunch()

#bottom.ox = [0,l_cubex,l_cubex+l_gap, l_cubex+l_gap+l_metal, l_cubex+l_gap+l_metal+l_verre, l_cubex+l_gap+l_metal+l_verre+l_PML]
bottom.ox = [0, l_cubex, l_cubex+l_gap, period, period+l_PML]
bottom.nx = bottom.ox
bottom.oy = [0,l_cubey]
bottom.ny = [0,l_cubey]
bottom.Mm=Mm
bottom.Nm=Nm
bottom.mu =  np.array([[1.,1., 1., 1.]])#,
#                  [1.]])
bottom.eps =  np.array([[1.,1., e_au, e_au]])#,
#                  [1.]])

bottom.eta=eta
bottom.pmlx=[0, 0,0,1]
bottom.pmly=[0]
   
k0 = 2*pi/wavelength
top.k0 = k0
bottom.k0 = k0

a = -k0 * np.sin(theta) * np.cos(phi)
top.a0 = a
bottom.a0 = a

list_beta = np.linspace(0.1,89,90) * np.pi / 180
R = np.empty(list_beta.size)
neff = np.empty(list_beta.size)

for i, beta in enumerate(list_beta):
    #b = -k0 * np.sin(theta) * np.sin(phi)
    b = beta
    top.b0 = b
    bottom.b0 = b
    
    #[Pair,Vair], ext = base.homogene(top, ext=1)
    
    #Vair_sort = Vair#[isort]
    #Vair_sort = np.real(Vair_sort) * (np.abs(np.real(Vair_sort))>1e-10) + 1.0j*(np.imag(Vair_sort) * (np.abs(np.imag(Vair_sort))>1e-10))
    
    #[Pgp,Vgp] = base.reseau(gp)
    #[Psub,Vsub], ext2 = base.homogene(bot, ext=1)
    #[Pspa,Vspa] = base.homogene(spa)
    #[Pml, Vml] = base.homogene(ml)

    [Ptop, Vtop] = base.reseau(top)
    [Pbottom, Vbottom] = base.reseau(bottom)

    S = base.c_bas(base.interface(Ptop, Pbottom), Vbottom, 0)

    # Creating the entry vector
    # print(ext)
    # a = np.cos(pol) * np.cos(theta) * np.cos(phi) - np.sin(pol) * np.sin(phi)
    # b = np.cos(pol) * np.cos(theta) * np.sin(phi) + np.sin(pol) * np.cos(phi)
    # c = top.eps[0,0] * top.mu[0,0] * top.k0**2
    # d = np.sqrt(c - top.a0**2 - top.b0**2)
    # e = ((c-top.b0**2)*np.abs(a)**2 + (c-top.a0**2)*np.abs(b)**2 + 2*top.a0*top.b0*np.real(a*b)) / (top.mu[0,0]*d)
    
    # V = np.zeros(4 * (2*Nm+1) *(2*Mm+1))
    # V[int(np.real(ext[3,0]))] = a/np.sqrt(e)
    # # ! Petite bizarrerie, ext[0,0] contient la moitié du nombre de 
    # # ! modes propagatifs en espace libre. Normalement. A cause de la polarisation :
    # # ! pour chaque mode, il y a deux polarisations !
    # V[int(np.real(ext[3,int(np.real(ext[0,0]))]))] = b/np.sqrt(e)

    # V = S @ V

    # reflechi = base.efficace(top, ext, V[:2 * (2*Nm+1) *(2*Mm+1)])
    # r[i] = reflechi[3,0]
    #print(lambd, r[i])

    # print(S)
    # b = np.argmin(np.imag(Vair))
    # print(b, Vair[b])
    # r[i] = S[b,b]
    # print(lambd, S[b,b], abs(S[b,b]))
#     # print(Vair[b])
#     # print(np.real(Vair[b])/gp.k0)
    pos_gp = np.argmin(np.imag(Vtop))
    neff[i] = np.real(Vtop[pos_gp]/top.k0)
    R[i] = np.abs(S[pos_gp,pos_gp])**2


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(list_beta * 180 / np.pi, R)
plt.xlabel("beta (°)")
plt.ylabel("$R_{GP}$")
plt.ylim([0,1])
plt.title("Reflexion of the gap plasmon")
plt.savefig("R_beta_renversed_Nmodes10_cube30_gap3_or100_PML10_wav700.png")
plt.show(block=False)

plt.figure(2)
plt.plot(list_beta * 180 / np.pi, neff)
plt.xlabel("beta (°)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of the gap plasmon")
plt.show(block=False)
plt.savefig("neff_beta_renversed_Nmodes10_cube30_gap3_or100_PML10_wav700.png")

np.savez("data_beta_renversed_Nmodes10_cube30_gap3_or100_PML10_wav700.npz", list_beta=list_beta, R=R, neff=neff)
