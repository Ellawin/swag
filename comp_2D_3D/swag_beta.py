import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

Mm = 10
Nm = 10
eta = 0.999 # stretching


hcube = 30.0
hspacer = 3.0
hlayer = 10.0
l_cubex = 30.0
l_cubey = 30.0
space_x = 101-l_cubex
space_y = 102-l_cubey
eps_env = 1.0 **2
eps_dielec = 1.41 **2
eps_glass = 1.5 **2
# fin_pml = 500.01
# deb_cube = 100.01

# nb_lamb = 75
lambdas = np.linspace(400,1800,200)
r = np.zeros(len(lambdas), dtype=complex)
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
pol = 90*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola

pi = np.pi

top = bunch.Bunch()

top.ox = [0,l_cubex,l_cubex+space_x]
top.nx = [0,l_cubex,l_cubex+space_x]
top.oy = [0,l_cubey,l_cubey+space_y]
top.ny = [0,l_cubey,l_cubey+space_y]
top.Mm=Mm
top.Nm=Nm
top.mu =  np.array([[1.,1.],
                  [1.,1.]])
top.eps =  np.array([[1.,1.],
                  [1.,1.]])

top.eta=eta
top.pmlx=[0, 0]
top.pmly=[0, 0]

bot = bunch.Bunch()

bot.ox = [0,l_cubex,l_cubex+space_x]
bot.nx = [0,l_cubex,l_cubex+space_x]
bot.oy = [0,l_cubey,l_cubey+space_y]
bot.ny = [0,l_cubey,l_cubey+space_y]
bot.Mm=Mm
bot.Nm=Nm
bot.mu = np.array([[1.,1],
                  [1.,1.]])

bot.eps = np.array([[eps_glass, eps_glass], 
                   [eps_glass, eps_glass]])

bot.eta=eta
bot.pmlx=[0, 0]
bot.pmly=[0, 0]

spa = bunch.Bunch()

spa.ox = [0,l_cubex,l_cubex+space_x]
spa.nx = [0,l_cubex,l_cubex+space_x]
spa.oy = [0,l_cubey,l_cubey+space_y]
spa.ny = [0,l_cubey,l_cubey+space_y]
spa.Mm=Mm
spa.Nm=Nm
spa.mu =  np.array([[1.,1],
                  [1.,1.]])
spa.eps =  np.array([[eps_dielec,eps_dielec],
                  [eps_dielec,eps_dielec]])
spa.eta=eta
spa.pmlx=[0, 0]
spa.pmly=[0, 0]

gp = bunch.Bunch()

gp.ox = [0,l_cubex,l_cubex+space_x]
gp.nx = [0,l_cubex,l_cubex+space_x]
gp.oy = [0,l_cubey,l_cubey+space_y]
gp.ny = [0,l_cubey,l_cubey+space_y]
gp.Mm = Mm
gp.Nm = Nm
gp.mu = np.array([[1.,1],
                  [1.,1.]])
gp.eta = eta
gp.pmlx=[0, 0]
gp.pmly=[0, 0]

ml = bunch.Bunch()

ml.ox = [0,l_cubex,l_cubex+space_x]
ml.nx = [0,l_cubex,l_cubex+space_x]
ml.oy = [0,l_cubey,l_cubey+space_y]
ml.ny = [0,l_cubey,l_cubey+space_y]
ml.Mm=Mm
ml.Nm=Nm
ml.mu =  np.array([[1.,1],
                  [1.,1.]])

ml.eta=eta
ml.pmlx=[0, 0]
ml.pmly=[0, 0]

for i, lambd in enumerate(lambdas):
    # print(lambd)
    e_au = mat.epsAubb(lambd)
    e_ag = mat.epsAgbb(lambd)
   
    k0 = 2*pi/lambd
    top.k0 = k0
    spa.k0 = k0
    bot.k0 = k0
    gp.k0 = k0
    ml.k0 = k0

    a = -k0 * np.sin(theta) * np.cos(phi)
    top.a0 = a
    bot.a0 = a
    spa.a0 = a
    gp.a0 = a
    ml.a0 = a 

    b = -k0 * np.sin(theta) * np.sin(phi)
    top.b0 = b
    bot.b0 = b
    spa.b0 = b
    gp.b0 = b
    ml.b0 = b 

    gp.eps =  np.array([[e_ag,1.],
                         [1.,1.]])
    ml.eps = np.array([[e_au,e_au],
                       [e_au,e_au]])
    
    [Pair,Vair], ext = base.homogene(top, ext=1)
    
    Vair_sort = Vair#[isort]
    Vair_sort = np.real(Vair_sort) * (np.abs(np.real(Vair_sort))>1e-10) + 1.0j*(np.imag(Vair_sort) * (np.abs(np.imag(Vair_sort))>1e-10))
    
    [Pgp,Vgp] = base.reseau(gp)
    [Psub,Vsub], ext2 = base.homogene(bot, ext=1)
    [Pspa,Vspa] = base.homogene(spa)
    [Pml, Vml] = base.homogene(ml)

    S = base.c_bas(base.interface(Pair, Pgp), Vgp, hcube)
    S = base.cascade(S, base.c_bas(base.interface(Pgp, Pspa), Vspa, hspacer))
    S = base.cascade(S, base.c_bas(base.interface(Pspa, Pml), Vml, hlayer))
    S = base.cascade(S, base.c_bas(base.interface(Pml, Psub), Vsub, 0))

    # Creating the entry vector
    # print(ext)
    a = np.cos(pol) * np.cos(theta) * np.cos(phi) - np.sin(pol) * np.sin(phi)
    b = np.cos(pol) * np.cos(theta) * np.sin(phi) + np.sin(pol) * np.cos(phi)
    c = top.eps[0,0] * top.mu[0,0] * top.k0**2
    d = np.sqrt(c - top.a0**2 - top.b0**2)
    e = ((c-top.b0**2)*np.abs(a)**2 + (c-top.a0**2)*np.abs(b)**2 + 2*top.a0*top.b0*np.real(a*b)) / (top.mu[0,0]*d)
    
    V = np.zeros(4 * (2*Nm+1) *(2*Mm+1))
    V[int(np.real(ext[3,0]))] = a/np.sqrt(e)
    # ! Petite bizarrerie, ext[0,0] contient la moitié du nombre de 
    # ! modes propagatifs en espace libre. Normalement. A cause de la polarisation :
    # ! pour chaque mode, il y a deux polarisations !
    V[int(np.real(ext[3,int(np.real(ext[0,0]))]))] = b/np.sqrt(e)

    V = S @ V

    reflechi = base.efficace(top, ext, V[:2 * (2*Nm+1) *(2*Mm+1)])
    r[i] = reflechi[3,0]
    print(lambd, r[i])

    # print(S)
    # b = np.argmin(np.imag(Vair))
    # print(b, Vair[b])
    # r[i] = S[b,b]
    # print(lambd, S[b,b], abs(S[b,b]))
#     # print(Vair[b])
#     # print(np.real(Vair[b])/gp.k0)

import matplotlib.pyplot as plt
plt.figure(2)
plt.plot(betas, np.abs(r)**2)
plt.xlabel("Angle (degree) ")
plt.ylabel("$R_{GP}$")
plt.ylim([0,1])
plt.savefig("truc.png")
plt.show(block=False)

### reprendre ici : on transforme ce code (qui fonctionne en variant la longueur d'onde avec le système cube 3D complet)
### en un code qui varie l'angle beta (longueur d'onde fixée) et en système renversé 