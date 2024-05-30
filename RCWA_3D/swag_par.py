import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

def reflectance(lambd):
    e_au = mat.epsAubb(lambd)
    e_ag = mat.epsAgbb(lambd)

    k0 = 2*np.pi/lambd
    super.k0 = k0
    gap.k0 = k0
    sub.k0 = k0
    cube.k0 = k0
    metal.k0 = k0

    a = -k0 * np.sin(theta) * np.cos(phi)
    super.a0 = a
    sub.a0 = a
    gap.a0 = a
    cube.a0 = a
    metal.a0 = a 

    b = -k0 * np.sin(theta) * np.sin(phi)
    super.b0 = b
    sub.b0 = b
    gap.b0 = b
    cube.b0 = b
    metal.b0 = b 

    cube.eps =  np.array([[e_ag,1.],
                         [1.,1.]])
    metal.eps = np.array([[e_au,e_au],
                       [e_au,e_au]])
    
    [Pair,Vair], ext = base.homogene(super, ext=1)
    [Pgp,Vgp] = base.reseau(cube)
    [Pspa,Vspa] = base.homogene(gap)
    [Pml, Vml] = base.homogene(metal)
    [Psub,Vsub], ext2 = base.homogene(sub, ext=1)

    S = base.c_bas(base.interface(Pair, Pgp), Vgp, hcube)
    S = base.cascade(S, base.c_bas(base.interface(Pgp, Pspa), Vspa, hspacer))
    S = base.cascade(S, base.c_bas(base.interface(Pspa, Pml), Vml, hlayer))
    S = base.cascade(S, base.c_bas(base.interface(Pml, Psub), Vsub, 0))

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

Mm = 10
Nm = 10
eta = 0.999 # stretching

### geometry
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

# top layer
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

# Bottom layer
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

# spacer
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

# cube layer
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

# metallic layer
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

### Wave
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
pol = 90*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola

lambdas = np.linspace(400,1800,100)

pool = ThreadPool(2)
res = pool.map(reflectance, lambdas)

plt.figure(1)
plt.plot(lambdas, np.abs(res))
plt.xlabel("Wavelength")
plt.ylabel("|r|")
plt.ylim([-0.1,1.1])
plt.savefig(f"Rup/cube30gold10gap3.png")
plt.show(block=False)