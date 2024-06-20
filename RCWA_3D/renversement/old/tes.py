from .. import RCWA_3D_python
#.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import PyMoosh as pm
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

def reflectance(lambd, period):
    e_ag = mat.epsAgbb(lambd)

    # top layer
    super.ox = [0,l_cubex,l_cubex + l_gap, l_cubex + l_gap + period]
    super.nx = [0,l_cubex,l_cubex + l_gap, l_cubex + l_gap + period]
    # super.oy = [0,l_cubex,l_cubey + l_gap, l_cubey + l_gap + period]
    # super.ny = [0,l_cubex,l_cubey + l_gap, l_cubey + l_gap + period]
    super.oy = [0, l_cubey + l_gap + period]
    super.ny = [0, l_cubey + l_gap + period]
    super.eps =  np.array([[e_ag,1., e_ag]])
                   #[e_ag,1.,1.]])

    # Bottom layer
    sub.ox = [0,l_cubex,l_cubex + l_gap, l_cubex + l_gap + period]
    sub.nx = [0,l_cubex,l_cubex + l_gap, l_cubex + l_gap + period]
    #sub.oy = [0,l_cubex,l_cubey + l_gap, l_cubey + l_gap + period]
    #sub.ny = [0,l_cubex,l_cubey + l_gap, l_cubey + l_gap + period]
    sub.oy = [0,l_cubex, l_cubey + l_gap + period]
    sub.ny = [0,l_cubex, l_cubey + l_gap + period]
    sub.eps =  np.array([[1.,1., e_ag]])
                   #[1.,1.,1.]])

    k0 = 2*np.pi/lambd
    super.k0 = k0
    sub.k0 = k0

    a = -k0 * np.sin(theta) * np.cos(phi)
    super.a0 = a
    sub.a0 = a

    b = -k0 * np.sin(theta) * np.sin(phi)
    super.b0 = b
    sub.b0 = b
    
    [Pup, Vup] = base.reseau(super)
    [Psub,Vsub] = base.reseau(sub)

    S = base.interface(Pup, Psub)

    b = np.argmin(np.imag(Vup))
    neff = np.real(Vup[b]/super.k0)
    
    d = np.argmin(np.imag(Vsub))
    #n_sp = np.real(Vsub[d]/sub.k0)

    R = np.abs(S[b,b])**2
    return R, neff

Mm = 10
Nm = 0
eta = 0.99 # stretching

### Wave
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
#pol = 0*np.pi/180. # 0 (TE) ou 90 (TM) pour fixer la pola 

eps_env = 1.0 **2
# fin_pml = 500.01
# deb_cube = 100.01

### geometry
hcube = 30.0
l_cubex = 30.0
l_cubey = 30.0
l_gap = 3.0

super = bunch.Bunch()
sub = bunch.Bunch()

super.Mm=Mm
super.Nm=Nm
super.mu =  np.array([[1.,1., 1.]])#, [1,1,1]])

super.eta=eta
super.pmlx=[0, 0, 0]
super.pmly=[0]

sub.Mm=Mm
sub.Nm=Nm
sub.mu =  np.array([[1.,1.,1.]])#, [1,1,1]])

sub.eta=eta
sub.pmlx=[0, 0, 0]
sub.pmly=[0]


wavelength = 600
e_ag = mat.epsAgbb(wavelength)

list_period = np.linspace(100,600,10)
R = np.empty(list_period.size)
neff = np.empty(list_period.size)

for i, period in enumerate(list_period):
    R[i], neff[i] = reflectance(wavelength, period)

plt.figure(1)
plt.plot(list_period, R)
plt.xlabel("Period (nm)")
plt.title("Reflexion of the gap plasmon")
plt.ylabel("Rgp")
plt.show(block=False)
plt.savefig("renversement/Rgp_period_30Mm.jpg")

plt.figure(2)
plt.plot(list_period, neff)
plt.xlabel("Period (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of the gap plasmon")
plt.show(block=False)
plt.savefig("renversement/Neff_period_30Mm.jpg")