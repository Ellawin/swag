import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import PyMoosh as pm
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

def reflectance(lambd, period, l_gap, beta):
    e_ag = mat.epsAgbb(lambd)

    # top layer
    super.ox = [0,l_pml, l_pml+l_cube,l_pml+l_cube + l_gap,l_pml+l_cube + l_gap + l_layer, l_pml+ l_cube+l_gap+l_layer+l_pml]
    
    Reprendre ici :)
    super.nx = [0,l_cube,l_cube + l_gap, l_cube + l_gap + period]
    super.oy = [0, l_cube + l_gap + period]
    super.ny = [0, l_cube + l_gap + period]
    super.eps =  np.array([[e_ag,1., e_ag]])
                   #[e_ag,1.,1.]])

    # Bottom layer
    sub.ox = [0,l_cube,l_cube + l_gap, l_cube + l_gap + period]
    sub.nx = [0,l_cube,l_cube + l_gap, l_cube + l_gap + period]
    sub.oy = [0,l_cube, l_cube + l_gap + period]
    sub.ny = [0,l_cube, l_cube + l_gap + period]
    sub.eps =  np.array([[1.,1., e_ag]])
                   #[1.,1.,1.]])

    k0 = 2*np.pi/lambd
    super.k0 = k0
    sub.k0 = k0

    alpha = -k0 * np.sin(theta) * np.cos(phi)
    super.a0 = alpha
    sub.a0 = alpha

    #beta = -k0 * np.sin(theta) * np.sin(phi)
    super.b0 = beta
    sub.b0 = beta
    
    [Pup, Vup] = base.reseau(super)
    [Psub,Vsub] = base.reseau(sub)

    S = base.interface(Pup, Psub)

    pos_gp = np.argmin(np.imag(Vup))
    neff = np.real(Vup[pos_gp]/super.k0)
    
    #d = np.argmin(np.imag(Vsub))
    #n_sp = np.real(Vsub[d]/sub.k0)

    R = np.abs(S[pos_gp,pos_gp])**2
    return R, neff

Mm = 30
Nm = 0
eta = 0.99 # stretching

### Wave
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
#pol = 90*np.pi/180. # 0 (TE) ou 90 (TM) pour fixer la pola 

eps_env = 1.0 **2
# fin_pml = 500.01
# deb_cube = 100.01

### geometry
l_cube = 30.0
l_pml = 10
l_layer = 30 
#l_gap = 3.0

super = bunch.Bunch()
sub = bunch.Bunch()

super.Mm=Mm
super.Nm=Nm
super.mu =  np.array([[1.,1.,1.,1., 1.]])#, [1,1,1]])

super.eta=eta
super.pmlx=[1,0, 0, 0,1]
super.pmly=[0]

sub.Mm=Mm
sub.Nm=Nm
sub.mu =  np.array([[1.,1.,1.,1.,1.]])#, [1,1,1]])

sub.eta=eta
sub.pmlx=[1,0, 0, 0,1]
sub.pmly=[0]

R = np.empty(100)
neff = np.empty(100)

#list_wavelengths = np.linspace(300,900,100)
wavelength = 600
e_ag = mat.epsAgbb(wavelength)
#R = np.empty(list_wavelengths.size)
#neff = np.empty(list_wavelengths.size)

list_period = np.linspace(100,600,100)
#period = 300
#R = np.empty(list_period.size)
#neff = np.empty(list_period.size)

#list_gaps = np.linspace(1, 10, 100)
l_gap = 3.0
#R = np.empty(list_gaps.size)
#neff = np.empty(list_gaps.size)

#list_beta = np.linspace(1,89,100) * np.pi / 180
beta = 0
#R = np.empty(list_beta.size)
#neff = np.empty(list_beta.size)

#for i, wavelength in enumerate(list_wavelengths):
for i, period in enumerate(list_period):
#for i, l_gap in enumerate(list_gaps): 
#for i, beta in enumerate(list_beta):
    print(i)
    R[i], neff[i] = reflectance(wavelength, period, l_gap, beta)

plt.figure(3)
plt.plot(list_period, R)
#plt.plot(list_wavelengths, R)
# plt.plot(list_gaps, R)
#plt.plot(list_beta * 180 / np.pi, R)
#plt.xlabel("Wavelength (nm)")
#plt.xlabel("gap (nm)")
#plt.xlabel("beta (°)")
plt.xlabel("Period (nm)")
plt.title("Reflexion of the gap plasmon")
plt.ylabel("Rgp")
plt.show(block=False)
#plt.savefig("renversement/Rgp_wavelength_30Mm_pol90.jpg")
#plt.savefig("renversement/Rgp_gap_30Mm_pol90.jpg")
plt.savefig("renversement/pml/Rgp_period_30Mm_pml.jpg")


plt.figure(4)
plt.plot(list_period, neff)
#plt.plot(list_wavelengths, neff)
#plt.plot(list_gaps, neff)
#plt.plot(list_beta * 180 / np.pi, neff)
# plt.xlabel("Wavelength (nm)")
#plt.xlabel("Gap (nm)")
#plt.xlabel("beta (°)")
plt.xlabel("Period (nm)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index of the gap plasmon")
plt.show(block=False)
plt.savefig("renversement/pml/Neff_period_30Mm_pml.jpg")