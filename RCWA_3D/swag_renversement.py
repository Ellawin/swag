import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import PyMoosh as pm
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

def reflectance(lambd, beta):
    e_ag = mat.epsAgbb(lambd)
    e_au = mat.epsAubb(lambd)

    top.eps =  np.array([[1.5, 1.5, e_au,1.41, e_ag]])
                   #[e_ag,1.,1.]])

    down.eps =  np.array([[1.5,1.5, e_au, 1.41, 1.41]])
                   #[1.,1.,1.]])

    k0 = 2*np.pi/lambd
    top.k0 = k0
    down.k0 = k0

    alpha = -k0 * np.sin(theta) * np.cos(phi)
    top.a0 = alpha
    down.a0 = alpha

    #beta = -k0 * np.sin(theta) * np.sin(phi)
    top.b0 = beta
    down.b0 = beta
    
    [Ptop, Vtop] = base.reseau(top)
    [Pdown,Vdown] = base.reseau(down)

    S = base.interface(Ptop, Pdown)

    pos_gp = np.argmin(np.imag(Vtop))
    neff = np.real(Vtop[pos_gp]/top.k0)
    
    #d = np.argmin(np.imag(Vsub))
    #n_sp = np.real(Vsub[d]/sub.k0)

    R = np.abs(S[pos_gp,pos_gp])**2
    return R, neff

Mm = 10
Nm = 0
eta = 0.99 # stretching

### Wave
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
#pol = 90*np.pi/180. # 0 (TE) ou 90 (TM) pour fixer la pola 

eps_env = 1.0 **2
#fin_pml = 500.01
#deb_cube = 100.01

### geometry
thick_pml = 30
thick_quartz = 100
thick_metal = 10
thick_gap = 3
thick_cube = 30

top = bunch.Bunch()
down = bunch.Bunch()

top.Mm=Mm
top.Nm=Nm
top.mu =  np.array([[1.,1.,1.,1., 1.]])#, [1,1,1]])

top.ox = [0,thick_pml, thick_pml+thick_quartz, thick_pml+thick_quartz+thick_metal, thick_pml+thick_quartz+thick_metal+thick_gap, thick_pml+thick_quartz+thick_metal+thick_gap+thick_cube]
top.nx = top.ox
top.oy = [0, 10]
top.ny = top.oy

top.eta=eta
top.pmlx=[1,0, 0, 0,0]
top.pmly=[0]

down.Mm=Mm
down.Nm=Nm
down.mu =  np.array([[1.,1.,1.,1.,1.]])#, [1,1,1]])

down.ox = [0,thick_pml, thick_pml+thick_quartz, thick_pml+thick_quartz+thick_metal, thick_pml+thick_quartz+thick_metal+thick_gap, thick_pml+thick_quartz+thick_metal+thick_gap+thick_cube]
down.nx = down.ox
down.oy = [0, 10]
down.ny = down.oy

down.eta=eta
down.pmlx=[1,0, 0, 0,0]
down.pmly=[0]

R = np.empty(100)
neff = np.empty(100)

#list_wavelengths = np.linspace(300,900,100)
wavelength = 700
e_ag = mat.epsAgbb(wavelength)
e_au = mat.epsAubb(wavelength)
#R = np.empty(list_wavelengths.size)
#neff = np.empty(list_wavelengths.size)


#list_period = np.linspace(100,600,100)
#period = 300
#R = np.empty(list_period.size)
#neff = np.empty(list_period.size)

#list_gaps = np.linspace(1, 10, 100)
#l_gap = 3.0
#R = np.empty(list_gaps.size)
#neff = np.empty(list_gaps.size)

list_beta = np.linspace(1,89,100) * np.pi / 180
#beta = 0
R = np.empty(list_beta.size)
neff = np.empty(list_beta.size)

#for i, wavelength in enumerate(list_wavelengths):
#for i, period in enumerate(list_period):
#for i, l_gap in enumerate(list_gaps): 
for i, beta in enumerate(list_beta):
    print(i)
    R[i], neff[i] = reflectance(wavelength, beta)

plt.figure(1)
#plt.plot(list_period, R)
#plt.plot(list_wavelengths, R, label = 'PML 50 nm')
# plt.plot(list_gaps, R)
plt.plot(list_beta * 180 / np.pi, R)
#plt.xlabel("Wavelength (nm)")
#plt.xlabel("gap (nm)")
plt.xlabel("beta (°)")
#plt.xlabel("Period (nm)")
plt.title("Reflexion of the gap plasmon")
plt.ylabel("Rgp")
plt.legend()
plt.show(block=False)
plt.savefig("renversement/pml/Rgp_beta_10Mm_pml30.jpg")
#plt.savefig("renversement/Rgp_gap_30Mm_pol90.jpg")
#plt.savefig("renversement/pml/Rgp_period_30Mm_pml.jpg")


plt.figure(2)
#plt.plot(list_period, neff)
#plt.plot(list_wavelengths, neff, label = 'PML 50 nm')
#plt.plot(list_gaps, neff)
plt.plot(list_beta * 180 / np.pi, neff)
plt.xlabel("Wavelength (nm)")
#plt.xlabel("Gap (nm)")
plt.xlabel("beta (°)")
#plt.xlabel("Period (nm)")
plt.ylabel("$n_{eff}$")
plt.legend()
plt.title("Effective index of the gap plasmon")
plt.show(block=False)
plt.savefig("renversement/pml/Neff_beta_10Mm_pml30.jpg")
