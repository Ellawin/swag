import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
#import RCWA_2D.base2D as base2D
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

def structure(period):
    pos_sans_pml = [0,l_cube,l_cube+period]
    pos_avec_pml = [0,l_cube,l_cube+period, l_cube+period+l_pml]
    
    mu_sans_pml = np.array([[1.,1.],
                  [1.,1.]])
    mu_avec_pml = np.array([[1.,1., 1.],
                  [1.,1., 1.]])
    
    ### Superstrat sans PML
    super_sans = bunch.Bunch()

    super_sans.ox = pos_sans_pml
    super_sans.nx = super_sans.ox
    super_sans.oy = pos_sans_pml
    super_sans.ny = super_sans.oy
    super_sans.Mm=Mm
    super_sans.Nm=Nm
    super_sans.mu = mu_sans_pml 
    super_sans.eps =  np.array([[1.,1.],
                  [1.,1.]])

    super_sans.eta=eta
    super_sans.pmlx=[0, 0]
    super_sans.pmly=[0, 0]

    ### Superstrat avec PML
    super_pml = bunch.Bunch()

    super_pml.ox = pos_avec_pml
    super_pml.nx = super_pml.ox
    super_pml.oy = pos_avec_pml
    super_pml.ny = super_pml.oy
    super_pml.Mm=Mm
    super_pml.Nm=Nm
    super_pml.mu = mu_avec_pml
    super_pml.eps =  np.array([[1.,1., 1.],
                    [1.,1., 1.]])

    super_pml.eta=eta
    super_pml.pmlx=[0, 0, 1]
    super_pml.pmly=[0, 0, 1]

    ### Substrat sans PML
    sub_sans = bunch.Bunch()

    sub_sans.ox = pos_sans_pml
    sub_sans.nx = sub_sans.ox
    sub_sans.oy = pos_sans_pml
    sub_sans.ny = sub_sans.oy
    sub_sans.Mm=Mm
    sub_sans.Nm=Nm
    sub_sans.mu =  mu_sans_pml
    sub_sans.eps =  np.array([[1.5,1.5],
                    [1.5,1.5]])

    sub_sans.eta=eta
    sub_sans.pmlx=[0, 0]
    sub_sans.pmly=[0, 0]

    ### Substrat avec PML
    sub_pml = bunch.Bunch()

    sub_pml.ox = pos_avec_pml
    sub_pml.nx = sub_pml.ox
    sub_pml.oy = pos_avec_pml
    sub_pml.ny = sub_pml.oy
    sub_pml.Mm=Mm
    sub_pml.Nm=Nm
    sub_pml.mu =  mu_avec_pml
    sub_pml.eps =  np.array([[1.5,1.5, 1.5],
                    [1.5,1.5, 1.5]])

    sub_pml.eta=eta
    sub_pml.pmlx=[0, 0, 1]
    sub_pml.pmly=[0, 0, 1]

    ### Gap sans PML
    gap_sans = bunch.Bunch()

    gap_sans.ox = pos_sans_pml
    gap_sans.nx = pos_sans_pml
    gap_sans.oy = pos_sans_pml
    gap_sans.ny = pos_sans_pml
    gap_sans.Mm=Mm
    gap_sans.Nm=Nm
    gap_sans.mu =  mu_sans_pml
    gap_sans.eps =  np.array([[eps_dielec,eps_dielec],
                    [eps_dielec,eps_dielec]])
    gap_sans.eta=eta
    gap_sans.pmlx=[0, 0]
    gap_sans.pmly=[0, 0]

    ### Gap avec PML
    gap_pml = bunch.Bunch()

    gap_pml.ox = pos_avec_pml
    gap_pml.nx = pos_avec_pml
    gap_pml.oy = pos_avec_pml
    gap_pml.ny = pos_avec_pml
    gap_pml.Mm=Mm
    gap_pml.Nm=Nm
    gap_pml.mu =  mu_avec_pml
    gap_pml.eps =  np.array([[eps_dielec,eps_dielec, eps_dielec],
                    [eps_dielec,eps_dielec, eps_dielec]])
    gap_pml.eta=eta
    gap_pml.pmlx=[0, 0, 1]
    gap_pml.pmly=[0, 0, 1]

    ### Cube sans PML
    cube_sans = bunch.Bunch()

    cube_sans.ox = pos_sans_pml
    cube_sans.nx = pos_sans_pml
    cube_sans.oy = pos_sans_pml
    cube_sans.ny = pos_sans_pml
    cube_sans.Mm = Mm
    cube_sans.Nm = Nm
    cube_sans.mu = mu_sans_pml
    cube_sans.eps =  np.array([[e_ag,1.],
                    [e_ag,1.]])
    cube_sans.eta = eta
    cube_sans.pmlx=[0, 0]
    cube_sans.pmly=[0, 0]

    ### Cube avec PML
    cube_pml = bunch.Bunch()

    cube_pml.ox = pos_avec_pml
    cube_pml.nx = pos_avec_pml
    cube_pml.oy = pos_avec_pml
    cube_pml.ny = pos_avec_pml
    cube_pml.Mm = Mm
    cube_pml.Nm = Nm
    cube_pml.mu = mu_avec_pml
    cube_pml.eps =  np.array([[e_ag,1., 1.],
                    [e_ag,1., 1.]])
    cube_pml.eta = eta
    cube_pml.pmlx=[0, 0, 1]
    cube_pml.pmly=[0, 0, 1]

    ### Couche métallique sans PML
    metal_sans = bunch.Bunch()

    metal_sans.ox = pos_sans_pml
    metal_sans.nx = pos_sans_pml
    metal_sans.oy = pos_sans_pml
    metal_sans.ny = pos_sans_pml
    metal_sans.Mm=Mm
    metal_sans.Nm=Nm
    metal_sans.mu =  np.array([[1.,1],
                    [1.,1.]])
    metal_sans.eps =  np.array([[e_au,e_au],
                    [e_au,e_au]])
    metal_sans.eta=eta
    metal_sans.pmlx=[0, 0]
    metal_sans.pmly=[0, 0]

    ### Couche métallique avec PML
    metal_pml = bunch.Bunch()

    metal_pml.ox = pos_avec_pml
    metal_pml.nx = pos_avec_pml
    metal_pml.oy = pos_avec_pml
    metal_pml.ny = pos_avec_pml
    metal_pml.Mm=Mm
    metal_pml.Nm=Nm
    metal_pml.mu =  np.array([[1.,1, 1.],
                    [1.,1., 1.]])
    metal_pml.eps =  np.array([[e_au,e_au, e_au],
                    [e_au,e_au, e_au]])
    metal_pml.eta=eta
    metal_pml.pmlx=[0, 0, 1]
    metal_pml.pmly=[0, 0, 1]

    return super_sans, super_pml, cube_sans, cube_pml, gap_sans, gap_pml, metal_sans, metal_pml, sub_sans, sub_pml

def reflectance_3D(super, cube, gap, metal, sub):
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
    
    [Psuper,Vsuper], ext = base.homogene(super, ext=1)
    [Pcube,Vcube] = base.reseau(cube)
    [Pgap,Vgap] = base.homogene(gap)
    [Pmetal, Vmetal] = base.homogene(metal)
    [Psub,Vsub], ext2 = base.homogene(sub, ext=1)

    S = base.c_bas(base.interface(Psuper, Pcube), Vcube, h_cube)
    S = base.cascade(S, base.c_bas(base.interface(Pcube, Pgap), Vgap, h_spacer))
    S = base.cascade(S, base.c_bas(base.interface(Pgap, Pmetal), Vmetal, h_layer))
    S = base.cascade(S, base.c_bas(base.interface(Pmetal, Psub), Vsub, 0))

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
    return r

### 3D
Mm = 5
Nm = 5
eta = 0.999 # stretching

theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
pol = 90*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola
wavelength = 800
pi = np.pi
k0 = 2*pi/wavelength

eps_env = 1.0 **2
eps_dielec = 1.41 **2
eps_glass = 1.5 **2
e_au = mat.epsAubb(wavelength)
e_ag = mat.epsAgbb(wavelength)

h_cube = 30.0
h_spacer = 3.0
h_layer = 10.0
l_cube = 30.0
l_pml = 30

list_period = np.linspace(40.01,1000.01,100)
r_sans_pml = np.empty(list_period.size, dtype = complex)
r_avec_pml = np.empty(list_period.size, dtype = complex)

#period = 100
for i, period in enumerate(list_period):
    print(i)
    super_sans, super_pml, cube_sans, cube_pml, gap_sans, gap_pml, metal_sans, metal_pml, sub_sans, sub_pml = structure(period)
    r_sans_pml[i] = reflectance_3D(super_sans, cube_sans, gap_sans, metal_sans, sub_sans)
    r_avec_pml[i] = reflectance_3D(super_pml, cube_pml, gap_pml, metal_pml, sub_pml)

plt.figure(2)
plt.plot(list_period, np.abs(r_sans_pml)**2, label = 'sans pml')
plt.plot(list_period, np.abs(r_avec_pml)**2, label = 'avec pml')
plt.xlabel("Period (nm)")
plt.ylabel("R")
plt.title("Total reflectance of 3D structure")
plt.legend()
plt.show(block=False)
plt.savefig("3D_pml_impact_period_40_500.jpg")