import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

# Modal 
Mm = 20
Nm = 0
eta = 0.999 # stretching

# Wave
wavelength = 600.123
#theta = 0.0 * np.pi/180. #latitude (z)
phi = 90.0 * np.pi/180. # précession (xy)
#pol = 0*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola
k0 = 2*np.pi/wavelength

# materials
eps_dielec = 1
eps_metal = mat.epsAgbb(wavelength)

# geometry
l_gap = 3.215
#list_pml = np.linspace(0.1, 100, 200)
l_pml = 40
list_periodx = np.linspace(100,500,200)
r_gp = np.empty(list_periodx.size, dtype = complex)
n_gp = np.empty(list_periodx.size)

for i, periodx in enumerate(list_periodx):
#l_pml = 0.1

    top_gp = bunch.Bunch()
    top_gp.Mm=Mm
    top_gp.Nm=Nm
    top_gp.mu =  np.array([[1.,1., 1., 1.]])#,
    top_gp.eps =  np.array([[eps_metal, eps_metal, eps_dielec, eps_metal]])#,
    top_gp.eta=eta
    top_gp.pmlx=[1, 0, 0, 0]
    top_gp.pmly=[0]
    top_gp.k0 = k0

    sub_sp = bunch.Bunch()
    sub_sp.Mm=Mm
    sub_sp.Nm=Nm
    sub_sp.mu = np.array([[1., 1.,1, 1.]])#,

    sub_sp.eps = np.array([[eps_metal, eps_metal, eps_dielec, eps_dielec]])

    sub_sp.eta=eta
    sub_sp.pmlx=[1, 0, 0, 0]
    sub_sp.pmly=[0]
    sub_sp.k0 = k0

    sub_sp_pml = bunch.Bunch()
    sub_sp_pml.Mm=Mm
    sub_sp_pml.Nm=Nm
    sub_sp_pml.mu = np.array([[1.,1, 1., 1.]])#,

    #periodx = 300
    periody = 10

    sub_sp.oy = [0,periody]
    sub_sp.ny = [0,periody]

    top_gp.oy = [0,periody]
    top_gp.ny = [0,periody]

    top_gp.ox = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]
    top_gp.nx = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]
    sub_sp.ox = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]
    sub_sp.nx = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]

    top_gp.a0 = 0
    top_gp.b0 = 0

    [P0, V0] = base.reseau(top_gp)
    index0 = np.argmin(np.imag(V0))
    neff = V0[index0] / top_gp.k0

    theta = 0

    a = -k0 * np.sin(theta) * np.cos(phi)
    top_gp.a0 = a
    sub_sp.a0 = a

    b = - k0 * np.sin(theta) * np.sin(phi)
    top_gp.b0 = b
    sub_sp.b0 = b
    
    [Pgp, Vgp] = base.reseau(top_gp)
    [Psp, Vsp] = base.reseau(sub_sp)
    index_gp = np.argmin(np.imag(Vgp))
    n_gp[i] = np.real(Vgp[index_gp] / top_gp.k0) # équivaut au kz

    S = base.interface(Pgp, Psp)
    r_gp[i] = S[index_gp, index_gp]

plt.figure(1)
plt.plot(list_periodx, n_gp, label = 'PML 40 nm')
plt.xlabel("Taille période (nm)")
plt.legend()
plt.ylabel("$n_{GP}$")
plt.title("Effective index")
plt.show(block=False)
plt.savefig("Ngp_period_gap3_20modes_lam600_phi90_pml40.jpg")

plt.figure(2)
plt.plot(list_periodx, np.abs(r_gp)**2, label = 'PML 40 nm')
plt.xlabel("Taille période (nm)")
plt.ylabel("$r_{GP}$")
plt.legend()
#plt.ylim([0,1])
plt.title("Reflexion of GP")
plt.show(block=False)
plt.savefig("Rgp_period_gap3_20modes_lam600_phi90_pml40.jpg")

np.savez("data_period_gap3_20modes_lam600_phi90_pml40.npz", list_periodx=list_periodx, r_gp = r_gp, n_gp = n_gp)
