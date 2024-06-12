import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

# Modal 
Mm = 5
Nm = 0
eta = 0.999 # stretching

# Wave
wavelength = 700.123
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
#pol = 0*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola
k0 = 2*np.pi/wavelength

# materials
eps_dielec = 1
eps_metal = mat.epsAgbb(wavelength)

# geometry
#l_pml = 30
l_gap = 3.215

top_gp = bunch.Bunch()
top_gp.Mm=Mm
top_gp.Nm=Nm
top_gp.mu =  np.array([[1.,1., 1.]])#,
                  #[1.,1.]])
top_gp.eps =  np.array([[eps_metal, eps_dielec, eps_metal]])#,
                  #[1.,1.]])
top_gp.eta=eta
top_gp.pmlx=[0, 0, 0]
top_gp.pmly=[0]
top_gp.k0 = k0

top_gp_pml = bunch.Bunch()
top_gp_pml.Mm=Mm
top_gp_pml.Nm=Nm
top_gp_pml.mu =  np.array([[1.,1., 1., 1.]])#,
                  #[1.,1.]])
top_gp_pml.eps =  np.array([[eps_metal, eps_metal, eps_dielec, eps_metal]])#,
                  #[1.,1.]])
top_gp_pml.eta=eta
top_gp_pml.pmly=[0]
top_gp_pml.k0 = k0

sub_sp = bunch.Bunch()
sub_sp.Mm=Mm
sub_sp.Nm=Nm
sub_sp.mu = np.array([[1.,1, 1.]])#,
                  #[1.,1.]])

sub_sp.eps = np.array([[eps_metal, eps_dielec, eps_dielec]])

sub_sp.eta=eta
sub_sp.pmlx=[0, 0, 0]
sub_sp.pmly=[0]
sub_sp.k0 = k0

sub_sp_pml = bunch.Bunch()
sub_sp_pml.Mm=Mm
sub_sp_pml.Nm=Nm
sub_sp_pml.mu = np.array([[1.,1, 1., 1.]])#,
                  #[1.,1.]])

sub_sp_pml.eps = np.array([[eps_metal, eps_metal, eps_dielec, eps_dielec]])

sub_sp_pml.eta=eta
sub_sp_pml.pmly=[0]
sub_sp_pml.k0 = k0

list_periodx = np.linspace(100, 1400, 100)
r_gp = np.empty(list_periodx.size, dtype = complex)
n_gp = np.empty(list_periodx.size)
r_gp_pml = np.empty(list_periodx.size, dtype = complex)
n_gp_pml = np.empty(list_periodx.size)

for i, periodx in enumerate(list_periodx):
    l_pml = 0.3*periodx

    periody = periodx

    sub_sp.oy = [0,periody]
    sub_sp.ny = [0,periody]

    sub_sp_pml.oy = [0,periody]
    sub_sp_pml.ny = [0,periody]

    top_gp_pml.oy = [0,periody]
    top_gp_pml.ny = [0,periody]

    top_gp.oy = [0,periody]
    top_gp.ny = [0,periody]

    top_gp.ox = [0, periodx / 2, periodx / 2 + l_gap, periodx]
    top_gp.nx = [0,periodx / 2, periodx / 2 + l_gap, periodx]
    sub_sp.ox = [0, periodx / 2, periodx / 2 + l_gap, periodx]
    sub_sp.nx = [0,periodx / 2, periodx / 2 + l_gap, periodx]
    
    top_gp_pml.ox = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]
    top_gp_pml.nx = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]
    sub_sp_pml.ox = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]
    sub_sp_pml.nx = [0, l_pml, periodx / 2, periodx / 2 + l_gap, periodx]

    sub_sp_pml.pmlx=[1, 0, 0, 0]
    top_gp_pml.pmlx=[1, 0, 0, 0]

    a = -k0 * np.sin(theta) * np.cos(phi)
    top_gp.a0 = a
    sub_sp.a0 = a
    top_gp_pml.a0 = a
    sub_sp_pml.a0 = a

    b = -k0 * np.sin(theta) * np.sin(phi)
    top_gp.b0 = b
    sub_sp.b0 = b
    top_gp_pml.b0 = b
    sub_sp_pml.b0 = b
    
    [Pgp, Vgp] = base.reseau(top_gp)
    [Psp, Vsp] = base.reseau(sub_sp)
    index_gp = np.argmin(np.imag(Vgp))
    n_gp[i] = np.real(Vgp[index_gp] / top_gp.k0) # équivaut au kz

    S = base.interface(Pgp, Psp)
    r_gp[i] = S[index_gp, index_gp]

    [Pgp_pml, Vgp_pml] = base.reseau(top_gp_pml)
    [Psp_pml, Vsp_pml] = base.reseau(sub_sp_pml)
    index_gp_pml = np.argmin(np.imag(Vgp_pml))
    n_gp_pml[i] = np.real(Vgp_pml[index_gp_pml] / top_gp_pml.k0) # équivaut au kz

    S_pml = base.interface(Pgp_pml, Psp_pml)
    r_gp_pml[i] = S[index_gp_pml, index_gp_pml]

plt.figure(1)
plt.plot(list_periodx, n_gp, label = 'without pml')
plt.plot(list_periodx, n_gp_pml, label = 'with pml')
plt.xlabel("Period (nm)")
plt.legend()
plt.ylabel("$n_{GP}$")
plt.title("Effective index")
plt.show(block=False)
plt.savefig("Ngp_period_variable_renversed_system_PML0.3*periodx_gap3_5modes_pml1first_lam700.jpg")

plt.figure(2)
plt.plot(list_periodx, np.abs(r_gp) ** 2, label = 'without pml')
plt.plot(list_periodx, np.abs(r_gp_pml) ** 2, label = 'with pml')
plt.xlabel("Period(nm)")
plt.ylabel("$r_{GP}$")
plt.legend()
plt.title("Reflexion of GP")
plt.show(block=False)
plt.savefig("Rgp_period_variable_renversed_system_PML0.3*periodx_gap3_5modespml1first_lam700.jpg")
