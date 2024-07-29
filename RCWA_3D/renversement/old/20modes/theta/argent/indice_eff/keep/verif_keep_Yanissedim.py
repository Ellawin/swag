import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
import PyMoosh as pm

np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

# Modal 
Mm = 150
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
l_gap = 10.215
#l_gap3 = 3.215
l_pml = 500

top_gp = bunch.Bunch()
top_gp.Mm=Mm
top_gp.Nm=Nm
top_gp.mu =  np.array([[1.,1., 1., 1.]])#,
                  #[1.,1.]])
top_gp.eps =  np.array([[eps_metal, eps_metal, eps_dielec, eps_metal]])#,
                  #[1.,1.]])
top_gp.eta=eta
top_gp.pmlx=[1, 0, 0, 0]
top_gp.pmly=[0]
top_gp.k0 = k0

sub_sp = bunch.Bunch()
sub_sp.Mm=Mm
sub_sp.Nm=Nm
sub_sp.mu = np.array([[1., 1.,1, 1.]])#,
                  #[1.,1.]])

sub_sp.eps = np.array([[eps_dielec, eps_dielec, eps_dielec, eps_metal]])

sub_sp.eta=eta
sub_sp.pmlx=[1, 0, 0, 0]
sub_sp.pmly=[0]
sub_sp.k0 = k0

sub_sp_pml = bunch.Bunch()
sub_sp_pml.Mm=Mm
sub_sp_pml.Nm=Nm
sub_sp_pml.mu = np.array([[1.,1, 1., 1.]])#,
                  #[1.,1.]])

periodx = 3300
#r_gp = np.empty(list_periodx.size, dtype = complex)
#n_gp = np.empty(list_periodx.size)

periody = 10

sub_sp.oy = [0,periody]
sub_sp.ny = [0,periody]

top_gp.oy = [0,periody]
top_gp.ny = [0,periody]

top_gp.ox = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
top_gp.nx = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
sub_sp.ox = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
sub_sp.nx = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]

a = 0
top_gp.a0 = a
sub_sp.a0 = a

list_theta = np.linspace(0,90, 100) / 180 * np.pi
r_gp_lgap10 = np.empty(list_theta.size, dtype = complex)
n_gp_lgap10 = np.empty(list_theta.size)

top_gp.a0 = 0
top_gp.b0 = 0

### départ avec pymoosh
material_list = [1., 'Silver']
layer_down = [1,0,1]
start_index_eff = 4
tol = 1e-12
step_max = 100000
thicknesses_down = [200,10,200]
Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
neff_pm = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, 1)
print("neff_pm = ", neff_pm)

[P0, V0] = base.reseau(top_gp)
index0 = np.argmin(abs(V0 - neff_pm * top_gp.k0))
neff = V0[index0] / top_gp.k0
print("neff = ", neff)

b = np.empty(list_theta.size)

for idx, theta in enumerate(list_theta):
    print(idx)

###kz = neff cos theta * k0 

    b[idx] = - np.real(neff) * k0 * np.sin(theta) # * np.sin(phi)
    top_gp.b0 = b[idx]
    sub_sp.b0 = b[idx]
    
    [Pgp, Vgp] = base.reseau(top_gp)
    [Psp, Vsp] = base.reseau(sub_sp)
    index_gp = np.argmin(abs(Vgp - neff * top_gp.k0))
    n_gp_lgap10[idx] = np.real(Vgp[index_gp] / top_gp.k0) # équivaut au kz/k0

    S = base.interface(Pgp, Psp)
    r_gp_lgap10[idx] = S[index_gp, index_gp]

"""
### Gap 3 à partir de maintenant
l_gap = 3

top_gp.ox = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
top_gp.nx = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
sub_sp.ox = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
sub_sp.nx = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]

a = 0
top_gp.a0 = a
sub_sp.a0 = a

list_theta = np.linspace(0,90, 100) / 180 * np.pi
r_gp_lgap3 = np.empty(list_theta.size, dtype = complex)
n_gp_lgap3 = np.empty(list_theta.size)

top_gp.a0 = 0
top_gp.b0 = 0

### départ avec pymoosh
material_list = [1., 'Silver']
layer_down = [1,0,1]
start_index_eff = 4
tol = 1e-12
step_max = 100000
thicknesses_down = [200,3,200]
Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
neff_pm = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, 1)

[P0, V0] = base.reseau(top_gp)
index0 = np.argmin(abs(V0 - neff_pm * top_gp.k0))
neff = V0[index0] / top_gp.k0

for idx, theta in enumerate(list_theta):
    print(idx)

###kz = neff cos theta * k0 

    b = - np.real(neff) * k0 * np.sin(theta) # * np.sin(phi)
    top_gp.b0 = b
    sub_sp.b0 = b
    
    [Pgp, Vgp] = base.reseau(top_gp)
    [Psp, Vsp] = base.reseau(sub_sp)
    index_gp = np.argmin(abs(Vgp - neff * top_gp.k0))
    n_gp_lgap3[idx] = np.real(Vgp[index_gp] / top_gp.k0) # équivaut au kz/k0

    S = base.interface(Pgp, Psp)
    r_gp_lgap3[idx] = S[index_gp, index_gp]
"""

### Les figures
plt.figure(8)
plt.plot(list_theta * 180 / np.pi, b)
plt.xlabel("Theta")
plt.ylabel("b")
plt.show(block=False) 

plt.figure(4)
plt.plot(list_theta * 180 / np.pi, n_gp_lgap10, label = 'Gap 10 nm')
#plt.plot(list_theta * 180 / np.pi, n_gp_lgap3, label = 'Gap 3 nm')
plt.xlabel("Theta")
plt.legend()
plt.ylabel("$n_{GP}$")
plt.title("Effective index")
plt.show(block=False)
#plt.savefig("Ngp_theta_gap10-3_150modes_lam600_period3300_phi90_pml500_KEEP_realneff_dimYanisse.jpg")

plt.figure(5)
plt.plot(list_theta * 180 / np.pi, np.abs(r_gp_lgap10) ** 2, label = 'Gap 10 nm')
#plt.plot(list_theta * 180 / np.pi, np.abs(r_gp_lgap3) ** 2, label = 'Gap 3 nm')
plt.xlabel("Theta")
plt.ylabel("$r_{GP}$")
plt.legend()
plt.ylim([0,1])
plt.title("Reflexion of GP")
plt.show(block=False)
#plt.savefig("Rgp_theta_gap10-3_150modes_lam600_period3300_phi90_pml500_KEEP_realneff_dimYanisse.jpg")

plt.figure(6)
plt.plot(list_theta * 180 / np.pi, np.angle(r_gp_lgap10), label = 'Gap 10 nm')
#plt.plot(list_theta * 180 / np.pi, np.angle(r_gp_lgap3), label = 'Gap 3 nm')
plt.xlabel("Theta")
plt.ylabel("$r_{GP}$")
plt.legend()
plt.title("Reflexion of GP (phase)")
plt.show(block=False)
plt.savefig("RgpPhi_theta_gap10-3_150modes_lam600_period3300_phi90_pml500_KEEP_realneff_dimYanisse.jpg")

#np.savez("data_theta_gap10-3_150modes_lam600_phi90_period3300_pml500_KEEP_realneff_dimYanisse.npz", list_theta=list_theta, r_gp_lgap10 = r_gp_lgap10, n_gp_lgap10 = n_gp_lgap10, r_gp_lgap3 = r_gp_lgap3, n_gp_lgap3 = n_gp_lgap3)
