import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
import PyMoosh as pm

np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

# materials
def nk_ITO(lam):
    tableau3D = []

    file = open('ITO.txt', "r")    
    lines = file.readlines()
    file.close()

    nb_lines = len(lines)
    for idx in range (2,nb_lines-2):
        values = lines[idx].split("\t")
        values = [float(val) for val in values]
        tableau3D.append(values)
    
    tableau3D = np.array(tableau3D)
    wl = []
    wl = tableau3D[:,0]
    n = []
    n = tableau3D[:,1]
    k = []
    k = tableau3D[:,2]

    n_int = np.interp(lam, wl, n)
    k_int = np.interp(lam, wl, k)

    ri = n_int + 1.0j * k_int
    return(ri)

# Modal 
Mm = 20
Nm = 0
eta = 0.999 # stretching

# Wave
wavelength = 700.123
#theta = 0.0 * np.pi/180. #latitude (z)
phi = 90.0 * np.pi/180. # précession (xy)
#pol = 0*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola
k0 = 2*np.pi/wavelength

# materials
eps_dielec = 1.45 ** 2
eps_ag = mat.epsAgbb(wavelength)
eps_au = mat.epsAubb(wavelength)
#eps_ito = nk_ITO(wavelength) ** 2
eps_al = mat.epsAlbb(wavelength)
eps_sio2 = 1.5 ** 2

# geometry
l_gap = 3.215
l_pml = 40

top_gp = bunch.Bunch()
top_gp.Mm=Mm
top_gp.Nm=Nm
top_gp.mu =  np.array([[1.,1., 1., 1., 1., 1.]])#,
                  #[1.,1.]])
top_gp.eps =  np.array([[eps_ag, eps_dielec, eps_au, eps_al, eps_sio2, eps_sio2]])#,
                  #[1.,1.]])
top_gp.eta=eta
top_gp.pmlx=[0, 0, 0, 0, 0, 1]
top_gp.pmly=[0]
top_gp.k0 = k0

sub_sp = bunch.Bunch()
sub_sp.Mm=Mm
sub_sp.Nm=Nm
sub_sp.mu = np.array([[1., 1.,1, 1., 1., 1.]])#,
                  #[1.,1.]])

sub_sp.eps = np.array([[eps_dielec, eps_dielec, eps_au, eps_al, eps_sio2, eps_sio2]])

sub_sp.eta=eta
sub_sp.pmlx=[0, 0, 0, 0, 0, 1]
sub_sp.pmly=[0]
sub_sp.k0 = k0

periody = 10

sub_sp.oy = [0,periody]
sub_sp.ny = [0,periody]

top_gp.oy = [0,periody]
top_gp.ny = [0,periody]

size_nanocube = 50
l_metal = 5
size_sub = 50
l_accroche = 3

top_gp.ox = [0, size_nanocube , size_nanocube + l_gap, size_nanocube + l_gap + l_metal, size_nanocube + l_gap, size_nanocube + l_gap + l_metal + l_accroche, size_nanocube + l_gap + l_metal + size_sub + l_accroche, l_accroche + size_nanocube + l_gap + l_metal + size_sub + l_pml]
top_gp.nx = [0, size_nanocube , size_nanocube + l_gap, size_nanocube + l_gap + l_metal, size_nanocube + l_gap, size_nanocube + l_gap + l_metal + l_accroche, size_nanocube + l_gap + l_metal + size_sub + l_accroche, l_accroche + size_nanocube + l_gap + l_metal + size_sub + l_pml]
sub_sp.ox = [0, size_nanocube , size_nanocube + l_gap, size_nanocube + l_gap + l_metal, size_nanocube + l_gap, size_nanocube + l_gap + l_metal + l_accroche, size_nanocube + l_gap + l_metal + size_sub + l_accroche, l_accroche + size_nanocube + l_gap + l_metal + size_sub + l_pml]
sub_sp.nx = [0, size_nanocube , size_nanocube + l_gap, size_nanocube + l_gap + l_metal, size_nanocube + l_gap, size_nanocube + l_gap + l_metal + l_accroche, size_nanocube + l_gap + l_metal + size_sub + l_accroche, l_accroche + size_nanocube + l_gap + l_metal + size_sub + l_pml]

a = 0
top_gp.a0 = a
sub_sp.a0 = a

list_theta = np.linspace(0,90, 100) / 180 * np.pi
r_gp_lgap = np.empty(list_theta.size, dtype = complex)
n_gp_lgap = np.empty(list_theta.size)

top_gp.a0 = 0
top_gp.b0 = 0

### départ avec pymoosh
material_list = [1.45, 'Silver', 'Gold'] #, np.sqrt(eps_ag), np.sqrt(eps_sio2)]
layer_down = [1,0,2] #, 3, 4]
start_index_eff = 4
tol = 1e-12
step_max = 100000
thicknesses_down = [200,l_gap,l_metal, l_accroche, 200]
Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
neff_pm = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, 1)

print("neff_pm = ", neff_pm)

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
    n_gp_lgap[idx] = np.real(Vgp[index_gp] / top_gp.k0) # équivaut au kz/k0

    S = base.interface(Pgp, Psp)
    r_gp_lgap[idx] = S[index_gp, index_gp]

### Les figures
plt.figure(1)
plt.plot(list_theta * 180 / np.pi, n_gp_lgap) #, label = 'Gap 10 nm')
plt.xlabel("Theta")
plt.legend()
plt.ylabel("$n_{GP}$")
plt.title("Effective index")
plt.show(block=False)
plt.savefig("Ngp_theta_gap3_20modes_lam700_phi90_pml40_realneff_si2o_accAl3_metal5.jpg")

plt.figure(2)
plt.plot(list_theta * 180 / np.pi, np.abs(r_gp_lgap) ** 2) #, label = 'Gap 10 nm')
plt.xlabel("Theta")
plt.ylabel("$r_{GP}$")
plt.legend()
plt.ylim([0,1])
plt.title("Reflexion of GP")
plt.show(block=False)
plt.savefig("Rgp_theta_gap3_20modes_lam700_phi90_pml40_realneff_si2o_accAl3_metal5.jpg")

plt.figure(3)
plt.plot(list_theta * 180 / np.pi, np.angle(r_gp_lgap)) #, label = 'Gap 10 nm')
plt.xlabel("Theta")
plt.ylabel("$r_{GP}$")
plt.legend()
plt.title("Reflexion of GP (phase)")
plt.show(block=False)
plt.savefig("RgpPhi_theta_gap3_20modes_lam700_phi90_pml40_realneff_si2o_accAl3_metal5.jpg")

np.savez("data_theta_gap3_20modes_lam700_phi90_pml40_realneff_si2o_accAl3_metal5.npz", list_theta=list_theta, r_gp_lgap = r_gp_lgap, n_gp_lgap = n_gp_lgap)
