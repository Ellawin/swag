import numpy as np 
import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import matplotlib.pyplot as plt 

### Modal parameters
Mm = 3 # pour test, sinon trop peu (viser plutôt 150 ici)
Nm = 0
eta = 0.999 # stretching

### Wave
wavelength = 600.123
#theta = 0.0 * np.pi/180. #latitude (z)
phi = 90.0 * np.pi/180. # précession (xy)
k0 = 2*np.pi/wavelength
list_theta = np.linspace(0,90, 100) / 180 * np.pi

### materials
eps_dielec = 1
eps_metal = mat.epsAgbb(wavelength)

### geometry 
#list_gap = np.linspace(3,11,10)
l_gap = 10.215 # dielectric gap
l_pml = 500 # Perfected Matched Layers pour isoler un systeme

# Couche supérieure avec nanocube et PML
top_gp = bunch.Bunch() 
top_gp.Mm=Mm 
top_gp.Nm=Nm
top_gp.mu =  np.array([[1.,1., 1., 1.]])
top_gp.eps =  np.array([[eps_metal, eps_metal, eps_dielec, eps_metal]])
top_gp.eta=eta
top_gp.pmlx=[1, 0, 0, 0]
top_gp.pmly=[0]
top_gp.k0 = k0

# Couche inférieure avec couche métallique seulement, et PML
sub_sp = bunch.Bunch()
sub_sp.Mm=Mm
sub_sp.Nm=Nm
sub_sp.mu = np.array([[1., 1.,1, 1.]])
sub_sp.eps = np.array([[eps_dielec, eps_dielec, eps_dielec, eps_metal]])
sub_sp.eta=eta
sub_sp.pmlx=[1, 0, 0, 0]
sub_sp.pmly=[0]
sub_sp.k0 = k0

# Periodes 
periodx = 3300
periody = 10

sub_sp.oy = [0,periody]
sub_sp.ny = [0,periody]

top_gp.oy = [0,periody]
top_gp.ny = [0,periody]

# Design
top_gp.ox = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
top_gp.nx = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
sub_sp.ox = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]
sub_sp.nx = [0, l_pml, l_pml + 2500 , l_pml + 2500 + l_gap, l_pml + 2500 + l_gap + 290]

top_gp.a0 = 0
top_gp.b0 = 0

sub_sp.a0 = 0
sub_sp.b0 = 0

### Indices effectifs PyMoosh (système 1D infini)
#neff_pm = 5.981385621498812+0.33776911019095557j # lam = 600, gap 3 nm
neff_pm = 2.797602583949803+0.08671471953322236j # lam = 600 nm, gap = 10 nm
print("neff_pm = ", neff_pm)

### RCWA 3D

# Initialisation couche supérieure
[P0, V0] = base.reseau(top_gp)
index0 = np.argmin(abs(V0 - neff_pm * top_gp.k0))
neff = V0[index0] / top_gp.k0
print("neff = ", neff)

### calcul le coefficient de réflexion pour un angle (donc un ky) donné
def reflexion_gp(ky):
    top_gp.b0 = ky
    sub_sp.b0 = ky

    [Pgp, Vgp] = base.reseau(top_gp)
    [Psp, Vsp] = base.reseau(sub_sp)
    index_gp_pm = np.argmin(abs(Vgp - neff * top_gp.k0))
    neff_gp = np.real(Vgp[index_gp_pm] / top_gp.k0)
    S = base.interface(Pgp, Psp)
    r_gp = S[index_gp_pm, index_gp_pm]
    return(r_gp, neff_gp)

### calcul la relation de dispersion
def dispersion(ky):
    r, n = reflexion_gp(ky)
    func = np.abs(1 - r ** 2 * np.exp(2j*np.sqrt(neff_pm**2 * k0**2 - ky**2) * l_gap))
    return func, r, n 

start_ky = -np.real(neff) * k0 * np.sin(60 * np.pi / 180)

### descente de gradient pour résoudre la relation de dispersion
def steepest(neff,tol,step_max):
    start_ky = -np.real(neff) * k0 * np.sin(60 * np.pi / 180)
    z = start_ky # ky
    delta = abs(z) * 0.001
    dz= 0.01 * delta
    step = 0
    current = dispersion(z)

    while (current > tol) and (step < step_max):

        grad = (
        dispersion(z+dz)
        -current
        +1j*(dispersion(z+1j*dz)
        -current)
        )/(dz)

        if abs(grad)!=0 :
            z_new = z - delta * grad / abs(grad)
        else:
            # We have a finishing condition not linked to the gradient
            # So if we meet a gradient of 0, we just divide the step by two
            delta = delta/2.
            z_new = z

        value_new = dispersion(z_new)
        if (value_new > current):
            # The path not taken
            delta = delta / 2.
            dz = dz / 2.
        else:
            current = value_new
            z = z_new
    #        print("Step", step, z,current)
        step = step + 1

    #print("End of the loop")
    if step == step_max:
        print("Warning: maximum number of steps reached")

    return z

# trouve les modes guidés dans une structure
def guided_modes(ky_min,ky_max,initial_points = 1):
    tolerance = 1e-8
    ky_start = np.linspace(ky_min,ky_max,initial_points,dtype=complex)
    modes=[]
    for idx, ky in enumerate(ky_start):
        print(idx)
        solution = steepest(ky,tolerance,1000)
        if (len(modes)==0):
            modes.append(solution)
        elif (min(abs(modes-solution))>1e-5*k0):
            modes.append(solution)
    return modes

list_ky = -np.real(neff) * k0 * np.sin(list_theta)

func = np.empty(list_theta.size, dtype = complex)
r = np.empty(list_theta.size, dtype = complex)
n = np.empty(list_theta.size, dtype = complex)

for idx_ky, ky in enumerate(list_ky):
    print(idx_ky)
    func[idx_ky], r[idx_ky], n[idx_ky] = dispersion(ky)

plt.figure(1)
plt.plot(list_theta * 180 / np.pi, np.abs(func)**2)
plt.xlabel("Theta")
plt.ylabel("Dispersion")
plt.show(block=False)
#plt.savefig("dispersion_theta_dimYanisse_neffgpreal_100points_gap3.jpg")

plt.figure(2)
plt.plot(list_theta * 180 / np.pi, np.abs(r)**2)
plt.xlabel("Theta")
plt.ylabel("Reflexion")
plt.show(block=False)
#plt.savefig("reflexion_theta_dimYanisse_neffgpreal_100points_gap3.jpg")

plt.figure(3)
plt.plot(list_theta * 180 / np.pi, np.abs(n)**2)
plt.xlabel("Theta")
plt.ylabel("Indice effectif")
plt.show(block=False)
#plt.savefig("neff_theta_dimYanisse_neffgpreal_100points_gap3.jpg")
