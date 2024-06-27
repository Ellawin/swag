import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv

def eps_ITO(lam):
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
    return(ri ** 2)

def eps_TiN(lam):
    tableau3D = []

    file = open('../TiN_nk.txt', "r")    
    lines = file.readlines()
    file.close()

    nb_lines = len(lines)
    for idx in range (2,nb_lines-2):
        values = lines[idx].split("\t")
        values = [float(val.replace(',','.')) for val in values]
        tableau3D.append(values)
    
    tableau3D = np.array(tableau3D)
    wl = []
    wl = tableau3D[:,0]
    n_thermal = []
    n_thermal = tableau3D[:,1]
    k_thermal = []
    k_thermal = tableau3D[:,2]
    n_nh3 = []
    n_nh3 = tableau3D[:,4]
    k_nh3 = []
    k_nh3 = tableau3D[:,5]
    n_n2 = []
    n_n2 = tableau3D[:,7]
    k_n2 = []
    k_n2 = tableau3D[:,8]

    n_thermal_int = np.interp(lam, wl, n_thermal)
    k_thermal_int = np.interp(lam, wl, k_thermal)
    n_nh3_int = np.interp(lam, wl, n_nh3)
    k_nh3_int = np.interp(lam, wl, k_nh3)
    n_n2_int = np.interp(lam, wl, n_n2)
    k_n2_int = np.interp(lam, wl, k_n2)

    ri_thermal = n_thermal_int + 1.0j * k_thermal_int
    ri_nh3 = n_nh3_int + 1.0j * k_nh3_int
    ri_n2 = n_n2_int + 1.0j * k_n2_int
    #return(ri_thermal, ri_nh3, ri_n2)
    return(ri_thermal ** 2)

def eps_Tibb(lam):
    tableau3D = []

    file = open('Ti_Rakic-BB.txt', "r")    
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
    return(ri ** 2)

def epsAlbb(lam):
    "Permet de caluler la permittivité de l'argent en une longueur d'onde lam donnée"
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.526
    Gamma0=0.047
    omega_p=14.98
    f=np.array([0.213,0.060,0.182,0.014])
    Gamma=np.array([0.312,0.315,1.587,2.145])
    omega=np.array([0.163,1.561,1.827,4.495])
    sigma=np.array([0.013,0.042,0.256,1.735])
    a=np.sqrt(w*(w+1.0j*Gamma))
    a=a*np.sign(np.real(a))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Conversion
    aha=1.0j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64))
    epsilon=1-omega_p**2*f0/(w*(w+1.0j*Gamma0))+np.sum(aha)
    return(epsilon)

def faddeeva(z,N):
    "Bidouille les signes et les parties réelles et imaginaires d'un nombre complexe --> à creuser"
    w=np.zeros(z.size,dtype=complex)

    idx=np.real(z)==0
    w[idx]=np.exp(np.abs(-z[idx]**2))*(1-erf(np.imag(z[idx])))
    idx=np.invert(idx)
    idx1=idx + np.imag(z)<0

    z[idx1]=np.conj(z[idx1])

    M=2*N
    M2=2*M
    k = np.arange(-M+1,M)
    L=np.sqrt(N/np.sqrt(2))

    theta=k*np.pi/M
    t=L*np.tan(theta/2)
    f=np.exp(-t**2)*(L**2+t**2)
    f=np.append(0,f)
    a=np.real(np.fft.fft(np.fft.fftshift(f)))/M2
    a=np.flipud(a[1:N+1])

    Z=(L+1.0j*z[idx])/(L-1.0j*z[idx])
    p=np.polyval(a,Z)
    w[idx]=2*p/(L-1.0j*z[idx])**2+(1/np.sqrt(np.pi))/(L-1.0j*z[idx])
    w[idx1]=np.conj(2*np.exp(-z[idx1]**2) - w[idx1])
    return(w)

material_list = ['Silver', 1.45 ** 2, 'Gold', epsAlbb, eps_ITO, 1.5 ** 2]
stack_glass = [0, 1,2,3, 5]
stack_ito = [0,1,2,4]
    
start_index_eff = 30
tol = 1e-12
step_max = 10000

# list_metal = np.linspace(1,50,100)
# list_gap = np.linspace(5,10,5)

# n_gp_glass = np.empty((list_gap.size, list_metal.size), dtype = complex)
# n_gp_ito = np.empty((list_gap.size, list_metal.size), dtype = complex)

polarization = 1

#list_wavelength = np.linspace(450,1000,200)
wavelength = 800

# for idx_gap, thick_gap in enumerate(list_gap):
#     for idx_metal, thick_metal in enumerate(list_metal):
#         thicknesses_glass = [200,thick_gap,thick_metal,3, 200]
#         thicknesses_ito = [200,thick_gap, thick_metal, 200]
#         Layers_glass = pm.Structure(material_list, stack_glass, thicknesses_glass)
#         Layers_ito = pm.Structure(material_list, stack_ito, thicknesses_ito)
#         n_gp_glass[idx_gap, idx_metal] = pm.steepest(4,1e-12,10000,Layers_glass,wavelength,polarization)
#         n_gp_ito[idx_gap, idx_metal] = pm.steepest(4,1e-12,10000,Layers_ito,wavelength,polarization)

# plt.figure(10)
# for idx_gap in np.arange(list_gap.size):
#     plt.plot(list_metal, np.abs(n_gp_glass[idx_gap])**2, label = f"thickness gap {int(list_gap[idx_gap])} nm")

# plt.legend()
# plt.ylabel("$n_{gp}$")
# plt.xlabel("Thickness Au (nm)")
# plt.title("Glass / Al / Au / Effective index / PyMoosh")

# plt.show(block=False)
# plt.savefig("ngp_Glass_Al_Au_gap_metal_lam600.pdf")
# plt.savefig("ngp_Glass_Al_Au_gap_metal_lam600.jpg")

# plt.figure(11)
# for idx_gap in np.arange(list_gap.size):
#     plt.plot(list_metal, np.abs(n_gp_ito[idx_gap])**2, label = f"thickness gap {int(list_gap[idx_gap])} nm")

# plt.legend()
# plt.ylabel("$n_{gp}$")
# plt.xlabel("Thickness Au (nm)")
# plt.title("ITO / Au / Effective index / PyMoosh")

# plt.show(block=False)
# plt.savefig("ngp_ITO_Au_gap_metal_lam600.pdf")
# plt.savefig("ngp_ITO_Au_gap_metal_lam600.jpg")

# np.savez("data_ngp_Au_gap_metal_lam600_Glass-ITO.npz", list_gap = list_gap, list_metal = list_metal, n_gp_glass = n_gp_glass, n_gp_ito = n_gp_ito)

thick_gap = 5
thick_metal = 10
thicknesses_ito = [100,thick_gap,thick_metal,100]
thicknesses_glass = [100,thick_gap,thick_metal,3,100]

Layers_ito = pm.Structure(material_list, stack_ito, thicknesses_ito)
Layers_glass = pm.Structure(material_list, stack_glass, thicknesses_glass)

n_gp_ito = pm.steepest(30,1e-12,10000,Layers_ito,wavelength,polarization)
x_ito,profil_ito = pm.profile(Layers_ito,n_gp_ito,wavelength,polarization,pixel_size = 0.05)

plt.figure(5)
plt.plot(x_ito,np.real(profil_ito),linewidth = 2)
plt.ylabel("Amplitude du champ")
plt.xlabel("Position in the structure(nm)")
plt.title("ITO - Au / Effective index profile / PyMoosh / lam 800 nm")

plt.show(block=False)
plt.savefig("Profile_ngp_ITO-Au_lam800_zoom.pdf")
plt.savefig("Profile_ngp_ITO-Au_lam800_zoom.jpg")

n_gp_glass = pm.steepest(30,1e-12,10000,Layers_glass,wavelength,polarization)
x_glass,profil_glass = pm.profile(Layers_glass,n_gp_glass,wavelength,polarization,pixel_size = 0.05)

plt.figure(6)
plt.plot(x_glass,np.real(profil_glass),linewidth = 2)
plt.ylabel("Amplitude du champ")
plt.xlabel("Position in the structure(nm)")
plt.title("Glass - Al - Au / Effective index profile / PyMoosh / lam 800 nm")

plt.show(block=False)
plt.savefig("Profile_ngp_Glass-Al-Au_lam800_zoom.pdf")
plt.savefig("Profile_ngp_Glass-Al-Au_lam800_zoom.jpg")

np.savez("data_Profile_ngp_Glass-Al-Au_lam800_zoom.npz", x_ito = x_ito, profil_ito = profil_ito, x_glass = x_glass, profil_glass = profil_glass)
