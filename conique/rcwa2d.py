import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

def epsAgbb(lam):
    "Permet de caluler la permittivité de l'argent en une longueur d'onde lam donnée"
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.821
    Gamma0=0.049
    omega_p=9.01
    f=np.array([0.050,0.133,0.051,0.467,4.000])
    Gamma=np.array([0.189,0.067,0.019,0.117,0.052])
    omega=np.array([2.025,5.185,4.343,9.809,18.56])
    sigma=np.array([1.894,0.665,0.189,1.170,0.516])
    a=np.sqrt(w*(w+i*Gamma))
    a=a*np.sign(np.real(a))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Conversion
    aha=i*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64))
    epsilon=1-omega_p**2*f0/(w*(w+i*Gamma0))+np.sum(aha)
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

    Z=(L+i*z[idx])/(L-i*z[idx])
    p=np.polyval(a,Z)
    w[idx]=2*p/(L-i*z[idx])**2+(1/np.sqrt(np.pi))/(L-i*z[idx])
    w[idx1]=np.conj(2*np.exp(-z[idx1]**2) - w[idx1])
    return(w)

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

def step(a,b,w,x0,n):
    '''Computes the Fourier series for a piecewise function having the value
    b over a portion w of the period, starting at position x0
    and the value a otherwise. The period is supposed to be equal to 1.
    Then returns the toeplitz matrix generated using the Fourier series.
    '''
    from scipy.linalg import toeplitz
    from numpy import sinc
    l=np.zeros(n,dtype=np.complex128)
    m=np.zeros(n,dtype=np.complex128)
    tmp=np.exp(-2*1j*np.pi*(x0+w/2)*np.arange(0,n))*sinc(w*np.arange(0,n))*w
    l=np.conj(tmp)*(b-a)
    m=tmp*(b-a)
    l[0]=l[0]+a
    m[0]=l[0]
    T=toeplitz(l,m)
    return T

def grating(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position.
    Warning : There is nothing checking that the blocks don't overlapp.
    '''
    n_blocs=blocs.shape[0];
    nmod=int(n/2)
    M1=e1*np.eye(n,n)
    M2=1/e1*np.eye(n,n)
    for k in range(0,n_blocs):
        M1=M1+step(0,e2-e1,blocs[k,0],blocs[k,1],n)
        M2=M2+step(0,1/e2-1/e1,blocs[k,0],blocs[k,1],n)
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))+0j
    if (pol==0):
        M=alpha*alpha-k0*k0*M1
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        T=np.linalg.inv(M2)
        M=np.matmul(np.matmul(np.matmul(T,alpha),np.linalg.inv(M1)),alpha)-k0*k0*T
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(M2,E),np.diag(L))]])
    return P,L

def homogene(k0,a0,pol,epsilon,n):
    nmod=int(n/2)
    valp=np.sqrt(epsilon*k0*k0-(a0+2*np.pi*np.arange(-nmod,nmod+1))**2+0j)
    valp=valp*(1-2*(valp<0))
    P=np.block([[np.eye(n)],[np.diag(valp*(pol/epsilon+(1-pol)))]])
    return P,valp

def reflectance(width_reso, width_gap, width_metallicLayer, period, perm_dielec, perm_metal, angle, wavelength, polarization, n_mod):    
    n = 2 * n_mod + 1
    
    ## trouver le mode du GP avec PyMoosh
    material_list = [1., 'Silver']
    layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    #start_index_eff = 4
    #tol = 1e-12
    #step_max = 50000
    #thicknesses_gp = [300,10,300]
    #Layer_gp = pm.Structure(material_list, layer_down, thicknesses_gp)
    #GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_gp, wavelength, polarization)
    #print("GP effective index = ", GP_effective_index)

    # Adimensionalisation
    wavelength_norm = wavelength / period
    
    width_reso = width_reso / period
    width_gap = width_gap / period
    width_metallicLayer = width_metallicLayer / period

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle * np.pi / 180)

    Pgp, Vgp = grating(k0, a0, polarization, perm_metal, perm_dielec, n, np.array([[width_gap, width_reso]]))
    Pplan, Vplan = grating(k0, a0, polarization, perm_metal, perm_dielec, n, np.array([[width_gap + width_reso, 0]]))

    S = interface(Pgp,Pplan)
    ## PM, quand on travaille à longueur d'onde fixée et qu'on a calculé l'indice effectif une fois pour toute
    #GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    #position_GP = np.argmin(abs(Vgp - GP_effective_index * k0))
    
    position_GP = np.argmin(np.imag(Vgp))
    neff_GP = np.real(Vgp[position_GP] / k0)

    position_SP = np.argmin(np.imag(Vplan))
    neff_plan = np.real(Vplan[position_SP] / k0)

    r = S[position_GP, position_GP]

    R_GP = abs(r) ** 2
    phase_R_GP = np.angle(r)

    return r, R_GP, phase_R_GP, position_GP, neff_GP

width_reso = 150
width_gap = 10
width_metallicLayer = 140

angle = 0
polarization = 1

perm_dielec = 1

n_mod = 150

# Etude de la valeur du coefficient de réflexion à longueur d'onde et période fixée pour comparaison avec oc_rgp_var_pml.m
period = 300
wavelength = 600
perm_metal = epsAgbb(wavelength)
r, RGP0, RGP_phase0, position_GP_func, neff_GP = reflectance(width_reso, width_gap, width_metallicLayer, period, perm_dielec, perm_metal, angle, wavelength, polarization, n_mod)

print("RGP0 = ", RGP0)
print("phase = ", RGP_phase0)

# ### Etude de la variation de la permittivité de l'argent en fonction de la longueur d'onde
# period = 300
# list_wavelength = np.linspace(300,900,100)
# perm_metal = np.empty(list_wavelength.size, dtype = complex)
# idx = 0
# for wavelength in list_wavelength:
#     perm_metal[idx] = epsAgbb(wavelength)
#     idx += 1

# plt.figure(1)
# plt.plot(list_wavelength, perm_metal)
# plt.ylabel("Real Permittivity")
# plt.title("Permittivity of the silver")
# plt.xlabel("Wavelength (nm)")
# plt.show(block=False)
# plt.savefig("silver_perm_python.jpg")

# ### Etude de la variation de la réflexion du GP en fonction de la periode
# wavelength = 600
# perm_metal = epsAgbb(wavelength)

# list_period = np.linspace(100,600,100)
# RGP = np.empty(list_period.size)
# RGP_phase = np.empty(list_period.size)
# idx = 0
# for period in list_period:
#     r, RGP[idx], RGP_phase[idx], position_GP_func = reflectance(width_reso, width_gap, width_metallicLayer, period, perm_dielec, perm_metal, angle, wavelength, polarization, n_mod)
#     idx += 1

# plt.figure(2)
# plt.subplot(211)
# plt.plot(list_period, RGP)
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# plt.subplot(212)
# plt.plot(list_period, RGP_phase)
# plt.xlabel("Period (nm)")
# plt.ylabel('Phase')
# plt.show(block=False)
# plt.savefig("R_gp_rcwa2D_depending_period.jpg")

# ## Validation de la fonction reflectance : OK

#     # avec la fonction
# r_func, R_GP_func, phase_R_GP_func, position_GP_func = reflectance(width_reso, width_gap, width_metallicLayer, period, perm_dielec, perm_metal, angle, wavelength, polarization, n_mod)

# print("avec la fonction")
# print("R_GPf = ", R_GP_func)
# print("phase_R_GPf =", phase_R_GP_func)
# print("rf = ", r_func)
# print("position GP =", position_GP_func)

### Etude de la variation de la réflexion du GP en fonction de la longueur d'onde
# period = 300
# list_wavelength = np.linspace(300,900,100)
# RGP = np.empty(list_wavelength.size)
# RGP_phase = np.empty(list_wavelength.size)
# idx = 0
# neff_GP = np.empty(list_wavelength.size, dtype = complex)
# for wavelength in list_wavelength:
#     perm_metal = epsAgbb(wavelength)
#     #print("perm_metal = ", perm_metal)
#     r, RGP[idx], RGP_phase[idx], position_GP_func, neff_GP[idx] = reflectance(width_reso, width_gap, width_metallicLayer, period, perm_dielec, perm_metal, angle, wavelength, polarization, n_mod)
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_wavelength, np.real(neff_GP))
# plt.ylabel("Real part")
# plt.title("Effectiv index of the GP")
# plt.subplot(212)
# plt.plot(list_wavelength, np.imag(neff_GP))
# plt.ylabel("Imaginary part")
# plt.xlabel("Wavelength (nm)")
# plt.show(block=False)
# plt.savefig("neff_GP_python.jpg")

# plt.figure(2)
# plt.subplot(211)
# plt.plot(list_wavelength, RGP)
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# plt.subplot(212)
# plt.plot(list_wavelength, RGP_phase)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel('Phase')
# plt.show(block=False)
# plt.savefig("R_gp_rcwa2D_depending_lambda.jpg")

# ### Validation de la fonction reflectance : OK

#     ## avec la fonction
# #r_func, R_GP_func, phase_R_GP_func, position_GP_func = reflectance(width_reso, width_gap, width_metallicLayer, period, perm_dielec, perm_metal, angle, wavelength, polarization, n_mod)

# # print("avec la fonction")
# # print("R_GPf = ", R_GP_func)
# # print("phase_R_GPf =", phase_R_GP_func)
# # print("rf = ", r_func)
# # print("position GP =", position_GP_func)

