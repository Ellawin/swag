import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv

def nk_TiN(lam):
    tableau3D = []

    file = open('TiN_nk.txt', "r")    
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
    # n_nh3 = []
    # n_nh3 = tableau3D[:,4]
    # k_nh3 = []
    # k_nh3 = tableau3D[:,5]
    # n_n2 = []
    # n_n2 = tableau3D[:,7]
    # k_n2 = []
    # k_n2 = tableau3D[:,8]

    n_thermal_int = np.interp(lam, wl, n_thermal)
    k_thermal_int = np.interp(lam, wl, k_thermal)
    # n_nh3_int = np.interp(lam, wl, n_nh3)
    # k_nh3_int = np.interp(lam, wl, k_nh3)
    # n_n2_int = np.interp(lam, wl, n_n2)
    #k_n2_int = np.interp(lam, wl, k_n2)

    ri_thermal = n_thermal_int + 1.0j * k_thermal_int
    #ri_nh3 = n_nh3_int + 1.0j * k_nh3_int
    #ri_n2 = n_n2_int + 1.0j * k_n2_int
    return(ri_thermal) #, ri_nh3, ri_n2)
    #return(ri_thermal ** 2)

def nk_Tibb(lam):
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
    return(ri)

def epsAubb(lam):

    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam

    f0 = 0.770
    Gamma0 = 0.050
    omega_p = 9.03
    f = np.array([0.054,0.050,0.312,0.719,1.648])
    Gamma = np.array([0.074,0.035,0.083,0.125,0.179])
    omega = np.array([0.218,2.885,4.069,6.137,27.97])
    sigma = np.array([0.742,0.349,0.830,1.246,1.795])

    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    # Conversion

    epsilon = 1-omega_p**2*f0/(w*(w+1.0j*Gamma0))+np.sum(1.0j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64)))
    return(epsilon) 

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
    a=np.sqrt(w*(w+1.0j*Gamma))
    a=a*np.sign(np.real(a))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Conversion
    aha=1.0j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64))
    epsilon=1-omega_p**2*f0/(w*(w+1.0j*Gamma0))+np.sum(aha)
    return(epsilon)

def epsCrbb(lam):
    "Permet de caluler la permittivité de l'argent en une longueur d'onde lam donnée"
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.154
    Gamma0=0.048
    omega_p=10.75
    f=np.array([0.338,0.261,0.817,0.105])
    Gamma=np.array([4.256,3.957,2.218,6.983])
    omega=np.array([0.281,0.584,1.919,6.997])
    sigma=np.array([0.115,0.252,0.225,4.903])
    a=np.sqrt(w*(w+1.0j*Gamma))
    a=a*np.sign(np.real(a))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Conversion
    aha=1.0j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64))
    epsilon=1-omega_p**2*f0/(w*(w+1.0j*Gamma0))+np.sum(aha)
    return(epsilon)

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

def epsNibb(lam):
    "Permet de caluler la permittivité de l'argent en une longueur d'onde lam donnée"
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.083
    Gamma0=0.022
    omega_p=15.92
    f=np.array([0.357,0.039,0.127,0.654])
    Gamma=np.array([2.820,0.120,1.822,6.637])
    omega=np.array([0.317,1.059,4.583,8.825])
    sigma=np.array([0.606,1.454,0.379,0.510])
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

### RCWA functions
def cascade(T,U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''
    n=int(T.shape[1]/2)
    J=np.linalg.inv(np.eye(n)-np.matmul(U[0:n,0:n],T[n:2*n,n:2*n]))
    K=np.linalg.inv(np.eye(n)-np.matmul(T[n:2*n,n:2*n],U[0:n,0:n]))
    S=np.block([[T[0:n,0:n]+np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,0:n]),T[n:2*n,0:n]),np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,n:2*n])],[np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,0:n]),U[n:2*n,n:2*n]+np.matmul(np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,n:2*n]),U[0:n,n:2*n])]])
    return S

def c_bas(A,V,h):
    ''' Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    '''
    n=int(A.shape[1]/2)
    D=np.diag(np.exp(1j*V*h))
    S=np.block([[A[0:n,0:n],np.matmul(A[0:n,n:2*n],D)],[np.matmul(D,A[n:2*n,0:n]),np.matmul(np.matmul(D,A[n:2*n,n:2*n]),D)]])
    return S

def c_haut(A,valp,h):
    n = int(A[0].size/2)
    D = np.diag(np.exp(1j*valp*h))
    S11 = np.dot(D,np.dot(A[0:n,0:n],D))
    S12 = np.dot(D,A[0:n,n:2*n])
    S21 = np.dot(A[n:2*n,0:n],D)
    S22 = A[n:2*n,n:2*n]
    S1 = np.append(S11,S12,1)
    S2 = np.append(S21,S22,1)
    S = np.append(S1,S2,0)
    return S    

def intermediaire(T,U):
    n = int(T.shape[0] / 2)
    H = np.linalg.inv( np.eye(n) - np.matmul(U[0:n,0:n],T[n:2*n,n:2*n]))
    K = np.linalg.inv( np.eye(n) - np.matmul(T[n:2*n,n:2*n],U[0:n,0:n]))
    a = np.matmul(K, T[n:2*n,0:n])
    b = np.matmul(K, np.matmul(T[n:2*n,n:2*n],U[0:n,n:2*n]))
    c = np.matmul(H, np.matmul(U[0:n,0:n],T[n:2*n,0:n]))
    d = np.matmul(H,U[0:n,n:2*n]) 
    S = np.block([[a,b],[c,d]])
    return S

def couche(valp, h):
    n = len(valp)
    AA = np.diag(np.exp(1j*valp*h))
    C = np.block([[np.zeros((n,n)),AA],[AA,np.zeros((n,n))]])
    return C

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
    n_blocs=blocs.shape[0]
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

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

def HErmes(T,U,V,P,Amp,ny,h,a0):
    n = int(np.shape(T)[0] / 2)
    nmod = int((n-1) / 2)
    nx = n
    X = np.matmul(intermediaire(T,cascade(couche(V,h),U)),Amp.reshape(Amp.size,1))
    D = X[0:n]
    X = np.matmul(intermediaire(cascade(T,couche(V,h)),U),Amp.reshape(Amp.size,1))
    E = X[n:2*n]
    M = np.zeros((ny,nx-1), dtype = complex)
    for k in range(ny):
        y = h / ny * (k+1)
        Fourier = np.matmul(P,np.matmul(np.diag(np.exp(1j*V*y)),D) + np.matmul(np.diag(np.exp(1j*V*(h-y))),E))
        MM = np.fft.ifftshift(Fourier[0:len(Fourier)-1])
        M[k,:] = MM.reshape(len(MM))
    M = np.conj(np.fft.ifft(np.conj(M).T, axis = 0)).T * n
    x, y = np.meshgrid(np.linspace(0,1,nx-1), np.linspace(0,1,ny))
    M = M * np.exp(1j * a0 * x)
    return(M)

### SWAG functions
def reflectance(geometry, wave, materials, n_mod):  
    period = geometry["period"]
    #thick_super = geometry["thick_super"] / period
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period
    thick_gap = geometry["thick_gap"] / period
    thick_func = geometry["thick_func"] / period
    thick_mol = geometry["thick_mol"] / period
    thick_metalliclayer = geometry["thick_metalliclayer"] / period
    thick_sub = geometry["thick_sub"] / period
    thick_accroche = geometry["thick_accroche"] / period 

    wavelength = wave["wavelength"] / period
    angle = wave["angle"] 
    polarization = wave["polarization"]

    perm_env = materials["perm_env"]
    perm_dielec = materials["perm_dielec"]
    perm_sub = materials["perm_sub"]
    perm_reso = materials["perm_reso"]
    perm_metalliclayer =  materials["perm_metalliclayer"]
    perm_accroche = materials["perm_accroche"]

    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])

    n = 2 * n_mod + 1

    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    Pup, Vup = homogene(k0, a0,polarization, perm_env, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])

    if thick_mol < (thick_gap - thick_func):
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso)
    
        P2, V2 = grating(k0, a0, polarization, perm_env, perm_dielec, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_gap - (thick_mol + thick_func))

        P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_mol + thick_func)

    else:
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_reso, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso - (thick_mol - (thick_gap - thick_func)))

        P2, V2 = grating(k0, a0, polarization, perm_dielec, perm_reso, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_mol - (thick_gap - thick_func))

        P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_gap)

    Pmetalliclayer, Vmetalliclayer = homogene(k0, a0, polarization, perm_metalliclayer, n)
    S = cascade(S, interface(P3, Pmetalliclayer))
    S = c_bas(S, Vmetalliclayer, thick_metalliclayer)

    Pacc, Vacc = homogene(k0, a0, polarization, perm_accroche, n)
    S = cascade(S, interface(Pmetalliclayer, Pacc))
    S = c_bas(S, Vacc, thick_accroche)

    Pdown, Vdown = homogene(k0, a0, polarization, perm_sub, n)
    S = cascade(S, interface(Pacc, Pdown))
    S = c_bas(S, Vdown, thick_sub)

    #Rup = abs(S[n_mod, n_mod]) ** 2 
    #Rdown = abs(S[n + n_mod, n + n_mod]) ** 2 

    Tup = abs(S[n + n_mod, n_mod]) ** 2 * np.real(Vdown[n_mod]) / (k0 * np.cos(angle)) / perm_sub * perm_env
    Tdown = abs(S[n_mod, n + n_mod]) ** 2 / np.real(Vdown[n_mod]) * perm_sub * k0 * np.cos(angle) / perm_env

    return Tup, Tdown #Rup, Rdown

### Swag-structure
thick_super = 200
width_reso = 30 # largeur du cube
thick_reso = width_reso # width_reso #hauteur du cube
thick_gap = 3 # hauteur de diéléctrique en dessous du cube
thick_func = 1 # présent partout tout le temps
thick_mol = 0 # si molécules détectées
thick_metal = 10 # hauteur de l'or au dessus du substrat
#thick_accroche = 1 # couche d'accroche 
period = 100.2153
#period = 300.2153
#period = 500.2153
thick_sub = 200

# A modifier selon le point de fonctionnement
#wavelength = 800.021635
angle = 0
polarization = 1

## Paramètres des matériaux
#perm_env = 1.33 ** 2
perm_env = 1.0
perm_dielec = 1.45 ** 2 # spacer
perm_sub = 1.5 ** 2 # substrat Glass
#perm_Ag = epsAgbb(wavelength) # argent
#perm_Au = epsAubb(wavelength) # or
#perm_Cr = epsCrbb(wavelength)

n_mod = 100 
n_mod_total = 2 * n_mod + 1

list_wavelength = np.linspace(450,1000, 200)

# R_cr1 = np.empty(list_wavelength.size)
# R_cr3 = np.empty(list_wavelength.size)
# R_al1 = np.empty(list_wavelength.size)
# R_al2 = np.empty(list_wavelength.size)
# R_al3 = np.empty(list_wavelength.size)
# R_ni1 = np.empty(list_wavelength.size)
# R_ni2 = np.empty(list_wavelength.size)
# R_ti1 = np.empty(list_wavelength.size)
# R_ti2 = np.empty(list_wavelength.size)

# R_cr1d = np.empty(list_wavelength.size)
# R_cr3d = np.empty(list_wavelength.size)
# R_al1d = np.empty(list_wavelength.size)
# R_al2d = np.empty(list_wavelength.size)
# R_al3d = np.empty(list_wavelength.size)
# R_ni1d = np.empty(list_wavelength.size)
# R_ni2d = np.empty(list_wavelength.size)
# R_ti1d = np.empty(list_wavelength.size)
# R_ti2d = np.empty(list_wavelength.size)

T_cr1 = np.empty(list_wavelength.size)
T_cr3 = np.empty(list_wavelength.size)
T_al1 = np.empty(list_wavelength.size)
T_al2 = np.empty(list_wavelength.size)
T_al3 = np.empty(list_wavelength.size)
T_ni1 = np.empty(list_wavelength.size)
T_ni2 = np.empty(list_wavelength.size)
T_ti1 = np.empty(list_wavelength.size)
T_ti2 = np.empty(list_wavelength.size)

T_cr1d = np.empty(list_wavelength.size)
T_cr3d = np.empty(list_wavelength.size)
T_al1d = np.empty(list_wavelength.size)
T_al2d = np.empty(list_wavelength.size)
T_al3d = np.empty(list_wavelength.size)
T_ni1d = np.empty(list_wavelength.size)
T_ni2d = np.empty(list_wavelength.size)
T_ti1d = np.empty(list_wavelength.size)
T_ti2d = np.empty(list_wavelength.size)


thick_acc1 = 1
thick_acc2 = 2
thick_acc3 = 3

geometry_acc1 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_metalliclayer": thick_metal, "thick_sub": thick_sub, "thick_accroche": thick_acc1, "period": period}
geometry_acc2 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_metalliclayer": thick_metal, "thick_sub": thick_sub, "thick_accroche": thick_acc2, "period": period}
geometry_acc3 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_metalliclayer": thick_metal, "thick_sub": thick_sub, "thick_accroche": thick_acc3, "period": period}

for idx_wav, wavelength in enumerate(list_wavelength):
    print(idx_wav)
    
    perm_Ag = epsAgbb(wavelength) 
    perm_Au = epsAubb(wavelength)
    perm_TiN = nk_TiN(wavelength) ** 2
    perm_Ti = nk_Tibb(wavelength) ** 2
    perm_Cr = epsCrbb(wavelength)
    perm_Al = epsAlbb(wavelength)
    perm_Ni = epsNibb(wavelength)

    materials_Cr = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_sub": perm_sub, "perm_reso": perm_Ag, "perm_metalliclayer": perm_Au, "perm_accroche": perm_Cr}
    materials_Al = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_sub": perm_sub, "perm_reso": perm_Ag, "perm_metalliclayer": perm_Au, "perm_accroche": perm_Al}
    materials_Ni = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_sub": perm_sub, "perm_reso": perm_Ag, "perm_metalliclayer": perm_Au, "perm_accroche": perm_Ni}
    materials_Ti = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_sub": perm_sub, "perm_reso": perm_Ag, "perm_metalliclayer": perm_Au, "perm_accroche": perm_Ti}

    wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
    T_cr1[idx_wav], T_cr1d[idx_wav] = reflectance(geometry_acc1, wave, materials_Cr, n_mod)
    T_cr3[idx_wav], T_cr3d[idx_wav] = reflectance(geometry_acc3, wave, materials_Cr, n_mod)
    T_al1[idx_wav], T_al1d[idx_wav] = reflectance(geometry_acc1, wave, materials_Al, n_mod)
    T_al2[idx_wav], T_al2d[idx_wav] = reflectance(geometry_acc2, wave, materials_Al, n_mod)
    T_al3[idx_wav], T_al3d[idx_wav] = reflectance(geometry_acc3, wave, materials_Al, n_mod)
    T_ni1[idx_wav], T_ni1d[idx_wav] = reflectance(geometry_acc1, wave, materials_Ni, n_mod)
    T_ni2[idx_wav], T_ni2d[idx_wav] = reflectance(geometry_acc2, wave, materials_Ni, n_mod)
    T_ti1[idx_wav], T_ti1d[idx_wav] = reflectance(geometry_acc1, wave, materials_Ti, n_mod)
    T_ti2[idx_wav], T_ti2d[idx_wav] = reflectance(geometry_acc2, wave, materials_Ti, n_mod)

# plt.figure(1)
# plt.plot(list_wavelength, R_cr1, label = 'Cr 1')
# plt.plot(list_wavelength, R_cr3, label = 'Cr 3')
# plt.plot(list_wavelength, R_al1, label = 'Al 1')
# plt.plot(list_wavelength, R_al2, label = 'Al 2')
# plt.plot(list_wavelength, R_al3, label = 'Al 3')
# plt.plot(list_wavelength, R_ni1, label = 'Ni 1')
# plt.plot(list_wavelength, R_ni2, label = 'Ni 2')
# plt.plot(list_wavelength, R_ti1, label = 'Ti 1')
# plt.plot(list_wavelength, R_ti2, label = 'Ti 2')
# plt.legend()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Reflectance ")
# plt.title("R up")
# plt.show(block=False)
# plt.savefig("Accroches_all_Rup.jpg")

# plt.figure(2)
# plt.plot(list_wavelength, R_cr1d, label = 'Cr 1')
# plt.plot(list_wavelength, R_cr3d, label = 'Cr 3')
# plt.plot(list_wavelength, R_al1d, label = 'Al 1')
# plt.plot(list_wavelength, R_al2d, label = 'Al 2')
# plt.plot(list_wavelength, R_al3d, label = 'Al 3')
# plt.plot(list_wavelength, R_ni1d, label = 'Ni 1')
# plt.plot(list_wavelength, R_ni2d, label = 'Ni 2')
# plt.plot(list_wavelength, R_ti1d, label = 'Ti 1')
# plt.plot(list_wavelength, R_ti2d, label = 'Ti 2')
# plt.legend()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Reflectance ")
# plt.title("R down")
# plt.show(block=False)
# plt.savefig("Accroches_all_Rdown.jpg")

plt.figure(3)
plt.plot(list_wavelength, T_cr1, label = 'Cr 1')
plt.plot(list_wavelength, T_cr3, label = 'Cr 3')
plt.plot(list_wavelength, T_al1, label = 'Al 1')
plt.plot(list_wavelength, T_al2, label = 'Al 2')
plt.plot(list_wavelength, T_al3, label = 'Al 3')
plt.plot(list_wavelength, T_ni1, label = 'Ni 1')
plt.plot(list_wavelength, T_ni2, label = 'Ni 2')
plt.plot(list_wavelength, T_ti1, label = 'Ti 1')
plt.plot(list_wavelength, T_ti2, label = 'Ti 2')
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.title("T up")
plt.show(block=False)
plt.savefig("Accroches_all_Tup.jpg")

plt.figure(4)
plt.plot(list_wavelength, T_cr1d, label = 'Cr 1')
plt.plot(list_wavelength, T_cr3d, label = 'Cr 3')
plt.plot(list_wavelength, T_al1d, label = 'Al 1')
plt.plot(list_wavelength, T_al2d, label = 'Al 2')
plt.plot(list_wavelength, T_al3d, label = 'Al 3')
plt.plot(list_wavelength, T_ni1d, label = 'Ni 1')
plt.plot(list_wavelength, T_ni2d, label = 'Ni 2')
plt.plot(list_wavelength, T_ti1d, label = 'Ti 1')
plt.plot(list_wavelength, T_ti2d, label = 'Ti 2')
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.title("T down")
plt.show(block=False)
plt.savefig("Accroches_all_Tdown.jpg")

#R = [R_cr1, R_cr3, R_al1, R_al2, R_al3, R_ni1, R_ni2, R_ti1, R_ti2, R_cr1d, R_cr3d, R_al1d, R_al2d, R_al3d, R_ni1d, R_ni2d, R_ti1d, R_ti2d]
T = [T_cr1, T_cr3, T_al1, T_al2, T_al3, T_ni1, T_ni2, T_ti1, T_ti2, T_cr1d, T_cr3d, T_al1d, T_al2d, T_al3d, T_ni1d, T_ni2d, T_ti1d, T_ti2d]

#np.savez("data_accroches_all_Rdown-Rup.npz", list_wavelength = list_wavelength, R = R)
np.savez("data_accroches_all_Tdown-Tup.npz", list_wavelength = list_wavelength, T = T)
