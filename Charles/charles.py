### Demande 15/07/2025 ####
# Nous avons récemment adapté notre settup pour pouvoir mesurer la réflexion et la transmission en même temps. Comme je n'ai pas encore beaucoup de recul sur les mesures en transmission, cela m'aiderait d'avoir des simulations de la "enhanced transmission" induite par la présence d'un AgNC. Pourriez vous s'il vous plaît effectuer les simulations suivantes : 
# • Au=[2 5 7.1 10 20 50] nm
# • PVP = [0 1 2 3 4 5 6 7] nm
# • Diamètre AgNCs = [55 85] nm
# • Couche d'Al2O3 (entre Au et PVP) : [0 1 1.5] nm
# PS : si vous pouviez faire ces calculs pour la réflexion aussi ce serait top. Je peux aussi vous envoyer des données brutes de gap plasmons en reflexion avec corrélation SEM si vous voulez les montrer sur un rapport de stage, comité de thèse ou autre.

import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

### Materials

def ri_Al2O3(lam):
    tableau3D = []

    file = open('Al2O3_Zhukovsky.txt', "r")
    lines = file.readlines()
    file.close()

    nb_lines = len(lines)
    for i in range (nb_lines):
        values = lines[i].split("\t")
        values[2] = values[2].rstrip("\n")
        values = [float(val) for val in values]
        tableau3D.append(values)
    
    tableau3D = np.array(tableau3D)
    wl = []
    wl = tableau3D[:,0]
    ri_exp = []
    ri_exp = tableau3D[:,1]
    ri = np.interp(lam * 10**(-3), wl, ri_exp)
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

    a = np.sqrt(w * (w + i * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    # Conversion

    epsilon = 1-omega_p**2*f0/(w*(w+i*Gamma0))+np.sum(i*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64)))
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
    a=np.sqrt(w*(w+i*Gamma))
    a=a*np.sign(np.real(a))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Conversion
    aha=i*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64))
    epsilon=1-omega_p**2*f0/(w*(w+i*Gamma0))+np.sum(aha)
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

    from scipy.linalg import toeplitz
    l=np.zeros(n,dtype=np.complex128)
    m=np.zeros(n,dtype=np.complex128)
    tmp=1/(2*np.pi*np.arange(1,n))*(np.exp(-2*1j*np.pi*p*np.arange(1,n))-1)*np.exp(-2*1j*np.pi*np.arange(1,n)*x)
    l[1:n]=1j*(a-b)*tmp
    l[0]=p*a+(1-p)*b
    m[0]=l[0]
    m[1:n]=1j*(b-a)*np.conj(tmp)
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
    thick_pvp = geometry["thick_pvp"] / period
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period
    thick_gap = geometry["thick_gap"] / period
    #thick_func = geometry["thick_func"] / period
    #thick_mol = geometry["thick_mol"] / period
    thick_gold = geometry["thick_gold"] / period
    thick_sub = geometry["thick_sub"] / period
    #thick_chrome = geometry["thick_chrome"] / period 

    wavelength = wave["wavelength"] / period
    angle = wave["angle"] 
    polarization = wave["polarization"]

    perm_env = materials["perm_env"]
    perm_dielec = materials["perm_dielec"]
    perm_Glass = materials["perm_Glass"]
    perm_Ag = materials["perm_Ag"]
    perm_Au =  materials["perm_Au"]
    #perm_Cr = materials["perm_Cr"]
    perm_pvp = materials["perm_pvp"]

    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])

    n = 2 * n_mod + 1

    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    Pup, Vup = homogene(k0, a0,polarization, perm_env, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])

    #if thick_mol < (thick_gap - thick_func):
    P1, V1 = grating(k0, a0, polarization, perm_env, perm_Ag, n, pos_reso)
    S = cascade(S, interface(Pup, P1))
    S = c_bas(S, V1, thick_reso)

    P2, V2 = grating(k0, a0, polarization, perm_env, perm_pvp, n, pos_reso)
    S = cascade(S, interface(P1, P2))
    S = c_bas(S, V1, thick_pvp)

    P3, V3 = homogene(k0, a0, polarization, perm_dielec, n) #Al2O3
    S = cascade(S, interface(P2, P3))
    S = c_bas(S, V3, thick_gap)

    Pgold, Vgold = homogene(k0, a0, polarization, perm_Au, n)
    S = cascade(S, interface(P3, Pgold))
    S = c_bas(S, Vgold, thick_gold)

    # Pcr, Vcr = homogene(k0, a0, polarization, perm_Cr, n)
    # S = cascade(S, interface(Pgold, Pcr))
    # S = c_bas(S, Vcr, thick_chrome)

    Pdown, Vdown = homogene(k0, a0, polarization, perm_Glass, n)
    S = cascade(S, interface(Pgold, Pdown))
    S = c_bas(S, Vdown, thick_sub)

    # reflexion quand on eclaire par le dessus
    #Rup = abs(S[n_mod, n_mod]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    #Tup = abs(S[n + n_mod, n_mod]) ** 2 * np.real(Vdown[n_mod]) / (k0 * np.cos(angle)) / perm_Glass * perm_env
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    Rdown = abs(S[n + n_mod, n + n_mod]) ** 2 
    # transmission quand on éclaire par le dessous
    #Tdown = abs(S[n_mod, n + n_mod]) ** 2 / np.real(Vdown[n_mod]) * perm_Glass * k0 * np.cos(angle) / perm_env

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[n_mod, n_mod])
    #phase_R_down = np.angle(S[n + n_mod, n + n_mod])
    #phase_T_up = np.angle(S[n + n_mod, n_mod])
    #phase_T_down = np.angle(S[n_mod, n + n_mod])
    return Rdown#, Rup, Tdown, Tup#, Rup #phase_R_down#, Rup, phase_R_up#, Tdown, phase_T_down, Tup, phase_T_up

### Swag-structure
thick_super = 200
width_reso = 55 # largeur du cube (nm)
thick_reso = width_reso # width_reso #hauteur du cube
thick_gap = 3 # hauteur de diéléctrique en dessous du cube (Al2O3)
#thick_func = 1 # présent partout tout le temps
#thick_mol = 3 # si molécules détectées
thick_gold = 10 # hauteur de l'or au dessus du substrat
#thick_cr = 1 # couche d'accroche 
period = 300.2153 # periode
thick_sub = 200 #verre
list_thick_pvp = np.arange(0,8,1)
#thick_pvp=0

# A modifier selon le point de fonctionnement
#wavelength = 800.021635
angle = 0 # incidence normale
polarization = 1 # polarisation TM (p-polarization)

#list_wavelength = np.linspace(1000, 2600, 100)

## Paramètres des matériaux
perm_env = 1 ** 2 # air 
perm_Glass = 1.5 ** 2 # substrat
perm_pvp = 1.45 ** 2 
#perm_Ag = epsAgbb(wavelength) # argent
#perm_Au = epsAubb(wavelength) # or
#perm_Cr = epsCrbb(wavelength)

n_mod = 100 
n_mod_total = 2 * n_mod + 1

## Etude de la dépendance de la réflexion à la longueur d'onde, influence de l'épaisseur de PVP

list_wavelength = np.linspace(1000, 2600, 100)
Rdown_pvp0_nomol = np.empty(list_wavelength.size)
Rdown_pvp1_nomol = np.empty(list_wavelength.size)
Rdown_pvp2_nomol = np.empty(list_wavelength.size)
Rdown_pvp3_nomol = np.empty(list_wavelength.size)
Rdown_pvp4_nomol = np.empty(list_wavelength.size)
Rdown_pvp5_nomol = np.empty(list_wavelength.size)
Rdown_pvp6_nomol = np.empty(list_wavelength.size)
Rdown_pvp7_nomol = np.empty(list_wavelength.size)

# Rup_pvp0_nomol = np.empty(list_wavelength.size)
# Rup_pvp1_nomol = np.empty(list_wavelength.size)
# Rup_pvp2_nomol = np.empty(list_wavelength.size)
# Rup_pvp3_nomol = np.empty(list_wavelength.size)
# Rup_pvp4_nomol = np.empty(list_wavelength.size)
# Rup_pvp5_nomol = np.empty(list_wavelength.size)
# Rup_pvp6_nomol = np.empty(list_wavelength.size)
# Rup_pvp7_nomol = np.empty(list_wavelength.size)

# Tdown_pvp0_nomol = np.empty(list_wavelength.size)
# Tdown_pvp1_nomol = np.empty(list_wavelength.size)
# Tdown_pvp2_nomol = np.empty(list_wavelength.size)
# Tdown_pvp3_nomol = np.empty(list_wavelength.size)
# Tdown_pvp4_nomol = np.empty(list_wavelength.size)
# Tdown_pvp5_nomol = np.empty(list_wavelength.size)
# Tdown_pvp6_nomol = np.empty(list_wavelength.size)
# Tdown_pvp7_nomol = np.empty(list_wavelength.size)

# Tup_pvp0_nomol = np.empty(list_wavelength.size)
# Tup_pvp1_nomol = np.empty(list_wavelength.size)
# Tup_pvp2_nomol = np.empty(list_wavelength.size)
# Tup_pvp3_nomol = np.empty(list_wavelength.size)
# Tup_pvp4_nomol = np.empty(list_wavelength.size)
# Tup_pvp5_nomol = np.empty(list_wavelength.size)
# Tup_pvp6_nomol = np.empty(list_wavelength.size)
# Tup_pvp7_nomol = np.empty(list_wavelength.size)

idx = 0

geometry_pvp0_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 0, "period": period}
geometry_pvp1_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 1, "period": period}
geometry_pvp2_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 2, "period": period}
geometry_pvp3_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 3, "period": period}
geometry_pvp4_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 4, "period": period}
geometry_pvp5_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 5, "period": period}
geometry_pvp6_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 6, "period": period}
geometry_pvp7_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_pvp": 7, "period": period}

print(idx)
for wavelength in list_wavelength:
    perm_Ag = epsAgbb(wavelength) # argent
    perm_Au = epsAubb(wavelength)
    #perm_Cr = epsCrbb(wavelength)
    perm_dielec = ri_Al2O3(wavelength) ** 2 # spacer en Al2O3

    materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_pvp": perm_pvp}
    wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
    # Rdown_pvp0_nomol[idx], Rup_pvp0_nomol[idx], Tdown_pvp0_nomol[idx], Tup_pvp0_nomol[idx] = reflectance(geometry_pvp0_nomol, wave, materials, n_mod)
    # Rdown_pvp1_nomol[idx], Rup_pvp1_nomol[idx], Tdown_pvp1_nomol[idx], Tup_pvp1_nomol[idx] = reflectance(geometry_pvp1_nomol, wave, materials, n_mod)
    # Rdown_pvp2_nomol[idx], Rup_pvp2_nomol[idx], Tdown_pvp2_nomol[idx], Tup_pvp2_nomol[idx] = reflectance(geometry_pvp2_nomol, wave, materials, n_mod)
    # Rdown_pvp3_nomol[idx], Rup_pvp3_nomol[idx], Tdown_pvp3_nomol[idx], Tup_pvp3_nomol[idx] = reflectance(geometry_pvp3_nomol, wave, materials, n_mod)
    # Rdown_pvp4_nomol[idx], Rup_pvp4_nomol[idx], Tdown_pvp4_nomol[idx], Tup_pvp4_nomol[idx] = reflectance(geometry_pvp4_nomol, wave, materials, n_mod)
    # Rdown_pvp5_nomol[idx], Rup_pvp5_nomol[idx], Tdown_pvp5_nomol[idx], Tup_pvp5_nomol[idx] = reflectance(geometry_pvp5_nomol, wave, materials, n_mod)
    # Rdown_pvp6_nomol[idx], Rup_pvp6_nomol[idx], Tdown_pvp6_nomol[idx], Tup_pvp6_nomol[idx] = reflectance(geometry_pvp6_nomol, wave, materials, n_mod)
    # Rdown_pvp7_nomol[idx], Rup_pvp7_nomol[idx], Tdown_pvp7_nomol[idx], Tup_pvp7_nomol[idx] = reflectance(geometry_pvp7_nomol, wave, materials, n_mod)
   
    Rdown_pvp0_nomol[idx] = reflectance(geometry_pvp0_nomol, wave, materials, n_mod)
    Rdown_pvp1_nomol[idx] = reflectance(geometry_pvp1_nomol, wave, materials, n_mod)
    Rdown_pvp2_nomol[idx] = reflectance(geometry_pvp2_nomol, wave, materials, n_mod)
    Rdown_pvp3_nomol[idx] = reflectance(geometry_pvp3_nomol, wave, materials, n_mod)
    Rdown_pvp4_nomol[idx] = reflectance(geometry_pvp4_nomol, wave, materials, n_mod)
    Rdown_pvp5_nomol[idx] = reflectance(geometry_pvp5_nomol, wave, materials, n_mod)
    Rdown_pvp6_nomol[idx] = reflectance(geometry_pvp6_nomol, wave, materials, n_mod)
    Rdown_pvp7_nomol[idx] = reflectance(geometry_pvp7_nomol, wave, materials, n_mod)
   
    idx += 1
    print(idx)


plt.figure(0)
plt.plot(list_wavelength, Rdown_pvp0_nomol,"r",  label = "0 nm")
plt.plot(list_wavelength, Rdown_pvp1_nomol,"b", label = "1 nm")
plt.plot(list_wavelength, Rdown_pvp2_nomol,"y", label = "2 nm")
plt.plot(list_wavelength, Rdown_pvp3_nomol,"c", label = "3 nm")
plt.plot(list_wavelength, Rdown_pvp4_nomol,"k", label = "4 nm")
plt.plot(list_wavelength, Rdown_pvp5_nomol,"g", label = "5 nm")
plt.plot(list_wavelength, Rdown_pvp6_nomol,"m", label = "6 nm")
plt.plot(list_wavelength, Rdown_pvp7_nomol,"--r", label = "7 nm")

plt.legend()
plt.ylabel("Reflectance down")
plt.title("PVP thickness")
plt.xlabel("Wavelength (nm)")

plt.show(block=False)
plt.savefig("pvp_Rd_test2.pdf")
plt.savefig("pvp_Rd_test2.jpg")

file = open(f"test2.txt", 'w')
file.write("Environnement : Air / n = 1 \n")
file.write("Cube : Argent / n(lambda) / 70 nm\n")
file.write("Gap diélectrique / n = 1.45 / 1-10 nm\n")
file.write("Fonctionnalisation diélectrique / n = 1.45 / 1 nm\n")
file.write("Couche métallique : Or / n(lambda) / 20 nm\n")
file.write("Substrat : SiO2 / n = 1.5 / 200 nm\n")
file.write("\n")
file.write("Wavelengths (nm) \t R_gap1_mol\t R_gap1_nomol\t R_gap2_mol\t R_gap2_nomol\t R_gap3_mol\t R_gap3_nomol\t R_gap4_mol\t R_gap4_nomol\t R_gap5_mol\t R_gap5_nomol\t R_gap10_mol\t R_gap10_nomol\n")

for i in range(len(list_wavelength)):
    file.write(f"{list_wavelength[i]}, {Rdown_pvp0_nomol[i]}\t,{Rdown_pvp1_nomol[i]}\t,{Rdown_pvp2_nomol[i]}\t,{Rdown_pvp3_nomol[i]}\t, {Rdown_pvp4_nomol[i]}\t {Rdown_pvp5_nomol[i]}\t {Rdown_pvp6_nomol[i]}\t {Rdown_pvp7_nomol[i]} \n")
file.close()

