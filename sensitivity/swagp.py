import numpy as np
import matplotlib.pyplot as plt
#import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

### Materials

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
    width_reso = geometry["width_reso"] / period
    thick_reso = geometry["thick_reso"] / period
    thick_gap = geometry["thick_gap"] / period
    thick_func = geometry["thick_func"] / period
    thick_mol = geometry["thick_mol"] / period
    thick_gold = geometry["thick_gold"] / period
    thick_sub = geometry["thick_sub"] / period
    thick_chrome = geometry["thick_chrome"] / period 

    wavelength = wave["wavelength"] / period
    angle = wave["angle"] 
    polarization = wave["polarization"]

    perm_env = materials["perm_env"]
    perm_dielec = materials["perm_dielec"]
    perm_Glass = materials["perm_Glass"]
    perm_Ag = materials["perm_Ag"]
    perm_Au =  materials["perm_Au"]
    perm_Cr = materials["perm_Cr"]
    perm_delta = materials["perm_delta"]
    perm_func = materials["perm_func"]

    pos_reso = np.array([[width_reso, (1 - width_reso) / 2]])

    n = 2 * n_mod + 1

    k0 = 2 * np.pi / wavelength
    a0 = k0 * np.sin(angle * np.pi / 180)

    Pup, Vup = homogene(k0, a0,polarization, perm_env, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])

    if thick_mol < (thick_gap - thick_func):
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_Ag, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso)
    
        P2, V2 = grating(k0, a0, polarization, perm_env, perm_dielec, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_gap - (thick_mol + thick_func))

        P3, V3 = homogene(k0, a0, polarization, perm_delta, n) # grating delta / dielec
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_mol)

    else:
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_Ag, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso - (thick_mol - (thick_gap - thick_func)))

        P2, V2 = grating(k0, a0, polarization, perm_delta, perm_Ag, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_mol - (thick_gap - thick_func))

        P3, V3 = homogene(k0, a0, polarization, perm_delta, n) # grating delta / dielec
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_gap-thick_func)

        # P2, V2 = grating(k0, a0, polarization, perm_dielec, perm_Ag, n, pos_reso)
        # S = cascade(S, interface(P1, P2))
        # S = c_bas(S, V2, thick_mol - (thick_gap - thick_func))

        # P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        # S = cascade(S, interface(P2, P3))
        # S = c_bas(S, V3, thick_gap)
    
    Pfunc,Vfunc = homogene(k0, a0, polarization, perm_func, n)
    S = cascade(S, interface(P3, Pfunc))
    S = c_bas(S, Vfunc, thick_func)

    Pgold, Vgold = homogene(k0, a0, polarization, perm_Au, n)
    S = cascade(S, interface(Pfunc, Pgold))
    S = c_bas(S, Vgold, thick_gold)

    Pcr, Vcr = homogene(k0, a0, polarization, perm_Cr, n)
    S = cascade(S, interface(Pgold, Pcr))
    S = c_bas(S, Vcr, thick_chrome)

    Pdown, Vdown = homogene(k0, a0, polarization, perm_Glass, n)
    S = cascade(S, interface(Pcr, Pdown))
    S = c_bas(S, Vdown, thick_sub)

    # reflexion quand on eclaire par le dessus
    Rup = abs(S[n_mod, n_mod]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    Tup = abs(S[n + n_mod, n_mod]) ** 2 * np.real(Vdown[n_mod]) / (k0 * np.cos(angle)) / perm_Glass * perm_env
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    Rdown = abs(S[n + n_mod, n + n_mod]) ** 2 
    # transmission quand on éclaire par le dessous
    Tdown = abs(S[n_mod, n + n_mod]) ** 2 / np.real(Vdown[n_mod]) * perm_Glass * k0 * np.cos(angle) / perm_env

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[n_mod, n_mod])
    #phase_R_down = np.angle(S[n + n_mod, n + n_mod])
    #phase_T_up = np.angle(S[n + n_mod, n_mod])
    #phase_T_down = np.angle(S[n_mod, n + n_mod])
    return Rdown, Rup, Tdown, Tup#, Rup #phase_R_down#, Rup, phase_R_up#, Tdown, phase_T_down, Tup, phase_T_up

### Swag-structure
thick_super = 200
width_reso = 70 # largeur du cube
thick_reso = width_reso # width_reso #hauteur du cube
thick_gap = 5 # hauteur de diéléctrique en dessous du cube
thick_func = 1 # présent partout tout le temps
thick_mol = 3 # si molécules détectées
#thick_gold = 20 # hauteur de l'or au dessus du substrat
thick_cr = 0 # couche d'accroche 
period = 300.2153 # periode
thick_sub = 200

# A modifier selon le point de fonctionnement
#wavelength = 950 #800.021635
angle = 0
polarization = 1

## Paramètres des matériaux
perm_env = 1.33 ** 2
#perm_env=1
perm_dielec = 1.45 ** 2 # spacer (polymer)
perm_Glass = 1.5 ** 2 # substrat
# perm_Ag = epsAgbb(wavelength) # argent
# perm_Au = epsAubb(wavelength) # or
# perm_Cr = epsCrbb(wavelength)
RI_delta = 0.01 # to add in the 'clean' interface
perm_delta = (1.33 + RI_delta)**2 # molecules
print("perm delta = ", perm_delta)
perm_func = perm_dielec # couche de fonctionalisation des molécules

n_mod = 80 
n_mod_total = 2 * n_mod + 1

# ##### Sensitivity - test 1

# geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# # TODO : changer l'indice de la couche molécule à Delta_n au lieu de n_molecules 
# geometry_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr, "perm_delta": perm_delta, "perm_func": perm_func}
# wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
# Rd, Ru, Td, Tu = reflectance(geometry, wave, materials, n_mod)
# Rd_mol, Ru_mol, Td_mol, Tu_mol = reflectance(geometry_mol, wave, materials, n_mod)

# Sensibility_Rd = (abs(Rd-Rd_mol)) / (RI_delta) # to be continued
# Sensibility_Ru = (abs(Ru-Ru_mol)) / (RI_delta) # to be continued
# Sensibility_Td = (abs(Td-Td_mol)) / (RI_delta) # to be continued
# Sensibility_Tu = (abs(Tu-Tu_mol)) / (RI_delta) # to be continued

# print(Sensibility_Rd)
# print(Sensibility_Ru)
# print(Sensibility_Td)
# print(Sensibility_Tu)

# Sensitivity - depending on wavelength
list_wavelength = np.linspace(850, 1500, 100)
# for a specific config, I want its sensitivity (Rup)

list_gold = np.linspace(5, 100, 20)
idx_gold = 0

lam_s_ru = np.empty(list_gold.size)
S_max_ru = np.empty(list_gold.size)

lam_s_rd = np.empty(list_gold.size)
S_max_rd = np.empty(list_gold.size)

lam_s_tu = np.empty(list_gold.size)
S_max_tu = np.empty(list_gold.size)

lam_s_td = np.empty(list_gold.size)
S_max_td = np.empty(list_gold.size)

for thick_gold in list_gold:
    plt.close()
    print("gold = ",idx_gold, "/", list_gold.size)
# the geometry does not depend on wavelength but on the gold thickness
    geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0, "thick_gold": list_gold[idx_gold], "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
    geometry_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": list_gold[idx_gold], "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# the materials, wave, reflectance and sensitivity depend on the wavelength
    Ru = np.empty(list_wavelength.size)
    Ru_mol = np.empty(list_wavelength.size)
    Sensitivity_Ru = np.empty(list_wavelength.size)

    Rd = np.empty(list_wavelength.size)
    Rd_mol = np.empty(list_wavelength.size)
    Sensitivity_Rd = np.empty(list_wavelength.size)

    Tu = np.empty(list_wavelength.size)
    Tu_mol = np.empty(list_wavelength.size)
    Sensitivity_Tu = np.empty(list_wavelength.size)

    Td = np.empty(list_wavelength.size)
    Td_mol = np.empty(list_wavelength.size)
    Sensitivity_Td = np.empty(list_wavelength.size)
    idx_lam = 0

    for wavelength in list_wavelength:
        print("lam = ", idx_lam, "/", list_wavelength.size)
        perm_Ag = epsAgbb(wavelength)
        perm_Au = epsAubb(wavelength)
        perm_Cr = epsCrbb(wavelength)
        materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr, "perm_delta": perm_delta, "perm_func": perm_func}
        wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
    # paramaters for reference config
        [Rd[idx_lam], Ru[idx_lam], Td[idx_lam], Tu[idx_lam]] = reflectance(geometry, wave, materials, n_mod)
    # parameters for sensitive config
        [Rd_mol[idx_lam], Ru_mol[idx_lam], Td_mol[idx_lam], Tu_mol[idx_lam]] = reflectance(geometry_mol, wave, materials, n_mod)
    # only on Rup for the first step
        Sensitivity_Ru[idx_lam] = (abs(Ru[idx_lam]-Ru_mol[idx_lam])) / (RI_delta)
        Sensitivity_Tu[idx_lam] = (abs(Tu[idx_lam]-Tu_mol[idx_lam])) / (RI_delta)
        Sensitivity_Rd[idx_lam] = (abs(Rd[idx_lam]-Rd_mol[idx_lam])) / (RI_delta)
        Sensitivity_Td[idx_lam] = (abs(Td[idx_lam]-Td_mol[idx_lam])) / (RI_delta)
        idx_lam+=1

    idx_max_ru = np.argmax(Sensitivity_Ru)
    S_max_ru[idx_gold] = max(Sensitivity_Ru)
    lam_s_ru[idx_gold] = list_wavelength[idx_max_ru]

    idx_max_tu = np.argmax(Sensitivity_Tu)
    S_max_tu[idx_gold] = max(Sensitivity_Tu)
    lam_s_tu[idx_gold] = list_wavelength[idx_max_tu]

    idx_max_rd = np.argmax(Sensitivity_Rd)
    S_max_rd[idx_gold] = max(Sensitivity_Rd)
    lam_s_rd[idx_gold] = list_wavelength[idx_max_rd]

    idx_max_td = np.argmax(Sensitivity_Td)
    S_max_td[idx_gold] = max(Sensitivity_Td)
    lam_s_td[idx_gold] = list_wavelength[idx_max_td]

# print("idx_max = ", idx_max)
# print("S max = ", S_max)
# print("Lam_r = ", lam_r)

    plt.figure(idx_gold)
    plt.subplot(3,1,1)
    plt.plot(list_wavelength, Sensitivity_Ru, label = "Ru")
    plt.plot(list_wavelength, Sensitivity_Rd, label = "Rd")
    plt.plot(list_wavelength, Sensitivity_Tu, label = "Tu")
    plt.plot(list_wavelength, Sensitivity_Td, label = "Td")
    plt.legend()
    plt.ylabel("$\Delta R$ (or T) / $\Delta n$")
    plt.xlabel("Wavelength (nm)")
    plt.subplot(3,1,2)
    plt.plot(list_wavelength, Ru,"r", label = "Ru")
    plt.plot(list_wavelength, Ru_mol,"--r" )#,label = "Ru with molecule")
    plt.plot(list_wavelength, Rd,"k", label = "Rd")
    plt.plot(list_wavelength, Rd_mol,"--k")# ,label = "Rd with molecule")
    plt.ylabel("R")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(list_wavelength, Tu,"g", label = "Tu")
    plt.plot(list_wavelength, Tu_mol, "--g")#,label = "Tu")
    plt.plot(list_wavelength, Td,"b", label = "Td")
    plt.plot(list_wavelength, Td_mol,"--b")# ,label = "Td with molecule")
    plt.legend()
    plt.ylabel("T")
    plt.xlabel("Wavelength (nm)")
    plt.show(block=False)
    plt.savefig(f"sensitivity_RTall_gold={thick_gold}.pdf")
    plt.savefig(f"sensitivity_RTall_gold={thick_gold}.jpg")


    idx_gold += 1

plt.figure(89)
plt.subplot(2,1,1)
plt.plot(list_gold,S_max_ru,label="Ru")
plt.plot(list_gold,S_max_rd,label="Rd")
plt.plot(list_gold,S_max_tu,label="Tu")
plt.plot(list_gold,S_max_td,label="Td")
plt.legend()

#plt.xlabel("Gold thickness (nm)")
plt.ylabel("Sensitivity")
plt.subplot(2,1,2)
plt.plot(list_gold, lam_s_ru, label="Ru")
plt.plot(list_gold, lam_s_rd, label="Rd")
plt.plot(list_gold, lam_s_tu, label="Tu")
plt.plot(list_gold, lam_s_td, label="Td")
plt.xlabel("Gold thickness (nm)")
plt.ylabel("Lambda resonance")
plt.legend()
plt.show(block=False)
plt.savefig("sensitivity_RTall_allgold.pdf")
plt.savefig("sensitivity_RTall_allgold.jpg")



### To have an idea of a wavelength of interest. Conclusion : lam = 950 nm

# list_wavelength = np.linspace(450, 1500, 80)
# Rd = np.empty(list_wavelength.size)
# Ru = np.empty(list_wavelength.size)
# Td = np.empty(list_wavelength.size)
# Tu = np.empty(list_wavelength.size)
# Rd_mol = np.empty(list_wavelength.size)
# Ru_mol = np.empty(list_wavelength.size)
# Td_mol = np.empty(list_wavelength.size)
# Tu_mol = np.empty(list_wavelength.size)
# idx = 0

# geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     print(idx)
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     Rd[idx], Ru[idx], Td[idx], Tu[idx] = reflectance(geometry, wave, materials, n_mod)
#     Rd_mol[idx], Ru_mol[idx], Td_mol[idx], Tu_mol[idx] = reflectance(geometry_mol, wave, materials, n_mod)
#     idx += 1
    

# plt.figure(1)
# plt.plot(list_wavelength, Rd, "r", label = "R down")
# plt.plot(list_wavelength, Rd_mol, "--") 
# plt.plot(list_wavelength, Ru, "b", label = "R up")
# plt.plot(list_wavelength, Ru_mol, "--") 
# plt.plot(list_wavelength, Td, "g", label = "T down")
# plt.plot(list_wavelength, Td_mol, "--") 
# plt.plot(list_wavelength, Tu, "o", label = "T up")
# plt.plot(list_wavelength, Tu_mol, "--") 
# plt.legend()
# plt.ylabel("Observable")
# plt.title("Sensibility")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("sensitivity_1.pdf")
# plt.savefig("sensitivity_1.jpg")



# file = open(f"Data_sensitivity_1.txt", 'w')
# file.write("Environnement : Air / n = 1 \n")
# file.write("Cube : Argent / n(lambda) / 70 nm\n")
# file.write("Gap diélectrique / n = 1.45 / 5 nm\n")
# file.write("Fonctionnalisation diélectrique / n = 1.45 / 1 nm\n")
# file.write("Couche métallique : Or / n(lambda) / 20 nm\n")
# file.write("Substrat : SiO2 / n = 1.5 / 200 nm\n")
# file.write("\n")
# file.write("Wavelengths (nm) \t Rd \t Rd_mol\t Ru \t Ru_mol \t Td \t Td_mol\t Tu \t Tu_mol \n")

# for i in range(len(list_wavelength)):
#     file.write(f"{list_wavelength[i]}, {Rd[i]}\t, {Rd_mol[i]}\t, {Ru[i]}\t,{Ru_mol[i]}\t,{Td[i]}\t,{Td_mol[i]}\t,{Tu[i]}\t,{Tu_mol[i]}\n")
# file.close()

