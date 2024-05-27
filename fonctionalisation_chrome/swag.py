import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
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

        P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_mol + thick_func)

    else:
        P1, V1 = grating(k0, a0, polarization, perm_env, perm_Ag, n, pos_reso)
        S = cascade(S, interface(Pup, P1))
        S = c_bas(S, V1, thick_reso - (thick_mol - (thick_gap - thick_func)))

        P2, V2 = grating(k0, a0, polarization, perm_dielec, perm_Ag, n, pos_reso)
        S = cascade(S, interface(P1, P2))
        S = c_bas(S, V2, thick_mol - (thick_gap - thick_func))

        P3, V3 = homogene(k0, a0, polarization, perm_dielec, n)
        S = cascade(S, interface(P2, P3))
        S = c_bas(S, V3, thick_gap)

    Pgold, Vgold = homogene(k0, a0, polarization, perm_Au, n)
    S = cascade(S, interface(P3, Pgold))
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
    return Rdown#, Rup, Tdown, Tup#, Rup #phase_R_down#, Rup, phase_R_up#, Tdown, phase_T_down, Tup, phase_T_up

### Swag-structure
thick_super = 200
width_reso = 70 # largeur du cube
thick_reso = width_reso # width_reso #hauteur du cube
thick_gap = 3 # hauteur de diéléctrique en dessous du cube
thick_func = 1 # présent partout tout le temps
thick_mol = 3 # si molécules détectées
#thick_gold = 20 # hauteur de l'or au dessus du substrat
thick_cr = 1 # couche d'accroche 
period = 300.2153 # periode
thick_sub = 200

# A modifier selon le point de fonctionnement
#wavelength = 800.021635
angle = 0
polarization = 1

## Paramètres des matériaux
perm_env = 1.33 ** 2
perm_dielec = 1.45 ** 2 # spacer
perm_Glass = 1.5 ** 2 # substrat
#perm_Ag = epsAgbb(wavelength) # argent
#perm_Au = epsAubb(wavelength) # or
#perm_Cr = epsCrbb(wavelength)

n_mod = 100 
n_mod_total = 2 * n_mod + 1

## Etude de la dépendance de la réflexion à la longueur d'onde, influence de l'épaisseur d'or
thick_gold1 = 1
thick_gold2 = 2
thick_gold3 = 3
thick_gold4 = 4
thick_gold5 = 5
thick_gold6 = 6
thick_gold7 = 7
#thick_gold8 = 8
# thick_gold20 = 20
# thick_gold30 = 30
# thick_gold40 = 40
# thick_gold50 = 50
# thick_gold100 = 100
# thick_gold200 = 200

list_wavelength = np.linspace(1000, 2600, 200)
R_gold1_mol = np.empty(list_wavelength.size)
R_gold1_nomol = np.empty(list_wavelength.size)
R_gold2_mol = np.empty(list_wavelength.size)
R_gold2_nomol = np.empty(list_wavelength.size)
R_gold3_mol = np.empty(list_wavelength.size)
R_gold3_nomol = np.empty(list_wavelength.size)
R_gold4_mol = np.empty(list_wavelength.size)
R_gold4_nomol = np.empty(list_wavelength.size)
R_gold5_mol = np.empty(list_wavelength.size)
R_gold5_nomol = np.empty(list_wavelength.size)
R_gold6_mol = np.empty(list_wavelength.size)
R_gold6_nomol = np.empty(list_wavelength.size)
R_gold7_mol = np.empty(list_wavelength.size)
R_gold7_nomol = np.empty(list_wavelength.size)
#R_gold8_mol = np.empty(list_wavelength.size)
#R_gold8_nomol = np.empty(list_wavelength.size)
idx = 0

geometry_gold1_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold1, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold1_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold1, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold2_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold2, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold2_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold2, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold3_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold3, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold3_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold3, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold4_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold4, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold4_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold4, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold5_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold5, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold5_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold5, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold6_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold6, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold6_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold6, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold7_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold7, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gold7_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold7, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
#geometry_gold8_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold8, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
#geometry_gold8_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold8, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

for wavelength in list_wavelength:
    perm_Ag = epsAgbb(wavelength) # argent
    perm_Au = epsAubb(wavelength)
    perm_Cr = epsCrbb(wavelength)
    materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
    wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
    R_gold1_mol[idx] = reflectance(geometry_gold1_mol, wave, materials, n_mod)
    R_gold1_nomol[idx] = reflectance(geometry_gold1_nomol, wave, materials, n_mod)
    R_gold2_mol[idx] = reflectance(geometry_gold2_mol, wave, materials, n_mod)
    R_gold2_nomol[idx] = reflectance(geometry_gold2_nomol, wave, materials, n_mod)
    R_gold3_mol[idx] = reflectance(geometry_gold3_mol, wave, materials, n_mod)
    R_gold3_nomol[idx] = reflectance(geometry_gold3_nomol, wave, materials, n_mod)
    R_gold4_mol[idx] = reflectance(geometry_gold4_mol, wave, materials, n_mod)
    R_gold4_nomol[idx] = reflectance(geometry_gold4_nomol, wave, materials, n_mod)
    R_gold5_mol[idx] = reflectance(geometry_gold5_mol, wave, materials, n_mod)
    R_gold5_nomol[idx] = reflectance(geometry_gold5_nomol, wave, materials, n_mod)
    R_gold6_mol[idx] = reflectance(geometry_gold6_mol, wave, materials, n_mod)
    R_gold6_nomol[idx] = reflectance(geometry_gold6_nomol, wave, materials, n_mod)  
    R_gold7_mol[idx] = reflectance(geometry_gold7_mol, wave, materials, n_mod)
    R_gold7_nomol[idx] = reflectance(geometry_gold7_nomol, wave, materials, n_mod)
    #R_gold8_mol[idx] = reflectance(geometry_gold8_mol, wave, materials, n_mod)
    #R_gold8_nomol[idx] = reflectance(geometry_gold8_nomol, wave, materials, n_mod)          
    idx += 1

plt.figure(0)
plt.plot(list_wavelength, R_gold1_mol, "r", label = "1 nm")
plt.plot(list_wavelength, R_gold1_nomol, "--r") #,  label = "1 nm")
plt.plot(list_wavelength, R_gold2_mol, "b", label = "2 nm")
plt.plot(list_wavelength, R_gold2_nomol,"--b") #,  label = "2 nm")
plt.plot(list_wavelength, R_gold3_mol,"y", label = "3 nm")
plt.plot(list_wavelength, R_gold3_nomol,"--y") #, label = "3 nm")
plt.plot(list_wavelength, R_gold4_mol,"c", label = "4 nm")
plt.plot(list_wavelength, R_gold4_nomol,"--c") #, label = "4 nm")
plt.plot(list_wavelength, R_gold5_mol,"k", label = "5 nm")
plt.plot(list_wavelength, R_gold5_nomol,"--k") #, label = "3 nm")
plt.plot(list_wavelength, R_gold6_mol,"g", label = "6 nm")
plt.plot(list_wavelength, R_gold6_nomol,"--g") #, label = "4 nm")
plt.plot(list_wavelength, R_gold7_mol,"m", label = "7 nm")
plt.plot(list_wavelength, R_gold7_nomol,"--m") #, label = "3 nm")
#plt.plot(list_wavelength, R_gold8_mol,"pink", label = "8 nm")
#plt.plot(list_wavelength, R_gold8_nomol,"--pink") #, label = "4 nm") # --pink does not work
plt.legend()
plt.ylabel("Reflectance")
plt.title("Gold thickness / with (lines) or without (dotted) molecules")
plt.xlabel("Wavelength (nm)")

plt.show(block=False)
plt.savefig("spacer3/gold/thiner/allgold.pdf")
plt.savefig("spacer3/gold/thiner/allgold.jpg")


# plt.figure(1)
# plt.plot(list_wavelength, R_gold1_mol, "r", label = " with molecules")
# plt.plot(list_wavelength, R_gold1_nomol, "--r", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 1 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold1.pdf")
# plt.savefig("spacer3/gold/thiner/gold1.jpg")

# plt.figure(10)
# plt.plot(list_wavelength, R_gold2_mol, "b", label = " with molecules")
# plt.plot(list_wavelength, R_gold2_nomol, "--b", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 2 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold2.pdf")
# plt.savefig("spacer3/gold/thiner/gold2.jpg")

# plt.figure(20)
# plt.plot(list_wavelength, R_gold3_mol, "y", label = " with molecules")
# plt.plot(list_wavelength, R_gold3_nomol, "--y", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 3 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold3.pdf")
# plt.savefig("spacer3/gold/thiner/gold3.jpg")

# plt.figure(30)
# plt.plot(list_wavelength, R_gold4_mol, "c", label = " with molecules")
# plt.plot(list_wavelength, R_gold4_nomol, "--c", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 4 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold4.pdf")
# plt.savefig("spacer3/gold/thiner/gold4.jpg")

# plt.figure(40)
# plt.plot(list_wavelength, R_gold5_mol, "k", label = " with molecules")
# plt.plot(list_wavelength, R_gold5_nomol, "--k", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 5 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold5.pdf")
# plt.savefig("spacer3/gold/thiner/gold5.jpg")

# plt.figure(50)
# plt.plot(list_wavelength, R_gold6_mol, "g", label = " with molecules")
# plt.plot(list_wavelength, R_gold6_nomol, "--g", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 6 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold6.pdf")
# plt.savefig("spacer3/gold/thiner/gold6.jpg")

# plt.figure(100)
# plt.plot(list_wavelength, R_gold7_mol, "m", label = " with molecules")
# plt.plot(list_wavelength, R_gold7_nomol, "--m", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gold 7 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gold/thiner/gold7.pdf")
# plt.savefig("spacer3/gold/thiner/gold7.jpg")

# ## Etude de la dépendance de la réflexion à la longueur d'onde, influence de la taille du cube
# thick_reso30 = 30
# thick_reso40 = 40
# thick_reso50 = 50
# thick_reso60 = 60
# thick_reso70 = 70
# thick_reso80 = 80

# width_reso30 = 30
# width_reso40 = 40
# width_reso50 = 50
# width_reso60 = 60
# width_reso70 = 70
# width_reso80 = 80

# list_wavelength = np.linspace(450, 1500, 200)
# R_reso30_mol = np.empty(list_wavelength.size)
# R_reso30_nomol = np.empty(list_wavelength.size)
# R_reso40_mol = np.empty(list_wavelength.size)
# R_reso40_nomol = np.empty(list_wavelength.size)
# R_reso50_mol = np.empty(list_wavelength.size)
# R_reso50_nomol = np.empty(list_wavelength.size)
# R_reso60_mol = np.empty(list_wavelength.size)
# R_reso60_nomol = np.empty(list_wavelength.size)
# R_reso70_mol = np.empty(list_wavelength.size)
# R_reso70_nomol = np.empty(list_wavelength.size)
# R_reso80_mol = np.empty(list_wavelength.size)
# R_reso80_nomol = np.empty(list_wavelength.size)
# idx = 0

# geometry_reso30_mol = {"thick_super": thick_super, "width_reso": width_reso30, "thick_reso": thick_reso30, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso30_nomol = {"thick_super": thick_super, "width_reso": width_reso30, "thick_reso": thick_reso30, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso40_mol = {"thick_super": thick_super, "width_reso": width_reso40, "thick_reso": thick_reso40, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso40_nomol = {"thick_super": thick_super, "width_reso": width_reso40, "thick_reso": thick_reso40, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso50_mol = {"thick_super": thick_super, "width_reso": width_reso50, "thick_reso": thick_reso50, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso50_nomol = {"thick_super": thick_super, "width_reso": width_reso50, "thick_reso": thick_reso50, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso60_mol = {"thick_super": thick_super, "width_reso": width_reso60, "thick_reso": thick_reso60, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso60_nomol = {"thick_super": thick_super, "width_reso": width_reso60, "thick_reso": thick_reso60, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso70_mol = {"thick_super": thick_super, "width_reso": width_reso70, "thick_reso": thick_reso70, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso70_nomol = {"thick_super": thick_super, "width_reso": width_reso70, "thick_reso": thick_reso70, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso80_mol = {"thick_super": thick_super, "width_reso": width_reso80, "thick_reso": thick_reso80, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_reso80_nomol = {"thick_super": thick_super, "width_reso": width_reso80, "thick_reso": thick_reso80, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_reso30_mol[idx] = reflectance(geometry_reso30_mol, wave, materials, n_mod)
#     R_reso30_nomol[idx] = reflectance(geometry_reso30_nomol, wave, materials, n_mod)
#     R_reso40_mol[idx] = reflectance(geometry_reso40_mol, wave, materials, n_mod)
#     R_reso40_nomol[idx] = reflectance(geometry_reso40_nomol, wave, materials, n_mod)
#     R_reso50_mol[idx] = reflectance(geometry_reso50_mol, wave, materials, n_mod)
#     R_reso50_nomol[idx] = reflectance(geometry_reso50_nomol, wave, materials, n_mod)
#     R_reso60_mol[idx] = reflectance(geometry_reso60_mol, wave, materials, n_mod)
#     R_reso60_nomol[idx] = reflectance(geometry_reso60_nomol, wave, materials, n_mod)
#     R_reso70_mol[idx] = reflectance(geometry_reso70_mol, wave, materials, n_mod)
#     R_reso70_nomol[idx] = reflectance(geometry_reso70_nomol, wave, materials, n_mod)
#     R_reso80_mol[idx] = reflectance(geometry_reso80_mol, wave, materials, n_mod)
#     R_reso80_nomol[idx] = reflectance(geometry_reso80_nomol, wave, materials, n_mod)    
#     idx += 1

# plt.figure(0)
# plt.plot(list_wavelength, R_reso30_mol, "r", label = "30 nm")
# plt.plot(list_wavelength, R_reso30_nomol, "--r") #,  label = "1 nm")
# plt.plot(list_wavelength, R_reso40_mol, "b", label = "40 nm")
# plt.plot(list_wavelength, R_reso40_nomol,"--b") #,  label = "2 nm")
# plt.plot(list_wavelength, R_reso50_mol,"y", label = "50 nm")
# plt.plot(list_wavelength, R_reso50_nomol,"--y") #, label = "3 nm")
# plt.plot(list_wavelength, R_reso60_mol,"c", label = "60 nm")
# plt.plot(list_wavelength, R_reso60_nomol,"--c") #, label = "4 nm")
# plt.plot(list_wavelength, R_reso70_mol,"k", label = "70 nm")
# plt.plot(list_wavelength, R_reso70_nomol,"--k") #, label = "3 nm")
# plt.plot(list_wavelength, R_reso80_mol,"g", label = "80 nm")
# plt.plot(list_wavelength, R_reso80_nomol,"--g") #, label = "4 nm")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube thickness / with (lines) or without (dotted) molecules")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/allreso.pdf")
# plt.savefig("spacer3/cubes/period300/allreso.jpg")


# plt.figure(1)
# plt.plot(list_wavelength, R_reso30_mol, "r", label = " with molecules")
# plt.plot(list_wavelength, R_reso30_nomol, "--r", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube 30 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/reso30.pdf")
# plt.savefig("spacer3/cubes/period300/reso30.jpg")

# plt.figure(4)
# plt.plot(list_wavelength, R_reso40_mol, "b", label = " with molecules")
# plt.plot(list_wavelength, R_reso40_nomol, "--b", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube 40 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/reso40.pdf")
# plt.savefig("spacer3/cubes/period300/reso40.jpg")


# plt.figure(5)
# plt.plot(list_wavelength, R_reso50_mol, "y", label = " with molecules")
# plt.plot(list_wavelength, R_reso50_nomol, "--y", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube 50 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/reso50.pdf")
# plt.savefig("spacer3/cubes/period300/reso50.jpg")


# plt.figure(6)
# plt.plot(list_wavelength, R_reso60_mol, "c", label = " with molecules")
# plt.plot(list_wavelength, R_reso60_nomol, "--c", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube 60 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/reso60.pdf")
# plt.savefig("spacer3/cubes/period300/reso60.jpg")


# plt.figure(7)
# plt.plot(list_wavelength, R_reso70_mol, "k", label = " with molecules")
# plt.plot(list_wavelength, R_reso70_nomol, "--k", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube 70 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/reso70.pdf")
# plt.savefig("spacer3/cubes/period300/reso70.jpg")


# plt.figure(8)
# plt.plot(list_wavelength, R_reso80_mol, "g", label = " with molecules")
# plt.plot(list_wavelength, R_reso80_nomol, "--g", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Cube 80 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/cubes/period300/reso80.pdf")
# plt.savefig("spacer3/cubes/period300/reso80.jpg")


## Etude de la dépendance de la réflexion à la longueur d'onde, influence de l'épaisseur de chrome
# thick_cr0 = 0
# thick_cr1 = 1
# thick_cr2 = 2
# thick_cr3 = 3

# list_wavelength = np.linspace(450, 1500, 200)
# R_cr0_mol = np.empty(list_wavelength.size)
# R_cr0_nomol = np.empty(list_wavelength.size)
# R_cr1_mol = np.empty(list_wavelength.size)
# R_cr1_nomol = np.empty(list_wavelength.size)
# R_cr2_mol = np.empty(list_wavelength.size)
# R_cr2_nomol = np.empty(list_wavelength.size)
# R_cr3_mol = np.empty(list_wavelength.size)
# R_cr3_nomol = np.empty(list_wavelength.size)
# idx = 0

# geometry_cr0_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr0, "period": period}
# geometry_cr0_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr0, "period": period}
# geometry_cr1_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr1, "period": period}
# geometry_cr1_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr1, "period": period}
# geometry_cr2_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr2, "period": period}
# geometry_cr2_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr2, "period": period}
# geometry_cr3_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr3, "period": period}
# geometry_cr3_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr3, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_cr0_mol[idx] = reflectance(geometry_cr0_mol, wave, materials, n_mod)
#     R_cr0_nomol[idx] = reflectance(geometry_cr0_nomol, wave, materials, n_mod)
#     R_cr1_mol[idx] = reflectance(geometry_cr1_mol, wave, materials, n_mod)
#     R_cr1_nomol[idx] = reflectance(geometry_cr1_nomol, wave, materials, n_mod)
#     R_cr2_mol[idx] = reflectance(geometry_cr2_mol, wave, materials, n_mod)
#     R_cr2_nomol[idx] = reflectance(geometry_cr2_nomol, wave, materials, n_mod)
#     R_cr3_mol[idx] = reflectance(geometry_cr3_mol, wave, materials, n_mod)
#     R_cr3_nomol[idx] = reflectance(geometry_cr3_nomol, wave, materials, n_mod)
#     idx += 1

# plt.figure(0)
# plt.plot(list_wavelength, R_cr0_mol, "r", label = "0 nm")
# plt.plot(list_wavelength, R_cr0_nomol, "--r") #,  label = "1 nm")
# plt.plot(list_wavelength, R_cr1_mol, "b", label = "1 nm")
# plt.plot(list_wavelength, R_cr1_nomol,"--b") #,  label = "2 nm")
# plt.plot(list_wavelength, R_cr2_mol,"y", label = "2 nm")
# plt.plot(list_wavelength, R_cr2_nomol,"--y") #, label = "3 nm")
# plt.plot(list_wavelength, R_cr3_mol,"c", label = "3 nm")
# plt.plot(list_wavelength, R_cr3_nomol,"--c") #, label = "4 nm")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Chrome thickness / with (lines) or without (dotted lines) molecules")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/chrome/allcr.pdf")
# plt.savefig("spacer3/chrome/allcr.jpg")


# plt.figure(1)
# plt.plot(list_wavelength, R_cr0_mol, "r", label = " with molecules")
# plt.plot(list_wavelength, R_cr0_nomol, "--r", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Chrome 0 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/chrome/cr0.pdf")
# plt.savefig("spacer3/chrome/cr0.jpg")


# plt.figure(2)
# plt.plot(list_wavelength, R_cr1_mol, "b", label = " with molecules")
# plt.plot(list_wavelength, R_cr1_nomol, "--b", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Chrome 1 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/chrome/cr1.pdf")
# plt.savefig("spacer3/chrome/cr1.jpg")


# plt.figure(3)
# plt.plot(list_wavelength, R_cr2_mol, "y", label = " with molecules")
# plt.plot(list_wavelength, R_cr2_nomol, "--y", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Chrome 2 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/chrome/cr2.pdf")
# plt.savefig("spacer3/chrome/cr2.jpg")


# plt.figure(4)
# plt.plot(list_wavelength, R_cr3_mol, "c", label = " with molecules")
# plt.plot(list_wavelength, R_cr3_nomol, "--c", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Chrome 3 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/chrome/cr3.pdf")
# plt.savefig("spacer3/chrome/cr3.jpg")

## Etude de la dépendance de la réflexion à la longueur d'onde, influence de la taille du gap
thick_gap1 = 1
thick_gap2 = 2
thick_gap3 = 3
thick_gap4 = 4
thick_gap5 = 5
thick_gap10 = 10

list_wavelength = np.linspace(450, 2000, 200)
R_gap1_mol = np.empty(list_wavelength.size)
R_gap1_nomol = np.empty(list_wavelength.size)
R_gap2_mol = np.empty(list_wavelength.size)
R_gap2_nomol = np.empty(list_wavelength.size)
R_gap3_mol = np.empty(list_wavelength.size)
R_gap3_nomol = np.empty(list_wavelength.size)
R_gap4_mol = np.empty(list_wavelength.size)
R_gap4_nomol = np.empty(list_wavelength.size)
R_gap5_mol = np.empty(list_wavelength.size)
R_gap5_nomol = np.empty(list_wavelength.size)
R_gap10_mol = np.empty(list_wavelength.size)
R_gap10_nomol = np.empty(list_wavelength.size)
idx = 0

geometry_gap1_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap1, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap1_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap1,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap2_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap2,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap2_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap2,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap3_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap3,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap3_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap3,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap4_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap4,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap4_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap4,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap5_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap5,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap5_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap5,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap10_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap10,"thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
geometry_gap10_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap10,"thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

for wavelength in list_wavelength:
    perm_Ag = epsAgbb(wavelength) # argent
    perm_Au = epsAubb(wavelength)
    perm_Cr = epsCrbb(wavelength)
    materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
    wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
    R_gap1_mol[idx] = reflectance(geometry_gap1_mol, wave, materials, n_mod)
    R_gap1_nomol[idx] = reflectance(geometry_gap1_nomol, wave, materials, n_mod)
    R_gap2_mol[idx] = reflectance(geometry_gap2_mol, wave, materials, n_mod)
    R_gap2_nomol[idx] = reflectance(geometry_gap2_nomol, wave, materials, n_mod)
    R_gap3_mol[idx] = reflectance(geometry_gap3_mol, wave, materials, n_mod)
    R_gap3_nomol[idx] = reflectance(geometry_gap3_nomol, wave, materials, n_mod)
    R_gap4_mol[idx] = reflectance(geometry_gap4_mol, wave, materials, n_mod)
    R_gap4_nomol[idx] = reflectance(geometry_gap4_nomol, wave, materials, n_mod)
    R_gap5_mol[idx] = reflectance(geometry_gap5_mol, wave, materials, n_mod)
    R_gap5_nomol[idx] = reflectance(geometry_gap5_nomol, wave, materials, n_mod)
    R_gap10_mol[idx] = reflectance(geometry_gap10_mol, wave, materials, n_mod)
    R_gap10_nomol[idx] = reflectance(geometry_gap10_nomol, wave, materials, n_mod)
    idx += 1

plt.figure(0)
plt.plot(list_wavelength, R_gap1_mol, "r", label = "1 nm")
plt.plot(list_wavelength, R_gap1_nomol, "--r") #,  label = "1 nm")
plt.plot(list_wavelength, R_gap2_mol, "b", label = "2 nm")
plt.plot(list_wavelength, R_gap2_nomol,"--b") #,  label = "2 nm")
plt.plot(list_wavelength, R_gap3_mol,"y", label = "3 nm")
plt.plot(list_wavelength, R_gap3_nomol,"--y") #, label = "3 nm")
plt.plot(list_wavelength, R_gap4_mol,"c", label = "4 nm")
plt.plot(list_wavelength, R_gap4_nomol,"--c") #, label = "4 nm")
plt.plot(list_wavelength, R_gap5_mol,"k", label = "5 nm")
plt.plot(list_wavelength, R_gap5_nomol,"--k") #, label = "5 nm")
plt.plot(list_wavelength, R_gap10_mol, "g", label = "10 nm")
plt.plot(list_wavelength, R_gap10_nomol,"--g") #, label = "10 nm")
plt.legend()
plt.ylabel("Reflectance")
plt.title("Gap sizes / with (lines) or without (dotted) molecules")
plt.xlabel("Wavelength (nm)")

plt.show(block=False)
plt.savefig("spacer3/gap/large/allgap.pdf")
plt.savefig("spacer3/gap/large/allgap.jpg")


# plt.figure(1)
# plt.plot(list_wavelength, R_gap1_mol, "r", label = " with molecules")
# plt.plot(list_wavelength, R_gap1_nomol, "--r", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gap 1 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gap/large/gap1.pdf")
# plt.savefig("spacer3/gap/large/gap1.jpg")


# plt.figure(2)
# plt.plot(list_wavelength, R_gap2_mol,"b", label = " with molecules")
# plt.plot(list_wavelength, R_gap2_nomol,"--b", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gap 2 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gap/large/gap2.pdf")
# plt.savefig("spacer3/gap/large/gap2.jpg")

# plt.figure(3)
# plt.plot(list_wavelength, R_gap3_mol,"y", label = " with molecules")
# plt.plot(list_wavelength, R_gap3_nomol,"--y", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gap 3 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gap/large/gap3.pdf")
# plt.savefig("spacer3/gap/large/gap3.jpg")


# plt.figure(4)
# plt.plot(list_wavelength, R_gap4_mol,"c", label = " with molecules")
# plt.plot(list_wavelength, R_gap4_nomol,"--c", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gap 4 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gap/large/gap4.pdf")
# plt.savefig("spacer3/gap/large/gap4.jpg")

# plt.figure(5)
# plt.plot(list_wavelength, R_gap5_mol,"k", label = " with molecules")
# plt.plot(list_wavelength, R_gap5_nomol,"--k", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gap 5 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gap/large/gap5.pdf")
# plt.savefig("spacer3/gap/large/gap5.jpg")


# plt.figure(10)
# plt.plot(list_wavelength, R_gap10_mol, "g", label = " with molecules")
# plt.plot(list_wavelength, R_gap10_nomol,"--g", label = "without")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Gap 10 nm")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/gap/large/gap10.pdf")
# plt.savefig("spacer3/gap/large/gap10.jpg")


## Etude de la dépendance de la réflexion à la longueur d'onde, influence de la période
# period300 = 300.01
# period400 = 400.01
# period500 = 500.01
# period600 = 600.01
# period700 = 700.01


# list_wavelength = np.linspace(500, 1600, 200)
# R_period300 = np.empty(list_wavelength.size)
# R_period400 = np.empty(list_wavelength.size)
# R_period500 = np.empty(list_wavelength.size)
# R_period600 = np.empty(list_wavelength.size)
# R_period700 = np.empty(list_wavelength.size)

# idx = 0

# geometry_period300 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period300}
# geometry_period400 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func,"thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period400}
# geometry_period500 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func,"thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period500}
# geometry_period600 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period600}
# geometry_period700 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period700}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_period300[idx] = reflectance(geometry_period300, wave, materials, n_mod)
#     R_period400[idx] = reflectance(geometry_period400, wave, materials, n_mod)
#     R_period500[idx] = reflectance(geometry_period500, wave, materials, n_mod)
#     R_period600[idx] = reflectance(geometry_period600, wave, materials, n_mod)
#     R_period700[idx] = reflectance(geometry_period700, wave, materials, n_mod)
#     idx += 1

# plt.figure(1)
# plt.plot(list_wavelength, R_period300, "r", label = "period 300 nm")
# plt.plot(list_wavelength, R_period400, "b", label = "period 400 nm")
# plt.plot(list_wavelength, R_period500,"g", label = "period 500 nm")
# plt.plot(list_wavelength, R_period600,"c", label = "period 600 nm")
# plt.plot(list_wavelength, R_period700,"k", label = "period 700 nm")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Reflectance (period sensibility)")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/period/period.jpg")
# plt.savefig("spacer3/period/period.pdf")


# # # Etude de la dépendance de tous les coefficients 
# perm_air = 1.0 ** 2
# perm_eau = 1.33 ** 2

# list_wavelength = np.linspace(450, 2000, 200)
# Rdown_air = np.empty(list_wavelength.size)
# Rup_air = np.empty(list_wavelength.size)
# Tdown_air = np.empty(list_wavelength.size)
# Tup_air = np.empty(list_wavelength.size)
# Rdown_eau = np.empty(list_wavelength.size)
# Rup_eau = np.empty(list_wavelength.size)
# Tdown_eau = np.empty(list_wavelength.size)
# Tup_eau = np.empty(list_wavelength.size)

# idx = 0

# geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func,"thick_mol": thick_mol, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials_air = {"perm_env": perm_air, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     materials_eau = {"perm_env": perm_eau, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     Rdown_air[idx], Rup_air[idx], Tdown_air[idx], Tup_air[idx] = reflectance(geometry, wave, materials_air, n_mod)
#     Rdown_eau[idx], Rup_eau[idx], Tdown_eau[idx], Tup_eau[idx] = reflectance(geometry, wave, materials_eau, n_mod)
#     idx += 1

# plt.figure(1)
# plt.plot(list_wavelength, Rdown_air, "b", label = "Rd A")
# plt.plot(list_wavelength, Rup_air, "r", label = "Ru A")
# plt.plot(list_wavelength, Tdown_air, "k", label = "Td A")
# plt.plot(list_wavelength, Tup_air, "g", label = "Tu A")
# plt.plot(list_wavelength, Rdown_eau, "--b", label = "Rd W")
# plt.plot(list_wavelength, Rup_eau, "--r", label = "Ru W")
# plt.plot(list_wavelength, Tdown_eau, "--k", label = "Td W")
# plt.plot(list_wavelength, Tup_eau, "--g", label = "Tu W")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Reflectance (environment sensibility)") # reflectance (functionnalization thickness)
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/allcoeffsAW.jpg")
# plt.savefig("spacer3/allcoeffsAW.pdf")

# plt.figure(2)
# plt.plot(list_wavelength, Rdown_air, "b", label = "Rd A")
# #plt.plot(list_wavelength, Rup_air, "r", label = "Ru A")
# #plt.plot(list_wavelength, Tdown_air, "k", label = "Td A")
# #plt.plot(list_wavelength, Tup_air, "g", label = "Tu A")
# plt.plot(list_wavelength, Rdown_eau, "--b", label = "Rd W")
# #plt.plot(list_wavelength, Rup_eau, "--r", label = "Ru W")
# #plt.plot(list_wavelength, Tdown_eau, "--k", label = "Td W")
# #plt.plot(list_wavelength, Tup_eau, "--g", label = "Tu W")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Reflectance (environment sensibility)") # reflectance (functionnalization thickness)
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/RdcoeffsAW.jpg")
# plt.savefig("spacer3/RdcoeffsAW.pdf")


# ## Etude de la dépendance de la réflexion à la longueur d'onde, influence de l'épaisseur de moélcules
# thick_mol0 = 0
# thick_mol1 = 1
# thick_mol2 = 2
# thick_mol3 = 3
# thick_mol4 = 4
# thick_mol5 = 5

# list_wavelength = np.linspace(1100, 1400, 200)
# R_mol0 = np.empty(list_wavelength.size)
# R_mol1 = np.empty(list_wavelength.size)
# R_mol2 = np.empty(list_wavelength.size)
# R_mol3 = np.empty(list_wavelength.size)
# R_mol4 = np.empty(list_wavelength.size)
# R_mol5 = np.empty(list_wavelength.size)
# idx = 0

# geometry_mol0 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_mol1 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol1, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_mol2 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol2, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_mol3 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol3, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_mol4 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol4, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_mol5 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap,"thick_func": thick_func, "thick_mol": thick_mol5, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_mol0[idx] = reflectance(geometry_mol0, wave, materials, n_mod)
#     R_mol1[idx] = reflectance(geometry_mol1, wave, materials, n_mod)
#     R_mol2[idx] = reflectance(geometry_mol2, wave, materials, n_mod)
#     R_mol3[idx] = reflectance(geometry_mol3, wave, materials, n_mod)
#     R_mol4[idx] = reflectance(geometry_mol4, wave, materials, n_mod)
#     R_mol5[idx] = reflectance(geometry_mol5, wave, materials, n_mod)        
#     idx += 1

# plt.figure(0)
# plt.plot(list_wavelength, R_mol0, "r", label = "0 nm")
# plt.plot(list_wavelength, R_mol1, "b", label = "1 nm")
# plt.plot(list_wavelength, R_mol2,"y", label = "2 nm")
# plt.plot(list_wavelength, R_mol3,"c", label = "3 nm")
# plt.plot(list_wavelength, R_mol4,"k", label = "4 nm")
# plt.plot(list_wavelength, R_mol5,"g", label = "5 nm")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Molecules thickness")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("spacer3/mol/allmol.pdf")
# plt.savefig("spacer3/mol/allmol.jpg")

# ## Etude de la dépendance de la réflexion à la longueur d'onde, influence de la couche d'or
# thick_gold0 = 0
# thick_gold10 = 10
# thick_gold20 = 20
# thick_gold30 = 30
# thick_gold40 = 40
# thick_gold50 = 50
# thick_gold100 = 100
# thick_gold200 = 200

# list_wavelength = np.linspace(750, 1600, 200)
# R_gold0_mol = np.empty(list_wavelength.size)
# R_gold0_nomol = np.empty(list_wavelength.size)
# R_gold10_mol = np.empty(list_wavelength.size)
# R_gold10_nomol = np.empty(list_wavelength.size)
# R_gold20_mol = np.empty(list_wavelength.size)
# R_gold20_nomol = np.empty(list_wavelength.size)
# R_gold30_mol = np.empty(list_wavelength.size)
# R_gold30_nomol = np.empty(list_wavelength.size)
# R_gold40_mol = np.empty(list_wavelength.size)
# R_gold40_nomol = np.empty(list_wavelength.size)
# R_gold50_mol = np.empty(list_wavelength.size)
# R_gold50_nomol = np.empty(list_wavelength.size)
# R_gold100_mol = np.empty(list_wavelength.size)
# R_gold100_nomol = np.empty(list_wavelength.size)
# R_gold200_mol = np.empty(list_wavelength.size)
# R_gold200_nomol = np.empty(list_wavelength.size)

# idx = 0

# geometry_gold0_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol, "thick_gold": thick_gold0, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold0_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0, "thick_gold": thick_gold0, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold10_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func,"thick_mol": thick_mol, "thick_gold": thick_gold10, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold10_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func,"thick_mol": 0, "thick_gold": thick_gold10, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold20_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func,"thick_mol": thick_mol, "thick_gold": thick_gold20, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold20_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0,"thick_gold": thick_gold20, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold30_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold30, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold30_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0,"thick_gold": thick_gold30, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold40_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold40, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold40_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0,"thick_gold": thick_gold40, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold50_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold50, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold50_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0,"thick_gold": thick_gold50, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold100_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold100, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold100_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0,"thick_gold": thick_gold100, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold200_mol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": thick_mol,"thick_gold": thick_gold200, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_gold200_nomol = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_mol": 0,"thick_gold": thick_gold200, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_gold0_mol[idx] = reflectance(geometry_gold0_mol, wave, materials, n_mod)
#     R_gold0_nomol[idx] = reflectance(geometry_gold0_nomol, wave, materials, n_mod)
#     R_gold10_mol[idx] = reflectance(geometry_gold10_mol, wave, materials, n_mod)
#     R_gold10_nomol[idx] = reflectance(geometry_gold10_nomol, wave, materials, n_mod)
#     R_gold30_mol[idx] = reflectance(geometry_gold30_mol, wave, materials, n_mod)
#     R_gold30_nomol[idx] = reflectance(geometry_gold30_nomol, wave, materials, n_mod)
#     R_gold20_mol[idx] = reflectance(geometry_gold20_mol, wave, materials, n_mod)
#     R_gold20_nomol[idx] = reflectance(geometry_gold20_nomol, wave, materials, n_mod)
#     R_gold40_mol[idx] = reflectance(geometry_gold40_mol, wave, materials, n_mod)
#     R_gold40_nomol[idx] = reflectance(geometry_gold40_nomol, wave, materials, n_mod)
#     R_gold50_mol[idx] = reflectance(geometry_gold50_mol, wave, materials, n_mod)
#     R_gold50_nomol[idx] = reflectance(geometry_gold50_nomol, wave, materials, n_mod)
#     R_gold100_mol[idx] = reflectance(geometry_gold100_mol, wave, materials, n_mod)
#     R_gold100_nomol[idx] = reflectance(geometry_gold100_nomol, wave, materials, n_mod)
#     R_gold200_mol[idx] = reflectance(geometry_gold200_mol, wave, materials, n_mod)
#     R_gold200_nomol[idx] = reflectance(geometry_gold200_nomol, wave, materials, n_mod)
#     idx += 1

# # plt.figure(1)
# # #plt.plot(list_wavelength, R_gold0_mol, "r", label = "Au 0")
# # #plt.plot(list_wavelength, R_gold0_nomol, label = "0 nm")
# # #plt.plot(list_wavelength, R_gold10_mol, "b", label = "Au 10")
# # #plt.plot(list_wavelength, R_gold10_nomol, label = "10 nm")
# # plt.plot(list_wavelength, R_gold20_mol, "r", label = "Au 20")
# # plt.plot(list_wavelength, R_gold20_nomol, "xr")#, label = "20 nm")
# # plt.plot(list_wavelength, R_gold30_mol,"b", label = "Au 30")
# # plt.plot(list_wavelength, R_gold30_nomol,"xb")#, label = "30 nm")
# # plt.plot(list_wavelength, R_gold40_mol,"g", label = "Au 40")
# # plt.plot(list_wavelength, R_gold40_nomol,"xg")#, label = "40 nm")
# # #plt.plot(list_wavelength, R_gold50_mol,"c", label = "Au 50")
# # #plt.plot(list_wavelength, R_gold50_nomol, label = "50 nm")
# # #plt.plot(list_wavelength, R_gold100_mol,"y", label = "Au 100")
# # #plt.plot(list_wavelength, R_gold100_nomol, label = "100 nm")
# # #plt.plot(list_wavelength, R_gold200_mol, label = "Au 200")
# # #plt.plot(list_wavelength, R_gold200_nomol, label = "200 nm")
# # plt.legend()
# # plt.ylabel("Reflectance")
# # plt.title("Gold thickness sensibility without molecules (lines) and with molecules (crosses)")
# # plt.xlabel("Wavelength (nm)")

# # plt.show(block=False)
# # plt.savefig("spacer3/thickness_gold/mol_nomol_widthgoldAll_longspectrum.jpg")

# plt.figure(2)
# plt.plot(list_wavelength, R_gold20_mol, "r", label = "With molecules")
# plt.plot(list_wavelength, R_gold20_nomol, "b", label = "without molecules")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Molecules sensibility (gold 20 nm)")
# plt.xlabel("Wavelength (nm)")
# plt.show(block=False)
# plt.savefig("spacer3/thickness_gold/mol_nomol_widthgold20_longspectrum.jpg")

# plt.figure(3)
# plt.plot(list_wavelength, R_gold30_mol, "r", label = "With molecules")
# plt.plot(list_wavelength, R_gold30_nomol, "b", label = "without molecules")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Molecules sensibility (gold 30 nm)")
# plt.xlabel("Wavelength (nm)")
# plt.show(block=False)
# plt.savefig("spacer3/thickness_gold/mol_nomol_widthgold30_longspectrum.jpg")

# plt.figure(4)
# plt.plot(list_wavelength, R_gold40_mol, "r", label = "With molecules")
# plt.plot(list_wavelength, R_gold40_nomol, "b", label = "without molecules")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Molecules sensibility (gold 40 nm)")
# plt.xlabel("Wavelength (nm)")
# plt.show(block=False)
# plt.savefig("spacer3/thickness_gold/mol_nomol_widthgold40_longspectrum.jpg")

# plt.figure(5)
# plt.plot(list_wavelength, R_gold5_func, label = "Au 5, with func")
# plt.plot(list_wavelength, R_gold5_nofunc, label = "Au 5 without func")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Dependance of width gold")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("width_gold_dep/reflectance_wav_func_nofunc_widthgold5.jpg")

# plt.figure(10)
# plt.plot(list_wavelength, R_gold10_func, label = "Au 10, with func")
# plt.plot(list_wavelength, R_gold10_nofunc, label = "Au 10 without func")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Dependance of width gold")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("width_gold_dep/reflectance_wav_func_nofunc_widthgold10.jpg")

# plt.figure(15)
# plt.plot(list_wavelength, R_gold15_func, label = "Au 15, with func")
# plt.plot(list_wavelength, R_gold15_nofunc, label = "Au 15 without func")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Dependance of width gold")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("width_gold_dep/reflectance_wav_func_nofunc_widthgold15.jpg")

# plt.figure(20)
# plt.plot(list_wavelength, R_gold20_func, label = "Au 20, with func")
# plt.plot(list_wavelength, R_gold20_nofunc, label = "Au 20 without func")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Dependance of width gold")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("width_gold_dep/reflectance_wav_func_nofunc_widthgold20.jpg")

# plt.figure(25)
# plt.plot(list_wavelength, R_gold25_func, label = "Au 25, with func")
# plt.plot(list_wavelength, R_gold25_nofunc, label = "Au 25 without func")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Dependance of width gold")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("width_gold_dep/reflectance_wav_func_nofunc_widthgold25.jpg")

## Etude de la dépendance de la réflexion à la longueur d'onde, influence du chrome
# thick_cr0 = 0
# thick_cr1 = 1
# thick_cr2 = 2
# thick_cr3 = 3

# list_wavelength = np.linspace(750, 1100, 200)
# R_cr0_func = np.empty(list_wavelength.size)
# R_cr0_nofunc = np.empty(list_wavelength.size)
# R_cr1_func = np.empty(list_wavelength.size)
# R_cr1_nofunc = np.empty(list_wavelength.size)
# R_cr2_func = np.empty(list_wavelength.size)
# R_cr2_nofunc = np.empty(list_wavelength.size)
# R_cr3_func = np.empty(list_wavelength.size)
# R_cr3_nofunc = np.empty(list_wavelength.size)
# idx = 0

# geometry_cr0_func = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr0, "period": period}
# geometry_cr0_nofunc = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr0, "period": period}
# geometry_cr1_func = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr1, "period": period}
# geometry_cr1_nofunc = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr1, "period": period}
# geometry_cr2_func = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr2, "period": period}
# geometry_cr2_nofunc = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr2, "period": period}
# geometry_cr3_func = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr3, "period": period}
# geometry_cr3_nofunc = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr3, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_cr0_func[idx] = reflectance(geometry_cr0_func, wave, materials, n_mod)
#     R_cr0_nofunc[idx] = reflectance(geometry_cr0_nofunc, wave, materials, n_mod)
#     R_cr1_func[idx] = reflectance(geometry_cr1_func, wave, materials, n_mod)
#     R_cr1_nofunc[idx] = reflectance(geometry_cr1_nofunc, wave, materials, n_mod)
#     R_cr2_func[idx] = reflectance(geometry_cr2_func, wave, materials, n_mod)
#     R_cr2_nofunc[idx] = reflectance(geometry_cr2_nofunc, wave, materials, n_mod)
#     R_cr3_func[idx] = reflectance(geometry_cr3_func, wave, materials, n_mod)
#     R_cr3_nofunc[idx] = reflectance(geometry_cr3_nofunc, wave, materials, n_mod)
#     idx += 1

# plt.figure(3)
# plt.plot(list_wavelength, R_cr0_func, label = "Cr 0, with func")
# plt.plot(list_wavelength, R_cr0_nofunc, label = "Cr 0 without func")
# plt.plot(list_wavelength, R_cr1_func, label = "Cr 1, with func")
# plt.plot(list_wavelength, R_cr1_nofunc, label = "Cr 1 without func")
# plt.plot(list_wavelength, R_cr2_func, label = "Cr 2, with func")
# plt.plot(list_wavelength, R_cr2_nofunc, label = "Cr 2 without func")
# plt.plot(list_wavelength, R_cr3_func, label = "Cr 3, with func")
# plt.plot(list_wavelength, R_cr3_nofunc, label = "Cr 3 without func")
# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Dependance of width chrome")
# plt.xlabel("Wavelength (nm)")

# plt.show(block=False)
# plt.savefig("width_chrome_dep/reflectance_wav_func_nofunc_widthchromeAll.jpg")

### Startstudies

# plt.subplot(212)
# plt.plot(list_wavelength, phase_R_up_NR, "r", label="phase R up NR")
# plt.plot(list_wavelength, phase_R_down_NR, "b", label="phase R down NR")
# plt.legend()
# plt.xlabel("Wavelength (nm) ")
# plt.ylabel("Phase of reflectance")
# plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("reflectance_dependanceWavelength_ref_dessus_dessous_phase_module.jpg")

## Etude de la dépendance de la réflexion à la longueur d'onde, en fonction de l'épaisseur de gold
# thick_gold0 = 0
# thick_gold10 = 10
# thick_gold20 = 20
# thick_gold30 = 30
# thick_gold40 = 40 
# thick_gold50 = 50
# thick_gold100 = 100
# thick_gold200 = 200

# list_wavelength = np.linspace(750, 1200, 200)
# R_gold0 = np.empty(list_wavelength.size)
# R_gold10 = np.empty(list_wavelength.size)
# R_gold20 = np.empty(list_wavelength.size)
# R_gold30 = np.empty(list_wavelength.size)
# R_gold40 = np.empty(list_wavelength.size)
# R_gold50 = np.empty(list_wavelength.size)
# R_gold100 = np.empty(list_wavelength.size)
# R_gold200 = np.empty(list_wavelength.size)
# idx = 0

# geometry0 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold0, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry10 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold10, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry20 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold20, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry30 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold30, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry40 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold40, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry50 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold50, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry100 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold100, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry200 = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold200, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}


# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength)
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_gold0[idx] = reflectance(geometry0, wave, materials, n_mod)
#     R_gold10[idx] = reflectance(geometry10, wave, materials, n_mod)
#     R_gold20[idx] = reflectance(geometry20, wave, materials, n_mod)
#     R_gold30[idx] = reflectance(geometry30, wave, materials, n_mod)
#     R_gold40[idx] = reflectance(geometry40, wave, materials, n_mod)
#     R_gold50[idx] = reflectance(geometry50, wave, materials, n_mod)
#     R_gold100[idx] = reflectance(geometry100, wave, materials, n_mod)
#     R_gold200[idx] = reflectance(geometry200, wave, materials, n_mod)
#     idx += 1

# plt.figure(1)
# plt.plot(list_wavelength, R_gold0, "r", label="R gold 0")
# plt.plot(list_wavelength, R_gold10, "b", label="R gold 10")
# plt.plot(list_wavelength, R_gold20, "k", label="R gold 20")
# plt.plot(list_wavelength, R_gold30, "g", label="R gold 30")
# plt.plot(list_wavelength, R_gold40, "r", label="R gold 40")
# plt.plot(list_wavelength, R_gold50, "m", label="R gold 50")
# plt.plot(list_wavelength, R_gold100, "y", label = "R gold 100")
# plt.plot(list_wavelength, R_gold200, "0.7", label="R gold 200")
# plt.legend()
# plt.xlabel("Wavelength (nm) ")
# plt.ylabel("Module of reflectance")
# plt.title("Wavelength dependance, gold thickness dependance")
# plt.show(block=False)
# plt.savefig("startstudies/reflectance_dependanceWavelength_ThicknessGold.jpg")


## influence largeur du résonateur
# list_width_reso = np.linspace(45, 350, 100)
# R_up = np.empty(list_width_reso.size)
# R_down = np.empty(list_width_reso.size)
# idx = 0

# materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
# wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}

# for width_reso in list_width_reso:
#     geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
#     R_down[idx], R_up[idx] = reflectance(geometry, wave, materials, n_mod)
#     idx += 1

# plt.figure(1)
# plt.plot(list_width_reso, R_down)
# plt.xlabel("Width of the rod (nm)")
# plt.ylabel("R down")
# plt.show(block=False)
# plt.title("Cavity size dependance of the reflection")
# plt.savefig("startstudies/r0_Lc_Rdown.jpg")

# plt.figure(2)
# plt.plot(list_width_reso, R_up)
# plt.xlabel("Width of the rod (nm)")
# plt.ylabel("R up")
# plt.show(block=False)
# plt.title("Cavity size dependance of the reflection")
# plt.savefig("startstudies/r0_Lc_Rup.jpg")

# list_wavelength = np.linspace(750, 1500, 500)
# R_up = np.empty(list_wavelength.size)
# R_down = np.empty(list_wavelength.size)
# R_up_phase = np.empty(list_wavelength.size)
# R_down_phase = np.empty(list_wavelength.size)
# T_up = np.empty(list_wavelength.size)
# T_down = np.empty(list_wavelength.size)
# T_up_phase = np.empty(list_wavelength.size)
# T_down_phase = np.empty(list_wavelength.size)
# idx = 0

# geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength)
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_down[idx], R_down_phase[idx], R_up[idx], R_up_phase[idx], T_down[idx], T_down_phase[idx], T_up[idx], T_up_phase[idx] = reflectance(geometry, wave, materials, n_mod)
#     idx += 1

# plt.figure(3)
# plt.title("Wavelength dependance of the reflection")

# #plt.subplot(211)
# plt.plot(list_wavelength, R_up, "r", label = "R up")
# plt.plot(list_wavelength, R_down, "b", label = "R down")
# plt.plot(list_wavelength, T_up, "g", label = "T up")
# plt.plot(list_wavelength, T_down, "k", label = "T down")
# plt.legend()
# plt.ylabel("Coefficients")

# # plt.subplot(212)
# # plt.plot(list_wavelength, R_up_phase, "r", label = "R up")
# # plt.plot(list_wavelength, R_down_phase, "b", label = "R down")
# # plt.plot(list_wavelength, T_up_phase, "g", label = "T up")
# # plt.plot(list_wavelength, T_down_phase, "k", label = "T down")
# # plt.legend()
# # plt.xlabel("Wavelength (nm)")
# # plt.ylabel("Phases")

# plt.show(block=False)
# plt.savefig("startstudies/ref_wav_all_coeffs.jpg")

### Influence de l'angle d'incidence
# list_wavelength = np.linspace(750, 1200, 100)
# R_theta0 = np.empty(list_wavelength.size)
# R_theta20 = np.empty(list_wavelength.size)
# R_theta0_phase = np.empty(list_wavelength.size)
# R_theta20_phase = np.empty(list_wavelength.size)
# idx = 0

# geometry = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave_theta0 = {"wavelength": wavelength, "angle": 0, "polarization": polarization}
#     wave_theta20 = {"wavelength": wavelength, "angle": 20, "polarization": polarization}
#     R_theta0[idx], R_theta0_phase[idx] = reflectance(geometry, wave_theta0, materials, n_mod)
#     R_theta20[idx], R_theta20_phase[idx] = reflectance(geometry, wave_theta20, materials, n_mod)
#     idx += 1

# plt.figure(2)
# plt.title("Reflectance depending incidence angle")
# plt.subplot(211)
# plt.plot(list_wavelength, R_theta0, label = "Normal incidence")
# plt.plot(list_wavelength, R_theta20, label = "20° incidence")
# plt.legend()
# plt.ylabel("Modulus")

# plt.subplot(212)
# plt.plot(list_wavelength, R_theta0_phase, label = "Normal incidence")
# plt.plot(list_wavelength, R_theta20_phase, label = "20° incidence")
# plt.legend()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Phase")
# plt.show(block=False)
# plt.savefig("startstudies/reflectance_wav_inc0_inc20.jpg")


## Etude de la dépendance de la réflexion à la longueur d'onde
# list_wavelength = np.linspace(750, 1100, 200)
# R_cr_func = np.empty(list_wavelength.size)
# R_cr_nofunc = np.empty(list_wavelength.size)
# R_nocr_func = np.empty(list_wavelength.size)
# R_nocr_nofunc = np.empty(list_wavelength.size)
# #Rphase = np.empty(list_wavelength.size)
# #Rnofunc = np.empty(list_wavelength.size)
# #Rnofunc_phase = np.empty(list_wavelength.size)
# idx = 0

# geometry_cr_func = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_cr_nofunc = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": thick_cr, "period": period}
# geometry_nocr_func = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": thick_func, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": 0, "period": period}
# geometry_nocr_nofunc = {"thick_super": thick_super, "width_reso": width_reso, "thick_reso": thick_reso, "thick_gap": thick_gap, "thick_func": 0, "thick_gold": thick_gold, "thick_sub": thick_sub, "thick_chrome": 0, "period": period}

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength) # argent
#     perm_Au = epsAubb(wavelength)
#     perm_Cr = epsCrbb(wavelength)
#     materials = {"perm_env": perm_env, "perm_dielec": perm_dielec, "perm_Glass": perm_Glass, "perm_Ag": perm_Ag, "perm_Au": perm_Au, "perm_Cr": perm_Cr}
#     wave = {"wavelength": wavelength, "angle": angle, "polarization": polarization}
#     R_cr_func[idx] = reflectance(geometry_cr_func, wave, materials, n_mod)
#     R_cr_nofunc[idx] = reflectance(geometry_cr_nofunc, wave, materials, n_mod)
#     R_nocr_func[idx] = reflectance(geometry_nocr_func, wave, materials, n_mod)
#     R_nocr_nofunc[idx] = reflectance(geometry_nocr_nofunc, wave, materials, n_mod)
#     idx += 1

# plt.figure(3)
# #plt.subplot(211)
# plt.plot(list_wavelength, R_cr_func, label = "Cr and func")
# plt.plot(list_wavelength, R_cr_nofunc, label = "Cr and no func")
# plt.plot(list_wavelength, R_nocr_func, label = "No cr and func")
# plt.plot(list_wavelength, R_nocr_nofunc, label = "No cr and no func")

# plt.legend()
# plt.ylabel("Reflectance")
# plt.title("Width chrome = 1 nm, width func = 3 nm")

# # plt.subplot(212)
# # plt.plot(list_wavelength, Rphase, label = "with func") 
# # plt.plot(list_wavelength, Rnofunc_phase, label = "without func")
# # plt.legend()
# plt.xlabel("Wavelength (nm)")
# # plt.ylabel("Phase of reflectance")

# plt.show(block=False)
# plt.savefig("reflectance_wav_func_nofunc_chrome_nocr.jpg")

# ### Etude de la permittivité du chrome

# test = epsCrbb(800)
# print(test)

# list_wavelength = np.linspace(100,10000,100)
# perm_Cr = np.empty(list_wavelength.size, dtype = complex)
# idx = 0
# for wavelength in list_wavelength:
#     perm_Cr[idx] = epsCrbb(wavelength)
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_wavelength, np.real(perm_Cr))
# #plt.xlabel("Wavelength")
# plt.ylabel("Real part")
# plt.title("Chrome permittivity")

# plt.subplot(212)
# plt.plot(list_wavelength, np.imag(perm_Cr))
# plt.ylabel("Imaginary part")
# plt.xlabel("Wavelength")
# plt.show(block=False)
# plt.savefig("permittivity_cr.jpg")