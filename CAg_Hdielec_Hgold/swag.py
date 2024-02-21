import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv
#from PyMoosh import *

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
    n=int(T.shape[1]/2)
    J=np.linalg.inv(np.eye(n)-np.matmul(U[0:n,0:n],T[n:2*n,n:2*n]))
    K=np.linalg.inv(np.eye(n)-np.matmul(T[n:2*n,n:2*n],U[0:n,0:n]))
    S=np.block([[T[0:n,0:n]+np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,0:n]),T[n:2*n,0:n]),np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,n:2*n])],[np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,0:n]),U[n:2*n,n:2*n]+np.matmul(np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,n:2*n]),U[0:n,n:2*n])]])
    return S

def c_bas(A,V,h):
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

def marche(a,b,p,n,x):
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

def creneau(k0,a0,pol,e1,e2,a,n,x0):
    nmod=int(n/2)
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))
    if (pol==0):
        M=alpha*alpha-k0*k0*marche(e1,e2,a,n,x0)
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        U=marche(1/e1,1/e2,a,n,x0)
        T=np.linalg.inv(U)
        M=np.matmul(np.matmul(np.matmul(T,alpha),np.linalg.inv(marche(e1,e2,a,n,x0))),alpha)-k0*k0*T
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(np.matmul(U,E),np.diag(L))]])
    return P,L

def homogene(k0,a0,pol,epsilon,n):
    nmod=int(n/2)
    valp=np.sqrt(epsilon*k0*k0-(a0+2*np.pi*np.arange(-nmod,nmod+1))**2+0j)
    valp=valp*(1-2*(valp<0))
    P=np.block([[np.eye(n)],[np.diag(valp*(pol/epsilon+(1-pol)))]])
    return P,valp

def interface(P,Q):
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
def reflectance(thick_super, width_reso, thick_gap, thick_reso, thick_gold, period, width_fiber, thick_sub, wavelength, angle, polarization, perm_dielec, perm_Glass, perm_Ag, perm_Au,  n_mod):    
    n = 2 * n_mod + 1
    wavelength_norm = wavelength / period
    
    thick_super = thick_super / period 
    width_reso = width_reso / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period
    thick_gap = thick_gap / period
    wavelength_norm = wavelength / period
    width_fiber = width_fiber / period
    thick_sub = thick_sub / period
    position_reso = 0.5 - width_reso / 2
    position_fiber = 0.5 - width_fiber / 2

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle * np.pi / 180)
    
    P, V = homogene(k0, a0, polarization, 1, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])
    
    Pc, Vc = creneau(k0, a0, polarization, perm_Ag, 1, width_reso, n, position_reso)
    S = cascade(S, interface(P, Pc))
    S = c_bas(S, Vc, thick_reso)
    
    Pdielec, Vdielec = homogene(k0, a0, polarization, perm_dielec, n)
    S = cascade(S, interface(Pc, Pdielec))
    S = c_bas(S, Vdielec, thick_gap)

    Pgold, Vgold = homogene(k0, a0, polarization, perm_Au, n)
    S = cascade(S, interface(Pdielec, Pgold))
    S = c_bas(S, Vgold, thick_gold)

    # P, V = creneau(k0, a0, polarization, perm_dielec, 1, width_fiber, n, position_fiber)
    # S = cascade(S, interface(Pc, P))
    # S = c_bas(S, V, thick_gap)

    # Pc, Vc = creneau(k0, a0, polarization, perm_Au, 1, width_fiber, n_mod_total, position_fiber)
    # S = cascade(S, interface(P, Pc))
    # S = c_bas(S, Vc, thick_gold)

    # P, V = creneau(k0, a0, polarization, perm_Glass, perm_Au, width_fiber, n_mod_total, position_fiber)
    # S = cascade(S, interface(Pc, P))
    # S = c_bas(S, V, thick_sub)

    R = abs(S[n_mod, n_mod]) ** 2 * np.real(V[n_mod]) / k0
    return R 

def Field(thick_super, width_reso, thick_gap, thick_reso, thick_gold, period, width_fiber, thick_sub, wavelength, angle, polarization, perm_dielec, perm_Glass, perm_Ag, perm_Au,  n_mod):
    thick_super = thick_super / period 
    width_reso = width_reso / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period
    thick_gap = thick_gap / period
    wavelength_norm = wavelength / period
    width_fiber = width_fiber / period
    thick_sub = thick_sub / period
    
    position_reso = 0.5 - width_reso / 2
    position_fiber = 0.5 - width_fiber / 2

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle)

    n_mod_total = 2 * n_mod + 1

    A = [] # matrice de stockage de tous les modes et valeurs propres

    # milieu incident, couche homogene d'air
    P_in, V_in = homogene(k0, a0, polarization, 1, n_mod_total)
    A.append([P_in.tolist(), V_in.tolist()])

    # couche 2 : cube d'argent dans couche d'air
    P_2, V_2 = creneau(k0, a0, polarization, perm_Ag, 1, width_reso, n_mod_total, position_reso)
    A.append([P_2.tolist(), V_2.tolist()])

    # couche 3 : couche de dielectrique au dessus de la fibre
    P_3, V_3 = creneau(k0, a0, polarization, perm_dielec, 1, width_fiber, n_mod_total, position_fiber)
    A.append([P_3.tolist(), V_3.tolist()])

    # couche 4 : couche d'or au dessus de la fibre
    P_4, V_4 = creneau(k0, a0, polarization, perm_Au, 1, width_fiber, n_mod_total, position_fiber)
    A.append([P_4.tolist(), V_4.tolist()])

    # couche 5 : fibre de verre
    P_5, V_5 = creneau(k0, a0, polarization, perm_Glass, 1, width_fiber, n_mod_total, position_fiber)
    A.append([P_5.tolist(), V_5.tolist()])

    thickness = np.array([thick_super, thick_reso, thick_gap, thick_gold, thick_sub])

    n_couches = thickness.size

    # matrice neutre pour l'opération de cascadage
    S11 = np.zeros((n_mod_total,n_mod_total))
    S12 = np.eye(n_mod_total)
    S1 = np.append(S11,S12,axis=0)
    S2 = np.append(S12,S11,axis=0)
    S0 = np.append(S1,S2,1)

    # matrices d'interface
    B = []
    for k in range(n_couches-1): # car nc - 1 interfaces dans la structure
        a = np.array(A[k][0])
        b = np.array(A[k+1][0])
        c = interface(a,b)
        c = c.tolist()
        B.append(c)

    S = []
    S0 = S0.tolist()
    S.append(S0)

    # Matrices montantes
    for k in range(n_couches-1):
        a = np.array(S[k])
        b = c_haut(np.array(B[k]),np.array(A[k][1]),thickness[k])
        S_new = cascade(a,b) 
        S.append(S_new.tolist())

    a = np.array(S[n_couches-1])
    b = np.array(A[n_couches-1][1])
    c = c_bas(a,b,thickness[n_couches-1])
    S.append(c.tolist())

    # Matrices descendantes
    Q = []
    Q.append(S0)

    for k in range(n_couches-1):
        a = np.array(B[n_couches-k-2])
        b = np.array(A[n_couches-(k+1)][1])
        c = thickness[n_couches-(k+1)]
        d = np.array(Q[k])
        Q_new = cascade(c_bas(a,b,c),d)
        Q.append(Q_new.tolist())

    a = np.array(Q[k])
    b = np.array(A[0][1])
    c = c_haut(a,b,thickness[n_couches-(k+1)])
    Q.append(c.tolist())

    stretch = period / (2 * n_mod + 1)

    exc = np.zeros(2*n_mod_total)
    # Eclairage par au dessus, onde plane
    #exc[n_mod] = 1
    # eclairage par en dessous, onde plane
    #exc[n_mod_total + n_mod] = 1
    # eclairage par en dessous, guide d'onde (le mode avec la plus grande partie réelle)
    position = np.argmax(np.real(V_5))
    exc[n_mod_total + position] = 1

    ny = np.floor(thickness * period / stretch)

    M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[0]), thickness[0], a0)
    
    for j in np.arange(1,n_couches):
        M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[j]), thickness[j], a0) 
        M = np.append(M,M_new, 0)

    Mfield = np.abs(M)**2
    return Mfield


def Field_resosimple(thick_super, width_reso, thick_reso, period, thick_sub, wavelength, angle, polarization, perm_Ag, n_mod):
    thick_super = thick_super / period 
    width_reso = width_reso / period
    thick_reso = thick_reso / period
    wavelength_norm = wavelength / period
    thick_sub = thick_sub / period
    
    position_reso = 0.5 - width_reso / 2

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle)

    n_mod_total = 2 * n_mod + 1

    A = [] # matrice de stockage de tous les modes et valeurs propres

    # milieu incident, couche homogene d'air
    P_in, V_in = homogene(k0, a0, polarization, 1, n_mod_total)
    A.append([P_in.tolist(), V_in.tolist()])

    # couche 2 : cube d'argent dans couche d'air
    P_2, V_2 = creneau(k0, a0, polarization, perm_Ag, 1, width_reso, n_mod_total, position_reso)
    A.append([P_2.tolist(), V_2.tolist()])

   # couche 3 : couche homogène d'air
    A.append([P_in.tolist(), V_in.tolist()])
    
    thickness = np.array([thick_super, thick_reso, thick_sub])

    n_couches = thickness.size

    # matrice neutre pour l'opération de cascadage
    S11 = np.zeros((n_mod_total,n_mod_total))
    S12 = np.eye(n_mod_total)
    S1 = np.append(S11,S12,axis=0)
    S2 = np.append(S12,S11,axis=0)
    S0 = np.append(S1,S2,1)

    # matrices d'interface
    B = []
    for k in range(n_couches-1): # car nc - 1 interfaces dans la structure
        a = np.array(A[k][0])
        b = np.array(A[k+1][0])
        c = interface(a,b)
        c = c.tolist()
        B.append(c)

    S = []
    S0 = S0.tolist()
    S.append(S0)

    # Matrices montantes
    for k in range(n_couches-1):
        a = np.array(S[k])
        b = c_haut(np.array(B[k]),np.array(A[k][1]),thickness[k])
        S_new = cascade(a,b) 
        S.append(S_new.tolist())

    a = np.array(S[n_couches-1])
    b = np.array(A[n_couches-1][1])
    c = c_bas(a,b,thickness[n_couches-1])
    S.append(c.tolist())

    # Matrices descendantes
    Q = []
    Q.append(S0)

    for k in range(n_couches-1):
        a = np.array(B[n_couches-k-2])
        b = np.array(A[n_couches-(k+1)][1])
        c = thickness[n_couches-(k+1)]
        d = np.array(Q[k])
        Q_new = cascade(c_bas(a,b,c),d)
        Q.append(Q_new.tolist())

    a = np.array(Q[k])
    b = np.array(A[0][1])
    c = c_haut(a,b,thickness[n_couches-(k+1)])
    Q.append(c.tolist())

    stretch = period / (2 * n_mod + 1)

    exc = np.zeros(2*n_mod_total)
    # Eclairage par au dessus, onde plane
    exc[n_mod] = 1
    # eclairage par en dessous, onde plane
    #exc[n_mod_total + n_mod] = 1
    # eclairage par en dessous, guide d'onde (le mode avec la plus grande partie réelle)
    #position = np.argmax(np.real(V_5))
    #exc[n_mod_total + position] = 1

    ny = np.floor(thickness * period / stretch)

    M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[0]), thickness[0], a0)
    
    for j in np.arange(1,n_couches):
        M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[j]), thickness[j], a0) 
        M = np.append(M,M_new, 0)

    Mfield = np.abs(M)**2
    return Mfield


def show_structure(thick, width, period):
    """ visualisation d'une structure constituée de deux matériaux uniquement -- à travailler """

    number_layers = len(thick)

    p = (period - np.array(width)) / 2

    X = np.array([thick, width, p])
    X = X.T

    h = np.array(thick) / period
    x = X / period

    n = 600
    H = sum(h)
    M0 = np.zeros((n))
    for j in range(number_layers):
        tmp = np.zeros(n)
        position = np.arange(1,n+1)/n
        for k in range(int((X.shape[1]-1)/2)):
            tmp = tmp + (position>x[j,2*k+2])*(position<x[j,2*k+1]+x[j,2*k+2])
        if (x[j,2*k+1]+x[j,2*k+2]>1):
            tmp = tmp + (position<x[j,2*k+1]+x[j,2*k+2]-1)
        cst = int(np.floor(h[j]*n))
        M2 = np.tile(tmp,(cst,1))
        if (j == 0):
            M = M2
        else:
            M = np.vstack((M,M2))

    M = 1 - M  
    return M

### Swag-structure

# Espace au dessus
thick_super = 200

# Cubes
width_reso = 75 # largeur du cube
thick_gap = 5 # hauteur de diéléctrique en dessous du cube
thick_reso = 75 #hauteur du cube
thick_gold = 200 # hauteur de l'or au dessus du substrat
period = 300.2153 # periode

# Largeur de la fibre
width_fiber = 0

# Espace en dessous de la fibre
thick_sub = 200

# A modifier selon le point de fonctionnement
wavelength = 1500.021635
angle = 0
polarization = 1

## Paramètres des matériaux
perm_dielec = 1.41 ** 2 # spacer
perm_Glass = 1.5 ** 2 # substrat
#perm_Ag = epsAgbb(wavelength) # argent
#perm_Au = epsAubb(wavelength) # or

n_mod = 50 
n_mod_total = 2 * n_mod + 1

### Pour avoir une représentation de la structure (WIP)
# thicknesses = np.array([thick_super, thick_reso, thick_gap, thick_gold, thick_sub]) # un résonateur de 75 nm de côté, un gap de 5 nm, et un substrat de metal
# widths = np.array([0,width_reso, width_fiber, width_fiber, width_fiber])
# M = show_structure(thicknesses, widths, period)

# plt.figure(3)
# plt.imshow(M*127, cmap = 'bone')
# plt.show(block = False)    
# plt.savefig("WIP_Visu_structure_GP_Lc_fiber.pdf")

# Rmq : la fonction n'est faite que pour 2 matériaux, il faudrait pouvoir améliorer ça 


### Pour tracer la carte de champ 

# Mfield2 = Field(thick_super, width_reso, thick_gap, thick_reso, thick_gold, period, width_fiber, thick_sub, wavelength, angle, polarization, perm_dielec, perm_Glass, perm_Ag, perm_Au, n_mod)
# plt.figure(1)
# plt.imshow(Mfield2, cmap = 'jet', aspect = 'auto')
# plt.colorbar()
# plt.title("Python X dessous Lc 75 fibre")
# plt.show(block=False)
# plt.savefig("Python_X_dessous_L75_fibre.jpg") 

# Mfield = Field_resosimple(thick_super, width_reso, thick_reso, period, thick_sub, wavelength, angle, polarization, perm_Ag, n_mod)
# plt.figure(7)
# plt.imshow(Mfield, cmap = 'jet', aspect = 'auto')
# plt.colorbar()
# plt.title("Python X dessous L 250 NR")
# plt.show(block=False)
# plt.savefig("Field_Xdessous_L250_NR.jpg") 


### Pour étudier l'influence de la largeur du résonateur

# list_L = np.linspace(50, 350, 350-50)
# R = np.empty(list_L.size)
# idx = 0

# for width_reso in list_L:   
#     R[idx] = reflectance(thick_super, width_reso, thick_gap, thick_reso, thick_gold, period, width_fiber, thick_sub, wavelength, angle, polarization, perm_dielec, perm_Glass, perm_Ag, perm_Au, n_mod)
#     idx += 1

# plt.figure(1)
# plt.plot(list_L, R) #, "xb", label="avec fonction" )
# plt.legend()
# plt.xlabel("Width of the rod")
# plt.ylabel("r0")
# plt.title("Cavity size dependance of the reflection")
# plt.show(block=False)
# plt.savefig("reflectance_dependanceWidthReso.jpg")

### version : fiber without fiber (thickness 0 nm) and Au substrate 

### Pour étudier la dépendance en longueur d'onde du coefficient de réflexion / tranmission selon si on éclaire par dessus / par dessous


list_wavelength = np.linspace(400, 1200, 100)
R = np.empty(list_wavelength.size)
idx = 0

for wavelength in list_wavelength:
    perm_Ag = epsAgbb(wavelength) # argent
    perm_Au = epsAubb(wavelength)
    R[idx] = reflectance(thick_super, width_reso, thick_gap, thick_reso, thick_gold, period, width_fiber, thick_sub, wavelength, angle, polarization, perm_dielec, perm_Glass, perm_Ag, perm_Au, n_mod)
    idx += 1

plt.figure(2)
plt.plot(list_wavelength, R) #, "xb", label="avec fonction" )
plt.legend()
plt.xlabel("Wavelength (nm) ")
plt.ylabel("r0")
plt.title("Wavelength dependance of the reflection")
plt.show(block=False)
plt.savefig("reflectance_dependanceWavelength_theta0.jpg")