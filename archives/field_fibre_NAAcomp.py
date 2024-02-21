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

def reseau(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position. #not anymore
    Warning : There is nothing checking that the blocks don't overlapp.
    Remark : 'reseau' is a version of 'creneau' taking account several blocs in a period
    '''
    n_blocs=blocs.shape[0];
    nmod=int(n/2)
    M1=marche(e2,e1,blocs[0,0],n,blocs[0,1])
    M2=marche(1/e2,1/e1,blocs[0,0],n,blocs[0,1])
    if n_blocs>1:
        for k in range(1,n_blocs):
            M1=M1+marche(e2-e1,0,blocs[k,0],n,blocs[k,1])
            M2=M2+marche(1/e2-1/e1,0,blocs[k,0],n,blocs[k,1])
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

def show_field(a0,A,thickness, exc,periode):
    n = len(A[0][1]) # n = 2 * n_mod + 1
    #n_couches = thickness.shape[0]
    n_couches = thickness.size
    S11=np.zeros((n,n))
    S12=np.eye(n)
    S1=np.append(S11,S12,axis=0)
    S2=np.append(S12,S11,axis=0)
    S0=np.append(S1,S2,1)

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

    # matrices montantes
    for k in range(n_couches-1):
        a = np.array(S[k])
        b = c_haut(np.array(B[k]),np.array(A[k][1]),thickness[k])
        S_new = cascade(a,b) 
        S.append(S_new.tolist())

    a = np.array(S[n_couches-1])
    b = np.array(A[n_couches-1][1])
    c = c_bas(a,b,thickness[n_couches-1])
    S.append(c.tolist())

    # matrices descendantes
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

    #ny = int(np.floor(thickness * period * n))
    ny = [20,75,5,50] * n
    #M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n,0:n],exc,int(np.floor(thickness[0] * periode / stretch)), thickness[0], a0)
    M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n,0:n],exc,ny[0], thickness[0], a0)
    
    #print("n_couches = ", n_couches)
    for j in np.arange(1,n_couches):
        M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n,0:n],exc,ny[j], thickness[j], a0) 
        M = np.append(M,M_new, 0)
    return M, S, Q

### SWAG functions
def reflectance(L, H, h, wavelength, period, polarization, perm_dielec, perm_metal, n, angle):
    cavity_size = L / period # taille du résonateur selon x, en nm
    H = H / period # taille du résonateur selon z, en nm. Cube au départ donc H = L
    h = h / period # taille du gap dans lequel résone le GP, dans la couche homogène de dielec. Equivalent à 'spacer'
    x0 = 0.5 - cavity_size / 2 # position du bloc dans la période, centrale
    wavelength_norm = wavelength / period 
    k0 = 2 * np.pi / wavelength_norm
    P, V = homogene(k0, angle, polarization, perm_dielec, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])
    Pc, Vc = creneau(k0, angle, polarization, perm_metal, perm_dielec, cavity_size, n, x0)
    S = cascade(S, interface(P, Pc))
    S = c_bas(S, Vc, H)
    S = cascade(S, interface(Pc, P))
    S = c_bas(S, V, h)
    Pc, Vc = homogene(k0, angle, polarization, perm_metal, n)
    S = cascade(S, interface(P, Pc))
    R = abs(S[n_mod, n_mod]) ** 2 * np.real(V[n_mod]) / k0
    return R 

def Field(H, L, h, period, wavelength, polarization, angle, perm_dielec, perm_metal, n_mod):
    L = L / period
    H = H / period 
    h = h / period 
    x0 = 0.5 - L / 2
    n = 2 * n_mod + 1

    wavelength_norm = wavelength / period 
    k0 = 2 * np.pi / wavelength_norm

    P_in, V_in = homogene(k0, angle, polarization, perm_dielec, n)
    A = []
    A.append([P_in.tolist(), V_in.tolist()])
    P_new, V_new = creneau(k0, angle, polarization, perm_metal, perm_dielec, L, n, x0)
    A.append([P_new.tolist(), V_new.tolist()])
    P_new, V_new = homogene(k0, angle, polarization, perm_dielec, n)
    A.append([P_new.tolist(), V_new.tolist()])
    P_out, V_out = homogene(k0, angle, polarization, perm_metal, n)
    A.append([P_out.tolist(), V_out.tolist()])
    #print(len(A))
    exc = np.zeros(2*n)
    exc[n_mod + 1] = 1

    #thicknesses = np.append(np.append(np.append(200/period,H), 10*h), 100/period) # 0 H h 0
    #thicknesses = np.array([100,100,100,100]) / period
    thicknesses = np.array([200 / period, H, h, 200 / period ])
    #print(thicknesses)
    M, S, Q = show_field(angle, A, thicknesses, exc, period)
    return M, A, thicknesses, S, Q

### Swag-structure

n_mod = 50 # inf à 50 sinon différences avec Octave
n_mod_total = 2 * n_mod + 1

# Espace au dessus
espace = 800

# Cubes
Largeur_cube = 92.6 # largeur du cube
Hauteur_dielec = 6.55 # hauteur de diéléctrique en dessous du cube
hauteur_cube = Largeur_cube #hauteur du cube
Hauteur_gold = 18 # hauteur de l'or au dessus du substrat
period = 1600.123 # periode

# Largeur de la fibre
Largeur_fibre = 400

# Espace en dessous de la fibre
espace2 = 500

# A modifier selon le point de fonctionnement
lam = 1150
theta = 0
pol = 1

## Paramètres des matériaux
e_d = 1.41 ** 2 # spacer
e_Glass = 1.5 ** 2 # substrat
e_Ag = epsAgbb(lam) # argent
e_Au = epsAubb(lam) # or

## Normalisation
espace = espace / period 
a = (period - Largeur_cube) / period
lc = Largeur_cube / period
hg = Hauteur_gold / period
hd = Hauteur_dielec / period
l = lam / period
lf = Largeur_fibre / period
espace2 = espace2 / period

## Calcul
k0 = 2 * np.pi / l
a0 = k0 * np.sin(theta)

# modes et valeurs propres pour chaque couche

A = [] # matrice de stockage de tous les modes et valeurs propres

# milieu incident, couche homogene d'air
P_in, V_in = homogene(k0, a0, pol, 1, n_mod_total)
A.append([P_in.tolist(), V_in.tolist()])

# couche 2 : cube d'argent dans couche d'air
P_2, V_2 = creneau(k0, a0, pol, e_Ag, 1, lc, n_mod_total, 0.5 - lc / 2)
#P_2, V_2 = homogene(k0, a0, pol, e_Glass, n_mod_total)
A.append([P_2.tolist(), V_2.tolist()])

# couche 3 : couche de dielectrique au dessus de la fibre
P_3, V_3 = creneau(k0, a0, pol, e_d, 1, lf, n_mod_total, 0.5 - lf / 2)
#P_3, V_3 = homogene(k0, a0, pol, e_d, n_mod_total)
A.append([P_3.tolist(), V_3.tolist()])

# couche 4 : couche d'au au dessus de la fibre
P_4, V_4 = creneau(k0, a0, pol, e_Au, 1, lf, n_mod_total, 0.5 - lf / 2)
A.append([P_4.tolist(), V_4.tolist()])

# couche 5 : couche de verre au dessus de la fibre
P_5, V_5 = creneau(k0, a0, pol, e_Glass, 1, lf, n_mod_total, 0.5 - lf / 2)
A.append([P_5.tolist(), V_5.tolist()])

thickness = np.array([espace, lc, hd, hg, espace2])
#thickness = np.array([espace, lc, hd])

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
exc[n_mod] = 1

ny = np.floor(thickness * period / stretch)


M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[0]), thickness[0], a0)
    
for j in np.arange(1,n_couches):
    M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[j]), thickness[j], a0) 
    M = np.append(M,M_new, 0)

Mfield = np.abs(M)**2
plt.figure(2)
plt.title("75 modes")
plt.imshow(Mfield, cmap = 'jet', aspect = 'auto')
plt.colorbar()
plt.show(block=False)
plt.savefig("fig/champ_comp_simu_octave75modes.pdf")


### on valide le code pour nmod < 50