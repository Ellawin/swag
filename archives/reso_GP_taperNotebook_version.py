import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

### Materials ### 
def epsAgbb(lam):
    "Permet de caluler la permittivité de l'argent en une longueur d'onde lam donnée"
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.821;
    Gamma0=0.049;
    omega_p=9.01;
    f=np.array([0.050,0.133,0.051,0.467,4.000])
    Gamma=np.array([0.189,0.067,0.019,0.117,0.052])
    omega=np.array([2.025,5.185,4.343,9.809,18.56])
    sigma=np.array([1.894,0.665,0.189,1.170,0.516])
    a=np.sqrt(w*(w+i*Gamma))
    a=a*np.sign(np.real(a));
    x=(a-omega)/(np.sqrt(2)*sigma);
    y=(a+omega)/(np.sqrt(2)*sigma);
    # Conversion
    aha=i*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(faddeeva(x,64)+faddeeva(y,64));
    epsilon=1-omega_p**2*f0/(w*(w+i*Gamma0))+np.sum(aha);
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

### RCWA 2D functions ###
def homogene(k0,a0,pol,epsilon,n):
    "Calcule les valeurs propres (valp) et les vecteurs propres (P) pour une couche homogène."
    "L'épaisseur de la couche ne semble pas être prise en compte."
    "k0 : vecteur d'onde incident"
    "a0 : prise en compte de l'angle d'incidence theta de l'onde qui arrive"
    "pol : polarisation de l'onde"
    "epsilon : permittivité de la couche"
    "n : nombre de modes (2*nmod +1 pour être plus précis)"
    nmod=np.floor(n/2)
    nmod_vect=np.arange(-nmod,nmod+1)
    valp=np.sqrt(epsilon*k0**2-(a0+2*np.pi*nmod_vect)**2+0j)
    valp=valp*(1-2*(np.imag(valp)<0))
    if pol==0:
        P=np.array([np.eye(n,n),np.diag(valp)])
    else:
        P=np.append(np.eye(n,n),np.diag(valp)/epsilon,axis=0)
    return P,valp

def cascade(T,U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''
    n = int(T.shape[1] / 2)
    J = np.linalg.inv( np.eye(n) - np.matmul(U[0:n,0:n],T[n:2*n,n:2*n] ) )
    K = np.linalg.inv( np.eye(n) - np.matmul(T[n:2*n,n:2*n],U[0:n,0:n] ) )
    S = np.block([[T[0:n,0:n] + np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),
    U[0:n,0:n]),T[n:2*n,0:n]),np.matmul(np.matmul(T[0:n,n:2*n],J),U[0:n,n:2*n])
    ],[np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,0:n]),U[n:2*n,n:2*n]
    + np.matmul(np.matmul(np.matmul(U[n:2*n,0:n],K),T[n:2*n,n:2*n]),U[0:n,n:2*n])
    ]])
    return S

def c_bas(A,V,h):
    ''' Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    '''
    n = int(A.shape[1]/2)
    D = np.diag(np.exp(1j*V*h))
    S = np.block([[A[0:n,0:n],np.matmul(A[0:n,n:2*n],D)],[np.matmul(D,A[n:2*n,0:n]),np.matmul(np.matmul(D,A[n:2*n,n:2*n]),D)]])
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

def step(a,b,w,x0,n):
    '''Computes the Fourier series for a piecewise function having the value
    b over a portion w of the period, starting at position x0
    and the value a otherwise. The period is supposed to be equal to 1.
    Then returns the toeplitz matrix generated using the Fourier series.
    '''
    from scipy.linalg import toeplitz
    from numpy import sinc
    l = np.zeros(n,dtype=np.complex128)
    m = np.zeros(n,dtype=np.complex128)
    tmp = np.exp(-2*1j*np.pi*(x0+w/2)*np.arange(0,n))*sinc(w*np.arange(0,n))*w
    l = np.conj(tmp)*(b-a)
    m = tmp*(b-a)
    l[0] = l[0]+a
    m[0] = l[0]
    T = toeplitz(l,m)
    return T

def fpml(q,g,n):
    from scipy.linalg import toeplitz
    from numpy import sinc,flipud
    x = np.arange(-n,n+1)
    v = -q/2*((1+g/4)*sinc(q*x)+(sinc(q*x-1)+sinc(q*x+1))*0.5-g*0.125*(sinc(q*x-2)+sinc(q*x+2)))
    v[n] = v[n]+1
    T = toeplitz(flipud(v[1:n+1]),v[n:2*n])
    return T

def aper(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position.
    Warning : There is nothing checking that the blocks don't overlapp.
    '''
    n_blocs = blocs.shape[0];
    nmod = int(n/2)
    M1 = e1*np.eye(n,n)
    M2 = 1/e1*np.eye(n,n)
    for k in range(0,n_blocs):
        M1 = M1+step(0,e2-e1,blocs[k,0],blocs[k,1],n)
        M2 = M2+step(0,1/e2-1/e1,blocs[k,0],blocs[k,1],n)
    alpha = np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))+0j
    g = 1/(1-1j);
    fprime = np.eye(n)
    if (pol==0):
        tmp = np.linalg.inv(fprime)
        M = np.matmul(tmp, np.matmul(alpha, np.matmul(tmp, alpha)))\
        -k0*k0*M1
        L,E = np.linalg.eig(M)
        L = np.sqrt(-L+0j)
        L = (1-2*(np.imag(L)<-1e-15))*L
        P = np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        M = np.matmul(np.linalg.inv(np.matmul(fprime, M2)),\
        -k0*k0*fprime+np.matmul(alpha, np.matmul(np.linalg.inv(np.matmul(M1, fprime)), alpha)))
        L,E = np.linalg.eig(M)
        L = np.sqrt(-L+0j)
        L = (1-2*(np.imag(L)<-1e-15))*L
        P = np.block([[E],[np.matmul(np.matmul(M2,E),np.diag(L))]])
    return P,L

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n = int(P.shape[1])
    S = np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
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

### Visualization of the structure
def show_structure(thick, width, period):
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

### Visualization of the field of the structure
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
    return M

def show_field(k0,a0,A,thickness, exc,periode):
    n = len(A[0][1])
    n_couches = thickness.shape[0]
    S11=np.zeros((n,n))
    S12=np.eye(n)
    S1=np.append(S11,S12,axis=0)
    S2=np.append(S12,S11,axis=0)
    S0=np.append(S1,S2,1)

    B = []
    for k in range(n_couches-1):
        a = np.array(A[k][0])
        b = np.array(A[k+1][0])
        c = interface(a,b)
        c = c.tolist()
        B.append(c)

    S = []
    S0 = S0.tolist()
    S.append(S0)

    for k in range(n_couches-1):
        a = np.array(S[k])
        b = c_haut(np.array(B[k]),np.array(A[k][1]),thickness[k])
        S_new = cascade(a,b) 
        S.append(S_new.tolist())

    a = np.array(S[n_couches-1])
    b = np.array(A[n_couches-1][1])
    c = c_bas(a,b,thickness[n_couches-1])
    S.append(c.tolist())

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

    stretch = periode / n

    M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n,0:n],exc,int(np.floor(thickness[0] * periode / stretch)), thickness[0], a0)

    for j in np.arange(1,n_couches):
        M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n,0:n],exc,int(np.floor(thickness[j] * periode / stretch)), thickness[j], a0) 
        M = np.append(M,M_new, 0)
    return M

def Field(thick, width, period, lam, pol, a0, e1, e2, nmod):
    n = 2*nmod+1
    number_layers = 1

    t = 75 / period 
    w = 75 / period 
    p = 0.5 - w / 2 

    X = np.array([thick, width, p])
    X = X.T

    h = np.array(thick) / period

    thickness = np.append(np.append(0 / period, t), 0 / period)

    l = lam / period 
    k0 = 2 * np.pi / l 

    P_in,V_in = homogene(k0, a0, pol, e1, n)
#    a = nmod

#    a = np.argmin(abs(V_in-2.35*k0)) 
    A = []
    A.append([P_in.tolist(),V_in.tolist()])
    for k in range(number_layers):
        reso = np.array([[w[k],p[k]]])
        P_new,V_new = aper(k0,a0,pol,e1,e2,n,reso)
        A.append([P_new.tolist(),V_new.tolist()])
#    P_out, V_out = aper(k0,a0,pol,e1,e2,n,np.array([[w_out,0.5-w_out/2]]))
    P_out, V_out = homogene(k0, a0, pol, e2, n)
    
#    a = np.argmin(abs(V_out-2.35*k0)) 
    A.append([P_out.tolist(),V_out.tolist()])
    exc = np.zeros(2*n)
    exc[a-1] = 1

    M = show_field(k0, a0, A, thickness, exc, period)
    return M 
    
### Structure definition
period =  501.2153 
n_layers = 1
n_reso = 1

wavelength = 700.021635 # in nanometers
n_mod = 50

pola = 1
angle = 0
eps_dielec = 1.
eps_metal = epsAgbb(wavelength)

#thicknesses = np.array([75, 5, 200]) # un résonateur de 75 nm de côté, un gap de 5 nm, et un substrat de metal
#widths = np.array([75, 0, period]) # un résonateur de 75 nm de côté, un gap de diélectrique homogène, un substrat de metal homogène

thick = [75]
width = [75]

#M = show_structure(thick, width, period)

#plt.figure(2)
#plt.imshow(M*127, cmap = 'bone')
#plt.show(block = False)    
#plt.savefig("Visu_structure_GP_3Lc.pdf")

Mfield = Field(thick, width, period, wavelength, pola, angle, eps_dielec, eps_metal, n_mod)

# Mfield = Field(best)
Mfield = np.abs(Mfield)**2
plt.figure(3)
plt.imshow(Mfield, cmap = 'jet', aspect = 'auto')
plt.colorbar()
plt.show(block=False)
plt.savefig("Field_GP_Lc_50modes.pdf")
plt.close()


# # Starting with neutral S matrix and dielectric homogene layer
# S = np.block([[np.zeros([n_mod, n_mod]), np.eye(n_mod, dtype = np.complex128)], [np.eye(n_mod), np.zeros([n_mod, n_mod])]])
# P,V = homogene(wave_vector_incident, angle, pola, eps_dielec, n_mod)

# # Compute the matrix elements for each layers and modes
# for n_l in range(0,n_layers):
#     reso = np.array([[widths[n_l], positions[n_l]]])
#     P_new, V_new = aper(wave_vector_incident, angle, pola, eps_dielec, eps_metal, n_mod, reso) # e2  indide e1 -> metal dans dielec
#     S = cascade(S, interface(P, P_new))
#     S = c_bas(S, V_new, thicknesses[n_l])
#     P, V = P_new, V_new

# # Ending with metallic homogene layer
# P_out, V_out = homogene(wave_vector_incident, angle, pola, eps_metal, n_mod)
# S = cascade(S, interface(P, P_out))

    
