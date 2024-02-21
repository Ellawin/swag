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

def reseau(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position. #not anymore
    Warning : There is nothing checking that the blocks don't overlapp.
    Remark : 'reseau' is a version of 'creneau' taking account several blocs in a period
    '''
    n_blocs=blocs.shape[0]
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

def fpml(q,g,n):
    from scipy.linalg import toeplitz
    from numpy import sinc,flipud
    x=np.arange(-n,n+1)
    v=-q/2*((1+g/4)*sinc(q*x)+(sinc(q*x-1)+sinc(q*x+1))*0.5-g*0.125*(sinc(q*x-2)+sinc(q*x+2)))
    v[n]=v[n]+1
    T=toeplitz(flipud(v[1:n+1]),v[n:2*n])
    return T

def aper(k0,a0,pol,e1,e2,n,blocs):
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
    g=1/(1-1j)
    #fprime=fpml(0.2001,g,n)
    fprime = fpml(0.1,g,n)
    if (pol==0):
        tmp=np.linalg.inv(fprime)
        M=np.matmul(tmp, np.matmul(alpha, np.matmul(tmp, alpha)))\
        -k0*k0*M1
        L,E=np.linalg.eig(M)
        L=np.sqrt(-L+0j)
        L=(1-2*(np.imag(L)<-1e-15))*L
        P=np.block([[E],[np.matmul(E,np.diag(L))]])
    else:
        M=np.matmul(np.linalg.inv(np.matmul(fprime, M2)),\
        -k0*k0*fprime+np.matmul(alpha, np.matmul(np.linalg.inv(np.matmul(M1, fprime)), alpha)))
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
def reflectance_creneau(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod):    
    n = 2 * n_mod + 1
    
    ## trouver le mode du GP avec PyMoosh
    material_list = [1., 'Silver']
    layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    start_index_eff = 4
    tol = 1e-12
    step_max = 50000
    thicknesses_down = [thick_reso,thick_gap,thick_gold]
    Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
    GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)
    #print("GP effective index = ", GP_effective_index)

    # Adimensionalisation
    wavelength_norm = wavelength / period
    
    thick_up = thick_up / period 
    thick_down = thick_down / period 
    thick_gap = thick_gap / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period

    wavelength_norm = wavelength / period

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle * np.pi / 180)

    Pup, Vup = creneau(k0, a0, polarization, perm_Ag, perm_dielec, thick_reso, n, 0)
    Pdown, Vdown = creneau(k0, a0, polarization, perm_dielec, perm_Ag, thick_gap, n, thick_reso)
    S = interface(Pup, Pdown)
 
    ## PM, quand on travaille à longueur d'onde fixée et qu'on a calculé l'indice effectif une fois pour toute
    #GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    position_GP = np.argmin(abs(Vdown - GP_effective_index * k0))
   
    # reflexion quand on eclaire par le dessus
    #Rup = abs(S[position_up, position_up]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    #Tup = abs(S[n + position_up, position_up]) ** 2 
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    Rdown_GP = abs(S[n + position_GP, n + position_GP]) ** 2 
    # transmission quand on éclaire par le dessous
    #Tdown = abs(S[position_down, n + position_down]) ** 2 
    
    # Les coefficients de transmission ont pas vraiment de sens ici, à cause de la normalisation différente
    # Comme on éclaire pas avec des ondes planes, qu'on vient pas de deux milieux homogènes, on a pas besoin des 
    # trucs bizarres après les coeffs de S

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[position_up, position_up])
    #phase_R_down = np.angle(S[n + position_down, n + position_down])
    phase_R_down_GP = np.angle(S[n + position_GP, n + position_GP])


    return Rdown_GP, phase_R_down_GP,  Vdown[position_GP] / k0, GP_effective_index #np.abs(Vdown[position_GP] / k0), np.abs(GP_effective_index)

def reflectance_reseau(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod):    
    n = 2 * n_mod + 1

     ### Éclairage par en dessous, guide d'onde

    ## Version 2 : PM, quand on ne sait pas la longueur d'onde
    material_list = [1., 'Silver']
    layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    start_index_eff = 4
    tol = 1e-12
    step_max = 10000

    thicknesses_down = [thick_reso,thick_gap,thick_gold]
    Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
    GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)

    # Normalisation
    wavelength_norm = wavelength / period
    
    thick_up = thick_up / period 
    thick_down = thick_down / period 
    thick_gap = thick_gap / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period

    wavelength_norm = wavelength / period

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle * np.pi / 180)

    blocs_1 = np.array([[thick_reso, 0]]) 
    blocs_2 = np.array([[thick_reso, 0], [thick_gold, thick_reso + thick_gap]])

    Pup, Vup = reseau(k0, a0, polarization, perm_dielec, perm_Ag, n, blocs_1)
    Pdown, Vdown = reseau(k0, a0, polarization, perm_dielec, perm_Ag, n, blocs_2)
    S = interface(Pup, Pdown)

    ## PM, quand on travaille à longueur d'onde fixée et qu'on a calculé l'indice effectif une fois pour toute
    #GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    
    position_GP = np.argmin(abs(Vdown - GP_effective_index * k0))
    #print("position GP = ", position_GP) 
   
    # reflexion quand on eclaire par le dessus
    #Rup = abs(S[position_up, position_up]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    #Tup = abs(S[n + position_up, position_up]) ** 2 
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    Rdown_GP = abs(S[n + position_GP, n + position_GP]) ** 2 
    # transmission quand on éclaire par le dessous
    #Tdown = abs(S[position_down, n + position_down]) ** 2 
    
    # Les coefficients de transmission ont pas vraiment de sens ici, à cause de la normalisation différente
    # Comme on éclaire pas avec des ondes planes, qu'on vient pas de deux milieux homogènes, on a pas besoin des 
    # trucs bizarres après les coeffs de S

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[position_up, position_up])
    #phase_R_down = np.angle(S[n + position_down, n + position_down])
    phase_R_down_GP = np.angle(S[n + position_GP, n + position_GP])


    return Rdown_GP, phase_R_down_GP

def reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod):    
    n = 2 * n_mod + 1

    ### Éclairage par au dessus, guide d'onde

    ## Version 2 : PM, quand on ne sait pas la longueur d'onde
    #material_list = [1., 'Silver']
    #layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    #start_index_eff = 4
    #tol = 1e-12
    #step_max = 10000

    #thicknesses_down = [thick_reso,thick_gap,thick_gold]
    #Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
    #GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)

    wavelength_norm = wavelength / period
    
    thick_up = thick_up / period 
    thick_down = thick_down / period 
    thick_gap = thick_gap / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period

    wavelength_norm = wavelength / period

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle * np.pi / 180)

    ### blocs de dielec dans de l'Ag
    blocs_1 = np.array([[(1 + thick_gap) / 2, (1 - thick_gap) / 2]]) 
    blocs_2 = np.array([[thick_gap, (1 - thick_gap) / 2]])

    ### blocs d'Ag dans du dielec
    #blocs_1 = np.array([[thick_reso, 0]]) 
    #blocs_2 = np.array([[thick_reso, 0], [thick_gold, thick_reso + thick_gap]])

    Pup, Vup = grating(k0, a0, polarization, perm_Ag, perm_dielec, n, blocs_1) # e2 dans e1
    Pdown, Vdown = grating(k0, a0, polarization, perm_Ag, perm_dielec, n, blocs_2)
    S = interface(Pup, Pdown)

    ## PM, quand on travaille à longueur d'onde fixée et qu'on a calculé l'indice effectif une fois pour toute
    GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    
    position_GP = np.argmin(abs(Vdown - GP_effective_index * k0))
    #print("position GP = ", position_GP) 
   
    # reflexion quand on eclaire par le dessus
    #Rup = abs(S[position_up, position_up]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    #Tup = abs(S[n + position_up, position_up]) ** 2 
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    Rdown_GP = abs(S[n + position_GP, n + position_GP]) ** 2 
    # transmission quand on éclaire par le dessous
    #Tdown = abs(S[position_down, n + position_down]) ** 2 
    
    # Les coefficients de transmission ont pas vraiment de sens ici, à cause de la normalisation différente
    # Comme on éclaire pas avec des ondes planes, qu'on vient pas de deux milieux homogènes, on a pas besoin des 
    # trucs bizarres après les coeffs de S

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[position_up, position_up])
    #phase_R_down = np.angle(S[n + position_down, n + position_down])
    phase_R_down_GP = np.angle(S[n + position_GP, n + position_GP])


    return Rdown_GP, phase_R_down_GP

def reflectance_aper(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod):    
    n = 2 * n_mod + 1

    ### Éclairage par au dessus, guide d'onde

    ## Version 2 : PM, quand on ne sait pas la longueur d'onde
    #material_list = [1., 'Silver']
    #layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    #start_index_eff = 4
    #tol = 1e-12
    #step_max = 10000

    #thicknesses_down = [thick_reso,thick_gap,thick_gold]
    #Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
    #GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)

    wavelength_norm = wavelength / period
    
    thick_up = thick_up / period 
    thick_down = thick_down / period 
    thick_gap = thick_gap / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period

    wavelength_norm = wavelength / period

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle * np.pi / 180)

    ### blocs de dielec dans de l'Ag
    blocs_1 = np.array([[(1 + thick_gap) / 2, (1 - thick_gap) / 2]]) 
    blocs_2 = np.array([[thick_gap, (1 - thick_gap) / 2]])

    ### blocs d'Ag dans du dielec
    #blocs_1 = np.array([[thick_reso, 0]]) 
    #blocs_2 = np.array([[thick_reso, 0], [thick_gold, thick_reso + thick_gap]])

    Pup, Vup = aper(k0, a0, polarization, perm_Ag, perm_dielec, n, blocs_1) # e2 dans e1
    Pdown, Vdown = aper(k0, a0, polarization, perm_Ag, perm_dielec, n, blocs_2)
    S = interface(Pup, Pdown)

    ## PM, quand on travaille à longueur d'onde fixée et qu'on a calculé l'indice effectif une fois pour toute
    GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    
    position_GP = np.argmin(abs(Vdown - GP_effective_index * k0))
    #print("position GP = ", position_GP) 
   
    # reflexion quand on eclaire par le dessus
    #Rup = abs(S[position_up, position_up]) ** 2 # correspond à ce qui se passe au niveau du SP layer up
    # transmission quand on éclaire par le dessus
    #Tup = abs(S[n + position_up, position_up]) ** 2 
    # reflexion quand on éclaire par le dessous
    #Rdown = abs(S[n + position_down, n + position_down]) ** 2 
    Rdown_GP = abs(S[n + position_GP, n + position_GP]) ** 2 
    # transmission quand on éclaire par le dessous
    #Tdown = abs(S[position_down, n + position_down]) ** 2 
    
    # Les coefficients de transmission ont pas vraiment de sens ici, à cause de la normalisation différente
    # Comme on éclaire pas avec des ondes planes, qu'on vient pas de deux milieux homogènes, on a pas besoin des 
    # trucs bizarres après les coeffs de S

    # calcul des phases du coefficient de réflexion
    #phase_R_up = np.angle(S[position_up, position_up])
    #phase_R_down = np.angle(S[n + position_down, n + position_down])
    phase_R_down_GP = np.angle(S[n + position_GP, n + position_GP])

    return Rdown_GP, phase_R_down_GP

def Field_aper(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod):
    
    #material_list = [1., 'Silver']
    #layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    #start_index_eff = 4
    #tol = 1e-12
    #step_max = 100000

    #thicknesses_down = [thick_reso,thick_gap,thick_gold]
    #Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
    #GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)
    
    wavelength_norm = wavelength / period
    
    thick_up = thick_up / period 
    thick_down = thick_down / period 
    thick_gap = thick_gap / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period

    ### blocs de dielec dans de l'Ag
    blocs_1 = np.array([[(1 + thick_gap) / 2, (1 - thick_gap) / 2]]) 
    blocs_2 = np.array([[thick_gap, (1 - thick_gap) / 2]])

    ### blocs d'Ag dans du dielec
    #blocs_1 = np.array([[thick_reso, 0]]) 
    #blocs_2 = np.array([[thick_reso, 0], [thick_gold, thick_reso + thick_gap]])    

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle)

    n_mod_total = 2 * n_mod + 1

    A = [] # matrice de stockage de tous les modes et valeurs propres

    # milieu incident, metal d'argent puis air # dielec (e2) dans Ag (e1)
    Pup, Vup = aper(k0, a0, polarization, perm_Ag, perm_dielec, n_mod_total, blocs_1)
    A.append([Pup.tolist(), Vup.tolist()])     

    # couche 2 : cube d'argent dans couche d'air # dielec (e2) dans Ag (e1)
    Pdown, Vdown = aper(k0, a0, polarization, perm_Ag, perm_dielec, n_mod_total, blocs_2)
    A.append([Pdown.tolist(), Vdown.tolist()])

    thickness = np.array([thick_up, thick_down])

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
    #position = np.argmax(np.real(Vdown))

    GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    position_GP = np.argmin(abs(Vdown - GP_effective_index * k0))
    
    exc[n_mod_total + position_GP] = 1

    ny = np.floor(thickness * period / stretch)

    M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[0]), thickness[0], a0)
    
    for j in np.arange(1,n_couches):
        M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[j]), thickness[j], a0) 
        M = np.append(M,M_new, 0)

    Mfield = np.abs(M)**2
    return Mfield

def Field_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod):
    #material_list = [1., 'Silver']
    #layer_down = [1,0,1]
    
    # Find the mode (it's unique) which is able to propagate in the GP gap
    #start_index_eff = 4
    #tol = 1e-12
    #step_max = 100000

    #thicknesses_down = [thick_reso,thick_gap,thick_gold]
    #Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
    #GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)
      
    wavelength_norm = wavelength / period
    
    thick_up = thick_up / period 
    thick_down = thick_down / period 
    thick_gap = thick_gap / period
    thick_reso = thick_reso / period
    thick_gold = thick_gold / period

    ### blocs de dielec dans de l'Ag
    blocs_1 = np.array([[(1 + thick_gap) / 2, (1 - thick_gap) / 2]]) 
    blocs_2 = np.array([[thick_gap, (1 - thick_gap) / 2]])

    ### blocs d'Ag dans du dielec
    #blocs_1 = np.array([[thick_reso, 0]]) 
    #blocs_2 = np.array([[thick_reso, 0], [thick_gold, thick_reso + thick_gap]])

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle)

    n_mod_total = 2 * n_mod + 1

    A = [] # matrice de stockage de tous les modes et valeurs propres

    # milieu incident, metal d'argent puis air # dielec (e2) dans Ag (e1)
    Pup, Vup = grating(k0, a0, polarization, perm_Ag, perm_dielec, n_mod_total, blocs_1)
    A.append([Pup.tolist(), Vup.tolist()])     

    # couche 2 : cube d'argent dans couche d'air # dielec (e2) dans Ag (e1)
    Pdown, Vdown = grating(k0, a0, polarization, perm_Ag, perm_dielec, n_mod_total, blocs_2)
    A.append([Pdown.tolist(), Vdown.tolist()])

    thickness = np.array([thick_up, thick_down])

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
    #position = np.argmax(np.real(Vdown))

    GP_effective_index = 3.87 + 0.13j # pour un lam de 700
    position_GP = np.argmin(abs(Vdown - GP_effective_index * k0))
    
    exc[n_mod_total + position_GP] = 1

    ny = np.floor(thickness * period / stretch)

    M = HErmes(np.array(S[0]), np.array(Q[n_couches-0-1]), np.array(A[0][1]), np.array(A[0][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[0]), thickness[0], a0)
    
    for j in np.arange(1,n_couches):
        M_new = HErmes(np.array(S[j]), np.array(Q[n_couches-j-1]), np.array(A[j][1]), np.array(A[j][0])[0:n_mod_total,0:n_mod_total],exc,int(ny[j]), thickness[j], a0) 
        M = np.append(M,M_new, 0)

    Mfield = np.abs(M)**2
    return Mfield


### Swag-structure

thick_up = 50
#thick_gap = 10 # hauteur de diéléctrique en dessous du cube
thick_reso = 75 #hauteur du cube
thick_gold = 20 # hauteur de l'or au dessus du substrat
#period = thick_reso + thick_gap + thick_gold # periode
period = 600
thick_down = 50

# A modifier selon le point de fonctionnement
wavelength = 700.021635
angle = 0
polarization = 1

## Paramètres des matériaux
#perm_dielec = 1.41 ** 2 # spacer
perm_dielec = 1
#perm_Glass = 1.5 ** 2 # substrat
perm_Ag = epsAgbb(wavelength) # argent
#perm_Au = epsAubb(wavelength) # or

n_mod = 200 
#n_mod_total = 2 * n_mod + 1

### Indice effectif du GP en fonction de l'épaisseur du gap, comparaison analytique / numérique

list_thick_gap = np.linspace(0.1, 5, 30)
neff_analytic = np.empty(list_thick_gap.size, dtype = complex)
neff_RCWA = np.empty(list_thick_gap.size, dtype = complex)
neff_Pymoosh = np.empty(list_thick_gap.size, dtype = complex)

# première approximation
neff_analytic = wavelength / (list_thick_gap * np.pi) * np.arctanh(- 1 / perm_Ag)

neff_quasistatic = - wavelength / (np.pi * perm_Ag * list_thick_gap) 

idx = 0
for thick_gap in list_thick_gap:
    R_down_GP, phase_R_down_GP, neff_RCWA[idx], neff_Pymoosh[idx] = reflectance_creneau(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
    idx += 1

# 2ème approximation

#thick_gap = 5

# k0 = np.linspace(1,10,10)
# kx = k0 ** 2 * neff_Pymoosh ** 2
# km_ref = np.sqrt(kx ** 2 - perm_Ag * k0 ** 2) 

# km_DL = kx - (perm_Ag * k0 ** 2) / (2 * kx)

# plt.figure(1)
# plt.plot(kx, km_ref, label = 'reference')
# plt.plot(kx, km_DL, label = 'DL')
# plt.legend()
# plt.xlabel("kx")
# plt.ylabel("Km")
# plt.title("Développement limité à l'ordre 1 de Km quand kx -> infini")
# plt.show(block=False)
# plt.savefig("DL_km.jpg")

plt.figure(4)
plt.subplot(211)
#plt.plot(list_thick_gap, np.real(neff_RCWA), "b", label = "RCWA")
plt.plot(list_thick_gap, np.real(neff_Pymoosh), "xr", label = "PyMoosh")
plt.plot(list_thick_gap, np.real(neff_analytic), "og", label = "Analytique")
plt.plot(list_thick_gap, np.real(neff_quasistatic), "*k", label = "Quasi static")
plt.ylabel("Real part")
plt.legend()
plt.title("Effective index of the GP")
plt.subplot(212)
#plt.plot(list_thick_gap, np.imag(neff_RCWA), "xb", label = "RCWA")
plt.plot(list_thick_gap, np.imag(neff_Pymoosh), "xr", label = "PyMoosh")
plt.plot(list_thick_gap, np.imag(neff_analytic), "og", label = "Analytique")
plt.plot(list_thick_gap, np.real(neff_quasistatic), "*k", label = "Quasi static")
plt.xlabel("Thickness of the gap (nm")
plt.ylabel("Imaginary part")
plt.legend()
plt.show(block=False)
plt.savefig("effective_index_comparaison_real_and_imag_static.jpg")

### Influence du nombre de modes et de l'épaisseur du gap sur le coefficient de réflexion du GP
# list_number_modes = np.linspace(10, 200, 100)

# R_down_GP_gap4 = np.empty(list_number_modes.size)
# phase_R_down_GP_gap4 = np.empty(list_number_modes.size)
# R_down_GP_gap6 = np.empty(list_number_modes.size)
# phase_R_down_GP_gap6 = np.empty(list_number_modes.size)
# R_down_GP_gap8 = np.empty(list_number_modes.size)
# phase_R_down_GP_gap8 = np.empty(list_number_modes.size)
# R_down_GP_gap10 = np.empty(list_number_modes.size)
# phase_R_down_GP_gap10 = np.empty(list_number_modes.size)
# idx = 0

# for n_mod in list_number_modes:
#     n_mod = int(n_mod)
#     R_down_GP_gap4[idx], phase_R_down_GP_gap4[idx] = reflectance_grating(thick_up, thick_down, 4, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     R_down_GP_gap6[idx], phase_R_down_GP_gap6[idx] = reflectance_grating(thick_up, thick_down, 6, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod)
#     R_down_GP_gap8[idx], phase_R_down_GP_gap8[idx] = reflectance_grating(thick_up, thick_down, 8, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     R_down_GP_gap10[idx], phase_R_down_GP_gap10[idx] = reflectance_grating(thick_up, thick_down, 10, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod)    
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_number_modes, R_down_GP_gap4, "g", label="Gap 4 nm")
# plt.plot(list_number_modes, R_down_GP_gap6, "r", label="Gap 6 nm")
# plt.plot(list_number_modes, R_down_GP_gap8, "b", label="Gap 8 nm")
# plt.plot(list_number_modes, R_down_GP_gap10, "k", label="Gap 10 nm")
# plt.legend()
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# #plt.show(block=False)
# #plt.savefig("Phase_and_modulus_reflexion_GP_depending_wavelength.jpg")

# plt.subplot(212)
# plt.plot(list_number_modes, phase_R_down_GP_gap4, "g", label="Gap 4 nm")
# plt.plot(list_number_modes, phase_R_down_GP_gap6, "r", label="Gap 6 nm")
# plt.plot(list_number_modes, phase_R_down_GP_gap8, "b", label="Gap 8 nm")
# plt.plot(list_number_modes, phase_R_down_GP_gap10, "k", label="Gap 10 nm")
# plt.legend()
# plt.xlabel("Number of modes")
# plt.ylabel("Phase")
# #plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("Phase_and_modulus_reflexion_GP_depending_NumberModes_thicknesses_gap_4_6_8_10.jpg")

### Influence du nombre de modes sur le coefficient de réflexion du GP

# list_number_modes = np.linspace(10, 200, 100)

# R_down_GP_Ag = np.empty(list_number_modes.size)
# phase_R_down_GP_Ag = np.empty(list_number_modes.size)
# R_down_GP_Au = np.empty(list_number_modes.size)
# phase_R_down_GP_Au = np.empty(list_number_modes.size)
# idx = 0

# for n_mod in list_number_modes:
#     n_mod = int(n_mod)
#     R_down_GP_Ag[idx], phase_R_down_GP_Ag[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     R_down_GP_Au[idx], phase_R_down_GP_Au[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod)
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_number_modes, R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_number_modes, R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# #plt.show(block=False)
# #plt.savefig("Phase_and_modulus_reflexion_GP_depending_wavelength.jpg")

# plt.subplot(212)
# plt.plot(list_number_modes, phase_R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_number_modes, phase_R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.xlabel("Number of modes")
# plt.ylabel("Phase")
# #plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("Phase_and_modulus_reflexion_GP_depending_NumberModes_Au_Ag.jpg")

### Influence de l'angle sur le coefficient de réflexion du GP

# list_angles = np.linspace(0, 90, 90)

# R_down_GP_Ag = np.empty(list_angles.size)
# phase_R_down_GP_Ag = np.empty(list_angles.size)
# R_down_GP_Au = np.empty(list_angles.size)
# phase_R_down_GP_Au = np.empty(list_angles.size)
# idx = 0

# for angle in list_angles:
#     R_down_GP_Ag[idx], phase_R_down_GP_Ag[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     R_down_GP_Au[idx], phase_R_down_GP_Au[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod)
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_angles, R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_angles, R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# #plt.show(block=False)
# #plt.savefig("Phase_and_modulus_reflexion_GP_depending_wavelength.jpg")

# plt.subplot(212)
# plt.plot(list_angles, phase_R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_angles, phase_R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.xlabel("Incidence angle (degree)")
# plt.ylabel("Phase")
# #plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("Phase_and_modulus_reflexion_GP_depending_Angle_Au_Ag.jpg")


### influence de la taille du gap sur le coefficient de réflexion du GP
# list_thick_gap = np.linspace(5, 30, 50)

# R_down_GP_Ag = np.empty(list_thick_gap.size)
# phase_R_down_GP_Ag = np.empty(list_thick_gap.size)
# R_down_GP_Au = np.empty(list_thick_gap.size)
# phase_R_down_GP_Au = np.empty(list_thick_gap.size)
# idx = 0

# for thick_gap in list_thick_gap:
#     R_down_GP_Ag[idx], phase_R_down_GP_Ag[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     R_down_GP_Au[idx], phase_R_down_GP_Au[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod)
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_thick_gap, R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_thick_gap, R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# #plt.show(block=False)
# #plt.savefig("Phase_and_modulus_reflexion_GP_depending_wavelength.jpg")

# plt.subplot(212)
# plt.plot(list_thick_gap, phase_R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_thick_gap, phase_R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.xlabel("Thickness gap (nm) ")
# plt.ylabel("Phase")
# #plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("Phase_and_modulus_reflexion_GP_depending_GapThickness_Au_Ag.jpg")


### Influence de la longueur d'onde sur l'indice effectif du GP
# list_wavelength = np.linspace(350, 800, 100)

# R_down_GP = np.empty(list_wavelength.size)
# phase_R_down_GP = np.empty(list_wavelength.size)
# Veff = np.empty(list_wavelength.size)
# neff_GP = np.empty(list_wavelength.size)

# idx = 0

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength)
#     R_down_GP[idx], phase_R_down_GP[idx], Veff[idx], neff_GP[idx] = reflectance_creneau(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     idx += 1

# plt.figure(7)
# plt.plot(list_wavelength, Veff, "b", label = "indice effectif le plus proche")
# plt.plot(list_wavelength, neff_GP, "r", label = "indice effectif PyMoosh")
# plt.xlabel("Wavelength (nm")
# plt.ylabel("Effective index module")
# plt.title("Effective index of the GP")
# plt.legend()
# plt.show(block=False)
# plt.savefig("effective_index.jpg")


### Etude de l'influence de la taille du domaine (période) sur la réflexion du GP
# list_period = np.linspace(100, 1500, 200)
# R_GP_grating_Ag = np.empty(list_period.size)
# phase_R_GP_grating_Ag = np.empty(list_period.size)
# R_GP_grating_Au = np.empty(list_period.size)
# phase_R_GP_grating_Au = np.empty(list_period.size)
# #R_GP_aper = np.empty(list_period.size)
# #phase_R_GP_aper = np.empty(list_period.size)

# idx = 0

# for period in list_period:
#     R_GP_grating_Au[idx], phase_R_GP_grating_Au[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod)
#     R_GP_grating_Ag[idx], phase_R_GP_grating_Ag[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod)
#     idx += 1

# plt.figure(6)
# plt.subplot(211)
# plt.plot(list_period, R_GP_grating_Ag, "b", label="Argent")
# plt.plot(list_period, R_GP_grating_Au, "r", label="Or")
# plt.legend()
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# #plt.show(block=False)
# #plt.savefig("Phase_and_modulus_reflexion_GP_depending_wavelength.jpg")

# plt.subplot(212)
# plt.plot(list_period, phase_R_GP_grating_Ag, "b", label="Argent")
# plt.plot(list_period, phase_R_GP_grating_Au, "r", label="Or")
# plt.legend()
# plt.xlabel("Period (nm) ")
# plt.ylabel("Phase")
# #plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("Phase_and_modulus_reflexion_GP_depending_period_Au_Ag.jpg")


### Etude du mode fondamental / mode GP
## Find the mode (it's unique) which is able to propagate in the GP gap

# wavelength = 700
# material_list = [1., 'Silver']
# layer_down = [1,0,1]

# start_index_eff = 4
# tol = 1e-12
# step_max = 100000

# thicknesses_down = [200,5,200]
# Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
# GP_effective_index = pm.steepest(start_index_eff, tol, step_max, Layer_down, wavelength, polarization)

# print("Indice effectif du GP =",GP_effective_index)

# x, prof = pm.profile(Layer_down, GP_effective_index, wavelength, polarization, pixel_size = 0.1)
# plt.figure(1)
# plt.plot(x, np.real(prof), label = "Partie réelle")
# plt.plot(x, np.abs(prof)**2, label = "Module") 
# plt.legend()
# plt.title("Mode of the GP, wavelength 400 nm")
# plt.xlabel("Position (nm)")
# plt.ylabel("Mode profile (a.u), intensity")
# plt.show(block=False)
# plt.savefig("GP_profile_lam400_comp_real_abs.jpg")

# pour une longueur d'onde donnée, on injecte la valeur du mode effectif trouvé et on trouve la position du mode par
# position = np.argmin(abs(Valeur_propre - Indice effectif complexe * k0)) 
# on utilise ça à la place de trouver la position en regardant la valeur propre d'amplitude maximale (c'est plus fiable)

### Pour étudier le champ en une longueur d'onde donnée

# period = 600
# Mfield2 = Field_aper(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, 1, perm_Ag, n_mod)
# plt.figure(5)
# plt.imshow(Mfield2, cmap = 'jet', aspect = 'auto')
# plt.colorbar()
# plt.title("PML 0.2")
# plt.show(block=False)
# plt.savefig("Champ_pml0.2.jpg") 

# Mfield_grating = Field_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, 1, perm_Ag, n_mod)
# plt.figure(4)
# plt.imshow(Mfield_grating, cmap = 'jet', aspect = 'auto')
# plt.colorbar()
# plt.title("Grating")
# plt.show(block=False)
# plt.savefig("Champ_grating.jpg") 

### Pour étudier l'influence de la longueur d'onde sur le coefficient de réflexion du GP
# list_wavelength = np.linspace(400, 800, 100)

# R_down_GP_Ag = np.empty(list_wavelength.size)
# phase_R_down_GP_Ag = np.empty(list_wavelength.size)

# R_down_GP_Au = np.empty(list_wavelength.size)
# phase_R_down_GP_Au = np.empty(list_wavelength.size)

# idx = 0

# for wavelength in list_wavelength:
#     perm_Ag = epsAgbb(wavelength)
#     perm_Au = epsAubb(wavelength) # or
#     R_down_GP_Ag[idx], phase_R_down_GP_Ag[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Ag, n_mod) 
#     R_down_GP_Au[idx], phase_R_down_GP_Au[idx] = reflectance_grating(thick_up, thick_down, thick_gap, thick_reso, thick_gold, period, wavelength, angle, polarization, perm_dielec, perm_Au, n_mod) 
#     idx += 1

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_wavelength, R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_wavelength, R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.ylabel("Module")
# plt.title("Reflectance of the GP")
# #plt.show(block=False)
# #plt.savefig("Phase_and_modulus_reflexion_GP_depending_wavelength.jpg")

# plt.subplot(212)
# plt.plot(list_wavelength, phase_R_down_GP_Ag, "b", label="Argent")
# plt.plot(list_wavelength, phase_R_down_GP_Au, "r", label="Or")
# plt.legend()
# plt.xlabel("Wavelength (nm) ")
# plt.ylabel("Phase")
# #plt.title("Wavelength dependance")
# plt.show(block=False)
# plt.savefig("Phase_and_modulus_reflexion_GP_depending_Wavelength_Au_Ag.jpg")

