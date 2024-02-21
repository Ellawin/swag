import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import toeplitz, inv
#from PyMoosh import *

i = complex(0,1)

### Materials
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
    valp=valp*(1-2*(valp<0))*(pol/epsilon+(1-pol))
    P=np.block([[np.eye(n)],[np.diag(valp)]])
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

# Swag-structure
wavelength = 700.021635 # in nanometers
polarization = 1 # 0 for TE, 1 for TM
period =  501.2153 
perm_dielec = 1 # material = air
perm_metal = epsAgbb(wavelength) # argent ou or, argent ici car accessible
n_mod = 200 # to do convergence study
n = 2 * n_mod + 1

### Base Morpho

## Pour L fixé, il faudra le faire varier si le code tourne

list_L = np.linspace(50, 350, 350-50)
R_creneau = np.empty(list_L.size)
idx_c = 0

for L in list_L:
    cavity_size = L / period # taille du résonateur selon x, en nm
    H = 75 / period # taille du résonateur selon z, en nm. Cube au départ donc H = L
    h = 5 / period # taille du gap dans lequel résone le GP, dans la couche homogène de dielec. Equivalent à 'spacer'
    x0 = [(1 - cavity_size )/ 2] # position du bloc dans la période. Central. 
    #x0 = 1 / 2

    wavelength_norm = wavelength / period 
    k0 = 2 * np.pi / wavelength_norm
    P, V = homogene(k0, 0, polarization, perm_dielec, n)
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex128)], [np.eye(n), np.zeros([n, n])]])
    Pc, Vc = creneau(k0, 0, polarization, perm_metal, perm_dielec, cavity_size, n, x0)
    S = cascade(S, interface(P, Pc))
    S = c_bas(S, Vc, H)
    S = cascade(S, interface(Pc, P))
    S = c_bas(S, V, h)
    Pc, Vc = homogene(k0, 0, polarization, perm_metal, n)
    S = cascade(S, interface(P, Pc))
    R_creneau[idx_c] = abs(S[n_mod, n_mod]) ** 2 * np.real(V[n_mod]) / k0
    idx_c += 1


plt.figure(3)
#plt.title("wavelength dependance")
plt.plot(list_L, R_creneau) #, "og", label = "creneau")
#plt.legend()
plt.xlabel("Width of the rod")
plt.ylabel("r0")
plt.show(block=False)
#plt.savefig("fig/r0_validation_base_Morpho_r0_fct_L_compare_reseau_creneau.pdf")
plt.savefig("fig/r0_validation_base_Morpho_r0_fct_L_200modes.pdf")

# ### Base Taper

# def show_structure(thick, width, period):
#     number_layers = len(thick)

#     p = (period - np.array(width)) / 2

#     X = np.array([thick, width, p])
#     X = X.T

#     h = np.array(thick) / period
#     x = X / period

#     n = 600
#     H = sum(h)
#     M0 = np.zeros((n))
#     for j in range(number_layers):
#         tmp = np.zeros(n)
#         position = np.arange(1,n+1)/n
#         for k in range(int((X.shape[1]-1)/2)):
#             tmp = tmp + (position>x[j,2*k+2])*(position<x[j,2*k+1]+x[j,2*k+2])
#         if (x[j,2*k+1]+x[j,2*k+2]>1):
#             tmp = tmp + (position<x[j,2*k+1]+x[j,2*k+2]-1)
#         cst = int(np.floor(h[j]*n))
#         M2 = np.tile(tmp,(cst,1))
#         if (j == 0):
#             M = M2
#         else:
#             M = np.vstack((M,M2))

#     M = 1 - M  
#     return M

# thick = [75,5,100]
# width = [75,0, period]
# period = 1500

# M = show_structure(thick, width, period)

# plt.figure(1)
# plt.imshow(M*127, cmap = 'bone')
# plt.show(block = False)    
# plt.savefig("fig/r0_validation_base_Taper_visu_structure.pdf")