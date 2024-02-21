import numpy as np
import matplotlib.pyplot as plt
from PyMoosh import *

### RCWA ###
def cascade(T,U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''
    n=int(T.shape[1] / 2)
    J=np.linalg.inv( np.eye(n) - np.matmul(U[0:n,0:n],T[n:2*n,n:2*n] ) )
    K=np.linalg.inv( np.eye(n) - np.matmul(T[n:2*n,n:2*n],U[0:n,0:n] ) )
    S=np.block([[T[0:n,0:n] + np.matmul(np.matmul(np.matmul(T[0:n,n:2*n],J),
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
    n=int(A.shape[1]/2)
    D=np.diag(np.exp(1j*V*h))
    S=np.block([[A[0:n,0:n],np.matmul(A[0:n,n:2*n],D)],[np.matmul(D,A[n:2*n,0:n]),np.matmul(np.matmul(D,A[n:2*n,n:2*n]),D)]])
    return S

def step(a,b,w,x0,n):
    '''Computes the Fourier series for a piecewise function having the value
    b over a portion w of the period, starting at position x0
    and the value a otherwise. The period is supposed to be equal to 1.
    Then returns the toeplitz matrix generated using the Fourier series.
    '''
    from scipy.linalg import toeplitz
    l=np.zeros(n,dtype=np.complex128)
    m=np.zeros(n,dtype=np.complex128)
    tmp=np.exp(-2*1j*np.pi*(x0+w/2)*np.arange(0,n))*np.sinc(w*np.arange(0,n))*w
    l=np.conj(tmp)*(b-a)
    m=tmp*(b-a)
    l[0]=l[0]+a
    m[0]=l[0]
    T=toeplitz(l,m)
    return T

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
    n_blocs=blocs.shape[0];
    nmod=int(n/2)
    M1=e1*np.eye(n,n)
    M2=1/e1*np.eye(n,n)
    for k in range(0,n_blocs):
        M1=M1+step(0,e2-e1,blocs[k,0],blocs[k,1],n)
        M2=M2+step(0,1/e2-1/e1,blocs[k,0],blocs[k,1],n)
    alpha=np.diag(a0+2*np.pi*np.arange(-nmod,nmod+1))+0j
    g=1/(1-1j);
#    fprime=fpml(0.2001,g,n) #fixe la proporsion de periode qui constitue la PML (à gauche et à droite)
#    fprime=fpml(0,g,n) #fixe la proporsion de periode qui constitue la PML (à gauche et à droite)
    fprime = np.eye(n)
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

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

number_modes = 201
period = 1500 # in nm, to isolate a period
number_layers = 1
number_resonators = 1
wavelength = 600 # pour un premier test, après on fait varier les lambda
polarization = 1 # on reste en TM si on veut voir des choses

a0 = 0 # je sais plus ce que c'est, en rapport avec l'angle et le k0 alpha 0
#material_metal = Material(['Gold'])
#eps_metal = material_metal.get_permittivity(wavelength)
#mu_metal = material_metal.get_permeability(wavelength)
eps_metal = -1.
eps_dielec = 1.

gap_size = 5 
thicknesses_list = np.array([gap_size]) / period
widths_list = np.array([0]) / period

positions_list = 0.5 - widths_list / 2
wavelength = wavelength / period 
k0 = 2*np.pi / wavelength

w_in = 75 / period 
w_out = period / period 

bloc_in = np.array([[w_in, 0.5 - w_in / 2]])
bloc_out = np.array([[w_out, 0.5 - w_out / 2]])

S = np.block([[np.zeros([number_modes,number_modes]),np.eye(number_modes,dtype=np.complex128)],[np.eye(number_modes),np.zeros([number_modes,number_modes])]])
P,V = aper(k0,a0,polarization,eps_dielec,eps_metal,number_modes,bloc_in)

a = np.argmin(abs(V-5.25*k0)) #guide entrée, 2.35 = indice effectif !!  plus vrai ! 5.25 d'après programme précédent pour gap de 5 nm (non toujours pas mais on tente comme ça quand même pour avoir un début)


for k in range(0,number_layers):
    resonators = np.array([[widths_list[k],positions_list[k]]])
    P_new,V_new = aper(k0,a0,polarization,eps_dielec,eps_metal,number_modes,resonators)
    S = cascade(S,interface(P,P_new))
    S = c_bas(S,V_new,thicknesses_list[k])
    P,V = P_new,V_new
P_out,V_out = aper(k0,a0,polarization,eps_dielec,eps_metal,number_modes,bloc_out)
S = cascade(S,interface(P,P_out))
b = np.argmin(abs(V_out-3.24033*k0)) # guide sortie, 3.24 = indice effectif (toujours pas bon)

ref = abs(S[b+number_modes, a])**2


