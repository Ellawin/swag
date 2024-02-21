import numpy as np
import matplotlib.pyplot as plt

### RCWA 2D ###
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

def marche(a,b,p,n,x):
    '''Computes the Fourier series for a piecewise function having the value
    a over a portion p of the period, starting at position x
    and the value b otherwise. The period is supposed to be equal to 1.
    Division by zero or very small values being not welcome, think about
    not taking round values for the period or for p. Then takes the toeplitz
    matrix generated using the Fourier series.
    '''
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

def reseau(k0,a0,pol,e1,e2,n,blocs):
    '''Warning: blocs is a vector with N lines and 2 columns. Each
    line refers to a block of material e2 inside a matrix of material e1,
    giving its size relatively to the period (first column) and its starting
    position. #not anymore
    Warning : There is nothing checking that the blocks don't overlapp.
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

def homogene(k0,a0,pol,epsilon,n):
    '''Generates the P matrix and the wavevectors exactly as for a
    periodic layer, just for an homogeneous layer. The results are
    analytic in that case.
    '''
    nmod=int(n/2)
    valp=np.sqrt(epsilon*k0*k0-(a0+2*np.pi*np.arange(-nmod,nmod+1))**2+0j)
    valp=valp*(1-2*(valp<0))*(pol/epsilon+(1-pol))
    P=np.block([[np.eye(n)],[np.diag(valp)]])
    return P,valp

def interface(P,Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n=int(P.shape[1])
    S=np.matmul(np.linalg.inv(np.block([[P[0:n,0:n],-Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]])),np.block([[-P[0:n,0:n],Q[0:n,0:n]],[P[n:2*n,0:n],Q[n:2*n,0:n]]]))
    return S

wavelength = 600
period = 600*1.4142135623730951
eps_metal = -1.
eps_dielec = 1.

normalized_wavelength = wavelength / period
wave_vector = 2 * np.pi / normalized_wavelength 

number_modes = 25
total_number_modes = 2 * number_modes + 1

number_layers = 2
number_cubes = 1

widths_list = np.array([75, 0]) / period
positions_list = 0.5 - widths_list / (2 * period)

thicknesses_list = np.array([75, 5]) / period

polarization = 1
a0 = 0

S = np.block([[np.zeros([total_number_modes,total_number_modes]),np.eye(total_number_modes,dtype=np.complex128)],[np.eye(total_number_modes),np.zeros([total_number_modes,total_number_modes])]])

P,V = homogene(wave_vector,a0,polarization,eps_dielec, total_number_modes)

for i_layer in range(0,number_layers):
    cubes = np.zeros((number_cubes,2))
    cubes[0] = widths_list[i_layer]
    cubes[1] = positions_list[i_layer]
    Pc,Vc = reseau(wave_vector,a0,polarization,eps_dielec,eps_metal,total_number_modes,cubes)
    S = cascade(S,interface(P,Pc))
    S = c_bas(S,Vc,thicknesses_list[i_layer])
    P = Pc
    V = Vc
Pc,Vc = homogene(wave_vector,a0,polarization,eps_metal,total_number_modes)
S = cascade(S,interface(P,Pc))
    
    # S[2*number_modes+1, 0]
    # Tm=np.zeros(3,dtype=np.float64)
    # for j in range(-1,2):
    #     Tm[j+1]=abs(S[j+nmod,n+nmod])**2*np.real(V[j+nmod])*1.46/vect_onde
    # #print(Tm)

    # cost=1-(Te[0]+Tm[2])/2
    # #cost = 1-Te[0]

    # return cost

