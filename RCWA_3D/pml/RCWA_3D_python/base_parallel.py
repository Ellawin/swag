import numpy as np
import scipy.linalg as lin

def cascade(T, U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''
    n = int(T.shape[0]/2)
    T11 = T[0:n,0:n]
    T12 = T[0:n,n:2*n]
    T21 = T[n:2*n,0:n]
    T22 = T[n:2*n,n:2*n]

    U11 = U[0:n,0:n]
    U12 = U[0:n,n:2*n]
    U21 = U[n:2*n,0:n]
    U22 = U[n:2*n,n:2*n]

    J = np.linalg.inv(np.eye(n) - U11 @ T22)
    K = np.linalg.inv(np.eye(n) - T22 @ U11)
    
    S = np.block([[T11 + T12 @ J @ U11 @ T21, T12 @ J @ U12],
                  [U21 @ K @ T21, U22 + U21 @ K @ T22 @ U12]])
    return S


def c_bas(A, V, h):
    ''' Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    '''
    n = int(A.shape[0]/2)
    D = np.diag(np.exp(1.0j*V*h))
    S = np.block([[A[0:n,0:n], A[0:n, n:2*n] @ D],
                  [D @ A[n:2*n,0:n], D @ A[n:2*n, n:2*n] @ D ]])
    return S


def c_haut(A,valp, h):
    n = int(A[0].size/2)
    D = np.diag(np.exp(1.0j*valp*h))
    S11 = D @ A[0:n,0:n] @ D
    S12 = D @ A[0:n, n:2*n]
    S21 = A[n:2*n,0:n] @ D
    S22 = A[n:2*n, n:2*n]
    S = np.block([[S11, S12],
                  [S21,S22]])
    return S    


def intermediaire(T, U):
    n = T.shape[0] // 2
    H = np.linalg.inv( np.eye(n) - U[0:n,0:n] @ T[n:2*n, n:2*n])
    K = np.linalg.inv( np.eye(n) - T[n:2*n, n:2*n] @ U[0:n,0:n])
    a = K @ T[n:2*n,0:n]
    b = K @ T[n:2*n, n:2*n] @ U[0:n, n:2*n]
    c = H @ U[0:n,0:n] @ T[n:2*n,0:n]
    d = H @ U[0:n, n:2*n]
    S = np.block([[a, b],
                  [c, d]])
    return S


def couche(valp, h):
    n = len(valp)
    AA = np.diag(np.exp(1.0j*valp*h))
    C = np.block([[np.zeros((n, n)),AA],[AA,np.zeros((n, n))]])
    return C


def interface(P, Q):
    '''Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, reseau or creneau.
    '''
    n = int(P.shape[1])
    A = np.block([[P[0:n,0:n],-Q[0:n,0:n]],
                  [P[n:2*n,0:n], Q[n:2*n,0:n]]])
    B = np.block([[-P[0:n,0:n], Q[0:n,0:n]],
                  [P[n:2*n,0:n], Q[n:2*n,0:n]]])
    S = np.linalg.inv(A) @ B
    return S



def eps11(s):
    """
        eps * g(y) / f(x)
    """
  
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + 1 / s.eps[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # f / eps
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = np.linalg.inv(toep(v)) # eps / f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # eps / f * g
            v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # eps / f * g

    return M


def eps22(s):
    """
        eps * f(x) / g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.eps[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # eps * f
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # eps * f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + 1 / T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])  # 1 / (eps * f) * g
            v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
            M[j*n:(j+1)*n, k*n:(k+1)*n] = np.linalg.inv(toep(v)) # eps * f / g

    return M


def eps33(s):
    """
        eps * f(x) * g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    # print(s.eps)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            # print(f"DEBUG eps33 l={l} j={j} tfx={tfx}")
            v = v + s.eps[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # eps * f
            # print(f"DEBUG eps33 l={l} j={j} v={v} eps={s.eps[l,j]}")
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # eps * f
    # print(f"DEBUG eps33 T={T}")
    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                # print(f"DEBUG eps33 j={j} k={k} l={l} tfy={tfy}")
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # eps * f * g
            v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # eps * f * g
            # print(f"DEBUG eps33 j={j} k={k} \n M={M}\n (j-1)*n:j*n={(j-1)*n}:{j*n} (k-1)*n:k*n={(k-1)*n}:{k*n}")
    # print(f"DEBUG eps33 M={M}")
    return M


def mu11(s):
    """
        mu * g(y) / f(x)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + 1 / s.mu[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # f / mu
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = np.linalg.inv(toep(v)) # mu / f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # mu / f * g
            v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # mu / f * g

    return M


def mu22(s):
    """
        mu * f(x) / g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            # print(f"DEBUG mu22 tfx={tfx}")
            v = v + s.mu[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # mu * f
        # print(f"DEBUG mu22 v={np.abs(v)}\n shape(v)={np.shape(v)} shape(T)={np.shape(T)} shape(toep(v))={np.shape(toep(v))}")
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # mu * f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + 1 / T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # 1  / (mu * f) * g
            v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
            M[j*n:(j+1)*n, k*n:(k+1)*n] = np.linalg.inv(toep(v)) # mu * f / g

    return M


def mu33(s):
    """
        mu * f(x) * g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.mu[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # mu * f
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # mu * f

    M = np.zeros((m*n, m*n), dtype=complex)
    # print(f"DEBUG mu33 ")
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # mu * f * g
            v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # mu * f * g

    return M


def g(s, y):
    """
        Stretching function along y
    """
    j = np.argmax((s.ny-y)>0) - 1

    new_diff = s.ny[j+1]-s.ny[j]
    new_k = 2*np.pi/new_diff
    old_diff = s.oy[j+1]-s.oy[j]

    val = s.oy[j] + old_diff/new_diff * (y - s.ny[j] - s.eta * np.sin(new_k*(y-s.ny[j])) / new_k)
    return val


def f(s, x):
    """
        Stretching function along x
    """
    j = np.argmax((s.nx-x)>0) - 1

    new_diff = s.nx[j+1]-s.nx[j]
    new_k = 2*np.pi/new_diff
    old_diff = s.ox[j+1]-s.ox[j]

    val = s.ox[j] + old_diff/new_diff * (x - s.nx[j] - s.eta * np.sin(new_k*(x-s.nx[j])) / new_k)
    return val
  

def reseau(s):
    """
        Computes the modes and eigenvalues in a structured layer
    """

    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    v_a = []
    for j in range(-s.Mm, s.Mm+1):
        a = s.a0 + 2*np.pi*j / s.nx[-1]
        v_a.extend([a] * n)
    alpha = 1.0j*np.diag(v_a)
    # print(f"DEBUG reseau v_a={v_a}")

    v_b = s.b0 + 2*np.pi * np.arange(-s.Nm, s.Nm+1) / s.ny[-1]
    # print(f"DEBUG reseau v_b={v_b}")
    v_b = np.tile(v_b, (m))
    # print(f"DEBUG reseau v_b={v_b}")
    beta = 1.0j*np.diag(v_b)
    # print("DEBUG", beta)

    inv_e33 = np.linalg.inv(eps33(s)) / (1.0j*s.k0)
    # print(f"DEBUG reseau inv_e33={inv_e33*1.0j*s.k0}")
    alpha_eps_beta = alpha @ inv_e33 @ beta 
    alpha_eps_alpha = alpha @ inv_e33 @ alpha
    beta_eps_beta = beta @ inv_e33 @ beta
    beta_eps_alpha = beta @ inv_e33 @ alpha
    # print(f"DEBUG reseau inv_e33={inv_e33}")
    Leh = np.block([[alpha_eps_beta, 1.0j*s.k0*mu22(s) - alpha_eps_alpha],
                    [-1.0j*s.k0*mu11(s) + beta_eps_beta, -beta_eps_alpha]])
    # print(f"DEBUG reseau L_eh={Leh}")
    

    inv_mu33 = np.linalg.inv(mu33(s)) / (1.0j*s.k0)
    alpha_mu_beta = alpha @ inv_mu33 @ beta
    alpha_mu_alpha = alpha @ inv_mu33 @ alpha
    beta_mu_beta = beta @ inv_mu33 @ beta
    beta_mu_alpha = beta @ inv_mu33 @ alpha
    Lhe = np.block([[-alpha_mu_beta, -1.0j*s.k0*eps22(s) + alpha_mu_alpha],
                    [1.0j*s.k0*eps11(s)-beta_mu_beta, beta_mu_alpha]])
    # print(f"DEBUG reseau Lhe={Lhe}")

    L = Leh @ Lhe
    L = L * (np.abs(L)>1e-18)
    # print(f"DEBUG reseau L={L}")
    [V, inv_L] = np.linalg.eig(L)
    for i_vp in range(inv_L.shape[1]):
        inv_L[:,i_vp] = inv_L[:,i_vp]/np.exp(1.0j*np.angle(inv_L[0,i_vp]))
    
    V = np.sqrt(-V)
    
    neg_val = np.imag(V) < 0
    V = V * (1 - 2*neg_val)
    # keep_im = np.abs(np.imag(V)) > 1e-15*np.abs(np.real(V))
    # V = np.real(V) + 1.0j*np.imag(V)*keep_im
    P = np.block([[inv_L],
                  [Lhe @ inv_L @ np.diag(1 / V)]])

    return (P, V)


def genere(ox, nx, eta, n):
    """
        Computes the fourier transform of coordinates, with stretching,
        for all interface coordinates
    """
    fp = []
    for i in range(len(ox)-1):
        fp.append(tfd(ox[i], ox[i+1], nx[i], nx[i+1], eta, nx[-1], n).T)
    return np.array(fp).T # TODO: check whether transposing is necessary


def homogene(s, ext=0):
    """
        Computing modes and eignevalues in a homogeneous layer
        Takes into account the possibility that it is the first or last layer
        (ext), in which case we match with propagative modes
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1
    global np # Why did I have to do this ??
    
    v_a = []
    for j in range(-s.Mm, s.Mm+1):
        a = s.a0 + 2*np.pi*j/s.nx[-1]
        v_a.extend([a] * n)
    alpha = 1.0j*np.diag(v_a)

    v_b = s.b0 + 2*np.pi*np.arange(-s.Nm, s.Nm+1)/s.ny[-1]
    v_b = np.tile(v_b, (m))
    beta = 1.0j*np.diag(v_b)

    i_eps33 = np.linalg.inv(eps33(s))
    i_mu33 = np.linalg.inv(mu33(s))
    eps1 = eps11(s)
    eps2 = eps22(s)
    mu1 = mu11(s)
    mu2 = mu22(s)
    print(f"DEBUG homogene, n={n}, m={m}, v_b={v_b}")
    print(f"DEBUG homogene, shape(mu1)={np.shape(mu1)}, shape(mu2)={np.shape(mu2)},\
          shape(eps1)={np.shape(eps1)}, shape(eps2)={np.shape(eps2)}, shape(alpha)={np.shape(alpha)}, shape(beta)={np.shape(beta)}")
    L = -s.k0**2 * mu2 @ eps1 - alpha @ i_eps33 @ alpha @ eps1 - mu2 @ beta @ i_mu33 @ beta
    print(f"DEBUG homogene, (mu1)={(mu1)}, (mu2)={(mu2)},\
          (eps1)={(eps1)}, (eps2)={(eps2)}, (alpha)={(alpha)}, (beta)={(beta)},\
            ieps33={i_eps33}, i_mu33={i_mu33} k0={s.k0}")
    [B, A] = np.linalg.eig(L)
    print(f"DEBUG homogene A={A}, B={B}, L={L}")

    L = -s.k0**2 * mu1 @ eps2 - beta @ i_eps33 @ beta @ eps2 - mu1 @ alpha @ i_mu33 @ alpha

    [D, C] = np.linalg.eig(L)

    E = np.block([[A, np.zeros((n*m, n*m))],
                  [np.zeros((n*m, n*m)), C]])
    print(f"DEBUG homogene, E={E}")

    inv_mu33 = i_mu33 / (1.0j*s.k0)
    alpha_mu_beta = alpha @ inv_mu33 @ beta
    alpha_mu_alpha = alpha @ inv_mu33 @ alpha
    beta_mu_beta = beta @ inv_mu33 @ beta
    beta_mu_alpha = beta @ inv_mu33 @ alpha
    Lhe = np.block([[-alpha_mu_beta, -1.0j*s.k0*eps22(s) + alpha_mu_alpha],
                    [1.0j*s.k0*eps11(s)-beta_mu_beta, beta_mu_alpha]])

    V = np.block([B, D])
    print(f"DEBUG homogene V={V}")

    # Ca dépend ! Soit c'est entre deux couches, soit c'est extérieur !
    if (ext):
        # This layer is a substrate or superstrate
        # -> we are interested in the Rayleigh decomposition of the modes

        V = np.sqrt(-V)

        print('exterieur')  

        # Finding real eigen values and their positions in V

        position = np.where((abs(np.angle(V))<1e-4))[0]

        # Compute the analytical eigen values (Rayleigh decomposition)

        dx = s.nx[-1]
        kx = 2*np.pi/dx
        dy = s.ny[-1]
        ky = 2*np.pi/dy
        k = np.sqrt(s.eps[1, 1]*s.mu[1, 1])*s.k0
        min_ord_x = int((k + s.a0)/kx)
        max_ord_x = int((k - s.a0)/kx)
        min_ord_y = int((k + s.b0)/ky)
        max_ord_y = int((k - s.b0)/ky)

        ana_kz = []
        # TODO: check is correct, changed max ord from dx to 1/kx
        i = 0
        for m in range(-min_ord_x, max_ord_x+1):
            for n in range(-min_ord_y, max_ord_y+1):
                gamma = np.sqrt(0j+k**2-(s.a0 + m*kx)**2-(s.b0 + n*ky)**2)
                # Computes the (n,m) diffracted order
                print(m, n, gamma)
                if (np.real(gamma) != 0):
                    # Keeping only propagative modes
                    if ((n == 0) and (m == 0)):
                        ana_kz.insert(0,[gamma, m, n, 0])
                        # Main propagative mode
                    else:
                        ana_kz.append([gamma, m, n, 0])
                    i+=1
        ana_kz = np.array(ana_kz).T


        if (np.shape(position)[1] == 2*np.shape(ana_kz)[1]):
            ana_kz = np.block([ana_kz, ana_kz])
            ana_kz[4, :] = position
        else:
            print('Missing modes! (homogene)')    

        for m in range(np.shape(ana_kz)[1]):
            # If the mode is propagative, it is in ana_kz
            # and we replace it in V (more precise?)
            V[ana_kz[3, m]] = ana_kz[1, m]

        print(f"DEBUG homogene V={V}")

        # Replacing modes

        n = 2 * s.Nm + 1
        m = 2 * s.Mm + 1
        E = np.zeros((2*n*m, 2*n*m))
        for j in range(np.shape(ana_kz)[1]/2):

            np = 2048
            pos_x = np.arange(np)/np*dx
            x = np.array(np)
            for k in range(np):
                x[k] = np.exp(1.0j*(s.a0 + ana_kz[2, j]*kx)*f(s, pos_x[k])-1.0j*s.a0*pos_x[k])
            x = dx*np.fft(x)/np
            x = [x[np-s.Mm:np], x[:s.Mm+1]]


            pos_y = np.arange(np)/np*dy
            y = np.array(np)
            for k in range(np):
                y[k] = np.exp(1.0j*(s.b0 + ana_kz[3, j]*ky)*g(s, pos_y[k])-1.0j*s.b0*pos_y[k])
            y = dy*np.fft(y)/np
            y = [y[np-s.Nm:np], y[:s.Nm+1]]

            vtmp = []
            for k in range(2 * s.Mm + 1):
                vtmp.append(x[k]*y)
            vtmp = np.array(vtmp).T


            E[:, ana_kz[3, j]] = np.block([[vtmp],
                                           [np.zeros((n*m, 1))]])
            E[:, ana_kz[3, j + np.shape(ana_kz)[1]//2]] = np.block([[np.zeros((n*m, 1))],
                                                                    [vtmp]])


        imag_val = np.angle(V)<-np.pi/2+0.00001
        V = (1-2*imag_val) * V
    else:
        # Not in a substrate/superstrate, we simply keep the modes
        V = np.sqrt(-V)
        ana_kz = []

    P = np.block([[E],
                  [Lhe @ E @ np.diag(1 / V)]])
    return (P, V, ana_kz)


def tfd(old_a, old_b, new_a, new_b, eta, d, N):
    """
        Computing fourier transform of coordinates with stretching
    """
    pi = np.pi
    old_ba = old_b - old_a
    new_ba = new_b - new_a

    # mode 0 is a problem so we split
    neg_modes = np.arange(-N, 0)
    pos_modes = np.arange(1, N+1)

    neg_sinc_prefac = old_ba * neg_modes / d * np.sinc(neg_modes/d * new_ba) * np.exp(-1.0j*np.pi*neg_modes * (new_b+new_a) / d)
    pos_sinc_prefac = old_ba * pos_modes / d * np.sinc(pos_modes/d * new_ba) * np.exp(-1.0j*np.pi*pos_modes * (new_b+new_a) / d)
    neg_fft = neg_sinc_prefac * (1/neg_modes + eta/2 * (new_ba/(d-neg_modes*new_ba)-new_ba/(d+neg_modes*new_ba)))
    pos_fft = pos_sinc_prefac * (1/pos_modes + eta/2 * (new_ba/(d-pos_modes*new_ba)-new_ba/(d+pos_modes*new_ba)))
    fft_zero = old_ba / d
    fft = np.concatenate((neg_fft, [fft_zero], pos_fft))
    # Once we've done this, there is only a problem if d-modes*new_ba=0 or d+modes*new_ba=0

    # This is the safe way. We could also probably check whether the period is a multiple of some part of the structure
    i_m_diff = -d / new_ba
    i_p_diff = d / new_ba
    if (i_m_diff == int(i_m_diff)):
        sinc_prefac = old_ba * i_m_diff / d * np.sinc(i_m_diff/d * new_ba) * np.exp(-1.0j*np.pi*i_m_diff * (new_b+new_a) / d)
        fft[int(i_m_diff)+N] = sinc_prefac * (1/i_m_diff + eta/2 * new_ba/(d-i_m_diff*new_ba)) - eta/2 * old_ba*np.exp(2.0j*pi*new_a/new_ba)/d
    if (i_p_diff == int(i_p_diff)):
        sinc_prefac = old_ba * i_p_diff / d * np.sinc(i_p_diff/d * new_ba) * np.exp(-1.0j*np.pi*i_p_diff * (new_b+new_a) / d)
        fft[int(i_p_diff)+N] = sinc_prefac * (1/i_p_diff + eta/2 * new_ba/(d+i_p_diff*new_ba)) - eta/2 * old_ba*np.exp(2.0j*pi*new_a/new_ba)/d

    return fft
"""
fft=0.;
for n=-N:N   
  if (n==0)
    fft(N+1) = (b-a)/d;
  elseif (d-n*(b1-a1)==0)
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n-eta/2*(b1-a1)/(d+n*(b1-a1)))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d))-eta/2*(b-a)*exp(-2*i*pi*a1/(b1-a1))/d;
  elseif (d+n*(b1-a1)==0)
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n+eta/2*(b1-a1)/(d-n*(b1-a1)))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d))-eta/2*(b-a)*exp(2*i*pi*a1/(b1-a1))/d;
  else
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n+eta/2*((b1-a1)/(d-n*(b1-a1))-(b1-a1)/(d+n*(b1-a1))))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d));
  endif
endfor


"""

def toep(v):
    """
        Computing Toeplitz matrix
    """
    n = (len(v)-1)//2
    a = v[n:0:-1]
    b = v[n:2*n]
    T = lin.toeplitz(b, a)
    # print(f"DEBUG toep v={v} shape(v)={np.shape(v)} n={n} a={a} b={b} \n toep={T}")
    return T