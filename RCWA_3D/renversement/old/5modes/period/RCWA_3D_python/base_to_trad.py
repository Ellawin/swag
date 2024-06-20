import numpy as np
import scipy.linalg as lin


def eps11(s):
    """
        eps * g / f
    """
  
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, np.shape(s.eps)[0]))
    for l in range(np.shape(s.eps)[0]):
        v = 0
        for j in range(np.shape(s.eps)[1]):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + 1 / s.eps[l, j] * tfx.T * (1 + 1.0j * s.pmlx[j])
        T[:, :, l] = np.linalg.inv(toep(v))

    M = np.zeros((m*n, m*n))
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(np.shape(s.eps)[0]):
                tfy = tfd(s.oy[l], s.oy[l+1], s.n[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])
            M[(j-1)*n:j*n, (k-1)*n:k*n] = toep(v)

    return M


def eps22(s):
    """
        eps * f / g
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, np.shape(s.eps)[0]))
    for l in range(np.shape(s.eps)[0]):
        v = 0
        for j in range(np.shape(s.eps)[1]):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.eps[l, j] * tfx.T * (1 + 1.0j * s.pmlx[j])
        T[:, :, l] = toep(v)

    M = np.zeros((m*n, m*n))
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(np.shape(s.eps)[0]):
                tfy = tfd(s.oy[l], s.oy[l+1], s.n[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + 1 / T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])
            M[(j-1)*n:j*n, (k-1)*n:k*n] = np.linalg.inv(toep(v))

    return M


def eps33(s):
    """
        eps * f * g
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, np.shape(s.eps)[0]))
    for l in range(np.shape(s.eps)[0]):
        v = 0
        for j in range(np.shape(s.eps)[1]):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.eps[l, j] * tfx.T * (1 + 1.0j * s.pmlx[j])
        T[:, :, l] = toep(v)

    M = np.zeros((m*n, m*n))
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(np.shape(s.eps)[0]):
                tfy = tfd(s.oy[l], s.oy[l+1], s.n[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])
            M[(j-1)*n:j*n, (k-1)*n:k*n] = toep(v)

    return M


def mu11(s):
    """
        mu * g / f
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, np.shape(s.eps)[0]))
    for l in range(np.shape(s.eps)[0]):
        v = 0
        for j in range(np.shape(s.eps)[1]):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + 1 / s.mu[l, j] * tfx.T * (1 + 1.0j * s.pmlx[j])
        T[:, :, l] = np.linalg.inv(toep(v))

    M = np.zeros((m*n, m*n))
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(np.shape(s.eps)[0]):
                tfy = tfd(s.oy[l], s.oy[l+1], s.n[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])
            M[(j-1)*n:j*n, (k-1)*n:k*n] = toep(v)

    return M


def mu22(s):
    """
        mu * f / g
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, np.shape(s.eps)[0]))
    for l in range(np.shape(s.eps)[0]):
        v = [np.zeros((2*m+1, 1))]
        for j in range(np.shape(s.eps)[1]):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.mu[l, j] * tfx.T * (1 + 1.0j * s.pmlx[j])
        T[:, :, l] = toep(v)

    M = np.zeros((m*n, m*n))
    for j in range(m):
        for k in range(m):
            v = [np.zeros((2*n+1, 1))]
            for l in range(np.shape(s.eps)[0]):
                tfy = tfd(s.oy[l], s.oy[l+1], s.n[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + 1 / T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])
            M[(j-1)*n:j*n, (k-1)*n:k*n] = np.linalg.inv(toep(v))

    return M


def mu33(s):
    """
        mu * f * g
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, np.shape(s.eps)[0]))
    for l in range(np.shape(s.eps)[0]):
        v = 0
        for j in range(np.shape(s.eps)[1]):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.mu[l, j] * tfx.T * (1 + 1.0j * s.pmlx[j])
        T[:, :, l] = toep(v)

    M = np.zeros((m*n, m*n))
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(np.shape(s.eps)[1]):
                tfy = tfd(s.oy[l], s.oy[l+1], s.n[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])
            M[(j-1)*n:j*n, (k-1)*n:k*n] = toep(v)

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

    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    v_a = []
    for j in range(-s.Mm, s.Mm+1):
        a = s.a0 + 2*np.pi*j / s.nx[-1]
        v_a.extend([a] * n)
    alpha = 1.0j*np.diag(v_a)

    v_b = s.b0 + 2*np.pi * np.arange(-s.Nm, s.Nm+1) / s.ny[-1]
    v_b = np.tile(v_b, (1, m))
    beta = 1.0j*np.diag(v_b)

    inv_e33 = np.linalg.inv(eps33(s)) / (1.0j*s.k0)
    alpha_eps_beta = alpha @ inv_e33 @ beta 
    alpha_eps_alpha = alpha @ inv_e33 @ alpha
    beta_eps_beta = beta @ inv_e33 @ beta
    beta_eps_alpha = beta @ inv_e33 @ alpha
    Leh = np.block([[alpha_eps_beta, 1.0j*s.k0*mu22(s) - alpha_eps_alpha],
                    [-1.0j*s.k0*mu11(s) + beta_eps_beta, -beta_eps_alpha]])

    inv_mu33 = np.linalg.inv(mu33(s)) / (1.0j*s.k0)
    alpha_mu_beta = alpha @ inv_mu33 @ beta
    alpha_mu_alpha = alpha @ inv_mu33 @ alpha
    beta_mu_beta = beta @ inv_mu33 @ beta
    beta_mu_alpha = beta @ inv_mu33 @ alpha
    Lhe = np.block([[-alpha_mu_beta, -1.0j*s.k0*eps22(s) + alpha_mu_alpha],
                    [1.0j*s.k0*eps11(s)-beta_mu_beta, beta_mu_alpha]])

    L = Leh @ Lhe
    [V, inv_L] = np.linalg.eig(L)
    
    V = np.sqrt(-V)
    neg_val = np.imag(V)< 0
    V = V * (1 - 2*neg_val)

    P = np.block([[inv_L],
                  [Lhe @ inv_L @ np.diag(1 / V)]])

    return (P, V)


def genere(ox, nx, eta, n):
    fp = []
    for i in range(len(ox)-1):
        fp.append(tfd(ox[i], ox[i+1], nx[i], nx[i+1], eta, nx[-1], n).T)
    return np.array(fp).T # TODO: check whether transposing is necessary


def homogene(s, ext=0):
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    v_a = []
    for j in range(-s.Mm, s.Mm+1):
        a = s.a0 + 2*np.pi*j/s.nx[-1]
        v_a.extend([a] * n)
    alpha = 1.0j*np.diag(v_a)

    v_b = s.b0 + 2*np.pi*np.arange(-s.Nm, s.Nm+1)/s.ny[-1]
    v_b = np.tile(v_b, (1, m))
    beta = 1.0j*np.diag(v_b)

    i_eps33 = np.linalg.inv(eps33(s))
    i_mu33 = np.linalg.inv(mu33(s))
    eps1 = eps11(s)
    eps2 = eps22(s)
    mu1 = mu11(s)
    mu2 = mu22(s)
    L = -s.k0**2 @ mu2 @ eps1 - alpha @ i_eps33 @ alpha @ eps1 - mu2 @ beta @ i_mu33 @ beta

    [B, A] = np.linalg.eig(L)

    L = -s.k0**2 @ mu1 @ eps2 - beta @ i_eps33 @ beta @ eps2 - mu1 @ alpha @ i_mu33 @ alpha

    [D, C] = np.linalg.eig(L)

    E = np.block([[A, np.zeros((n*m, n*m))],
                  [np.zeros((n*m, n*m)), C]])

    inv_mu33 = i_mu33 / (1.0j*s.k0)
    alpha_mu_beta = alpha @ inv_mu33 @ beta
    alpha_mu_alpha = alpha @ inv_mu33 @ alpha
    beta_mu_beta = beta @ inv_mu33 @ beta
    beta_mu_alpha = beta @ inv_mu33 @ alpha
    Lhe = np.block([[-alpha_mu_beta, -1.0j*s.k0*eps22(s) + alpha_mu_alpha],
                    [1.0j*s.k0*eps11(s)-beta_mu_beta, beta_mu_alpha]])

    V = np.block([B, D])

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

    P = np.block([[E],
                  [Lhe @ E @ np.diag(1 / V)]])
    return (P, V, ana_kz)


def tfd(old_a, old_b, new_a, new_b, eta, d, N):
    np.pi = np.np.pi
    fft = np.zeros(2*N+1)
    old_ba = old_b - old_a
    new_ba = new_b - new_a
    k_na = 2.0j*np.pi*new_a/d
    k_nb = 2.0j*np.pi*new_b/d
    prefac = -1 / (2.0j*np.pi)*(old_ba)/(new_ba)
    for i_mod in range(-N, N):
        if (i_mod ==0):
            fft[N] = old_ba / d
        elif (d-i_mod*(new_ba) ==0):
            fft[i_mod + N] = (1/i_mod-eta/2*(new_ba)/(d+i_mod*(new_ba))) * (np.np.exp(-k_nb*i_mod)-np.np.exp(-k_na*i_mod))-eta/2*(old_ba)*np.np.exp(-2.0j*np.pi*new_a/(new_ba))/d
        elif (d+i_mod*(new_ba) ==0):
            fft[i_mod + N] = (1/i_mod + eta/2*(new_ba)/(d-i_mod*(new_ba))) * (np.np.exp(-k_nb*i_mod)-np.np.exp(-k_na*i_mod))-eta/2*(old_ba)*np.np.exp(2.0j*np.pi*new_a/(new_ba))/d
        else:
            fft[i_mod + N] = (1/i_mod + eta/2*((new_ba)/(d-i_mod*(new_ba))-(new_ba)/(d+i_mod*(new_ba))))*(np.np.exp(-k_nb*i_mod)-np.np.exp(-k_na*i_mod))
    fft = prefac * fft
    return fft


def toep(v):
    n = (max(np.shape(v))-1)/2
    a = v[n:-1:2]
    b = v[n:2*n]
    T = lin.toeplitz(b, a)

    return T