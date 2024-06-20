    Ex = np.cos(pol) * np.cos(theta) * np.cos(phi) - np.sin(pol) * np.sin(phi) # Ex incident
    Ey = np.cos(pol) * np.cos(theta) * np.sin(phi) + np.sin(pol) * np.cos(phi) # Ey incident
    eps_k2 = top.eps[0,0] * top.mu[0,0] * top.k0**2 # eps k^2
    d = np.sqrt(eps_k2 - top.a0**2 - top.b0**2) # norme k
    # e = normalisation E
    norm = ((eps_k2-top.b0**2)*np.abs(Ex)**2 + (eps_k2-top.a0**2)*np.abs(Ey)**2 + 2*top.a0*top.b0*np.real(Ex*Ey)) / (top.mu[0,0]*d)
    
    V_inc = np.zeros(4 * (2*Nm+1) *(2*Mm+1))
    V_inc[int(np.real(ext[3,0]))] = Ex/np.sqrt(norm)
    V_inc[int(np.real(ext[3,int(np.real(ext[0,0]))]))] = Ey/np.sqrt(norm)

    V_out = S @ V_inc # outgoing fields

    V_r = V_out[:2 * (2*Nm+1) *(2*Mm+1)]
    V_t = V_out[2 * (2*Nm+1) *(2*Mm+1):]

    reflechi = base.efficace(top, ext_top, V_r)
    r[i] = reflechi[3,0]

    transm = base.efficace_t(bot, ext_bot, V_t)
    t[i] = transm[3,0]