
def eos_to_func(w_q, N_arr):

    def eos(N_val):
        idx = np.argwhere(N_arr>=N_val)[0][0]
        return w_q[idx]

    eos = np.vectorize(eos)

    return eos

w_powfunc = eos_to_func(w_pow, N)


def H(N, w_q, density_parameters):
    Ω_q0, Ω_r0, Ω_m0 = density_parameters

    I = np.trapz(w_q, N)
    H = np.sqrt(Ω_m0*np.exp(-3*N) + Ω_r0*np.exp(-4*N) + Ω_q0*np.exp(-3*(N+I)))

    return H

        H_exp2 = H(N, w_exp, Ω0_exp)
        H_pow2 = H(N, w_pow, Ω0_pow)
    ax.plot(z, H_exp2, "--", label="H_exp")
    ax.plot(z, H_pow2, "--", label="H_pow")

def H(N, w_q, density_parameters):

    I = np.zeros(N)
    for i, N in enumerate(N):
        integral = 3*(1 + w_q[:i+1])
        I[i] = np.trapz(integral)

    #coeff = np.array([np    sol = odeint(eom, x_init, N, args=(Γ,))
    #x1, x2, x3 = sol[:,0], sol[:,1], sol[:,2]

    #H = H0*np.exp(-1.5*(1 + x1**2 - x2**2 + x3**2/3) )

    #return H.exp(I), (1+z)**4, (1+z)**3], dtype=np.ndarray)

    Ω_q0, Ω_r0, Ω_m0 = density_parameters

    HH0 = np.sqrt( Ω_m0*np.exp(-3*N) + Ω_r0*np.exp(-4*N) Ω_q0*np.exp(I))

    return HH0

"""
H_exp = H(z, w_exp, Ω0_exp)
H_pow = H(z, w_pow, Ω0_pow)
"""
def H(N, w_q, density_parameters):
    """
    Uses eq.(7) to calculate dimensionless Hubble parameter H/H0 for quintessence
    models, given density parameters & eqs of state.
    """
    Ω_q0, Ω_r0, Ω_m0 = density_parameters


    H = np.zeros(len(N))
    for i, _N in enumerate(N):
        integral = 1 + w_q[:i+1:-1]
        I = np.trapz(integral, N[:i+1:-1])
        H[i] = np.sqrt(Ω_m0*np.exp(-3*_N) + Ω_r0*np.exp(-4*_N) + Ω_q0*np.exp(-3*I))

    return H
