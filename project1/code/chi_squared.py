chisquare(data, model, error):
    s = (model - data)**2/error

    return np.sum(s)

def fit_ΛCDM(obs, err, N):
    densities = np.linspace(0,1,10)
    z = np.exp(-N) - 1

    n_data  = len(obs)
    n_model = len(N)
    step = int( n_model/n_data )

    stop = step*n_data

    H = H_ΛCDM(N, 0)
    d_exp = d_L(z, H)[:stop:step]

    X = np.zeros(len(densities))

    for i, Ω_m0 in enumerate(densities):
        H = H_ΛCDM(N, Ω_m0)
        d_model = d_L(z, H)[:stop:step]
        X[i] = chisquare(obs, d_model*c/H0, err)

    idx_min = np.argmin(X)

    X_best = X[idx_min]
    Ω_best = densities[idx_min]
    print(X)
    """
    Ω_best = 0
    X_best = chisquare(obs, d_exp, err)

    for Ω_m0 in densities:
        H = H_ΛCDM(N, Ω_m0)*c/H0
        d_model = d_L(z, H)[:stop:step]
        X = chisquare(obs, d_model*c/H0, err)


        print(X<X_best)

        if X<X_best:
            X_best = X
            Ω_best = Ω_m0
    """
    return X_best, Ω_best
