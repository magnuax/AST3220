import numpy as np
import matplotlib.pyplot as plt
from eom import eom, solve
from problem_10 import H_ΛCDM, solve_H
from scipy import integrate
plt.style.use("ggplot")


data = np.loadtxt("sndata.txt") # [], [Gpc], [Gpc]
z_data, d_data, err = data[:,0], data[:,1], data[:,2]

pow_init = [5e-5, 1e-8, 0.9999, 1e9]
exp_init = [0, 5e-13, 0.9999, 1.5]


n_points = 10000
#Ni = -np.log(3)
Ni = -np.log(1+z_data[-1])
N = np.linspace(Ni, 0, n_points)
z = np.exp(-N) - 1

pow_solution = solve(N, pow_init, Γ=2)
exp_solution = solve(N, exp_init, Γ=1)

w_pow = pow_solution[3]
Ω0_pow = pow_solution[:3, -1]

w_exp = exp_solution[3]
Ω0_exp = exp_solution[:3, -1]

#H_exp = solve_H(N, exp_init, Γ=1)
#H_pow = solve_H(N, pow_init, Γ=2)
H_Λ   = H_ΛCDM(N)


def arr_to_func(y_array, x_array):

    def y_func(x):
        idx = np.argwhere(x_array <= x)[0][0]
        return y_array[idx]

    y_func = np.vectorize(y_func)

    return y_func



H_exp = solve_H(N, w_exp, Ω0_exp)
H_pow = solve_H(N, w_pow, Ω0_pow)

z = np.linspace(2, 0,10000)
c = 299792.458 # km/s

def d_L(z, H):
    d_L = np.zeros(len(z))
    for i, _z in enumerate(z):
        I = np.trapz(1/H[:i+1:-1], z[:i+1:-1])
        d_L[i] = (1+_z)*I

    return d_L

H0 = 100*0.7*1e3 # km s^-1 Gpc^-1

d_exp = d_L(z, H_exp)*c/H0
d_pow = d_L(z, H_pow)*c/H0

d_exp = arr_to_func(d_exp, z)
d_pow = arr_to_func(d_pow, z)


n_data = len(z_data)
step = int(n_points/n_data)


def chisquare(obs, exp, err):
    s = (exp - obs)**2/err

    return np.sum(s)

"""
X_exp = chisquare(d_data, d_exp(z_data)*c/H0, err)
X_pow = chisquare(d_data, d_pow(z_data)*c/H0, err)

print(X_exp, X_pow)
"""

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

X_best, Ω_best = fit_ΛCDM(d_data, err, N)

print(X_best, Ω_best)



plt.plot(z, d_exp(z), label="exp")
plt.plot(z, d_pow(z), "--", label="pow")
plt.plot(z, d_L(z, H_Λ)*c/H0, label="Λ")

plt.errorbar(z_data, d_data, err, label="data")
plt.fill_between(z_data, d_data-err, d_data+err, alpha=0.2, label="errorbar")
#plt.xlim(0,1.3)
plt.xlabel("z")
plt.ylabel("Gpc")
plt.legend()
plt.show()
