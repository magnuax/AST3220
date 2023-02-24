import numpy as np
import matplotlib.pyplot as plt
from eom import eom, solve, solve_H
from problem_10 import solve_H2, H_ΛCDM
plt.style.use("ggplot")


def H1(N, w_q, density_parameters):
    """
    Uses eq.(7) to calculate dimensionless Hubble parameter H/H0 for quintessence
    models, given density parameters & eqs of state.
    """
    Ω_q0, Ω_r0, Ω_m0 = density_parameters

    dN = N[1]-N[0]
    H = np.zeros(len(N))
    for i, _N in enumerate(N):
        integral = 1 + w_q[:i+1:-1]
        I = np.trapz(integral, N[:i+1:-1])
        H = np.sqrt(Ω_m0*np.exp(-3*_N) + Ω_r0*np.exp(-4*_N) + Ω_q0*np.exp(-3*I))

    return H


def age(H):
    """
    Calculates current dimensionless age of H0t0 of the universe.
    """
    """
    Ω_q0, Ω_r0, Ω_m0 = density_parameters


    H = np.zeros(len(x))
    for i, _x in enumerate(x):
        integral = 1 + w_q[i+1:-1]
        I = np.trapz(integral)
        H[i] = np.sqrt( Ω_m0*_x**(-3) + Ω_r0*_x**(-4) + Ω_q0*np.exp(-3*I))


    """
    x = np.linspace(0, 1, len(H))
    H0t0 = np.trapz(1/(H))#, x=x)
    return H0t0


pow_init = [5e-5, 1e-8, 0.9999, 1e9]
exp_init = [0, 5e-13, 0.9999, 1.5]

Ni = -7*np.log(20)
N = np.linspace(Ni, 0, 10000)
z = np.exp(-N) - 1

pow_solution = solve(N, pow_init, Γ=2)
exp_solution = solve(N, exp_init, Γ=1)

w_pow = pow_solution[3]
Ω0_pow = pow_solution[:3, -1]

w_exp = exp_solution[3]
Ω0_exp = exp_solution[:3, -1]


H_exp = solve_H1(N, w_exp, Ω0_exp)
H_pow = solve_H1(N, w_pow, Ω0_pow)
H_Λ   = H_ΛCDM(N)




x = np.linspace(0,1,10000)
age_exp = age(H_exp)
age_pow = age(H_pow)
age_Λ   = age(H_Λ)

print(age_exp, age_pow, age_Λ)
