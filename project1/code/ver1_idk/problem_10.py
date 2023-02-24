import numpy as np
import matplotlib.pyplot as plt
from eom import eom, solve
from scipy import integrate
plt.style.use("ggplot")


def eos_to_func(w_q, N_arr):

    def eos(N_val):
        idx = np.argwhere(N_arr>=N_val)[0][0]
        return w_q[idx]

    eos = np.vectorize(eos)

    return eos

def H_ΛCDM(N, Ω_m0=0.3):
    """
    Calculates dimensionless Hubble parameter H/H0 for ΛCDM model.
    """
    H = np.sqrt( Ω_m0*np.exp(-3*N) + (1-Ω_m0))
    return H

def I(N, N_array):
    idx = np.argwhere()


def solve_H(N, w_q, density_parameters):
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




if __name__ == "__main__":
    # Define initial parameters and time/redshift arrays:
    pow_init = [5e-5, 1e-8, 0.9999, 1e9]
    exp_init = [0, 5e-13, 0.9999, 1.5]

    Ni = -np.log(1+2e7)
    N = np.linspace(Ni, 0, 10000)
    z = np.exp(-N) - 1

    # Solve for equations of state & density parameters
    pow_solution = solve(N, pow_init, Γ=2)
    exp_solution = solve(N, exp_init, Γ=1)

    w_pow = pow_solution[3]
    Ω0_pow = pow_solution[:3, -1]

    w_exp = exp_solution[3]
    Ω0_exp = exp_solution[:3, -1]


    # Calculate dimensionless Hubble parameters H/H0
    H_exp = solve_H(N, w_exp, Ω0_exp)
    H_pow = solve_H(N, w_pow, Ω0_pow)
    H_Λ   = H_ΛCDM(N)


    print(H_pow[-1], H_exp[-1], H_Λ[-1])


    # Various plotting mumbo jumbo
    fig, ax = plt.subplots(figsize=(8,5))

    fig.supylabel("h km s^-1 Mpc^-1")
    fig.suptitle(r"$H(t)/H_{0}$")

    ax.plot(z, H_exp,label="H_exp")
    ax.plot(z, H_pow, label="H_pow")
    ax.plot(z, H_Λ, label="H_ΛCDM")

    ax.set_xlabel("Redshift z")

    ax.invert_xaxis()
    plt.legend()
    plt.show()
