import numpy as np
from scipy.integrate import odeint


def eom(x_init, N, Γ):
    """
    Calculates rhs of equations of motion. Takes vector X(N) = [x1, x2, x3, λ] and
    time variable N as input.
    """
    x1, x2, x3, λ = x_init

    Hx = 0.5*(3 + 3*x1**2 - 3*x2**2 + x3**2)

    d1 = -3*x1 + np.sqrt(1.5)*λ*x2**2 + x1*Hx
    d2 = -np.sqrt(1.5)*λ*x1*x2 + x2*Hx
    d3 = -2*x3 + x3*Hx

    dλ = -np.sqrt(6)*(λ**2)*(Γ-1)*x1

    dqdN = np.array([d1, d2, d3, dλ])

    return dqdN

def solve(N, x_init, Γ):
    """
    Solves equations of motion and returns density parameters and the quintessence
    equation of state parameter.
    """
    sol = odeint(eom, x_init, N, args=(Γ,))
    x1, x2, x3 = sol[:,0], sol[:,1], sol[:,2]

    # Calculate density parameters for quintessence (q), radiation (r) and mass (m):
    Ω_q = x1**2 + x2**2
    Ω_r = x3**2
    Ω_m = 1 - Ω_q - Ω_r

    w_q = (x1**2 - x2**2)/(x1**2 + x2**2)

    return np.array([Ω_q, Ω_r, Ω_m, w_q])

def solve_H(N, x_init, Γ):
    sol = odeint(eom, x_init, N, args=(Γ,))
    x1, x2, x3 = sol[:,0], sol[:,1], sol[:,2]


    """

    BRUK SOLVE IVP

    """
    #H = np.exp(1.5*(1 + x1**2 - x2**2 + 1/3*x3**2) )
    H = np.exp(0.5*(3 + 3*x1**2 - 3*x2**2 + x3**2))
    return H
