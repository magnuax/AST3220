import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from eom import eom, solve

pow_init = [5e-5, 1e-8, 0.9999, 1e9]
exp_init = [0, 5e-13, 0.9999, 1.5]

Ni = -np.log(1+2e7)
N = np.linspace(Ni, 0, 10000)
z = np.exp(-N) - 1


def plot(Ω_q, Ω_r, Ω_m, ax):
    """
    Self explanatory - Plots density parameters against redshift. Also plots
    the sum of the density parameters (=1) for quality control.
    """
    Ω = Ω_q + Ω_r + Ω_m
    ax.plot(z, Ω, "--k", label=r"$\sum_i Ω_i$")
    ax.plot(z, Ω_q, label=r"$\Omega_{\phi}$")
    ax.plot(z, Ω_r, label=r"$\Omega_{r}$")
    ax.plot(z, Ω_m, label=r"$\Omega_{m}$")

    ax.set_ylabel(r"$\Omega_i(z)$")

exp_solution = solve(N, exp_init, Γ=1)
pow_solution = solve(N, pow_init, Γ=2)

Ω_pow = pow_solution[:3]
w_pow = pow_solution[-1]

Ω_exp = exp_solution[:3]
w_exp = exp_solution[-1]

fig, ax = plt.subplots(nrows=2, figsize=(8,5), sharex=True, tight_layout=True)
fig.suptitle("Density parameters")
ax[1].set_xlim(z[0], z[-1])

ax[0].set_title(r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")
plot(*Ω_pow, ax[0])

#fig, ax = plt.figure(figsize=(8,5))
ax[1].set_title(r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
plot(*Ω_exp, ax[1])


ax[1].set_xlabel("Redshift z")
plt.legend()

fig, ax = plt.subplots(figsize=(8,5))
ax.set_title(r"Equation of state $w_{\phi}$")

ax.set_xlim(z[0], z[-1])
ax.plot(z, w_exp, label=r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$") #, \ \ \zeta=\frac{3}{2}$")
ax.plot(z, w_pow, label=r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")  #, \ \ \alpha=1$")

ax.set_xlabel("Redshift z")
plt.legend()
plt.show()
