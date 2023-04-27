import numpy as np
from scipy.integrate import quad, solve_ivp
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
plt.style.use("bmh")
dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.dirname(dir)
dir = dir + "/latex/figs/"


class BBN:
    def __init__(self):
        self.τ    = 1700        # [s]
        self.q    = 2.53        # []
        self.h    = 0.7         # []
        self.H0   = 100*self.h  # [km s^-1 Mpc^-1 ]
        self.T0   = 2.725       # [K]
        self.N_eff = 3
        self.Ω_r0 = 8*np.pi**3/45 * const.G/self.H0**2 * (const.k*self.T0)**4 /(
                    const.hbar**3*const.c**5)*(1 + self.N_eff*7/8*(4/11)**(4/3))

    def t(self, T):
        """
        Cosmic time as a function of photon temperature
        """
        t = 1/(2*self.H0*np.sqrt(self.Ω_r0)) * (self.T0/T)**2
        return t

    def a(self, t):
        """
        Scale factor as function of cosmic time
        """
        a = np.sqrt(2*self.H0*t)*self.Ω_r0**(1/4)
        return a

    def H(self, t):
        """
        Hubble parameter as function of cosmic time
        """
        a = self.a(t)
        H = self.H0*np.sqrt(self.Ω_r0)/(a**2)
        return H

    def decay_rate_n2p(self, T, q):
        """
        Neutron-to-proton decay rate
        """
        Γ = quad(self.I_dec, 1, np.inf, args=(q,T))
        return Γ[0]

    def I_dec(self, x, q, T):
        Z  = 5.93/(T*1e-9)#const.m_e*const.c**2/(const.k*T)
        Tν = np.cbrt(4/11)*T
        Zν = 5.93/(Tν*1e-9)#const.m_e*const.c**2/(const.k*Tν)

        I_1 = (x+q)**2*np.sqrt(x**2-1)*x/((1+np.exp(x*Z))*(1+np.exp(-(x+q)*Zν)))
        I_2 = (x-q)**2*np.sqrt(x**2-1)*x/((1+np.exp(-x*Z))*(1+np.exp((x-q)*Zν)))
        I = (I_1 + I_2)/self.τ
        return I

    def density_equations(self, lnT, Y):
        """
        Differential equations for evolution of relative neutron/proton density
        """
        Y_n, Y_p = Y

        Γ_n2p = self.decay_rate_n2p(np.exp(lnT), self.q)
        Γ_p2n = self.decay_rate_n2p(np.exp(lnT), -self.q)

        H = self.H(self.t(np.exp(lnT)))

        dY_n = -(Y_p*Γ_p2n - Y_n*Γ_n2p)/H
        dY_p = -(Y_n*Γ_n2p - Y_p*Γ_p2n)/H

        return [dY_n, dY_p]

    def initial_conditions(self, T_i):
        """
        Calculates initial conditions for method density_equations
        """
        Y_n = 1/(1 + np.exp((const.m_n-const.m_p)*const.c**2/(const.k*T_i)))
        Y_p = 1 - Y_n
        return np.array([Y_n, Y_p])

    def solve(self, T_i, T_f):
        """
        Finds neutron/proton densities as function of photon temperature T
        """
        Y_init = self.initial_conditions(T_i)
        Y_eom  = self.density_equations
        lnT_i = np.log(T_i)
        lnT_f = np.log(T_f)

        self.sol = solve_ivp(Y_eom, [lnT_i, lnT_f], y0=Y_init, method="Radau",
                            rtol=1e-12, atol=1e-12)

    def plot_Y(self):
        """
        Plots relative densities of neutron/proton against temperature
        """
        T = np.exp(self.sol.t)
        Y_n, Y_p  = self.sol.y
        Yi_n, Yi_p = self.initial_conditions(T)

        fig, ax = plt.subplots(figsize=(8,5), tight_layout=True)
        ax.set_yscale("log")

        ax.semilogx(T, Y_n, "k", label="n")
        ax.semilogx(T, Yi_n, "k", linestyle="dotted")
        ax.semilogx(T, Y_p, "r", label="p")
        ax.semilogx(T, Yi_p, "r", linestyle="dotted")

        ax.set_ylim(1e-3, 2)
        ax.set_xlim(1e11, 1e8)
        ax.set_xlabel(r"$T$ [K]")
        ax.set_ylabel(r"$Y_i$")
        plt.legend()


if __name__=="__main__":

    model = BBN()
    model.solve(T_i=100e9, T_f=0.1e9)
    model.plot_Y()
    plt.savefig(dir+"densities.png")
