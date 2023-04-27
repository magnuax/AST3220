import numpy as np
from scipy.integrate import quad, solve_ivp
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
plt.style.use("bmh")
dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.dirname(dir)
dir = dir + "/latex/figs/"

class BBN:
    def __init__(self):
        self.set_physical_constants()
        # Parameters used to calculate reaction rates
        self.τ    = 1700                 # [s]
        self.q    = 2.53                 # [ ]
        # No. of neutrino species:
        self.N_eff = 3                   # [ ]
        # Critical density & baryon/radiation density parameters:
        self.ρ_c0 = 9.2e-27 * 1e-3   # [g/cm^3]
        self.Ω_b0 = 0.05
        self.Ω_r0 = 8*np.pi**3/45 * self.G/self.H0**2 * (self.k_B*self.T0)**4 /(
                    self.hbar**3*self.c**5)*(1 + self.N_eff*7/8*(4/11)**(4/3))

    def set_physical_constants(self):
        self.T0   = 2.725                       # [K]     Photon temp. today
        self.k_B  = const.k_B.cgs.value         # [erg/K] Boltzmann constant
        self.hbar = const.hbar.cgs.value        # [erg s] Reduced planck constant
        self.m_u  = const.u.cgs.value           # [g]     Atomic mass unit
        self.m_n  = const.m_n.cgs.value         # [g]     Neutron mass
        self.m_p  = const.m_p.cgs.value         # [g]     Proton mass
        self.c    = const.c.cgs.value           # [cm/s]  Speed of light
        self.G    = const.G.cgs.value           # [??]    Gravitational constant
        self.H0   = cosmo.H(0).cgs.value        # [s^-1]  Hubble constants


    def t(self, T):
        """
        Cosmic time as a function of photon temperature
        """
        t = 1/(2*self.H0*np.sqrt(self.Ω_r0)) * (self.T0/T)**2
        return t

    def a(self, t):
        """
        Scale factor as function of cosmic time [s]
        """
        a = np.sqrt(2*self.H0*t)*self.Ω_r0**(1/4) # [cm]
        return a

    def H(self, t):
        """
        Hubble parameter as function of cosmic time
        """
        a = self.a(t)
        H = self.H0*np.sqrt(self.Ω_r0)/(a**2)
        return H

    def rate_np(self, T, q):
        """
        Neutron-to-proton decay rate
        """
        Γ = quad(self.I_dec, 1, np.inf, args=(q,T))
        return Γ[0]

    def rate_pn_Dγ(self, T, ρ_b):
        T9 = T*1e-9
        pn = ρ_b*2.5e4
        λ = 4.68e9 * pn/ρ_b * T9**(1.5) * np.exp(-25.82/T9)
        return pn, λ

    def I_dec(self, x, q, T):
        Z  = 5.93/(T*1e-9)      #const.m_e*const.c**2/(const.k_B*T)
        Tν = np.cbrt(4/11)*T
        Zν = 5.93/(Tν*1e-9)     #const.m_e*const.c**2/(const.k_B*Tν)

        I_1 = (x+q)**2*np.sqrt(x**2-1)*x/((1+np.exp(x*Z))*(1+np.exp(-(x+q)*Zν)))
        I_2 = (x-q)**2*np.sqrt(x**2-1)*x/((1+np.exp(-x*Z))*(1+np.exp((x-q)*Zν)))
        I = (I_1 + I_2)/self.τ
        return I

    def density_equations(self, lnT, Y, pbar, state):
        """
        Differential equations for evolution of relative neutron/proton density
        (pbar & state are used to generate progress bar)
        """
        # Update progress bar:
        last_T, dT = state
        n = int((lnT-last_T)/dT)
        pbar.update(n)
        state[0] = last_T + dT * n

        # Find baryon density and reaction rates:
        Y_n, Y_p, Y_D = Y

        T = np.exp(lnT)                   # [K]       Photon temperature (CGS)
        t = self.t(T)                     # [s]       Cosmic time (CGS)
        H = self.H(t)                     # [s^-1]    Hubbøle parameter (CGS)
        a = self.a(t)                     # [cm]      Scale factor (CGS)
        ρ_b = self.Ω_b0*self.ρ_c0/(a**3)  # [g/cm**3] Baryon density (CGS)

        pn = ρ_b*2.5e4
        rate_np     = self.rate_np(T, self.q)
        rate_pn     = self.rate_np(T, -self.q)
        pn, rate_Dn = self.rate_pn_Dγ(T, ρ_b)

        # Weak interactions:
        dY_p = -1/H*(-Y_p*rate_pn + Y_n*rate_np)
        dY_n = -1/H*( Y_p*rate_pn - Y_n*rate_np)

        # Strong & EM interactions:
        dY_n = dY_n - 1/H*(-Y_p*Y_n*pn + Y_D*rate_Dn)
        dY_p = dY_p - 1/H*(-Y_p*Y_n*pn + Y_D*rate_Dn)
        dY_D = 0    - 1/H*( Y_p*Y_n*pn - Y_D*rate_Dn)



        return [dY_n, dY_p, dY_D]

    def initial_conditions(self, T_i):
        """
        Calculates initial conditions for method density_equations
        """
        Y_n = 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T_i)))
        Y_p = 1 - Y_n
        return [Y_n, Y_p]

    def solve(self, T_i, T_f):
        """
        Finds neutron/proton densities as function of photon temperature T
        """
        Y_ni, Y_pi = self.initial_conditions(T_i)
        Y_init = [Y_ni, Y_pi, 0]
        Y_eom  = self.density_equations
        lnT_i = np.log(T_i)
        lnT_f = np.log(T_f)
        lnT_span = [lnT_i, lnT_f]

        with tqdm(total=1000, unit="‰") as pbar:
            self.sol = solve_ivp(Y_eom, lnT_span,
                            y0 = Y_init,
                            args = [pbar, [lnT_i, (lnT_f-lnT_i)/1000]],
                            method = "Radau",
                            rtol = 1e-12,
                            atol = 1e-12)

        print(self.sol.message)


    def plot_Y(self):
        """
        Plots relative densities of neutron/proton against temperature
        """
        T = np.exp(self.sol.t)
        Y_n, Y_p, Y_D  = self.sol.y
        Yi_n, Yi_p = self.initial_conditions(T)

        fig, ax = plt.subplots(figsize=(8,5), tight_layout=True)
        ax.set_yscale("log")

        ax.semilogx(T, Y_n, "k", label="n")
        ax.semilogx(T, Yi_n, "k", linestyle="dotted")
        ax.semilogx(T, Y_p, "r", label="p")
        ax.semilogx(T, Yi_p, "r", linestyle="dotted")
        ax.semilogx(T, Y_D, "b", label="D")

        ax.set_ylim(1e-3, 2)
        ax.set_xlim(1e11, 1e8)
        ax.set_xlabel(r"$T$ [K]")
        ax.set_ylabel(r"$Y_i$")
        plt.legend()


if __name__=="__main__":

    model = BBN()
    model.solve(T_i=100e9, T_f=0.1e9)
    model.plot_Y()
    plt.savefig(dir+"densities_h.png")
