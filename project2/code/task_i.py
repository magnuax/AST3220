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

    def rate_pn_Dγ(self, T9, Tν9, ρ_b):
        # a1) p + n <-> D + γ
        pn = ρ_b*2.5e4
        λ = 4.68e9 * pn/ρ_b * T9**(1.5) * np.exp(-25.82/T9)
        return pn, λ

    def rate_pD_He3(self, T9, Tν9, ρ_b):
        # 2) p + D <-> He3 + γ
            return pD, λ

    def calculate_reaction_rates(self, T9, Tν9, ρ_b):
        """
        Calculates reaction rates for reactions (1)-(11) and ... in table 2
        """
        """Table 2a) Weak interactions"""         # 1) n + nu <-> p + e-
        # 2) n + e- <-> p + nu_bar
        # 3) n <-> p + e- + nu_bar
        self.rate_λw_n = quad(self.I_dec, 1, np.inf, args=(self.q,T9, Tν9))[0]
        self.rate_λw_p = quad(self.I_dec, 1, np.inf, args=(-self.q,T9, Tν9))[1]

        """Table 2b) Strong & EM interactions"""
        # a1) p + n <-> D + γ
        self.rate_pn   = 2.5e4 * ρ_b
        self.rate_λγ_D = 4.68e9 * self.rate_pn/ρ_b * T9**(1.5) * np.exp(-25.82/T9)
        # 2) p + D <-> He3 + γ
        self.rate_pD     = 2.23e3*ρ_b*T9**(-2/3)*np.exp(-3.72*T9**(-1/3))*(1 +
                           0.112*T9**(1/3) + 3.38*T9**(2/3) + 2.65*T9)
        self.rate_λγ_He3 = 1.63e10*self.rate_pD/ρ_b*T9**(1.5)*np.exp(-63.75/T9)
        # 3) n + D <-> T + γ
        self.rate_nD   = ρ_b*(75.5 + 1250*T9)
        self.rate_λγ_T = 1.63e10*self.rate_nD/ρ_b*T9**(1.5)*np.exp(-72.62/T9)
        # 4) n + He3 <-> p + T
        self.rate_nHe3_p = 7.06e8*ρ_b
        self.rate_pT_n   = self.rate_nHe3_p*np.exp(-8.864/T9)
        # 5) p + T <-> He4 + γ
        self.rate_pT_γ     = 2.87e4*ρ_b*T9**(-2/3)*np.exp(-3.87*T9**(-1/3))*(1 +
                             0.108*T9**(1/3) + 0.466*T9**(2/3) + 0.352*T9 +
                             0.300*T9**(4/3) + 0.576*T9**(5/3))
        self.rate_λγ_He4_p = 2.59e10*self.rate_pT_γ/ρ_b*T9**(1.5)*np.exp(-229.9/T9)
        # 6) n + He3 <-> He4 + γ
        self.rate_nHe3_γ   = 6e3*ρ_b*T9
        self.rate_λγ_He4_n = 2.6e10*self.rate_nHe3_γ/ρ_b*T9**(1.5)*np.exp(-238.8/T9)
        # 7) D + D <-> n + He3
        self.rate_DD_n   = 3.9e8*ρ_b*T9**(-2/3)*np.exp(-4.26*T9**(-1/3))*(1 +
                           0.0979*T9**(1/3) + 0.642*T9**(2/3) + 0.440*T9)
        self.rate_nHe3_D = 1.73*self.rate_DD_n*np.exp(-37.94/T9)
        # 8) D + D <-> p + T
        self.rate_DD_p = self.rate_DD_n
        self.rate_pT_D = 1.73*self.rate_DD_p*np.exp(-46.80/T9)
        # 9) D + D <-> He4 + γ
        self.rate_DD_γ     = 24.1*ρ_b*T9**(-2/3)*np.exp(-4.26*T9**(-1/3))*(T9**(2/3)
                             + 0.685*T9 + 0.152*T9**(4/3) + 0.265*T9**(5/3))
        self.rate_λγ_He4_D = 4.5e10*self.rate_DD_γ/ρ_b*T9**(1.5)*np.exp(-276/T9)
        # 10) D + He3 <-> He4 + p
        self.rate_DHe3 = 2.6e9*ρ_b*T9**(-1.5)*np.exp(-2.99/T9)
        self.rate_He4p = 5.5*self.rate_DHe3*np.exp(-213.0/T9)
        # 11) D + T <-> He4 + n
        self.rate_DT   = 1.38e9*ρ_b*T9**(-1.5)*np.exp(-0.745/T9)
        self.rate_He4n = 5.50*self.rate_DT*np.exp(-204.1 /T9)
        # 15) He3 + T <-> He4 + D
        self.rate_He3T_D = 3.88e9*ρ_b*T9**(-2/3)*np.exp(-7.72*T9**(-1/3))*(1 +
                           0.0540*T9**(1/3))
        self.rate_He4D   = 1.59*self.rate_He3T_D*np.exp(-166.2/T9)
        # 16) He3 + He4 <-> Be7 + +γ
        self.rate_He3He4 = 4.8e6*ρ_b*T9**(-2/3)*np.exp(-12.8*T9**(-1/3))*(1 +
                           0.0326*T9**(1/3) - 0.219*T9**(2/3) - 0.0499*T9 +
                           0.0258*T9**(4/3) + 0.015*T9**(5/3))
        self.rate_λγ_Be7 = 1.12e10*self.rate_He3He4/ρ_b*T9**(1.5)*np.exp(-18.42/T9)
        # 17) T + He4 <-> Li7 + γ
        self.rate_THe4   = 5.28e5*ρ_b*T9**(-2/3)*np.exp(-8.08*T9**(-1/3))*(1 +
                           0.0516*T9**(1/3))
        self.rate_λγ_Li7 = 1.12e10*self.rate_THe4/ρ_b*T9**(1.5)*np.exp(-28.63/T9)
        # 18) n + Be7 <-> p + Li7
        self.rate_nBe7_p = 6.74e9*ρ_b
        self.rate_pLi7_n = self.rate_nBe7_p*np.exp(-19.07/T9)
        # 20) p + Li7 <-> He4 + He4
        self.rate_pLi7_He4 = 1.42e9*ρ_b*T9**(-2/3)*np.exp(-8.47*T9**(-1/3))*(1 +
                             0.0493*T9**(1/3))
        self.rate_He4He4_p = 4.64*self.rate_pLi7_He4*np.exp(-201.3/T9)
        # 21) n + Be7 <-> He4 + He4
        self.rate_nBe7_He4 = 1.2e7*ρ_b*T9
        self.rate_He4He4_n = 4.64*self.rate_nBe7_He4*np.exp(-220.4/T9)






    def I_dec(self, x, q, T9, Tν9):
        Z  = 5.93/(T9)           #const.m_e*const.c**2/(const.k_B*T)
        Zν = 5.93/(Tν9)          #const.m_e*const.c**2/(const.k_B*Tν)

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
        Y_p, Y_n, Y_D, Y_T, Y_He3, Y_He4, Y_Li7, Y_Be7 = Y
        dY_p, dY_n, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7 = 0,0,0,0,0,0,0,0


        T = np.exp(lnT)                   # [K]       Photon temperature (CGS)
        t = self.t(T)                     # [s]       Cosmic time (CGS)
        H = self.H(t)                     # [s^-1]    Hubbøle parameter (CGS)
        a = self.a(t)                     # [cm]      Scale factor (CGS)
        ρ_b = self.Ω_b0*self.ρ_c0/(a**3)  # [g/cm**3] Baryon density (CGS)

        T9  = T*1e-9
        Tν9 = np.cbrt(4/11)*T9
        """
        rate_λw_p     = self.rate_np(T, self.q)
        rate_λw_n     = self.rate_np(T, -self.q)

        rate_pn, rate_λγ_D   = self.rate_pn_Dγ(T9, Tν9, ρ_b)
        rate_pD, rate_λγ_He3 = self.rate_pD_He3(T9, Tν9, ρ_b)

        pD, rate_pD
        rate_pLi7_He4, rate_He4He4_p = cmpt_pLi7_to_He4He4(T9, Tnu9, rhob)
        """
        self.calculate_reaction_rates(T9, Tν9, ρ_b)



        """ a) Weak interactions"""
        # 1) n + nu <-> p + e-
        # 2) n + e- <-> p + nu_bar
        # 3) n <-> p + e- + nu_bar
        dY_p = -1/H*(-Y_p*self.rate_λw_p + Y_n*self.rate_λw_n)
        dY_n = -1/H*( Y_p*self.rate_λw_p - Y_n*self.rate_λw_n)

        """ b) Strong & EM interactions"""
        # 1) p + n <-> D + γ
        dY_n -= 1/H*(-Y_p*Y_n*self.rate_pn + Y_D*self.rate_λγ_D)
        dY_p -= 1/H*(-Y_p*Y_n*self.rate_pn + Y_D*self.rate_λγ_D)
        dY_D -= 1/H*( Y_p*Y_n*self.rate_pn - Y_D*self.rate_λγ_D)
        # 2) p + D <-> He3 + γ
        dY_p   -= 1/H*(-Y_p*Y_D*self.rate_pD + Y_He3*self.rate_λγ_He3)
        dY_D   -= 1/H*(-Y_p*Y_D*self.rate_pD + Y_He3*self.rate_λγ_He3)
        dY_He3 -= 1/H*( Y_p*Y_D*self.rate_pD - Y_He3*self.rate_λγ_He3)
        # 3) n + D <-> T + γ
        dY_n -= 1/H*(-Y_n*Y_D*self.rate_nD + Y_T*self.rate_λγ_T)
        dY_D -= 1/H*(-Y_n*Y_D*self.rate_nD + Y_T*self.rate_λγ_T)
        dY_T -= 1/H*( Y_n*Y_D*self.rate_nD - Y_T*self.rate_λγ_T)
        # 4) n + He3 <-> p + T
        dY_n   -= 1/H*(-Y_n*Y_He3*self.rate_nHe3_p + Y_p*Y_T*self.rate_pT_n)
        dY_He3 -= 1/H*(-Y_n*Y_He3*self.rate_nHe3_p + Y_p*Y_T*self.rate_pT_n)
        dY_p   -= 1/H*( Y_n*Y_He3*self.rate_nHe3_p - Y_p*Y_T*self.rate_pT_n)
        dY_T   -= 1/H*( Y_n*Y_He3*self.rate_nHe3_p - Y_p*Y_T*self.rate_pT_n)
        # 5) p + T <-> He4 + γ
        dY_p   -= 1/H*(-Y_p*Y_T*self.rate_pT_γ + Y_He4*self.rate_λγ_He4_p)
        dY_T   -= 1/H*(-Y_p*Y_T*self.rate_pT_γ + Y_He4*self.rate_λγ_He4_p)
        dY_He4 -= 1/H*( Y_p*Y_T*self.rate_pT_γ - Y_He4*self.rate_λγ_He4_p)
        # 6) n + He3 <-> He4 + γ
        dY_n   -= 1/H*(-Y_n*Y_He3*self.rate_nHe3_γ + Y_He4*self.rate_λγ_He4_n)
        dY_He3 -= 1/H*(-Y_n*Y_He3*self.rate_nHe3_γ + Y_He4*self.rate_λγ_He4_n)
        dY_He4 -= 1/H*( Y_n*Y_He3*self.rate_nHe3_γ - Y_He4*self.rate_λγ_He4_n)
        # 7) D + D <-> n + He3
        dY_D   -= 1/H*(   -Y_D*Y_D*self.rate_DD_n + 2*Y_n*Y_He3*self.rate_nHe3_D)
        dY_n   -= 1/H*(0.5*Y_D*Y_D*self.rate_DD_n -   Y_n*Y_He3*self.rate_nHe3_D)
        dY_He3 -= 1/H*(0.5*Y_D*Y_D*self.rate_DD_n -   Y_n*Y_He3*self.rate_nHe3_D)
        # 8) D + D <-> p + T
        dY_D -= 1/H*(   -Y_D*Y_D*self.rate_DD_p + 2*Y_p*Y_T*self.rate_pT_D)
        dY_p -= 1/H*(0.5*Y_D*Y_D*self.rate_DD_p -   Y_p*Y_T*self.rate_pT_D)
        dY_T -= 1/H*(0.5*Y_D*Y_D*self.rate_DD_p -   Y_p*Y_T*self.rate_pT_D)
        # 9) D + D <-> He4 + γ
        dY_D   -= 1/H*(   -Y_D*Y_D*self.rate_DD_γ + 2*Y_He4*self.rate_λγ_He4_D)
        dY_He4 -= 1/H*(0.5*Y_D*Y_D*self.rate_DD_γ -   Y_He4*self.rate_λγ_He4_D)
        # 10) D + He3 <-> He4 + p
        dY_D   -= 1/H*(-Y_D*Y_He3*self.rate_DHe3 + Y_He4*Y_p*self.rate_He4p)
        dY_He3 -= 1/H*(-Y_D*Y_He3*self.rate_DHe3 + Y_He4*Y_p*self.rate_He4p)
        dY_He4 -= 1/H*( Y_D*Y_He4*self.rate_DHe3 - Y_He4*Y_p*self.rate_He4p)
        dY_p   -= 1/H*( Y_D*Y_He3*self.rate_DHe3 - Y_He4*Y_p*self.rate_He4p)
        # 11) D + T <-> He4 + n
        dY_D   -= 1/H*(-Y_D*Y_T*self.rate_DT + Y_He4*Y_n*self.rate_He4n)
        dY_T   -= 1/H*(-Y_D*Y_T*self.rate_DT + Y_He4*Y_n*self.rate_He4n)
        dY_He4 -= 1/H*( Y_D*Y_T*self.rate_DT - Y_He4*Y_n*self.rate_He4n)
        dY_n   -= 1/H*( Y_D*Y_T*self.rate_DT - Y_He4*Y_n*self.rate_He4n)
        # 15) He3 + T <-> He4 + D
        dY_He3 -= 1/H*(-Y_He3*Y_T*self.rate_He3T_D + Y_He4*Y_D*self.rate_He4D)
        dY_T   -= 1/H*(-Y_He3*Y_T*self.rate_He3T_D + Y_He4*Y_D*self.rate_He4D)
        dY_He3 -= 1/H*( Y_He3*Y_T*self.rate_He3T_D - Y_He4*Y_D*self.rate_He4D)
        dY_D   -= 1/H*( Y_He3*Y_T*self.rate_He3T_D - Y_He4*Y_D*self.rate_He4D)
        # 16) He3 + He4 <-> Be7 + +γ
        dY_He3 -= 1/H*(-Y_He3*Y_He4*self.rate_He3He4 + Y_Be7*self.rate_λγ_Be7)
        dY_He4 -= 1/H*(-Y_He3*Y_He4*self.rate_He3He4 + Y_Be7*self.rate_λγ_Be7)
        dY_Be7 -= 1/H*( Y_He3*Y_He4*self.rate_He3He4 - Y_Be7*self.rate_λγ_Be7)
        # 17) T + He4 <-> Li7 + γ
        dY_T   -= 1/H*(-Y_T*Y_He4*self.rate_THe4 + Y_Li7*self.rate_λγ_Li7)
        dY_He4 -= 1/H*(-Y_T*Y_He4*self.rate_THe4 + Y_Li7*self.rate_λγ_Li7)
        dY_Li7 -= 1/H*( Y_T*Y_He4*self.rate_THe4 - Y_Li7*self.rate_λγ_Li7)
        # 18) n + Be7 <-> p + Li7
        dY_n   -= 1/H*(-Y_n*Y_Be7*self.rate_nBe7_p + Y_p*Y_Li7*self.rate_pLi7_n)
        dY_Be7 -= 1/H*(-Y_n*Y_Be7*self.rate_nBe7_p + Y_p*Y_Li7*self.rate_pLi7_n)
        dY_p   -= 1/H*( Y_n*Y_Be7*self.rate_nBe7_p - Y_p*Y_Li7*self.rate_pLi7_n)
        dY_Li7 -= 1/H*( Y_n*Y_Be7*self.rate_nBe7_p - Y_p*Y_Li7*self.rate_pLi7_n)
        # 20) p + Li7 <-> He4 + He4
        dY_p   -= 1/H*( -Y_p*Y_Li7*self.rate_pLi7_He4 + 0.5*Y_He4*Y_He4*self.rate_He4He4_p)
        dY_Li7 -= 1/H*( -Y_p*Y_Li7*self.rate_pLi7_He4 + 0.5*Y_He4*Y_He4*self.rate_He4He4_p)
        dY_He4 -= 1/H*(2*Y_p*Y_Li7*self.rate_pLi7_He4 -     Y_He4*Y_He4*self.rate_He4He4_p)
        # 21) n + Be7 <-> He4 + He4
        dY_n   -= 1/H*( -Y_n*Y_Be7*self.rate_nBe7_He4 + 0.5*Y_He4*Y_He4*self.rate_He4He4_n)
        dY_Be7 -= 1/H*( -Y_n*Y_Be7*self.rate_nBe7_He4 + 0.5*Y_He4*Y_He4*self.rate_He4He4_n)
        dY_He4 -= 1/H*(2*Y_n*Y_Be7*self.rate_nBe7_He4 +     Y_He4*Y_He4*self.rate_He4He4_n)

        return [dY_p, dY_n, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7]

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
        Y_init = [Y_ni, Y_pi, 0, 0, 0, 0, 0, 0]
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
        Y_p, Y_n, Y_D, Y_T, Y_He3, Y_He4, Y_Li7, Y_Be7  = self.sol.y
        Yi_n, Yi_p = self.initial_conditions(T)

        fig, ax = plt.subplots(figsize=(8,5), tight_layout=True)
        ax.set_yscale("log")

        ax.semilogx(T, Y_n, "k", label="n")
        ax.semilogx(T, Yi_n, "k", linestyle="dotted")
        ax.semilogx(T, Y_p, "r", label="p")
        ax.semilogx(T, Yi_p, "r", linestyle="dotted")
        ax.semilogx(T, Y_D, "b", label="D")
        ax.semilogx(T, Y_T, label="Y_T")
        ax.semilogx(T, Y_He3, label="Y_He3")
        ax.semilogx(T, Y_He4, label="Y_He4")
        ax.semilogx(T, Y_Li7, label="Y_Li7")
        ax.semilogx(T, Y_Be7, label="Y_Be7")


        ax.set_ylim(1e-3, 1)
        ax.set_xlim(1e11, 1e7)
        ax.set_xlabel(r"$T$ [K]")
        ax.set_ylabel(r"$Y_i$")
        plt.legend()


if __name__=="__main__":

    model = BBN()
    model.solve(T_i=100e9, T_f=0.1e9)
    model.plot_Y()
    plt.savefig(dir+"densities_i.png")
