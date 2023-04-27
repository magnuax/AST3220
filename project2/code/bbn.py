import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
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

    def calculate_reaction_rates(self, T9, Tν9, ρ_b):
        """
        Calculates reaction rates for reactions (1)-(11) and ... in table 2
        """
        """Table 2a) Weak interactions"""
        # 1) n + nu <-> p + e-
        # 2) n + e- <-> p + nu_bar
        # 3) n <-> p + e- + nu_bar
        self.rate_λw_n = quad(self.I_dec, 1, np.inf, args=(self.q,T9, Tν9))[0]
        self.rate_λw_p = quad(self.I_dec, 1, np.inf, args=(-self.q,T9, Tν9))[0]

        """Table 2b) Strong & EM interactions"""
        # 1) p + n <-> D + γ
        self.rate_pn   = 2.5e4*ρ_b
        self.rate_λγ_D = 4.68e9*self.rate_pn/ρ_b*T9**(1.5)*np.exp(-25.82/T9)
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
        self.rate_λγ_He4_D = 4.5e10*self.rate_DD_γ/ρ_b*T9**(1.5)*np.exp(-276.7/T9)
        # 10) D + He3 <-> He4 + p
        self.rate_DHe3 = 2.6e9*ρ_b*T9**(-1.5)*np.exp(-2.99/T9)
        self.rate_He4p = 5.50*self.rate_DHe3*np.exp(-213.0/T9)
        # 11) D + T <-> He4 + n
        self.rate_DT   = 1.38e9*ρ_b*T9**(-1.5)*np.exp(-0.745/T9)
        self.rate_He4n = 5.50*self.rate_DT*np.exp(-204.1/T9)
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

    def density_equations(self, lnT, Y, Ω_b0, pbar, state):
        """
        Differential equations for evolution of relative neutron/proton density
        (pbar & state are used to generate progress bar,
        https://stackoverflow.com/questions/59047892/how-to-monitor-the-process-of-scipy-odeint)
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
        ρ_b = Ω_b0*self.ρ_c0/(a**3)       # [g/cm**3] Baryon density (CGS)

        T9  = T*1e-9
        Tν9 = np.cbrt(4/11)*T9

        self.calculate_reaction_rates(T9, Tν9, ρ_b)

        """ a) Weak interactions"""
        # 1) n + nu <-> p + e-
        # 2) n + e- <-> p + nu_bar
        # 3) n <-> p + e- + nu_bar
        dY_p = -1/H*(-Y_p*self.rate_λw_p + Y_n*self.rate_λw_n)
        dY_n = -1/H*( Y_p*self.rate_λw_p - Y_n*self.rate_λw_n)

        """ b) Strong & EM interactions"""
        # 1) p + n <-> D + γ
        dY_p -= 1/H*(-Y_p*Y_n*self.rate_pn + Y_D*self.rate_λγ_D)
        dY_n -= 1/H*(-Y_p*Y_n*self.rate_pn + Y_D*self.rate_λγ_D)
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
        dY_He4 -= 1/H*( Y_D*Y_He3*self.rate_DHe3 - Y_He4*Y_p*self.rate_He4p)
        dY_p   -= 1/H*( Y_D*Y_He3*self.rate_DHe3 - Y_He4*Y_p*self.rate_He4p)
        # 11) D + T <-> He4 + n
        dY_D   -= 1/H*(-Y_D*Y_T*self.rate_DT + Y_He4*Y_n*self.rate_He4n)
        dY_T   -= 1/H*(-Y_D*Y_T*self.rate_DT + Y_He4*Y_n*self.rate_He4n)
        dY_He4 -= 1/H*( Y_D*Y_T*self.rate_DT - Y_He4*Y_n*self.rate_He4n)
        dY_n   -= 1/H*( Y_D*Y_T*self.rate_DT - Y_He4*Y_n*self.rate_He4n)
        # 15) He3 + T <-> He4 + D
        dY_He3 -= 1/H*(-Y_He3*Y_T*self.rate_He3T_D + Y_He4*Y_D*self.rate_He4D)
        dY_T   -= 1/H*(-Y_He3*Y_T*self.rate_He3T_D + Y_He4*Y_D*self.rate_He4D)
        dY_He4 -= 1/H*( Y_He3*Y_T*self.rate_He3T_D - Y_He4*Y_D*self.rate_He4D)
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
        Calculates initial conditions for density_equations
        """
        Y_n = 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T_i)))
        Y_p = 1 - Y_n
        return [Y_p, Y_n, 0, 0, 0, 0, 0, 0]

    def solve(self, T_i, T_f, Ω_b0=0.05, save=False):
        """
        Finds mass fractions as function of photon temperature T
        """
        Y_init = self.initial_conditions(T_i)
        Y_eom  = self.density_equations
        lnT_i = np.log(T_i)
        lnT_f = np.log(T_f)
        lnT_span = [lnT_i, lnT_f]

        with tqdm(total=1000, unit="‰", colour="green") as pbar:
            self.sol = solve_ivp(Y_eom, lnT_span,
                            y0 = Y_init,
                            args = [Ω_b0, pbar, [lnT_i, (lnT_f-lnT_i)/1000]],
                            method = "Radau",
                            rtol = 1e-12,
                            atol = 1e-12)

        if save:
            mass_fractions = [self.sol.t, *self.sol.y]
            np.save("mass_fractions.npy", mass_fractions)

    def import_solution(self, filename="mass_fractions.npy"):
        self.sol = np.load(filename)
        self.log_T = self.sol[0]
        self.Y     = self.sol[1:]

    def plot_mass_fractions(self):
        """
        Plots relative densities of neutron/proton against temperature
        """
        sol = np.load("mass_fractions.npy")
        T = np.exp(self.log_T)

        # Create arrays with rel. densities, atomic number and element names
        Y_i = self.Y
        A_i = np.array([1, 1, 2, 3, 3, 4, 7, 7])
        X_i = np.array([r"$n$"   , r"$p$"   , r"$D$"   , r"$T$",
                        r"$He^3$", r"$He^4$", r"$Li^7$", r"$Be^7$"])
        # Use numpy broadcasting to calculate mass fraction
        AiYi = A_i[:,None]*Y_i

        fig, ax = plt.subplots(figsize=(8,5), tight_layout=True)
        ax.set_yscale("log")

        ax.semilogx(T, np.sum(AiYi, axis=0), "--k", label=r"$\Sigma_i A_i Y_i$")

        for A, Y, X in zip(A_i, Y_i, X_i):
            ax.semilogx(T, A*Y, label=X)

        ax.set_ylim(1e-11, 10)
        ax.set_xlim(1e11, 1e7)
        ax.set_xlabel(r"$T$ [K]")
        ax.set_ylabel(r"Mass fraction $A_i Y_i$")
        plt.legend()

    def calculate_relic_abundances(self):
        N = 10

        T_i = 100e9
        T_f = 0.01e9

        self.Y0 = np.zeros((8,N))

        Ω_b0 = np.linspace(0.01, 1, N)
        for i, _Ω in enumerate(Ω_b0):
            print(f"Calculating relic abundance {i}/20 ... Ω_b0={_Ω} ")
            self.solve(T_i, T_f, _Ω)
            self.Y0[:,i] = self.sol.y[:,-1]

        print("Finished!")

        # Save results as .npy file:
        Y_p   = self.Y0[0,:]
        Y_n   = self.Y0[1,:]
        Y_D   = self.Y0[2,:]
        Y_T   = self.Y0[3,:]
        Y_He3 = self.Y0[4,:]
        Y_He4 = self.Y0[4,:]
        Y_Li7 = self.Y0[6,:]
        Y_Be7 = self.Y0[7,:]

        relic_abundances = np.array([Ω_b0,  Y_p,   Y_n,   Y_D,   Y_T,
                                            Y_He3, Y_He4, Y_Li7, Y_Be7])
        np.save("relic_abundances.npy", relic_abundances)

    def chi_square(self, data, model, error):
        s = (model - data)**2/(error**2)
        return np.sum(s)

    def plot_relic_abundances(self):
        Y0 = np.load("relic_abundances.npy")

        Y0 = np.where(Y0<1e-20, 1e-20, Y0)

        Ω_b0  = Y0[0,:]
        Y_p   = Y0[1,:]
        Y_n   = Y0[2,:]
        Y_D   = Y0[3,:]
        Y_T   = Y0[4,:]
        Y_He3 = Y0[5,:]
        Y_He4 = Y0[6,:]
        Y_Li7 = Y0[7,:]
        Y_Be7 = Y0[8,:]

        Y_Li7 = Y_Li7 + Y_Be7
        Y_He3 = Y_He3 + Y_T

        logY_D_p   = interp1d(Ω_b0, np.log(Y_D/Y_p)  , kind="cubic")
        logY_Li7_p = interp1d(Ω_b0, np.log(Y_Li7/Y_p), kind="cubic")
        logY_4xHe4 = interp1d(Ω_b0, np.log(4*Y_He4)  , kind="cubic")
        logY_He3   = interp1d(Ω_b0, np.log(Y_He3/Y_p), kind="cubic")

        N = 1000
        Ω_b0 = np.linspace(0.01, 1, N)

        Y_D_p   = np.exp( logY_D_p(Ω_b0)   )
        Y_Li7_p = np.exp( logY_Li7_p(Ω_b0) )
        Y_4xHe4 = np.exp( logY_4xHe4(Ω_b0) )
        Y_He3   = np.exp( logY_He3(Ω_b0)   )

        model = np.array([Y_D_p, Y_Li7_p, Y_4xHe4])
        error = np.array([0.03e-5, 0.3e-10, 0.003])
        data  = np.array([2.57e-5, 1.6e-10, 0.254])

        χ = np.zeros(N)
        for i in range(N):
            χ[i] = self.chi_square(data, model[:,i], error)

        #χ = np.sqrt(χ)

        P = 1/np.sqrt(2*np.pi*np.prod(error**2)) * np.exp(-0.5*χ)
        P = P/np.max(P)

        idx = np.argmax(P)
        Ω_best = Ω_b0[idx]

        fig, ax = plt.subplots(nrows=3,
                            sharex=True,
                            tight_layout=True,
                            figsize=(7,8),
                            gridspec_kw={"height_ratios":[1,3,1]})

        ax[0].semilogx([Ω_best, Ω_best], [0.325, 0.20], ":k")
        ax[1].semilogx([Ω_best, Ω_best], [1e-11, 1e-3], ":k")

        ax[0].semilogx(Ω_b0, Y_4xHe4, label="$\mathrm{He}^4$")
        ax[1].semilogx(Ω_b0, Y_Li7_p, label="$\mathrm{Li}^7$")
        ax[1].semilogx(Ω_b0, Y_D_p  , label="$\mathrm{D}$"   )
        ax[1].semilogx(Ω_b0, Y_He3  , label="$\mathrm{He}^3$")
        ax[2].semilogx(Ω_b0, P, color="k")

        """
        ax[0].fill_between([Ω_b0[0], Ω_b0[-1]], [data[0]-error[0], data[0]+error[0]])
        ax[1].fill_between([Ω_b0[0], Ω_b0[-1]], [data[1]-error[1], data[1]+error[1]])
        ax[1].fill_between([Ω_b0[0], Ω_b0[-1]], [data[2]-error[2], data[2]+error[2]])
        """

        ax[0].set_ylabel(r"$4Y_{\mathrm{He}^4}$")
        ax[1].set_ylabel(r"$Y_{i} / Y_{p}$")
        ax[2].set_ylabel(f"Normalized\nprobability")
        ax[2].set_xlabel(r"$\Omega_{b0}$")

        ax[1].set_yscale("log")
        ax[1].set_ylim(1e-11, 1e-3)
        ax[2].set_xlim(1e-2 , 1e0)

        ax[0].legend()
        ax[1].legend()

        plt.show()


if __name__=="__main__":

    model = BBN()
    """
    model.solve(T_i=100e9, T_f=0.01e9)
    model.import_solution()
    model.plot_mass_fractions()
    plt.savefig(dir+"densities_i.png")
    """
    model.calculate_relic_abundances()
    model.plot_relic_abundances()
