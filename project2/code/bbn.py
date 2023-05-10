import numpy as np
from scipy.integrate import quad, simpson, solve_ivp
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from time import perf_counter
import warnings
import matplotlib as mpl

class BigBangNucleosynthesis:
    def __init__(self):
        self._set_physical_constants()
        # Parameters used to calculate reaction rates
        self.τ    = 1700                 # [s]
        self.q    = 2.53                 # [ ]
        # Effective no. of neutrino species:
        self.N_eff = 3                   # [ ]
        # Critical density & baryon/radiation density parameters:
        self.ρ_c0 = 9.2e-27 * 1e-3   # [g/cm^3]
        self.Ω_r0 = 8*np.pi**3/45 * self.G/self.H0**2 * (self.k_B*self.T0)**4 /(
                    self.hbar**3*self.c**5)*(1 + self.N_eff*7/8*(4/11)**(4/3))

    def _set_physical_constants(self):
        self.T0   = 2.725                       # [K]     Photon temp. today
        self.k_B  = const.k_B.cgs.value         # [erg/K] Boltzmann constant
        self.hbar = const.hbar.cgs.value        # [erg s] Reduced planck constant
        self.m_u  = const.u.cgs.value           # [g]     Atomic mass unit
        self.m_n  = const.m_n.cgs.value         # [g]     Neutron mass
        self.m_p  = const.m_p.cgs.value         # [g]     Proton mass
        self.c    = const.c.cgs.value           # [cm/s]  Speed of light
        self.G    = const.G.cgs.value           # [??]    Gravitational constant
        self.H0   = cosmo.H(0).cgs.value        # [s^-1]  Hubble constants

    def _get_cosmic_time(self, T):
        """
        Calculates cosmic time as a function of photon temperature
        """
        t = 1/(2*self.H0*np.sqrt(self.Ω_r0)) * (self.T0/T)**2
        return t

    def _get_scale_factor(self, t):
        """
        Calculates scale factor as function of cosmic time [s]
        """
        a = np.sqrt(2*self.H0*t)*self.Ω_r0**(1/4) # [cm]
        return a

    def _get_hubble_parameter(self, t):
        """
        Calculates Hubble parameter as function of cosmic time
        """
        a = self._get_scale_factor(t)
        H = self.H0*np.sqrt(self.Ω_r0)/(a**2)
        return H

    def _calculate_reaction_rates(self, T9, Tν9, ρ_b):
        """
        Calculates reaction rates for reactions (1)-(11) and ... in table 2
        """
        """Table 2a) Weak interactions"""
        # 1) n + nu <-> p + e-
        # 2) n + e- <-> p + nu_bar
        # 3) n <-> p + e- + nu_bar
        x = np.linspace(1,250,10001)
        I_dec_n = self._I_dec(x, self.q, T9, Tν9)
        I_dec_p = self._I_dec(x,-self.q, T9, Tν9)

        self.rate_λw_n = simpson(I_dec_n, x=x) #quad(self._I_dec, 1, 250, args=(self.q,T9, Tν9))[0]
        self.rate_λw_p = simpson(I_dec_p, x=x) #quad(self._I_dec, 1, 250, args=(-self.q,T9, Tν9))[0]

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

    def _I_dec(self, x, q, T9, Tν9):
        Z  = 5.93/(T9)           #const.m_e*const.c**2/(const.k_B*T)
        Zν = 5.93/(Tν9)          #const.m_e*const.c**2/(const.k_B*Tν)

        I_1 = (x+q)**2*np.sqrt(x**2-1)*x/((1+np.exp(x*Z))*(1+np.exp(-(x+q)*Zν)))
        I_2 = (x-q)**2*np.sqrt(x**2-1)*x/((1+np.exp(-x*Z))*(1+np.exp((x-q)*Zν)))
        I = (I_1 + I_2)/self.τ
        return I

    def _density_equations(self, lnT, Y, Ω_b0, pbar, state):
        """
        Differential equations for evolution of relative neutron/proton density
        (pbar & state are used to generate progress bar, thanks to this guy:
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
        t = self._get_cosmic_time(T)                     # [s]       Cosmic time (CGS)
        H = self._get_hubble_parameter(t)                     # [s^-1]    Hubbøle parameter (CGS)
        a = self._get_scale_factor(t)                     # [cm]      Scale factor (CGS)
        ρ_b = Ω_b0*self.ρ_c0/(a**3)       # [g/cm**3] Baryon density (CGS)

        T9  = T*1e-9
        Tν9 = np.cbrt(4/11)*T9

        self._calculate_reaction_rates(T9, Tν9, ρ_b)

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

    def _get_initial_conditions(self, T_i):
        """
        Calculates initial conditions for density_equations
        """
        Y_n = 1/(1 + np.exp((self.m_n-self.m_p)*self.c**2/(self.k_B*T_i)))
        Y_p = 1 - Y_n
        return [Y_p, Y_n, 0, 0, 0, 0, 0, 0]

    def solve(self, T_i, T_f, Ω_b0=0.05, filename=None, pbar_desc=None):
        """
        Finds mass fractions as function of photon temperature T
        """
        Y_init = self._get_initial_conditions(T_i)
        Y_eom  = self._density_equations
        lnT_i = np.log(T_i)
        lnT_f = np.log(T_f)
        lnT_span = [lnT_i, lnT_f]

        with tqdm(total=1000, unit="‰", colour="green", leave=False, desc=pbar_desc) as pbar:
            self.sol = solve_ivp(Y_eom, lnT_span,
                            y0 = Y_init,
                            args = [Ω_b0, pbar, [lnT_i, (lnT_f-lnT_i)/1000]],
                            method = "Radau",
                            rtol = 1e-12,
                            atol = 1e-12)

        if filename:
            self._store_solution(self.sol.t, self.sol.y, filename)
        else:
            return self.sol.t, self.sol.y

    def _store_solution(self, param, number_densities, filename):
        """
        For <number_densities> as function of <param>, stores solution arrays
        as a .npy file.
        """
        solution = np.array([param, *number_densities])
        np.save(filename, solution)
        print(f"Results saved to file <{filename}>")

    def calculate_relic_abundances_Ω_b0(self, filename):
        """
        Calculates relic abundances for different values of <Ω_b0>
        """
        N   = 10
        T_i = 100e9
        T_f = 0.01e9

        Y_relic = np.zeros((8,N))
        Ω_b0    = np.geomspace(0.01, 1, N)

        t_start = perf_counter()
        print("\nCalulculating relic abundances:\n")
        for i, Ω in enumerate(Ω_b0):
            #print(f"Calculating relic abundance {i+1}/{N} ... Ω_b0={Ω}", end="\r")
            pbar_desc = f"Iteration {i+1}/{N},  Ω_b0={Ω:.4f}"
            sol_t, sol_y = self.solve(T_i, T_f, Ω_b0=Ω, pbar_desc=pbar_desc)
            Y_relic[:,i] = sol_y[:,-1]

        t_stop = perf_counter()
        print(f"Done, time elapsed: {t_stop-t_start} seconds")

        self._store_solution(Ω_b0, Y_relic, filename)

    def calculate_relic_abundances_N_eff(self, filename):
        """
        Calculates relic abundances for different values of <N_eff>
        """
        N   = 10
        T_i = 100e9
        T_f = 0.01e9

        Y_relic = np.zeros((8,N))
        N_eff   = np.linspace(1,5,N)

        t_start = perf_counter()
        print("\nCalulculating relic abundances:\n")
        for i, _N in enumerate(N_eff):
            pbar_desc = f"Iteration {i+1}/{N},  N={_N:.4f}"
            self.N_eff = _N                   # [ ]
            self.Ω_r0 = 8*np.pi**3/45 * self.G/self.H0**2 * (self.k_B*self.T0)**4 /(
                        self.hbar**3*self.c**5)*(1 + self.N_eff*7/8*(4/11)**(4/3))

            sol_t, sol_y = self.solve(T_i, T_f, pbar_desc=pbar_desc)
            Y_relic[:,i] = sol_y[:,-1]

        t_stop = perf_counter()
        print(f"Done, time elapsed: {t_stop-t_start} seconds")
        self._store_solution(N_eff, Y_relic, filename)
        self.N_eff = 3

    def _chi_square(self, data, model, error):
        """
        Calculates chi^2, used to find optimal values of <Ω_b0> and <N_eff>
        """
        s = (model - data)**2/(error**2)
        return np.sum(s)

    def _find_best_fit(self, parameter, model, data, error):
        """
        Finds element in <parameter> that yields best fit of <model> to <data>
        """
        N = len(parameter)
        χ = np.zeros(N)
        for i in range(N):
            χ[i] = self._chi_square(data, model[:,i], error)

        probability = 1/np.sqrt(2*np.pi*np.prod(error**2)) * np.exp(-0.5*χ)
        probability = probability/np.max(probability)

        idx_best   = np.argmax(probability)
        param_best = parameter[idx_best]
        χ_best     = χ[idx_best]

        return probability, param_best, χ_best

    def plot_mass_fractions(self, filename):
        """
        Plots relative densities of neutron/proton against temperature
        """
        sol = np.load(filename)
        T   = np.exp(sol[0])
        # Create arrays with rel. densities, atomic number and element names
        Y_i = sol[1:]
        A_i = np.array([1, 1, 2, 3, 3, 4, 7, 7])
        X_i = np.array([r"$p$"   , r"$n$"   , r"$D$"   , r"$T$",
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
        plt.show()

    def _interpolate_relic_abundances(self, solution, N=1000):
        """
        Interpolates array of calculated relic abundances
        """
        param = solution[0,:]
        Y_p   = solution[1,:]
        Y_n   = solution[2,:]
        Y_D   = solution[3,:]
        Y_T   = solution[4,:]
        Y_He3 = solution[5,:]
        Y_He4 = solution[6,:]
        Y_Li7 = solution[7,:]
        Y_Be7 = solution[8,:]

        Y_Li7 = Y_Li7 + Y_Be7
        Y_He3 = Y_He3 + Y_T

        Y_raw = np.array([Y_D/Y_p, Y_Li7/Y_p, 4*Y_He4, Y_He3/Y_p])

        logY_D_p   = interp1d(param, np.log(Y_raw[0])  , kind="cubic")
        logY_Li7_p = interp1d(param, np.log(Y_raw[1]), kind="cubic")
        logY_4xHe4 = interp1d(param, np.log(Y_raw[2])  , kind="cubic")
        logY_He3   = interp1d(param, np.log(Y_raw[3]), kind="cubic")

        param_interp = np.linspace(param[0], param[-1], N)

        Y_D_p   = np.exp( logY_D_p(param_interp)   )
        Y_Li7_p = np.exp( logY_Li7_p(param_interp) )
        Y_4xHe4 = np.exp( logY_4xHe4(param_interp) )
        Y_He3   = np.exp( logY_He3(param_interp)   )

        Y_interp = np.array([Y_D_p, Y_Li7_p, Y_4xHe4, Y_He3])

        return param_interp, Y_interp, Y_raw

    def plot_relic_abundances_Ω(self, filename):
        """
        Plots relic abundances as function of Ω_b0
        """
        solution = np.load(filename)

        Ω_b0, Y_relic, Y_raw = self._interpolate_relic_abundances(solution)
        Y_D_p, Y_Li7_p, Y_4xHe4, Y_He3 = Y_relic

        model = np.array([Y_D_p, Y_Li7_p, Y_4xHe4])
        error = np.array([0.03e-5, 0.3e-10, 0.003])
        data  = np.array([2.57e-5, 1.6e-10, 0.254])

        P, Ω_best, χ_best = self._find_best_fit(Ω_b0, model, data, error)
        print(f"Optimal Ω_b0: {Ω_best}")

        fig, ax = plt.subplots(nrows=3,
                            sharex=True,
                            tight_layout=True,
                            figsize=(7,8),
                            gridspec_kw={"height_ratios":[1,3,1]})

        ax[0].vlines(Ω_best, 0.32, 0.20, "k", ":")
        ax[1].vlines(Ω_best, 1e-11, 1e-3, "k", ":")
        ax[2].vlines(Ω_best, -0.1, 1.1, "k", ":", label=f"$\chi^2=${χ_best:.3f}\n"+
                                                 r"$\Omega_{b0}=$"+f"{Ω_best:.3f}")
        # Plot interpolated solution:
        ax[0].semilogx(Ω_b0, Y_4xHe4, color="C3", label="$\mathrm{He}^4$")
        ax[1].semilogx(Ω_b0, Y_D_p  , color="C0", label="$\mathrm{D}$"   )
        ax[1].semilogx(Ω_b0, Y_He3  , color="C4", label="$\mathrm{He}^3$")
        ax[1].semilogx(Ω_b0, Y_Li7_p, color="C1", label="$\mathrm{Li}^7$")
        ax[2].semilogx(Ω_b0, P, color="k")

        # Plot observational data:
        axes   = [ax[1], ax[1], ax[0]]
        colors = ["C0", "C1", "C3"]
        for i, (axs, c) in enumerate(zip(axes, colors)):
            y1 = data[i]+error[i]
            y2 = data[i]-error[i]
            axs.fill_between(Ω_b0, y1, y2, color=c, alpha=0.4)

        # Plot solution points used for interpolation:
        Ω_b0 = solution[0,:]
        Y_D_p, Y_Li7_p, Y_4xHe4, Y_He3 = Y_raw
        ax[0].scatter(Ω_b0, Y_4xHe4, color="C3", marker="|")
        ax[1].scatter(Ω_b0, Y_D_p  , color="C0", marker="|")
        ax[1].scatter(Ω_b0, Y_He3  , color="C4", marker="|")
        ax[1].scatter(Ω_b0, Y_Li7_p, color="C1", marker="|")

        ax[0].set_ylabel(r"$4Y_{\mathrm{He}^4}$")
        ax[1].set_ylabel(r"$Y_{i} / Y_{p}$")
        ax[2].set_ylabel(f"Normalized\nprobability")
        ax[2].set_xlabel(r"$\Omega_{b0}$")

        ax[1].set_yscale("logit")
        ax[0].set_ylim(0.20, 0.32)
        ax[1].set_ylim(1e-11, 1e-3)
        ax[2].set_xlim(1e-2 , 1e0)
        ax[2].set_ylim(-0.1, 1.1)


        for i in range(3):
            ax[i].legend()

        plt.show()

    def plot_relic_abundances_N(self, filename):
        """
        Plots relic abundances as function of N_eff
        """
        solution = np.load(filename)

        N_eff, Y_relic, Y_raw = self._interpolate_relic_abundances(solution)
        Y_D_p, Y_Li7_p, Y_4xHe4, Y_He3 = Y_relic

        model = np.array([Y_D_p, Y_Li7_p, Y_4xHe4])
        error = np.array([0.03e-5, 0.3e-10, 0.003])
        data  = np.array([2.57e-5, 1.6e-10, 0.254])

        P, N_best, χ_best = self._find_best_fit(N_eff, model, data, error)
        print(f"Optimal N_eff: {N_best}")

        fig, ax = plt.subplots(nrows=4,
                            sharex=True,
                            tight_layout=True,
                            figsize=(7,8))

        ax[0].vlines(N_best, 0.22, 0.30, "k", ":")
        ax[1].vlines(N_best, 1e-5, 4e-5, "k", ":")
        ax[2].vlines(N_best, 5e-10, 1e-10, "k", ":")
        ax[3].vlines(N_best, -0.1, 1.1, "k", ":", label=f"$\chi^2=${χ_best:.3f}\n"+
                                               r"$N_{eff}=$"+f"{N_best:.3f}")
        # Plot interpolated solution:
        ax[0].plot(N_eff, Y_4xHe4, color="C3", label="$\mathrm{He}^4$")
        ax[1].plot(N_eff, Y_D_p  , color="C0", label="$\mathrm{D}$"   )
        ax[1].plot(N_eff, Y_He3  , color="C4", label="$\mathrm{He}^3$")
        ax[2].plot(N_eff, Y_Li7_p, color="C1", label="$\mathrm{Li}^7$")
        ax[3].plot(N_eff, P, color="k")

        # Plot observational data:
        axes   = [ax[1], ax[2], ax[0]]
        colors = ["C0", "C1", "C3"]
        for i, (axs, c) in enumerate(zip(axes, colors)):
            y1 = data[i]+error[i]
            y2 = data[i]-error[i]
            axs.fill_between(N_eff, y1, y2, color=c, alpha=0.4)

        # Plot solution points used for interpolation:
        Ω_b0 = solution[0,:]
        Y_D_p, Y_Li7_p, Y_4xHe4, Y_He3 = Y_raw
        ax[0].scatter(Ω_b0, Y_4xHe4, color="C3", marker="|")#, linewidth=0.8)
        ax[1].scatter(Ω_b0, Y_D_p  , color="C0", marker="|")#, linewidth=0.8)
        ax[1].scatter(Ω_b0, Y_He3  , color="C4", marker="|")#, linewidth=0.8)
        ax[2].scatter(Ω_b0, Y_Li7_p, color="C1", marker="|")#, linewidth=0.8)

        ax[0].set_ylabel(r"$4Y_{\mathrm{He}^4}$")
        ax[1].set_ylabel(r"$Y_{i} / Y_{p}$")
        ax[2].set_ylabel(r"$Y_{i} / Y_{p}$")
        ax[3].set_ylabel(f"Normalized\nprobability")
        ax[3].set_xlabel(r"$N_{eff}$")

        ax[0].set_ylim(0.22, 0.30)
        ax[1].set_ylim(1e-5, 4e-5)
        ax[2].set_ylim(1e-10, 5e-10)
        ax[3].set_xlim(1, 5)
        ax[3].set_ylim(-0.1, 1.1)

        for i in range(4):
            ax[i].legend()

        plt.show()

if __name__=="__main__":
    # Remove overflow warnings from  integrating <BBN._I_dec> hehehehe
    warnings.filterwarnings("ignore")

    # Tweak plot appearance
    small = 14
    big   = 16
    plt.style.use("bmh")
    plt.rc('font', size=small)
    plt.rc('axes', titlesize=big)
    plt.rc('axes', labelsize=big)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=big)

    # ---------
    # PROBLEMS:
    # ---------

    model = BigBangNucleosynthesis()

    # Problem d):
    headers      = np.array(["T [K]", "t(T) [s]"])
    temperatures = np.array([1e10, 1e9, 1e8])
    cosmic_times = model._get_cosmic_time(temperatures)
    table = zip(temperatures, cosmic_times)
    table = tabulate(table, headers=headers, tablefmt="github", floatfmt=".4e")
    print(table)

    # Problems f) through i)
    model.solve(T_i=100e9, T_f=0.01e9, filename="mass_fractions.npy")
    model.plot_mass_fractions("mass_fractions.npy")

    # Problem j)
    model.calculate_relic_abundances_Ω_b0("relic_abundances_j.npy")
    model.plot_relic_abundances_Ω("relic_abundances_j.npy")

    # Problem k)
    model.calculate_relic_abundances_N_eff("relic_abundances_k.npy")
    model.plot_relic_abundances_N("relic_abundances_k.npy")
