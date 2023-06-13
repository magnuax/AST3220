import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import solve_ivp, cumulative_trapezoid


class SlowRollApproximation:
    def set_initial_field_value(self, N_tot):
        self.ψ_i = np.sqrt( 1/(2*np.pi)*(N_tot + 0.5) )

    def _get_field_value(self, τ):
        ψ = self.ψ_i - τ/(4*np.pi*self.ψ_i)
        return ψ

    def _get_field_value_derivative(self, τ):
        dψdτ = 1/(4*np.pi*self.ψ_i)
        return dψdτ

class Inflation:
    def __init__(self):
        self.E_p = np.sqrt(const.hbar*const.c**5/const.G)
        self.m_p = np.sqrt(const.hbar*const.c/const.G)
        self.l_p = np.sqrt(const.hbar*const.G/const.c**3)

        self.SlowRoll = SlowRollApproximation()

        self.subclassName = type(self).__name__

    def _get_hubble_param(self, τ, ψ, dψdτ):
        """
        calculates dimensionless hubble parameter
        """
        v = self._get_potential(ψ)
        h = np.sqrt(8*np.pi/3 * (0.5*dψdτ**2 + v) )
        return h

    def _get_total_efolds(self, τ, ψ, dψdτ):
        """
        calculates total number of efolds
        """
        h = self._get_hubble_param(τ, ψ, dψdτ)
        N_tot = np.trapz(h, x=τ)

        return N_tot

    def _get_eq_of_state(self, ψ, dψdτ):
        v = self._get_potential(ψ)
        w_ϕ = (0.5*dψdτ**2 - v)/((0.5*dψdτ**2 + v))
        return w_ϕ

    def _eom(self, t, y):
        τ    = t
        ψ, ξ = y

        h = self._get_hubble_param(τ, ψ, ξ)
        dvdψ = self._get_potential_diff(ψ)

        dξdτ = -3*h*ξ - dvdψ
        dψdτ = ξ

        return [dψdτ, dξdτ]

    def solve(self, τ_span):
        y_init = [self.ψ_i, 0]
        self.sol = solve_ivp(self._eom, τ_span, y_init, rtol=1e-10, atol=1e-10)

        self.τ = self.sol.t
        self.ψ  = self.sol.y[0]
        self.dψ = self.sol.y[1]

        self.ϵ = self._get_slow_roll_param_eps(self.ψ)
        self.η = self._get_slow_roll_param_eta(self.ψ)

        self.h = self._get_hubble_param(self.τ, self.ψ, self.dψ)
        self.ln_a = cumulative_trapezoid(self.h, x=self.τ, initial=0)
        self.N_tot = self._get_total_efolds(self.τ, self.ψ, self.dψ)
        self.N = self.N_tot - self.ln_a

    def plot_field_value(self):
        plt.figure(figsize=(10,5), tight_layout=True)
        plt.title("Value of scalar field $\psi$, for " + self.V_repr)

        plt.plot(self.τ, self.ψ, label="Numerical solution")
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\psi$")
        plt.legend()
        plt.savefig(f"{self.subclassName}_field-value")

    def plot_scale_factor(self):
        plt.figure(figsize=(10,5), tight_layout=True)
        plt.title("Scale factor, for " + self.V_repr)
        plt.plot(self.τ, self.ln_a, label="Numerical solution")
        plt.legend()
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\ln\left( a/a_i \right)$")
        plt.savefig(f"{self.subclassName}_scale-factor")

    def plot_slow_roll_param_against_tau(self):
        plt.figure(figsize=(10,5), tight_layout=True)
        plt.title(r"Slow roll parameters against $\tau$, for " + self.V_repr)
        plt.semilogy(self.τ, self.ϵ, label=r"$\epsilon$")
        plt.semilogy(self.τ, self.η, label=r"$\eta$")
        plt.axhline(1, color="k", linestyle="--")

        plt.xlabel(r"$\tau$")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-tau")

    def plot_slow_roll_param_against_N(self):
        idx_end = np.where(self.ϵ>1)[0][0]

        plt.figure(figsize=(10,5), tight_layout=True)
        plt.title(r"Slow roll parameters against $N$, for " + self.V_repr)

        plt.semilogx(self.N[:idx_end], self.ϵ[:idx_end], label=r"$\epsilon$")
        plt.semilogx(self.N[:idx_end], self.η[:idx_end], label=r"$\eta$")
        plt.axhline(1, color="k", linestyle="--")

        ax = plt.gca()
        ax.invert_xaxis()
        plt.xlabel(r"Remaining e-folds $N$")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-N")

    def plot_field_eos(self):
        w_ϕ = self._get_eq_of_state(self.ψ, self.dψ)

        fig = plt.figure(figsize=(10,5), tight_layout=True)
        plt.title(r"Equation of state $w_\phi$, for " + self.V_repr)
        plt.plot(self.τ, w_ϕ)
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$w_\phi$")
        plt.savefig(f"{self.subclassName}_field-eos")

    def plot_slow_roll_param_nr_plane(self):
        ϵ = self._get_slow_roll_param_eps(self.ψ)
        η = self._get_slow_roll_param_eta(self.ψ)
        N = self.N_tot - self.ln_a

        idx = np.logical_and(N>50, N<60)
        ϵ = ϵ[idx]
        η = η[idx]
        n = 1 - 6*ϵ + 2*η
        r = 16*ϵ

        plt.figure(figsize=(10,5), tight_layout=True)
        plt.title("TITLE")
        plt.plot(n, r, label="Numerical solution")
        plt.legend()
        plt.xlabel("n")
        plt.ylabel("r")
        plt.savefig(f"{self.subclassName}_slowroll-nr")

class QuadraticPotential(Inflation):
    def __init__(self):
        super().__init__()
        self.SlowRoll.set_initial_field_value(N_tot=500)
        self.ψ_i = self.SlowRoll.ψ_i
        self.V_repr = r"$V = \frac{1}{2}\frac{m^2c^4}{(\hbar c)^3} \phi^2$"

    def _get_potential(self, ψ):
        v = 3/(8*np.pi) * (ψ/self.ψ_i)**2 # * (self.m_p*const.c**2/self.E_p)**2
        return v

    def _get_potential_diff(self, ψ):
        dvdψ = 3/(4*np.pi) * ψ/(self.ψ_i**2)
        return dvdψ

    def _get_slow_roll_param_eps(self, ψ):
        ϵ = 1/(4*np.pi*ψ)
        return ϵ

    def _get_slow_roll_param_eta(self, ψ):
        η = 1/(4*np.pi*ψ**2)
        return η

    def plot_field_value(self):
        super().plot_field_value()

        ψ_slowroll = self.SlowRoll._get_field_value(self.τ)
        plt.plot(self.τ, ψ_slowroll, "--", label="Slow roll approximation")
        plot_inset(plt.gca(), [0.5,0.05,0.35,0.35], xlim=[1050,1500], ylim=[-0.2,0.2])

        plt.legend()
        plt.savefig(f"{self.subclassName}_field-value")

    def plot_scale_factor(self):
        super().plot_scale_factor()
        ψ_slowroll  = self.SlowRoll._get_field_value(self.τ)
        dψ_slowroll = self.SlowRoll._get_field_value_derivative(self.τ)
        h_slowroll  = self._get_hubble_param(self.τ, ψ_slowroll, dψ_slowroll)
        ln_a_slowroll = cumulative_trapezoid(h_slowroll, x=self.τ, initial=0)
        plt.plot(self.τ, ln_a_slowroll, "--", label="Slow-roll approximation")
        plt.savefig(f"{self.subclassName}_scale-factor")

    def plot_slow_roll_param_against_tau(self):
        super().plot_slow_roll_param_against_tau()
        ψ_slowroll = self.SlowRoll._get_field_value(self.τ)
        ϵ_slowroll = self._get_slow_roll_param_eps(ψ_slowroll)
        η_slowroll = self._get_slow_roll_param_eta(ψ_slowroll)

        plt.plot(self.τ, ϵ_slowroll, "--", label="$\epsilon$ - slow-roll approximation")
        plt.plot(self.τ, η_slowroll, "--", label="$\eta$ - slow-roll approximation")

        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-tau")

    def plot_slow_roll_param_against_N(self):
        super().plot_slow_roll_param_against_N()
        ϵ = self._get_slow_roll_param_eps(self.ψ)
        N = self.N_tot - self.ln_a
        idx_end = np.where(ϵ>1)[0][0]

        ψ_slowroll = self.SlowRoll._get_field_value(self.τ)
        ϵ = self._get_slow_roll_param_eps(ψ_slowroll)
        η = self._get_slow_roll_param_eta(ψ_slowroll)

        plt.semilogy(N[:idx_end], ϵ[:idx_end], "--", label="$\epsilon$ - slow-roll approximation")
        plt.semilogy(N[:idx_end], η[:idx_end], "--", label="$\eta$ - slow-roll approximation")
        plt.legend()
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-N")

    def plot_slow_roll_param_nr_plane(self):
        super().plot_slow_roll_param_nr_plane()
        ψ_slowroll = self.SlowRoll._get_field_value(self.τ)
        ϵ = self._get_slow_roll_param_eps(ψ_slowroll)
        η = self._get_slow_roll_param_eta(ψ_slowroll)

        n = 1 - 6*ϵ + 2*η
        r = 16*ϵ

        idx = np.logical_and(self.N>50, self.N<60)
        plt.plot(n[idx], r[idx], "--", label="Slow-roll approximation")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-nr")

class StarobinskyPotential(Inflation):
    def __init__(self):
        super().__init__()
        self.ψ_i = 2
        self.V_repr = r"$V = \frac{3M^2M_p^2}{4}\left(1-e^{-\sqrt{\frac{2}{3}}\frac{\phi}{M_p}}\right)^2$"

    def _get_potential(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        y_i = -np.sqrt(16*np.pi/3)*self.ψ_i

        v = 3/(8*np.pi) * (1-np.exp(y))**2/(1-np.exp(y_i))**2
        return v

    def _get_potential_diff(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        y_i = -np.sqrt(16*np.pi/3)*self.ψ_i

        dvdψ = np.sqrt(3/np.pi)*(np.exp(y) - np.exp(2*y))/(1 - np.exp(y_i))**2
        return dvdψ

    def _get_slow_roll_param_eps(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        ϵ = 4/3 * np.exp(2*y) / (1 - np.exp(y))**2
        return ϵ

    def _get_slow_roll_param_eta(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        η = 4/3 * (2*np.exp(2*y) - np.exp(y)) / (1 - np.exp(y))**2
        return η

    def plot_field_value(self):
        super().plot_field_value()
        plot_inset(plt.gca(), [0.2,0.2,0.4,0.4], xlim=[2680,2750], ylim=[-0.06,0.06])
        plt.savefig(f"{self.subclassName}_field-value")

    def plot_field_eos(self):
        super().plot_field_eos()
        plot_inset(plt.gca(), [0.045,0.455,0.5,0.5], xlim=[2690, 2720])
        plt.savefig(f"{self.subclassName}_field-eos")

    def plot_slow_roll_param_against_N(self):
        super().plot_slow_roll_param_against_N()
        ϵ = self._get_slow_roll_param_eps(self.ψ)
        idx_end = np.where(ϵ>1)[0][0]

        y = -np.sqrt(16*np.pi/3)*self.ψ
        N = 3/4*np.exp(-y)

        ϵ = 3/(4*N**2)
        η = -1/N

        plt.plot(N[:idx_end], ϵ[:idx_end], "--", label="$\epsilon$ - slow-roll approximation")
        plt.plot(N[:idx_end], η[:idx_end], "--", label="$\eta$ - slow-roll approximation")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-N")

    def plot_slow_roll_param_nr_plane(self):
        super().plot_slow_roll_param_nr_plane()
        y = -np.sqrt(16*np.pi/3)*self.ψ
        N = 3/4*np.exp(-y)
        n = 1 - 2/N
        r = 12/N**2

        idx = np.logical_and(N>50, N<60)
        plt.plot(n[idx], r[idx], "--", label="Slow-roll approximation")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-nr")

def plot_inset(axis, bounds, xlim=None, ylim=None):
    axin = axis.inset_axes(bounds)

    lines = axis.lines

    for line in lines:
        x, y = line.get_data()
        color = line.get_color()
        style = line.get_linestyle()

        axin.plot(x, y, color=color, linestyle=style)

    if xlim: axin.set_xlim(xlim)
    if ylim: axin.set_ylim(ylim)

    axin.set_xscale( axis.get_xscale() )
    axin.set_yscale( axis.get_yscale() )
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_xticklabels([])
    axin.set_yticklabels([])

    axis.indicate_inset_zoom(axin, edgecolor="k", alpha=0.5)



if __name__=="__main__":
    small = 14
    big   = 16
    plt.style.use("bmh")
    plt.rc('font', size=small)
    plt.rc('axes', titlesize=big)
    plt.rc('axes', labelsize=big)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=big)


    ###### Quadratic potential ######
    InflationQuadratic = QuadraticPotential()
    InflationQuadratic.solve(τ_span=[0, 2000])

    # e) & f)
    InflationQuadratic.plot_field_value()
    InflationQuadratic.plot_scale_factor()

    # g)
    InflationQuadratic.plot_slow_roll_param_against_tau()

    # h)
    InflationQuadratic.plot_field_eos()

    # j)
    InflationQuadratic.plot_slow_roll_param_against_N()

    # k)
    InflationQuadratic.plot_slow_roll_param_nr_plane()


    ###### Starobinsky potential ######
    InflationStarobinsky = StarobinskyPotential()
    InflationStarobinsky.solve(τ_span=[0, 3000])

    # m)
    InflationStarobinsky.plot_field_value()
    InflationStarobinsky.plot_scale_factor()

    #InflationStarobinsky.plot_field_eos()

    # n) & o)
    InflationStarobinsky.plot_slow_roll_param_against_N()
    InflationStarobinsky.plot_slow_roll_param_nr_plane()
