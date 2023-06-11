import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import solve_ivp, cumulative_trapezoid


class SlowRollApproximation:
    def set_initial_field_value(self, N_tot):
        self.ψ_i = np.sqrt( 1/(2*np.pi)*(N_tot + 0.5) )

    def get_field_value(self, τ):
        ψ = self.ψ_i - τ/(4*np.pi*self.ψ_i)
        return ψ

class Inflation:
    def __init__(self):
        self.E_p = np.sqrt(const.hbar*const.c**5/const.G)
        self.m_p = np.sqrt(const.hbar*const.c/const.G)
        self.l_p = np.sqrt(const.hbar*const.G/const.c**3)

        self.SlowRoll = SlowRollApproximation()

        self.subclassName = type(self).__name__

    def get_hubble_param(self, τ, ψ, dψdτ):
        """
        calculates dimensionless hubble parameter
        """
        v = self.get_potential(ψ)
        h = np.sqrt(8*np.pi/3 * (0.5*dψdτ**2 + v) )
        return h

    def get_total_efolds(self, τ, ψ, dψdτ):
        """
        calculates total number of efolds
        """
        h = self.get_hubble_param(τ, ψ, dψdτ)
        N_tot = np.trapz(h, x=τ)

        return N_tot

    def eom(self, t, y):
        τ    = t
        ψ, ξ = y

        h = self.get_hubble_param(τ, ψ, ξ)
        dvdψ = self.get_potential_diff(ψ)

        dξdτ = -3*h*ξ - dvdψ
        dψdτ = ξ

        return [dψdτ, dξdτ]

    def solve(self, τ_span):
        y_init = [self.ψ_i, 0]
        self.sol = solve_ivp(self.eom, τ_span, y_init, rtol=1e-12, atol=1e-12)

    def plot_field_value(self, plot_slowRoll=False):
        ψ_exact = self.sol.y[0]
        τ = self.sol.t

        plt.figure(figsize=(10,5), tight_layout=True)

        if plot_slowRoll:
            ψ_slowroll = self.SlowRoll.get_field_value(τ)
            plt.plot(τ, ψ_slowroll, label="slow roll")

        plt.plot(τ, ψ_exact, label="exact")
        plt.legend()
        plt.savefig(f"{self.subclassName}_field-value")

    def plot_scale_factor(self):
        τ = self.sol.t
        ψ  = self.sol.y[0]
        dψ = self.sol.y[1]

        h = self.get_hubble_param(τ, ψ, dψ)
        ln_a = cumulative_trapezoid(h, x=τ, initial=0)

        plt.figure(figsize=(10,5), tight_layout=True)
        plt.plot(τ, ln_a)
        plt.ylabel("ln(a/a_i)")
        plt.savefig(f"{self.subclassName}_scale-factor")

    def plot_slow_roll_param_against_tau(self):
        τ = self.sol.t
        ψ = self.sol.y[0]
        ϵ = self.get_slow_roll_param_eps(ψ)
        η = self.get_slow_roll_param_eta(ψ)

        plt.figure(figsize=(10,5), tight_layout=True)
        plt.semilogy(τ, η, label="η")
        plt.semilogy(τ, ϵ, label="ϵ")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-tau")

    def plot_slow_roll_param_against_N(self):
        τ = self.sol.t
        ψ  = self.sol.y[0]
        dψ = self.sol.y[1]
        ϵ = self.get_slow_roll_param_eps(ψ)
        η = self.get_slow_roll_param_eta(ψ)
        h = self.get_hubble_param(τ, ψ, dψ)
        N_tot = self.get_total_efolds(τ, ψ, dψ)
        #print(f"N_tot = {N_tot}")
        N = N_tot - cumulative_trapezoid(h, x=τ, initial=0)


        plt.figure(figsize=(10,5), tight_layout=True)
        ax = plt.gca()

        plt.semilogx(N, ϵ, label="eps")
        plt.semilogy(N, η, label="eta")
        plt.axhline(1, color="k", linestyle="--")

        ax.invert_xaxis()
        plt.xlabel("Remaining e-folds")
        plt.legend()
        plt.savefig(f"{self.subclassName}_slowroll-N")

    def plot_field_eos(self):
        τ = self.sol.t
        ψ  = self.sol.y[0]
        dψ = self.sol.y[1]
        v = self.get_potential(ψ)

        w_ϕ = (0.5*dψ**2 - v)/((0.5*dψ**2 + v))

        fig = plt.figure(figsize=(10,5), tight_layout=True)
        plt.plot(τ, w_ϕ, label="equation of state")
        #plt.xlim(2650, 2750)
        plt.legend()
        plt.savefig(f"{self.subclassName}_field_eos")

    def plot_slow_roll_param_nr_plane(self):
        τ = self.sol.t
        ψ  = self.sol.y[0]
        dψ = self.sol.y[1]
        ϵ = self.get_slow_roll_param_eps(ψ)
        η = self.get_slow_roll_param_eta(ψ)
        h = self.get_hubble_param(τ, ψ, dψ)
        N_tot = self.get_total_efolds(τ, ψ, dψ)
        N = N_tot - cumulative_trapezoid(h, x=τ, initial=0)

        idx = np.logical_and(N>50, N<60)
        ϵ = ϵ[idx]
        η = η[idx]

        n = 1 - 6*ϵ + 2*η
        r = 16*ϵ

        plt.figure(figsize=(10,5), tight_layout=True)
        plt.plot(n, r)
        plt.xlabel("n")
        plt.ylabel("r")
        plt.savefig(f"{self.subclassName}_slowroll-nr")

class QuadraticPotential(Inflation):
    def __init__(self):
        super().__init__()
        self.SlowRoll.set_initial_field_value(N_tot=500)
        self.ψ_i = self.SlowRoll.ψ_i

    def get_potential(self, ψ):
        v = 3/(8*np.pi) * (ψ/self.ψ_i)**2 # * (self.m_p*const.c**2/self.E_p)**2
        return v

    def get_potential_diff(self, ψ):
        dvdψ = 3/(4*np.pi) * ψ/(self.ψ_i**2)
        return dvdψ

    def get_slow_roll_param_eps(self, ψ):
        ϵ = 1/(4*np.pi*ψ)
        return ϵ

    def get_slow_roll_param_eta(self, ψ):
        η = 1/(4*np.pi*ψ**2)
        return η

class StarobinskyPotential(Inflation):
    def __init__(self):
        super().__init__()
        self.ψ_i = 2

    def get_potential(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        y_i = -np.sqrt(16*np.pi/3)*self.ψ_i

        v = 3/(8*np.pi) * (1-np.exp(y))**2/(1-np.exp(y_i))**2
        return v

    def get_potential_diff(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        y_i = -np.sqrt(16*np.pi/3)*self.ψ_i

        dvdψ = np.sqrt(3/np.pi)*(np.exp(y) - np.exp(2*y))/(1 - np.exp(y_i))**2
        return dvdψ

    def get_slow_roll_param_eps(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        y_i = -np.sqrt(16*np.pi/3)*self.ψ_i

        ϵ = 4/3 * np.exp(2*y) / (1 - np.exp(y))**2
        return ϵ

    def get_slow_roll_param_eta(self, ψ):
        y   = -np.sqrt(16*np.pi/3)*ψ
        y_i = -np.sqrt(16*np.pi/3)*self.ψ_i

        η = 4/3 * (2*np.exp(2*y) - np.exp(y)) / (1 - np.exp(y))**2
        return η

def plot_inset(axis, bounds, xlim=None, ylim=None):
    axin = axis.inset_axes(bounds)

    lines = axis.lines

    for line in lines:
        x = line.get_xdata()
        y = line.get_ydata()
        axin.plot(x,y)

    if xlim: axin.set_xlim(xlim)
    if ylim: axin.set_ylim(ylim)

    axin.set_xticks([])
    axin.set_yticks([])

    axis.indicate_inset_zoom(axin)
    plt.tight_layout()
    plt.show()

def zoom_inset():
    zoomed_inset_axes(plt.gca(), 2.5)

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

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    #model = QuadraticPotential()
    model = StarobinskyPotential()
    model.solve(τ_span=[0, 3000])

    model.plot_field_eos()
    exit()
    axis = plt.gca()
    plot_inset(axis, bounds=[0.02,0.98,0.4,0.4], xlim=[2650,2750])
    model.plot_scale_factor()
    model.plot_field_value()
    model.plot_slow_roll_param_against_N()
    model.plot_slow_roll_param_nr_plane()
    model.plot_slow_roll_param_against_tau()
