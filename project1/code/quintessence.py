import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad


class QuintessenceModel():
    def __init__(self, Γ):
        self.Γ = Γ



    def eom(self, N, X, Γ):
        """
        Calculates rhs of equations of motion. Takes vector X(N) = [x1, x2, x3, λ] and
        time variable N as input.
        """
        x1, x2, x3, λ = X

        Hx = 0.5*(3 + 3*x1**2 - 3*x2**2 + x3**2)
        d1 = -3*x1 + np.sqrt(1.5)*λ*x2**2 + x1*Hx
        d2 = -np.sqrt(1.5)*λ*x1*x2 + x2*Hx
        d3 = -2*x3 + x3*Hx
        dλ = -np.sqrt(6)*(λ**2)*(Γ-1)*x1

        return np.array([d1, d2, d3, dλ])

    def solve_eom(self, N, x_init):
        """
        Solves equations of motion and returns density parameters and the quintessence
        equation of state parameter.
        """
        N_span = (N[0], N[-1])
        self.sol = solve_ivp(self.eom, N_span, x_init, args=(self.Γ,),
                            method="DOP853", dense_output=True)


    def H(self, N):
        """
        Uses eq.(7) to calculate dimensionless Hubble parameter H/H0 for quintessence
        models, given density parameters & eqs of state.
        """
        Ω_q0, Ω_r0, Ω_m0 = self.density_parameters(0)


        H = np.zeros(len(N))
        w = self.eos(N)
        for i, _N in enumerate(N):
            integrand = 1 + w[:i+1]
            I = np.trapz(integrand, N[:i+1])
            H[i] = np.sqrt(Ω_m0*np.exp(-3*_N) + Ω_r0*np.exp(-4*_N) + Ω_q0*np.exp(-3*I))

        #   integrand = lambda N: 1 + self.eos(N)
        #I, err = quad(self.eos, N, 0)
        #H = np.sqrt(Ω_m0*np.exp(-3*N) + Ω_r0*np.exp(-4*N) + Ω_q0*np.exp(-3*(1+I)))

        return H

    def age(self):
        #x = 1/(1+z)
        integrand = lambda z: 1/((1+z)*self.H(z))
        H0t = quad(integrand, 0, np.inf)

        return H0t

    def d_L(self, z):
        """
        Calculates the dimensionless luminosity distance as function of distance.
        """
        N = -np.log(z+1)
        H = self.H(N)


        d_L = np.zeros(len(z))
        for i, _z in enumerate(z):
            d_L[i] = (1+_z)*np.trapz(1/H[:i+1:-1], z[:i+1:-1])
        #integrand = lambda N: self.H(N)
        #I, err = quad(self.H, 0, N)

        return d_L

    def density_parameters(self, N):
        x1, x2, x3 = self.sol.sol(N)[:3]

        Ω_q = x1**2 + x2**2
        Ω_r = x3**2
        Ω_m = 1 - Ω_q - Ω_r

        return Ω_q, Ω_r, Ω_m

    def eos(self, N):
        x1, x2 = self.sol.sol(N)[:2]
        w = (x1**2 - x2**2)/(x1**2 + x2**2)
        return w

    def plot_Ω(self, N, ax):
        """
        Self explanatory - Plots density parameters against redshift. Also plots
        the sum of the density parameters (=1) for quality control.
        """
        Ω_q, Ω_r, Ω_m = self.density_parameters(N)
        Ω = Ω_q + Ω_r + Ω_m

        z = np.exp(-N)-1
        ax.plot(z, Ω, "--k", label=r"$\sum_i Ω_i$")
        ax.plot(z, Ω_q, label=r"$\Omega_{\phi}$")
        ax.plot(z, Ω_r, label=r"$\Omega_{r}$")
        ax.plot(z, Ω_m, label=r"$\Omega_{m}$")
        ax.set_ylabel(r"$\Omega_i(z)$")

    def plot_eos(self, N, ax, label):
        z = np.exp(-N)-1
        w = self.eos(N)

        ax.plot(z, w, label=label)
        ax.set_xlabel("Redshift z")

if __name__=="__main__":
    plt.style.use("ggplot")

    pow_init = [5e-5, 1e-8, 0.9999, 1e9]
    exp_init = [0, 5e-13, 0.9999, 1.5]

    Ni = -np.log(1+2e7)
    N_span = (Ni, 0)
    N = np.linspace(Ni, 0, 100000)

    exp_model = QuintessenceModel(Γ=1)
    pow_model = QuintessenceModel(Γ=2)

    exp_model.solve_eom(N, exp_init)
    pow_model.solve_eom(N, pow_init)


    # Problem 9:

    # Plot density parameters
    fig, ax = plt.subplots(nrows=2, figsize=(8,5), sharex=True, tight_layout=True)

    pow_model.plot_Ω(N, ax[0])
    exp_model.plot_Ω(N, ax[1])

    ax[1].set_title(r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
    ax[0].set_title(r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")
    ax[1].set_xlabel("Redshift z")
    ax[1].invert_xaxis()
    plt.legend()

    # Plot equation of state parameter for quintessence
    fig, ax = plt.subplots(figsize=(8,5))

    exp_model.plot_eos(N, ax, label=r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
    pow_model.plot_eos(N, ax, label=r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")  #, \ \ \alpha=1$")

    ax.set_title(r"Equation of state $w_{\phi}$")
    ax.invert_xaxis()
    plt.legend()
    plt.show()

    age_exp = exp_model.age()
    age_pow = pow_model.age()
    H0 = 100*0.7*1e3 # km s^-1 Gpc^-1

    print(age_exp, age_pow)
