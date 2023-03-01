import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
import warnings

from quintessence import QuintessenceModel
from lambdaCDM import H_ΛCDM, d_L_ΛCDM

warnings.filterwarnings("ignore") # Hide matplotlib warnings lol
plt.style.use("seaborn")

# Define physical parameters
h  = 0.7                     # []               dimensionless Hubble constant
H0 = 100*h                   # [km s^-1 Mpc^-1] Hubble constant
c  = 299792.458              # [km/s]           speed of light in vacuum

# Set initial conditions
exp_init = [0, 5e-13, 0.9999, 1.5]
pow_init = [5e-5, 1e-8, 0.9999, 1e9]

steps = 10000
Ni = -np.log(1+2e7)
N_span = (Ni, 0)
N = np.linspace(Ni, 0, steps)

# Solve equations of motions
exp_model = QuintessenceModel(Γ=1)
pow_model = QuintessenceModel(Γ=2)

exp_model.solve_eom(N, exp_init)
pow_model.solve_eom(N, pow_init)

# -----------
# Problem 9:
# -----------

# Plot density parameters
fig, ax = plt.subplots(nrows=2, figsize=(8,5), sharex=True, tight_layout=True)

pow_model.plot_Ω(N, ax[0])
exp_model.plot_Ω(N, ax[1])

ax[1].set_title(r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
ax[0].set_title(r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")
ax[1].set_xlabel("Redshift z")
ax[1].invert_xaxis()
ax[1].set_xlim(2e7, 0)
plt.legend()
plt.savefig("figs/density_parameters.png")
# Plot equation of state parameter for quintessence
fig, ax = plt.subplots(figsize=(8,5))

exp_model.plot_eos(N, ax, label=r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
pow_model.plot_eos(N, ax, label=r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")  #, \ \ \alpha=1$")

ax.set_title(r"Equation of state $w_{\phi}$")
ax.invert_xaxis()
ax.set_xlim(2e7, 0)
plt.legend()
plt.savefig("figs/eos.png")

# -----------
# Problem 10:
# -----------
z = np.exp(-N)-1
h=0.7
H0 = 100*h*1e3 # km s^-1 Gpc^-1
H_exp = exp_model.H(N)
H_pow = pow_model.H(N)
H_Λ   = H_ΛCDM(z)
#H_Λ1 = cosmo.H(z)/(100)

fig, ax = plt.subplots(figsize=(8,5))

fig.supylabel("h km s^-1 Mpc^-1")
fig.suptitle(r"$H(t)/H_{0}$")

ax.plot(z, H_exp, label="H_exp")
ax.plot(z, H_pow, label="H_pow")
ax.plot(z, H_Λ, label="H_ΛCDM")
#ax.plot(z, H_Λ1, "--",label="FASIT")

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel("Redshift z")

ax.invert_xaxis()
plt.legend()
plt.savefig("figs/Hubble_param.png")


# -----------
# Problem 12:
# -----------


z = np.linspace(2,0,steps)
N = -np.log(1+z)

d_exp = exp_model.d_L(z)
d_pow = pow_model.d_L(z)
d_Λ   = d_L_ΛCDM(z)

# Plotting the dimensionless luminosity distances
fig, ax = plt.subplots(figsize=(8,5))

plt.plot(z, d_exp, label=r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
plt.plot(z, d_pow, label=r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")
plt.plot(z, d_Λ, label="ΛCDM")

plt.title(r"$H_{0} d_L(z)/c$")
plt.legend()
plt.savefig("figs/lum_distance.png")




# -----------
# Problem 13:
# -----------

def chisquare(data, model, error):
    s = (model - data)**2/(error**2)
    return np.sum(s)

# Import luminosity distance data:
data = np.loadtxt("sndata.txt") # [], [Gpc], [Gpc]

z_data = data[:,0][::-1]
d_data = data[:,1][::-1]
err    = data[:,2][::-1]

H0 = 100*h*1e3          # [km s^-1 Gpc^-1] Hubble parameter, different units

# Interpolate luminosity distance arrays in order to evaluate at datapoints
d_exp = interp1d(z, d_exp*c/H0)
d_pow = interp1d(z, d_pow*c/H0)

X_exp = chisquare(d_data, d_exp(z_data)*c/H0, err)
X_pow = chisquare(d_data, d_pow(z_data)*c/H0, err)
# Create formatted table
table = zip(["Exponential", "Power law"], [X_exp, X_pow])
table = tabulate(table, headers=["Potential", "χ2"], tablefmt="github")
print("\nProblem 13:\n\n", table, "\n\n", sep="")

# -----------
# Problem 14:
# -----------

steps = 10000
z = np.linspace(2, 0, steps)
N = -np.log(1+z)

densities = np.linspace(0,1,10000)

X = np.zeros(len(densities))
for i, Ω_m0 in enumerate(tqdm(densities)):
    d_model = d_L_ΛCDM(z, Ω_m0)*c/H0
    d_model = interp1d(z, d_model)

    X[i] = chisquare(d_data, d_model(z_data), err)

# Find least X^2:
idx_min = np.argmin(X)
X_best = X[idx_min]
Ω_best = densities[idx_min]

print(f"\nBest fit for ΛCDM: \nΩ_m0 = {Ω_best:.5f} \nχ2   = {X_best:.5e}")

d_model = d_L_ΛCDM(z, Ω_best)*c/H0
fig, ax = plt.subplots(figsize=(8,5))

# Plot that shit
plt.plot(z, d_model, label=f"Best fit, Ω_m0 = {Ω_best:.4f}")

plt.errorbar(z_data, d_data, err,  fmt="--k", capsize=2,
                markeredgewidth=1, linewidth=0.6, elinewidth=0.6)
plt.fill_between(z_data, (d_data-err), (d_data+err),
                alpha=0.2, color="k", label="data")

plt.title(r"$\chi^2$ fit of $\Omega_{m0}$ for $\Lambda CDM$")
plt.ylabel(r"$d_L(z)$ [Gpc]")
plt.xlabel("Redshift z")
plt.legend()
plt.savefig("figs/lum_distance_fit.png")
