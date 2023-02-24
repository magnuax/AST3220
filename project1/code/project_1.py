import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from quintessence import QuintessenceModel
from lambdaCDM import H_ΛCDM, d_L_ΛCDM
from astropy.cosmology import WMAP9 as cosmo

plt.style.use("seaborn")

# Import luminosity distance data: ()
data = np.loadtxt("sndata.txt") # [], [Gpc], [Gpc]
z_data, d_data, err = data[:,0][::-1], data[:,1][::-1], data[:,2][::-1]

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
plt.legend()
plt.savefig("figs/density_parameters.png")
# Plot equation of state parameter for quintessence
fig, ax = plt.subplots(figsize=(8,5))

exp_model.plot_eos(N, ax, label=r"$V(\phi) = V_{0} e^{-\kappa \zeta \phi}$")
pow_model.plot_eos(N, ax, label=r"$V(\phi) = M^{4+\alpha} \phi^{-\alpha}$")  #, \ \ \alpha=1$")

ax.set_title(r"Equation of state $w_{\phi}$")
ax.invert_xaxis()
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
#H_Λ = cosmo.H(z)/H0

fig, ax = plt.subplots(figsize=(8,5))

fig.supylabel("h km s^-1 Mpc^-1")
fig.suptitle(r"$H(t)/H_{0}$")

ax.plot(z, H_exp, label="H_exp")
ax.plot(z, H_pow, label="H_pow")
ax.plot(z, H_Λ*H0, label="H_ΛCDM")

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
steps = 100
z = np.linspace(z_data[0], z_data[-1], steps)
N = -np.log(1+z)
print("z=",z)
print()
def chisquare(data, model, error):
    s = (model - data)**2/(error**2)
    return np.sum(s)

H0 = 100*h*1e3 # km s^-1 Gpc^-1

step = int(steps/len(z_data))
stop = step*len(z_data)

X_exp = chisquare(d_data, d_exp[:stop:step]*c/H0, err)
X_pow = chisquare(d_data, d_pow[:stop:step]*c/H0, err)
print(X_exp, X_pow)

densities = np.linspace(0,1,500)
H = H_ΛCDM(z, 0.0)*0.7*100
d_model = d_L_ΛCDM(z, H)[:stop:step]


X = np.zeros(len(densities))
for i, Ω_m0 in enumerate(tqdm(densities)):
    H = H_ΛCDM(z, Ω_m0)
    d_model = d_L_ΛCDM(z, H)[:stop:step]*c/H0
    X[i] = chisquare(d_data, d_model, err)

idx_min = np.argmin(X)
X_best = X[idx_min]
Ω_best = densities[idx_min]

print(X)
print(X_best, Ω_best)

H = H_ΛCDM(z, Ω_best)
d_model = d_L_ΛCDM(z, H)*c/H0

fig, ax = plt.subplots(figsize=(8,5))

plt.plot(z, d_model, label=f"Best fit, Ω = {Ω_best}")
plt.errorbar(z_data, d_data, err,  fmt="--k", capsize=2,
                markeredgewidth=1, linewidth=0.6, elinewidth=0.6)
plt.fill_between(z_data, (d_data-err), (d_data+err),
                alpha=0.2, color="k", label="data")

H = H_ΛCDM(z, 0.3)
d_3 = d_L_ΛCDM(z, H)*c/H0
plt.plot(z, d_3, label="Ω_m0 = 0.3")

plt.title(r"$\chi^2$ fit of $\Omega_{m0}$ for $\Lambda CDM$")
plt.ylabel(r"$d_L(z)$ [Gpc]")
plt.xlabel("Redshift z")
plt.legend()
plt.savefig("figs/lum_distance_fit.png")


# [km s^-1 Gpc^-1]
