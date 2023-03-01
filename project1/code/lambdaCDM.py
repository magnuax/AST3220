import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.cosmology import WMAP9 as cosmo

h  = 0.7
H0 = 100*h*1e3 # km s^-1 Gpc^-1


def H_ΛCDM(z, Ω_m0=0.3):
    """
    Calculates dimensionless Hubble parameter H/H0 for ΛCDM model.
    """
    #H = np.sqrt( Ω_m0*np.exp(-3*N) + (1-Ω_m0))
    H = np.sqrt(Ω_m0*(1+z)**3 + (1-Ω_m0))
    return H

def d_L_ΛCDM(z, Ω_m0=0.3):
    """
    Calculates the dimensionless luminosity distance
    """
    z = z[::-1]
    H = H_ΛCDM(z, Ω_m0)
    d = np.zeros(len(z))
    d = (1+z)*integrate.cumulative_trapezoid(1/H, z, initial=0)

    return d[::-1]
