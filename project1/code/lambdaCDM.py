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
    H = np.sqrt( Ω_m0*(1+z) + (1-Ω_m0))
    return H

def d_L_ΛCDM(z, Ω_m0=0.3):
    """
    Calculates the dimensionless luminosity distance
    """
    N = -np.log(z+1)
    H = H_ΛCDM(z, Ω_m0)
    d = np.zeros(len(z))
    for i, _z in enumerate(z):
        d[i] = (1+_z)*np.trapz(1/H[:i+1:-1], z[:i+1:-1])


    return d
