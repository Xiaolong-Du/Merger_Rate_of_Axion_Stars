import numpy as np
from scipy import special

from constants import *

hubble                = 0.6736
OmegaM                = 0.3153
OmegaL                = 1.0-OmegaM

# Critical density of the universe.
rho_crit              = 3.0*(hubble*100.0*kilo/Mpc)**2/(8.0*np.pi*gravitationalConstant)
rho_mean              = OmegaM*rho_crit

rho_mean_code         = rho_mean/Msun*Mpc**3

# Useful functions.
def OmegaM_z(z):
    return OmegaM*(1.0+z)**3/(OmegaL+OmegaM*(1.0+z)**3)

# Virial density contrast.
def xi(z):
    return (18.0*np.pi**2+82.0*(OmegaM_z(z)-1.0)-39.0*(OmegaM_z(z)-1.0)**2)/OmegaM_z(z)

# Growth factor of linear density perturbations (Percival 2005).
D0 = special.hyp2f1(1.0/3.0, 1.0, 11.0/6.0, -OmegaL/OmegaM)

def D_growth(z):
    return 1.0/(1.0+z)*special.hyp2f1(1.0/3.0, 1.0, 11.0/6.0, -OmegaL/OmegaM/(1.0+z)**3)/D0

def D_growth_rate(z):
    return -D_growth(z)/(1.0+z)                                               \
           +6.0/11.0*OmegaL/OmegaM/(1.0+z)**5/D0                              \
           *special.hyp2f1(4.0/3.0, 2.0, 17.0/6.0, -OmegaL/OmegaM/(1.0+z)**3)

# Critical overdensity for collapse (Kitayama et al. 1996).
deltaC0 = 3.0/20.0*(12.0*np.pi)**(2.0/3.0)

def deltaC(z):
    return deltaC0*(1.0+0.0123*np.log10(OmegaM_z(z)))/D_growth(z)

