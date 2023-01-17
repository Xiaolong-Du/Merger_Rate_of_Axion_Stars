import numpy as np

from constants import *
from cosmology import *

# Jeans wavenumber.
def k_J(ma, z):
    return (6.0*OmegaM/(1.0+z))**0.25*(ma*eV*hubble*100.0*kilo/Mpc/hbar/light_speed**2)**0.5*Mpc

# Jeans mass.
def M_J(ma,z):
    return 4.0/3.0*np.pi*(np.pi/k_J(ma, z))**3*rho_mean_code

# Half-mode wavenumber. (Note that the definition is T(k_half)=1/2, \
# which is different from that in Hu et al. 2000)
def k_half(ma):
    return 1.108*4.5*(ma/1.0e-22)**(4.0/9.0)

# Half-mode mass.
def M_half(ma):
    return 4.0/3.0*np.pi*(np.pi/k_half(ma))**3*rho_mean_code

# Minimum halo mass at z.
def M_halo_min(ma, z):
    return 4.4e7*(1.0+z)**(3.0/4.0)*(xi(z)/xi(0.0))**0.25*(ma/1.0e-22)**(-1.5)

# Core mass given the halo mass.
def M_core(Mh, ma, z, alpha=1.0/3.0):
    Mh_min = M_halo_min(ma, z)
    return 0.25*(Mh/Mh_min)**alpha*Mh_min

# Star mass given the halo mass.
def M_star(Mh, ma, z, alpha=1.0/3.0):
    return 4.0*M_core(Mh, ma, z, alpha)

# Halo mass given the core mass.
def M_halo_from_core_mass(Mc, ma, z, alpha=1.0/3.0):
    Mh_min = M_halo_min(ma, z)
    return (4.0*Mc/Mh_min**(1.0-alpha))**(1.0/alpha)

# Halo mass given the star mass.
def M_halo_from_star_mass(Mstar, ma, z, alpha=1.0/3.0):
    return M_halo_from_core_mass(Mstar/4.0, ma, z, alpha)

# Critical star mass.
def M_decay(ma,ga_gamma):
    return 8.4e-5*(1.0e-11/ga_gamma)*(1.0e-13/ma)

