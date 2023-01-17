import numpy as np

from axion import *

# sigma(M)
# alpha = dln sigma / dln M

# Fitting parameters.
fit_a = 11.29340696
fit_b =  0.02318951
fit_c =  0.08005103
fit_d =  0.12590226
fit_e =  0.32915862
fit_f = -2.41133187
fit_g =  2.86571184

# CDM
def sigmaM_fit_CDM(M):
    # M is in units of Msun.    
    return fit_a * (M/1.0e3)**(-fit_b) * (1.0 - fit_c*np.log(1.0 + fit_d*np.sqrt(M/1.0e3)))

def alphaM_fit_CDM(M):
    # M is in units of Msun.
    return -(fit_b + 0.5*fit_c*(1.0 - 1.0       / (1.0 + fit_d*np.sqrt(M/1.0e3))) \
                              /(1.0 - fit_c*np.log(1.0 + fit_d*np.sqrt(M/1.0e3))) \
            )

# FDM
def sigmaM_fit_supression(M, ma):
    # M is in units of Msun.
    # ma is in units of eV.    
    M0 = M_half(ma)
    
    return ( 1.0 + (M / (fit_e * M0))**fit_f )**(fit_b / fit_f)

def alphaM_fit_supression(M, ma):
    # M is in units of Msun.
    # ma is in units of eV.
    M0 = M_half(ma)
    
    return ( 1.0 + (M / (fit_e * M0))**fit_f )**(fit_g / fit_f)

def sigmaM_fit_FDM(M, ma):
    # M is in units of Msun.
    # ma is in units of eV.
    return sigmaM_fit_CDM(M) * sigmaM_fit_supression(M, ma)

def alphaM_fit_FDM(M, ma):
    # M is in units of Msun.
    # ma is in units of eV.
    return alphaM_fit_CDM(M) * alphaM_fit_supression(M, ma)

