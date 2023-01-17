import numpy as np
import scipy.integrate as integrate
from scipy import interpolate

from cosmology import *
from sigmaM import *
from axion import *
from extended_Press_Schechter import *

# Halo mass function.
# CDM
def HMF_ST_CDM(M, z):
    S_base = sigmaM_fit_CDM(M)**2
    return  f_ST(S_base, deltaC(z))*rho_mean_code/M \
           *np.abs(alphaM_fit_CDM(M))*2.0*S_base/M

# FDM
def HMF_ST_FDM(M, z, ma):
    S_base = sigmaM_fit_FDM(M, ma)**2
    return  f_ST(S_base, deltaC(z))*rho_mean_code/M    \
           *np.abs(alphaM_fit_FDM(M, ma))*2.0*S_base/M

# Halo formation rate (Mitra et al. 2011).
# CDM
def Formation_Rate_CDM(M, z, dz=1.0e-3, epsL=1.0e-3, epsU=1.0e-3):
    S_base  = sigmaM_fit_CDM(M           )**2
    S_lower = sigmaM_fit_CDM(M*(1.0-epsL))**2
    S_upper = sigmaM_fit_CDM(M*     epsU )**2
    
    return  f(S_base, deltaC(z))                                       \
           *( f12_cumulative(S_lower, S_base, deltaC(z+dz), deltaC(z)) \
             -f12_cumulative(S_upper, S_base, deltaC(z+dz), deltaC(z)) \
            )/dz                                                       \
           *rho_mean_code/M                                            \
           *np.abs(alphaM_fit_CDM(M))*2.0*S_base/M

def Destruction_Rate_CDM(M, z, dz=1.0e-3, eps=1.0e-3):
    S_base  = sigmaM_fit_CDM(M          )**2
    S_upper = sigmaM_fit_CDM(M*(1.0+eps))**2
    
    return  f(S_base, deltaInterp(z+dz))                             \
           *f21_cumulative(S_upper, S_base, deltaC(z), deltaC(z+dz)) \
           /dz                                                       \
           *rho_mean_code/M                                          \
           *np.abs(alphaM_fit_CDM(M))*2.0*S_base/M

def f_Rate_CDM(M, z, dz=1.0e-3, eps=1.0e-3):
    S_base = sigmaM_fit_CDM(M)**2

    return  (f(S_base, deltaC(z))-f(S_base, deltaC(z+dz))) \
           /dz                                             \
           *rho_mean_code/M                                \
           *np.abs(alphaM_fit_CDM(M))*2.0*S_base/M

# Include the modification from Parkinson et al. 2008.
def Formation_Rate_Parkinson_2008_CDM(M, z, dz=1.0e-3, epsL=1.0e-3, epsU=1.0e-3):
    S_base  = sigmaM_fit_CDM(M           )**2
    S_lower = sigmaM_fit_CDM(M*(1.0-epsL))**2
    S_upper = sigmaM_fit_CDM(M*     epsU )**2
    
    return  f_ST(S_base, deltaC(z))                                                   \
           *( f12_Parkinson_2008_cumulative(S_lower, S_base, deltaC(z+dz), deltaC(z)) \
             -f12_Parkinson_2008_cumulative(S_upper, S_base, deltaC(z+dz), deltaC(z)) \
            )/dz                                                                      \
           *rho_mean_code/M                                                           \
           *np.abs(alphaM_fit_CDM(M))*2.0*S_base/M

def f_ST_Rate_CDM(M, z, dz=1.0e-3, eps=1.0e-3):
    S_base = sigmaM_fit_CDM(M)**2

    return  (f_ST(S_base, deltaC(z))-f_ST(S_base, deltaC(z+dz))) \
           /dz                                                   \
           *rho_mean_code/M                                      \
           *np.abs(alphaM_fit_CDM(M))*2.0*S_base/M

# FDM
def Formation_Rate_FDM(M, z, ma, dz=1.0e-3, epsL=1.0e-3, epsU=1.0e-3):
    S_base  = sigmaM_fit_FDM(M           , ma)**2
    S_lower = sigmaM_fit_FDM(M*(1.0-epsL), ma)**2
    S_upper = sigmaM_fit_FDM(M*     epsU , ma)**2
    
    return  f(S_base, deltaC(z))                                       \
           *( f12_cumulative(S_lower, S_base, deltaC(z+dz), deltaC(z)) \
             -f12_cumulative(S_upper, S_base, deltaC(z+dz), deltaC(z)) \
            )/dz                                                       \
           *rho_mean_code/M                                            \
           *np.abs(alphaM_fit_FDM(M, ma))*2.0*S_base/M

def Destruction_Rate_FDM(M, z, ma, dz=1.0e-3, eps=1.0e-3):
    S_base  = sigmaM_fit_FDM(M          , ma)**2
    S_upper = sigmaM_fit_FDM(M*(1.0+eps), ma)**2
    
    return  f(S_base, deltaInterp(z+dz))                             \
           *f21_cumulative(S_upper, S_base, deltaC(z), deltaC(z+dz)) \
           /dz                                                       \
           *rho_mean_code/M                                          \
           *np.abs(alphaM_fit_FDM(M, ma))*2.0*S_base/M

def f_Rate_FDM(M, z, ma, dz=1.0e-3, eps=1.0e-3):
    S_base = sigmaM_fit_FDM(M, ma)**2

    return  (f(S_base, deltaC(z))-f(S_base, deltaC(z+dz))) \
           /dz                                             \
           *rho_mean_code/M                                \
           *np.abs(alphaM_fit_FDM(M, ma))*2.0*S_base/M

# Include the modification from Parkinson et al. 2008.
def Formation_Rate_Parkinson_2008_FDM(M, z, ma, dz=1.0e-3, epsL=1.0e-3, epsU=1.0e-3):
    S_base  = sigmaM_fit_FDM(M           , ma)**2
    S_lower = sigmaM_fit_FDM(M*(1.0-epsL), ma)**2
    S_upper = sigmaM_fit_FDM(M*     epsU , ma)**2
    
    return  f_ST(S_base, deltaC(z))                                                   \
           *( f12_Parkinson_2008_cumulative(S_lower, S_base, deltaC(z+dz), deltaC(z)) \
             -f12_Parkinson_2008_cumulative(S_upper, S_base, deltaC(z+dz), deltaC(z)) \
            )/dz                                                                      \
           *rho_mean_code/M                                                           \
           *np.abs(alphaM_fit_FDM(M, ma))*2.0*S_base/M

def f_ST_Rate_FDM(M, z, ma, dz=1.0e-3, eps=1.0e-3):
    S_base = sigmaM_fit_FDM(M, ma)**2

    return  (f_ST(S_base, deltaC(z))-f_ST(S_base, deltaC(z+dz))) \
           /dz                                                   \
           *rho_mean_code/M                                      \
           *np.abs(alphaM_fit_FDM(M, ma))*2.0*S_base/M

####################################################################################################

# Compute df/dz.
# CDM
def Compute_df_dz_Parkinson_2008_Major_Merger_CDM(z, ma, ga_gamma, alpha, Meps=1.0e-3, dz=1.0e-3):
    # Critical halo mass.
    Mhalo_crit = M_halo_from_star_mass(M_decay(ma, ga_gamma), ma, z, alpha)
    
    # Mass resolution.
    Mres       = Meps * Mhalo_crit
    
    # Minimum halo mass at z.
    Mhalo_min  = M_halo_min(ma, z)
    
    # Major merger definition (M1 / M).
    # Assuming binary mergers: M = M1 + M2
    # (3/7)**alpha < M1/M2 < (7/3)**alpha
    majorMergerRatio = 3.0**(1.0/alpha) / (3.0**(1.0/alpha) + 7.0**(1.0/alpha))
    
    # Number density of newly formed halos with masses between [Mhalo_crit, 2 * Mhalo_crit].
    NTab       = 100
    MTab       = np.linspace(Mhalo_crit, 2.0*Mhalo_crit, NTab)
    MstarTab   = M_star(MTab, ma, z, alpha)
    nFormation = np.zeros(NTab)
    
    for i in range (NTab):
        # Assuming binary mergers: M = M1 + M2
        # Mres < M1, M2 < Mhalo_crit
        # (3/7)**alpha < M1/M2 < (7/3)**alpha
        MLow  = max(        Mhalo_min, MTab[i]-Mhalo_crit,      majorMergerRatio *MTab[i])
        MHigh = min(MTab[i]-Mhalo_min,         Mhalo_crit, (1.0-majorMergerRatio)*MTab[i])
        
        if (MLow >= MHigh or MTab[i] < Mhalo_min):
            nFormation[i] = 0.0
        else:
            S_base   = sigmaM_fit_CDM(MTab[i])**2
            S_lower  = sigmaM_fit_CDM(MHigh  )**2
            S_upper  = sigmaM_fit_CDM(MLow   )**2
            
            nFormation[i] =  f_ST(S_base, deltaC(z))                                                   \
                            *( f12_Parkinson_2008_cumulative(S_lower, S_base, deltaC(z+dz), deltaC(z)) \
                              -f12_Parkinson_2008_cumulative(S_upper, S_base, deltaC(z+dz), deltaC(z)) \
                             )/dz                                                                      \
                            *rho_mean_code/MTab[i]                                                     \
                            *np.abs(alphaM_fit_CDM(MTab[i]))*2.0*S_base/MTab[i]
    
    nFormationInterp = interpolate.interp1d(MTab, nFormation, kind='cubic',               \
                                            fill_value='extrapolate', bounds_error=False)
    
    # Mass contained in newly formed halos with mass [Mhalo_crit, 2 * Mhalo_crit].
    massFraction    , err = integrate.quad(lambda M:  nFormationInterp(M)*M,                \
                                           Mhalo_crit,  2.0*Mhalo_crit)
    
    # Star mass contained in newly formed halos with mass [Mhalo_crit, 2 * Mhalo_crit].
    starMassFraction, err = integrate.quad(lambda M:  nFormationInterp(M)*M_star(M, ma, z, alpha), \
                                           Mhalo_crit,  2.0*Mhalo_crit)
    
    return massFraction/rho_mean_code, starMassFraction/rho_mean_code

# FDM
def Compute_df_dz_Parkinson_2008_Major_Merger_FDM(z, ma, ga_gamma, alpha, Meps=1.0e-3, dz=1.0e-3):
    # Critical halo mass.
    Mhalo_crit = M_halo_from_star_mass(M_decay(ma, ga_gamma), ma, z, alpha)
    
    # Mass resolution.
    Mres       = Meps * Mhalo_crit
    
    # Minimum halo mass at z.
    Mhalo_min  = M_halo_min(ma, z)
    
    # Major merger definition (M1 / M).
    # Assuming binary mergers: M = M1 + M2
    # (3/7)**alpha < M1/M2 < (7/3)**alpha
    majorMergerRatio = 3.0**(1.0/alpha) / (3.0**(1.0/alpha) + 7.0**(1.0/alpha))
    
    # Number density of newly formed halos with masses between [Mhalo_crit, 2 * Mhalo_crit].
    NTab       = 100
    MTab       = np.linspace(Mhalo_crit, 2.0*Mhalo_crit, NTab)
    MstarTab   = M_star(MTab, ma, z, alpha)
    nFormation = np.zeros(NTab)
    
    for i in range (NTab):
        # Assuming binary mergers: M = M1 + M2
        # Mres < M1, M2 < Mhalo_crit
        # (3/7)**alpha < M1/M2 < (7/3)**alpha
        MLow  = max(        Mhalo_min, MTab[i]-Mhalo_crit,      majorMergerRatio *MTab[i])
        MHigh = min(MTab[i]-Mhalo_min,         Mhalo_crit, (1.0-majorMergerRatio)*MTab[i])
        
        if (MLow >= MHigh or MTab[i] < Mhalo_min):
            nFormation[i] = 0.0
        else:
            S_base   = sigmaM_fit_FDM(MTab[i], ma)**2
            S_lower  = sigmaM_fit_FDM(MHigh  , ma)**2
            S_upper  = sigmaM_fit_FDM(MLow   , ma)**2
            
            nFormation[i] =  f_ST(S_base, deltaC(z))                                                   \
                            *( f12_Parkinson_2008_cumulative(S_lower, S_base, deltaC(z+dz), deltaC(z)) \
                              -f12_Parkinson_2008_cumulative(S_upper, S_base, deltaC(z+dz), deltaC(z)) \
                             )/dz                                                                      \
                            *rho_mean_code/MTab[i]                                                     \
                            *np.abs(alphaM_fit_FDM(MTab[i], ma))*2.0*S_base/MTab[i]
    
    nFormationInterp = interpolate.interp1d(MTab, nFormation, kind='cubic',               \
                                            fill_value='extrapolate', bounds_error=False)
    
    # Mass contained in newly formed halos with mass [Mhalo_crit, 2 * Mhalo_crit].
    massFraction    , err = integrate.quad(lambda M:  nFormationInterp(M)*M,                \
                                           Mhalo_crit,  2.0*Mhalo_crit)
    
    # Star mass contained in newly formed halos with mass [Mhalo_crit, 2 * Mhalo_crit].
    starMassFraction, err = integrate.quad(lambda M:  nFormationInterp(M)*M_star(M, ma, z, alpha), \
                                           Mhalo_crit,  2.0*Mhalo_crit)
    
    return massFraction/rho_mean_code, starMassFraction/rho_mean_code
