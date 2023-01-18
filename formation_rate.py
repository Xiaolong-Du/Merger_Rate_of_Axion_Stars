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

# Fitting functions used to compute df/dz based on the halo mass function.
#
# df/dz = xi * dn/dlnM * M_star / rho_m
#
def xi_fit(Mh_crit, a, b, c):
    # Note that the fitting function is only accurate for 5 < z < 100, M_half < Mh_crit < 1.0e5 Msun.
    return (a + b * np.log10(Mh_crit/1.0e-6) ) / (1.0 + c * (Mh_crit/1.0e5)**0.39)

def Fit_Parms(z, a1, a2, a3):
    return a1 + a2 * z + a3 * z**2

def aFit(z, alpha=1.0/3.0):
    # Redshift dependence of the coefficient a in function xi_fit.
    # Here we provide the best-fit values for three cases, i.e. alpha=1/3, 2/5, 3/5.
    if   (alpha == 1.0/3.0):
        aFit_a1 =  0.1073693502105049
        aFit_a2 = -7.550823119030968e-5
        aFit_a3 = -9.394346936634254e-7
    elif (alpha == 2.0/5.0):
        aFit_a1 =  0.0986048913412091
        aFit_a2 = -7.077992922728815e-5
        aFit_a3 = -9.54236491420358e-7
    elif (alpha == 3.0/5.0):
        aFit_a1 =  0.08004088067732483
        aFit_a2 = -6.229715972804411e-5
        aFit_a3 = -9.316841492655746e-7
    else:
        print("Best-fit values for this alpha is not availble yet! Use the function Compute_df_dz_Parkinson_2008_Major_Merger_FDM instead.")
        sys.exit()

    return Fit_Parms(z, aFit_a1, aFit_a2, aFit_a3)

def bFit(z, alpha=1.0/3.0):
    # Redshift dependence of the coefficient b in function xi_fit.
    # Here we provide the best-fit values for three cases, i.e. alpha=1/3, 2/5, 3/5.
    if   (alpha == 1.0/3.0):
        bFit_a1 =  0.006846928532021523
        bFit_a2 = -1.4018636030025444e-05
        bFit_a3 = -1.8508182226681663e-07
    elif (alpha == 2.0/5.0):
        bFit_a1 =  0.006287499772726985
        bFit_a2 = -1.3638100526511303e-05
        bFit_a3 = -1.8643402238679904e-07
    elif (alpha == 3.0/5.0):
        bFit_a1 =  0.005100772893734679
        bFit_a2 = -1.3186809244289469e-05
        bFit_a3 = -1.6858987952414607e-07
    else:
        print("Best-fit values for this alpha is not availble yet! Use the function Compute_df_dz_Parkinson_2008_Major_Merger_FDM instead.")
        sys.exit()

    return Fit_Parms(z, bFit_a1, bFit_a2, bFit_a3)

def cFit(z, alpha=1.0/3.0):
    # Redshift dependence of the coefficient c in function xi_fit.
    # Here we provide the best-fit values for three cases, i.e. alpha=1/3, 2/5, 3/5.
    if   (alpha == 1.0/3.0):
        cFit_a1 =  0.2710837775787107
        cFit_a2 =  0.002475119558150311
        cFit_a3 =  6.897719788729967e-05
    elif (alpha == 2.0/5.0):
        cFit_a1 =  0.2658292263813366
        cFit_a2 =  0.002676604452159939
        cFit_a3 =  7.771099558819248e-05
    elif (alpha == 3.0/5.0):
        cFit_a1 =  0.25777613079473727
        cFit_a2 =  0.0027679089360191024
        cFit_a3 =  0.000102357782379973
    else:
        print("Best-fit values for this alpha is not availble yet! Use the function Compute_df_dz_Parkinson_2008_Major_Merger_FDM instead.")
        sys.exit()

    return Fit_Parms(z, cFit_a1, cFit_a2, cFit_a3)

