import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from axion import *
from formation_rate import *

mpl.rcParams['mathtext.fontset'] = 'cm'

# Compare df/dz with HMF_ST(M_crit)*M_star.
# As can be seen, in most cases df/dz is proportional to the latter.
m_a   = 1.0e-11

#Here we fix the slope of core-halo mass relation at 1/3.
slp   = 1.0/3.0

zTab  = np.linspace(0.0, 100.0, 101)

massFractionTab1      = np.zeros(len(zTab))
starMassFractionTab1  = np.zeros(len(zTab))

massFractionTab2      = np.zeros(len(zTab))
starMassFractionTab2  = np.zeros(len(zTab))

massFractionTab3      = np.zeros(len(zTab))
starMassFractionTab3  = np.zeros(len(zTab))

massFractionTab4      = np.zeros(len(zTab))
starMassFractionTab4  = np.zeros(len(zTab))

for i in range(len(zTab)):
    g_a = 1.0e-10
    massFractionTab1[i], starMassFractionTab1[i] = \
      Compute_df_dz_Parkinson_2008_Major_Merger_FDM(zTab[i], ma=m_a, ga_gamma=g_a, alpha=slp, Meps=0.0, dz=1.0e-3)
    
    g_a = 1.0e-11
    massFractionTab2[i], starMassFractionTab2[i] = \
      Compute_df_dz_Parkinson_2008_Major_Merger_FDM(zTab[i], ma=m_a, ga_gamma=g_a, alpha=slp, Meps=0.0, dz=1.0e-3)
    
    g_a = 1.0e-12
    massFractionTab3[i], starMassFractionTab3[i] = \
      Compute_df_dz_Parkinson_2008_Major_Merger_FDM(zTab[i], ma=m_a, ga_gamma=g_a, alpha=slp, Meps=0.0, dz=1.0e-3)
    
    g_a = 1.0e-13
    massFractionTab4[i], starMassFractionTab4[i] = \
      Compute_df_dz_Parkinson_2008_Major_Merger_FDM(zTab[i], ma=m_a, ga_gamma=g_a, alpha=slp, Meps=0.0, dz=1.0e-3)

plt.figure(1)

plt.plot(zTab, starMassFractionTab1 , \
         label="$m_a=10^{-11}{\\rm eV},\,g_{a\\gamma\\gamma}=1.0\\times 10^{-10}{\\rm GeV}^{-1}$")
plt.plot(zTab, starMassFractionTab2, \
         label="$m_a=10^{-11}{\\rm eV},\,g_{a\\gamma\\gamma}=1.0\\times 10^{-11}{\\rm GeV}^{-1}$")
plt.plot(zTab, starMassFractionTab3, \
         label="$m_a=10^{-11}{\\rm eV},\,g_{a\\gamma\\gamma}=1.0\\times 10^{-12}{\\rm GeV}^{-1}$")
plt.plot(zTab, starMassFractionTab4, \
         label="$m_a=10^{-11}{\\rm eV},\,g_{a\\gamma\\gamma}=1.0\\times 10^{-13}{\\rm GeV}^{-1}$")

g_a = 1.0e-10
plt.plot(zTab, HMF_ST_FDM(M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp), zTab, m_a) \
              *M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp)                        \
              *M_decay(m_a, g_a)/rho_mean_code*0.12, ls='dashed', c='k',label="$\\rm Fitting\,\,function$")

g_a = 1.0e-11
plt.plot(zTab, HMF_ST_FDM(M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp), zTab, m_a) \
              *M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp)                        \
              *M_decay(m_a, g_a)/rho_mean_code*0.12, ls='dashed', c='k')

g_a = 1.0e-12
plt.plot(zTab, HMF_ST_FDM(M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp), zTab, m_a) \
              *M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp)                        \
              *M_decay(m_a, g_a)/rho_mean_code*0.12, ls='dashed', c='k')

g_a = 1.0e-13
plt.plot(zTab, HMF_ST_FDM(M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp), zTab, m_a) \
              *M_halo_from_star_mass(M_decay(m_a, g_a), m_a, zTab, alpha=slp)                        \
              *M_decay(m_a, g_a)/rho_mean_code*0.12, ls='dashed', c='k')

plt.xlabel("$z$"    , fontsize=14)
plt.ylabel("$df/dz$", fontsize=14)

plt.legend(fontsize=11)

plt.yscale("log")

#plt.savefig("df_dz_ma_12.pdf")
plt.show()

