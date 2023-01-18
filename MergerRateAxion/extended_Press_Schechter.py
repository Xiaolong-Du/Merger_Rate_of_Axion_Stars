import numpy as np
from scipy import special

# Extended Press-Schechter theory (Lacey et al. 1993).

def f(S, delta):
    nu = delta/np.sqrt(S)
    return 1.0/np.sqrt(2.0*np.pi)*nu*np.exp(-0.5*nu**2)/S

def f12(S1, S2, delta1, delta2):
    return np.where(S1 <= S2, 0.0, f(S1-S2, delta1-delta2))

def f21(S2, S1, delta2, delta1):
    return np.where(S1 <= S2, 0.0, f12(S1, S2, delta1, delta2)*f(S2, delta2)/f(S1, delta1))

def f12_cumulative(S1, S2, delta1, delta2):
    return np.where(S1 <= S2,                                         \
                    0.0,                                              \
                    special.erf((delta1-delta2)/np.sqrt(2.0*(S1-S2))) \
                   )

def f21_cumulative(S2, S1, delta2, delta1):
    return np.where(S1 <= S2,
                    0.0,
                    +0.5*(delta1-2.0*delta2)/delta1*np.exp(2.0*delta2*(delta1-delta2)/S1)            \
                        *special.erfc((S1*delta2+S2*(delta1-2.0*delta2))/np.sqrt(2.0*S1*S2*(S1-S2))) \
                    +0.5*special.erfc((S1*delta2-S2* delta1            )/np.sqrt(2.0*S1*S2*(S1-S2)))
                   )

# Sheth-Tormen
def f_ST(S, delta):
    A  = 0.3221836349
    q  = 0.707
    p  = 0.3
    nu = delta/np.sqrt(S)
    
    nuPrime = np.sqrt(q)*nu
    
    return A/np.sqrt(2.0*np.pi)*nuPrime*(1.0+nuPrime**(-2.0*p)) \
            *np.exp(-0.5*nuPrime**2)/S

# Modification on the merging rate from Parkinson et al. 2008.
G0     = +0.57
gamma1 = +0.38
gamma2 = -0.01
    
def f_mod_Parkinson_2008(S1, S2, delta1, delta2):    
    sigam1 = np.sqrt(S1)
    sigma2 = np.sqrt(S2)
    
    return np.where(S1 <= S2,                                          \
                    0.0,                                               \
                    G0*(sigam1/sigma2)**gamma1*(delta2/sigma2)**gamma2 \
                   )

def f12_Parkinson_2008(S1, S2, delta1, delta2):
    return f12(S1, S2, delta1, delta2)*f_mod_Parkinson_2008(S1, S2, delta1, delta2)

def f12_Parkinson_2008_cumulative_First_Order(S1, S2, delta1, delta2):
    # Valid only when  S1-S2 >> (delta1-delta2)**2
    if (np.any((delta1-delta2)**2/2.0/(S1-S2) > 1.0e-3)):
        print("Inaccurate Results!")
        sys.exit()
    
    sigam1 = np.sqrt(S1)
    sigma2 = np.sqrt(S2)
    return np.where(S1 <= S2,                                                       \
                    0.0,                                                            \
                    -G0*(sigam1/sigma2)**gamma1*(delta2/sigma2)**gamma2             \
                       *np.sqrt(2.0/np.pi)                                          \
                       *S1*(delta1-delta2)                                          \
                       /(S1-S2)**(3.0/2.0)                                          \
                       *special.hyp2f1(1.0,3.0/2.0,3.0/2.0-gamma1/2.0, -S2/(S1-S2)) \
                       /(gamma1-1.0)                                                \
                   )

def f12_Parkinson_2008_cumulative(S1, S2, delta1, delta2):
    # Valid only when  S1-S2 >> (delta1-delta2)**2
    if (np.any((delta1-delta2)**2/2.0/(S1-S2) > 1.0e-1)):
        print("Inaccurate Results!")
        sys.exit()
    
    sigam1 = np.sqrt(S1)
    sigma2 = np.sqrt(S2)
    
    return np.where(S1 <= S2,                                                         \
                    0.0,                                                              \
                    -G0*(sigam1/sigma2)**gamma1*(delta2/sigma2)**gamma2               \
                       *np.sqrt(2.0/np.pi)                                            \
                       *S1*(delta1-delta2)                                            \
                       /(S1-S2)**(3.0/2.0)                                            \
                       *(+special.hyp2f1(1.0,3.0/2.0,3.0/2.0-gamma1/2.0, -S2/(S1-S2)) \
                         /(gamma1-1.0)                                                \
                         -0.5*(delta1-delta2)**2/(S1-S2)                              \
                         *special.hyp2f1(1.0,5.0/2.0,5.0/2.0-gamma1/2.0, -S2/(S1-S2)) \
                         /(gamma1-3.0)                                                \
                        )
                   )

