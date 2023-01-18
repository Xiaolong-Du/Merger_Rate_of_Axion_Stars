# Merger Rate of Axion Stars
The python scripts compute the formation rate of dark matter halos at a certain mass and redshift. Combined with a relation between the halo mass and the mass of the axion star in the center of the halo, the results can be converted to the merger rate of axion stars. The details physics is described in the paper [citation].

## Files contained in the package

`MergerRateAxion/constants.py` : contains definitions of physical and astronomical constants

`MergerRateAxion/cosmology.py` : cosmological parameters and some useful functions

`MergerRateAxion/axion.py`     : things related to axion dark matter model, e.g. the core-halo mass relation

`MergerRateAxion/sigmaM.py`    : fitting functions for $\sigma(M)$ and $\alpha(M) = \frac{d\ln \sigma}{d\ln M}$

`MergerRateAxion/extended_Press_Schechter.py` : extended Press-Schechter formalism

`MergerRateAxion/formation_rate.py`: functions computing the formation rate of halos and axion stars

`test.py`      : an example of computing the axion dark matter fractional decay rate to photons due to soliton major mergers and parametric resonance (see [citation] for more details)

**Note that for computing $\sigma(M)$ we have taken the cosmological parameters from [Planck 2018 resutls](https://arxiv.org/abs/1807.06211):**

$\Omega_m = 0.3153$, $\Omega_b=0.04930$, $\Omega_\Lambda  = 0.6847$, $\sigma_8 =0.8111$, $n_s=0.9649$, $h=0.6736$.
