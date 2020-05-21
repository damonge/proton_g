import numpy as np
import utils as ut
import pyccl as ccl
import matplotlib.pyplot as plt


# Initialize a Cosmology
cosmo = ccl.Cosmology(Omega_b=0.05,
                      Omega_c=0.25,
                      sigma8=0.8,
                      h=0.67,
                      n_s=0.96,
                      Omega_k=0)

# First, initialize two tracers, one for galaxies, one for UHECRs
g = ut.Gals("2MRS", n_gals=1E5)
# Here, passing "2MRS" will ensure that you use the 2MRS window function.
u = ut.UHECRs(E_cut=63, att_fname='data/attenuation_A1_e10.0_E20.0_g2.6.dat',
              n_crs=200, fwhm_deg=1.)
# Let's make another one that uses optimal weights for a
# given UHECR tracers.
g_opt = ut.Gals("2MRS", n_gals=1E5, t_other=u,
                cosmo=cosmo)

# Not let's combine them to get C_ells.
# use_hm makes sure you use the halo model to compute the power spectrum.
# Otherwise it'll use a linear bias for whatever bias you pass (as b1 and b2).
ls = np.unique(np.geomspace(2, 1000, 100).astype(int)).astype(float)
cl_uu = ut.get_cl(ls, cosmo, t1=u, t2=u, b1=1., b2=1., use_hm=True)
cl_ug = ut.get_cl(ls, cosmo, t1=u, t2=g, b1=1., b2=1., use_hm=True)
cl_gg = ut.get_cl(ls, cosmo, t1=g, t2=g, b1=1., b2=1., use_hm=True)
cl_ug_opt = ut.get_cl(ls, cosmo, t1=u, t2=g_opt, b1=1., b2=1., use_hm=True)

# Make a plot
plt.plot(ls, cl_uu, 'k-', label='UxU')
plt.plot(ls, cl_ug, 'r-', label='UxG')
plt.plot(ls, cl_gg, 'b-', label='GxG')
plt.plot(ls, cl_ug_opt, 'y--', label='UxG, optimized')
plt.loglog()
plt.legend()
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$C_\ell$', fontsize=15)
plt.show()
