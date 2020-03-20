import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from scipy.interpolate import interp1d
from utils import UHECRs, Gals, get_cl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

cols = ['#0000AA', '#AAAA00', '#AA0000']
us = []
us.append(UHECRs(100, "data/attenuation_A1_e10.0_E20.0_g2.6.dat", 30, 1.))
us.append(UHECRs(59, "data/attenuation_A1_e10.0_E20.0_g2.6.dat", 200, 1.))
us.append(UHECRs(39, "data/attenuation_A1_e10.0_E20.0_g2.6.dat", 1000, 1.))
g = Gals("2MRS", 1E5)

plot_nz = True
plot_cl_nl = True
plot_cl_el = True
plot_sn = True
ls_all = np.arange(2, 1000)

cosmo=ccl.Cosmology(Omega_b=0.05,
                    Omega_c=0.25,
                    sigma8=0.8,
                    h=0.67,
                    n_s=0.96,
                    Omega_k=0)

if plot_nz:
    zz = np.linspace(0.000001, 0.15, 1024)
    plt.figure()
    nz = g.get_nz(zz, cosmo)
    plt.plot(zz, nz/np.amax(nz), 'k-', label=r'$\phi_g,\,\,{\rm 2MRS}$')
    for u, c in zip(us, cols):
        nz = u.get_nz(zz, cosmo)
        plt.plot(zz, nz/np.amax(nz), '-', color=c,
                 label=r'$\phi_{\rm CR},\,\,E_{\rm cut}= %d\,{\rm EeV}$' % u.E_cut)
    plt.legend(loc='upper right', fontsize=13)
    plt.gca().tick_params(labelsize="large")
    plt.xlim([0,0.15])
    plt.ylim([0,1.1])
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$\phi(z)\,\,({\rm normalized})$', fontsize=15)
    plt.savefig("figures/phi_v2.pdf", bbox_inches='tight')

if plot_cl_nl:
    plt.figure()
    for u, c in zip(us, cols):
        plt.fill_between(ls_all,
                         get_cl(ls_all, cosmo, u, 1.),
                         get_cl(ls_all, cosmo, u, 2.),
                         alpha=0.5, color=c,
                         label=r'$E_{\rm cut}=%d\,{\rm EeV}$' % u.E_cut)
        plt.plot(ls_all, u.get_nl(n_ells=len(ls_all)), '--', color=c)
    plt.ylim([5E-6, 5])
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower left', fontsize=13, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'${\cal S}^{\rm CR\,CR}_\ell$', fontsize=15)
    plt.savefig("figures/cl_nl_cc.pdf", bbox_inches='tight')

    plt.figure()
    for u, c in zip(us, cols):
        plt.fill_between(ls_all,
                         get_cl(ls_all, cosmo, u, 1., g, 1.5),
                         get_cl(ls_all, cosmo, u, 2., g, 1.5),
                         alpha=0.5, color=c,
                         label=r'$E_{\rm cut}=%d\,{\rm EeV}$' % u.E_cut)
        plt.plot(ls_all, u.get_nl(n_ells=len(ls_all)), '--', color=c)
    plt.ylim([5E-6, 0.99])
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower left', fontsize=13, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'${\cal S}^{g,{\rm CR}}_\ell$', fontsize=15)
    plt.savefig("figures/cl_nl_gc.pdf", bbox_inches='tight')

if plot_cl_el:
    plt.figure()
    for u, c in zip(us, cols):
        bl = u.get_beam(ls_all)
        sl_cc = get_cl(ls_all, cosmo, u, 1.5) * bl**2
        nl_cc = u.get_nl()
        cl_cc = sl_cc + nl_cc
        el_cc = np.sqrt(2 * cl_cc**2 / (2*ls_all + 1))
        plt.fill_between(ls_all, sl_cc - el_cc, sl_cc+el_cc,
                         alpha=0.5, color=c,
                         label=r'$E_{\rm cut}=%d\,{\rm EeV}$' % u.E_cut)
        plt.plot(ls_all, sl_cc, '-', color=c)
    plt.ylim([5E-6, 4])
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower left', fontsize=13, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'${\cal S}^{\rm CR\,CR}_\ell$', fontsize=15)
    plt.savefig("figures/cl_el_cc.pdf", bbox_inches='tight')

    plt.figure()
    for u, c in zip(us, cols):
        bl = u.get_beam(ls_all)
        sl_cc = get_cl(ls_all, cosmo, u, 1.5) * bl**2
        sl_gc = get_cl(ls_all, cosmo, u, 1.5, g, 1.5) * bl
        sl_gg = get_cl(ls_all, cosmo, g, 1.5)
        nl_cc = u.get_nl()
        nl_gg = g.get_nl()
        cl_cc = sl_cc + nl_cc
        cl_gg = sl_gg + nl_gg
        cl_gc = sl_gc
        el_gc = np.sqrt((cl_cc * cl_gg + cl_gc**2) / (2*ls_all + 1))
        plt.fill_between(ls_all, sl_gc - el_gc, sl_gc+el_gc,
                         alpha=0.5, color=c,
                         label=r'$E_{\rm cut}=%d\,{\rm EeV}$' % u.E_cut)
        plt.plot(ls_all, sl_gc, '-', color=c) 
    plt.ylim([5E-6, 0.4])
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower left', fontsize=13, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'${\cal S}^{g,{\rm CR}}_\ell$', fontsize=15)
    plt.savefig("figures/cl_el_cc.pdf", bbox_inches='tight')

if plot_sn:
    def get_sn(d, c, reverse=False):
        cc = np.transpose(c, axes=[2,0,1])  # [Nl, Nt, Nt]
        dd = d.T  # [Nl, Nt]
        idd = np.linalg.solve(cc, dd) # [Nl, Nt]
        sn_l = np.sum(dd * idd, axis=1) # [Nl]
        if reverse:
            sn = np.sqrt(np.cumsum(sn_l[::-1]))[::-1]
        else:
            sn = np.sqrt(np.cumsum(sn_l))
        return sn
    
    plt.figure(1001)
    plt.figure(1002)
    for u, c in zip(us, cols):
        bl = u.get_beam(ls_all)
        sl_cc = get_cl(ls_all, cosmo, u, 1.5) * bl**2
        sl_gc = get_cl(ls_all, cosmo, u, 1.5, g, 1.5) * bl
        sl_gg = get_cl(ls_all, cosmo, g, 1.5)
        nl_cc = u.get_nl()
        nl_gg = g.get_nl()
        cl_cc = sl_cc + nl_cc
        cl_gc = sl_gc
        cl_gg = sl_gg + nl_gg
        dcl_A = np.array([2 * sl_cc])
        cov_A = np.array([[2 * cl_cc**2 / (2*ls_all + 1)]])
        dcl_X = np.array([sl_gc])
        cov_X = np.array([[(cl_cc * cl_gg + cl_gc**2) / (2*ls_all + 1)]])
        dcl_T = np.array([2* sl_cc, sl_gc, 0 * sl_gg])
        cov_T = np.array([[2*cl_cc**2 / (2*ls_all + 1),
                           2*cl_cc*cl_gc / (2*ls_all + 1),
                           2*cl_gc**2 / (2*ls_all + 1)],
                          [2*cl_cc*cl_gc / (2*ls_all + 1),
                           (cl_cc * cl_gg + cl_gc**2) / (2*ls_all + 1),
                           2*cl_gg*cl_gc / (2*ls_all + 1)],
                          [2*cl_gc**2/ (2*ls_all + 1),
                           2*cl_gg*cl_gc / (2*ls_all + 1),
                           2*cl_gg**2 / (2*ls_all + 1)]])
        sn_A_lmax = get_sn(dcl_A, cov_A, reverse=False)
        sn_A_lmin = get_sn(dcl_A, cov_A, reverse=True)
        sn_X_lmax = get_sn(dcl_X, cov_X, reverse=False)
        sn_X_lmin = get_sn(dcl_X, cov_X, reverse=True)
        sn_T_lmax = get_sn(dcl_T, cov_T, reverse=False)
        sn_T_lmin = get_sn(dcl_T, cov_T, reverse=True)

        plt.figure(1001)
        plt.plot(ls_all, sn_A_lmax, '-.', color=c)
        plt.plot(ls_all, sn_X_lmax, '--', color=c)
        plt.plot(ls_all, sn_T_lmax, '-', color=c,
                 label=r'$E_{\rm cut}=%d\,{\rm EeV}$' % u.E_cut)
        plt.figure(1002)
        plt.plot(ls_all, sn_A_lmin, '-.', color=c)
        plt.plot(ls_all, sn_X_lmin, '--', color=c)
        plt.plot(ls_all, sn_T_lmin, '-', color=c,
                 label=r'$E_{\rm cut}=%d\,{\rm EeV}$' % u.E_cut)

    for i in [1001, 1002]:
        plt.figure(i)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim([0.5, 100])
        plt.xlim([np.amin(ls_all), np.amax(ls_all)])
        plt.plot([-1,-1],[-1,-1],'k-.', label=r'${\rm Auto-only}$')
        plt.plot([-1,-1],[-1,-1],'k--', label=r'${\rm Cross-only}$')
        plt.plot([-1,-1],[-1,-1],'k-', label=r'${\rm Auto + cross}$')
        plt.plot(ls_all, 3.*np.ones_like(ls_all), 'k--', lw=1)
        plt.gca().tick_params(labelsize="large")
    plt.figure(1001)
    plt.xlabel(r'$\ell_{\rm max}$', fontsize=15)
    plt.ylabel(r'$\left(\frac{S}{N}\right)_{\ell \leq \ell_{\rm max}}$', fontsize=15)
    plt.legend(loc='upper left', fontsize=13, labelspacing=0.1, ncol=2)
    plt.savefig("figures/sn_lmax_v2.pdf", bbox_inches='tight')
    plt.figure(1002)
    plt.xlabel(r'$\ell_{\rm min}$', fontsize=15)
    plt.ylabel(r'$\left(\frac{S}{N}\right)_{\ell \geq \ell_{\rm min}}$', fontsize=15)
    plt.legend(loc='upper right', fontsize=13, labelspacing=0.1, ncol=2)
    plt.savefig("figures/sn_lmin_v2.pdf", bbox_inches='tight')
plt.show()
