import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from scipy.interpolate import interp1d
from utils import UHECRs, Gals

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


n_gal = 1E5
n_crs = 1E2
plot_nz = True
plot_cl_nl = True
plot_cl_el = True
plot_sn = True
ells = np.unique(np.geomspace(2, 1000, 100).astype(int)).astype(float)
ls_all = np.arange(2, 1000)

def get_cl_all_ell(t1, t2):
    cl = ccl.angular_cl(csm, t1, t2, ells)
    cli = interp1d(ells, np.log(cl))
    return np.exp(cli(ls_all))

csm=ccl.Cosmology(Omega_b=0.05,
                  Omega_c=0.25,
                  sigma8=0.8,
                  h=0.67,
                  n_s=0.96,
                  Omega_k=0)

u = UHECRs("data/att_2.4_proton.dat", n_crs, 2.)
g = Gals("2MRS", n_gal)

if plot_nz:
    zz = np.linspace(0.000001, 0.15, 1024)
    plt.figure()
    nz = g.get_nz(zz)
    plt.plot(zz, nz/np.amax(nz), 'k-', label=r'$\phi_g,\,\,{\rm 2MRS}$')
    for Ecut, c in zip(['60', '50', '40'],['r','b','y']):
        nz = u.get_nz(zz, Ecut, csm)
        plt.plot(zz, nz/np.amax(nz), c+'-',
                 label=r'$\phi_{\rm CR},\,\,E_{\rm cut}= %s\,{\rm EeV}$' % Ecut)
    plt.legend(loc='upper right', fontsize=13)
    plt.gca().tick_params(labelsize="large")
    plt.xlim([0,0.15])
    plt.ylim([0,1.1])
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$\phi(z)\,\,({\rm normalized})$', fontsize=15)
    plt.savefig("figures/phi.pdf", bbox_inches='tight')

if plot_cl_nl:
    tracer_gal = g.get_tracer(csm, bias=1.5)
    def get_cl(bias_cr, type='auto'):
        tracer_crs = u.get_tracer(csm, '60', bias=bias_cr)
        bl = 1#u.get_beam(ls_all)
        if type=='auto':
            cl = get_cl_all_ell(tracer_crs, tracer_crs) * bl**2
        else:
            cl = get_cl_all_ell(tracer_gal, tracer_crs) * bl
        return cl
    plt.figure()
    plt.fill_between(ls_all, get_cl(1.,'auto'), get_cl(2.,'auto'),
                     alpha=0.5, color='#FF1A00',
                     label=r'${\cal S}^{\rm CR\,CR}_\ell$')
    plt.fill_between(ls_all, get_cl(1.,'cross'), get_cl(2.,'cross'),
                     alpha=0.5, color='#1A21FF',
                     label=r'${\cal S}^{g\,{\rm CR}}_\ell$')
    plt.plot(ls_all, np.ones_like(ls_all) * 4 * np.pi / 1E2, 'k-',
             label=r'${\cal N}^{\rm CR\,CR}_\ell,\,\,(N_{\rm CR}=10^2)$')
    plt.plot(ls_all, np.ones_like(ls_all) * 4 * np.pi / 1E3, 'k--',
             label=r'${\cal N}^{\rm CR\,CR}_\ell,\,\,(N_{\rm CR}=10^3)$')
    plt.plot(ls_all, np.ones_like(ls_all) * 4 * np.pi / 1E4, 'k-.',
             label=r'${\cal N}^{\rm CR\,CR}_\ell,\,\,(N_{\rm CR}=10^4)$')
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.ylim([6E-6, 0.5])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower left', fontsize=13, ncol=2, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'$C_\ell$', fontsize=15)
    plt.savefig("figures/cl_nl.pdf", bbox_inches='tight')

if plot_cl_el:
    tracer_gal = g.get_tracer(csm, bias=1.5)
    tracer_crs = u.get_tracer(csm, '60', bias=1.5)
    bl = u.get_beam(ls_all)
    sl_cc = get_cl_all_ell(tracer_crs, tracer_crs) * bl**2
    sl_gc = get_cl_all_ell(tracer_gal, tracer_crs) * bl
    sl_gg = get_cl_all_ell(tracer_gal, tracer_gal)
    nl_cc_2 = np.ones_like(sl_cc) * 4 * np.pi / 1E2
    nl_cc_3 = np.ones_like(sl_cc) * 4 * np.pi / 1E3
    nl_gc = np.zeros_like(sl_gc)
    nl_gg = np.ones_like(sl_gg) * 4 * np.pi / 1E5
    cl_cc_2 = sl_cc + nl_cc_2
    cl_cc_3 = sl_cc + nl_cc_3
    cl_gc = sl_gc + nl_gc
    cl_gg = sl_gg + nl_gg
    el_cc_2 = np.sqrt(2 * cl_cc_2**2 / (2*ls_all + 1))
    el_gc_2 = np.sqrt((cl_cc_2 * cl_gg + cl_gc**2) / (2*ls_all + 1))
    el_cc_3 = np.sqrt(2 * cl_cc_3**2 / (2*ls_all + 1))
    el_gc_3 = np.sqrt((cl_cc_3 * cl_gg + cl_gc**2) / (2*ls_all + 1))
    el_gg = np.sqrt(2 * cl_gg**2 / (2*ls_all + 1))

    plt.figure()
    plt.fill_between(ls_all, sl_gc - el_gc_2, sl_gc + el_gc_2,
                     alpha=0.8, color='#8488FF', label=r'$N_{\rm CR}=10^2$')
    plt.fill_between(ls_all, sl_gc - el_gc_3, sl_gc + el_gc_3,
                     alpha=0.5, color='#1A21FF', label=r'$N_{\rm CR}=10^3$')
    plt.plot(ls_all, sl_gc, 'k-')
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.ylim([6E-6, 0.5])
    plt.gca().tick_params(labelsize="large")
    plt.loglog()
    plt.legend(loc='upper right', fontsize=13, ncol=2, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'${\cal S}^{g\,{\rm CR}}_\ell$', fontsize=15)
    plt.savefig("figures/cl_el_auto.pdf", bbox_inches='tight')

    plt.figure()
    plt.fill_between(ls_all, sl_cc - el_cc_2, sl_cc + el_cc_2,
                     alpha=0.8, color='#FFAAAA', label=r'$N_{\rm CR}=10^2$')
    plt.fill_between(ls_all, sl_cc - el_cc_3, sl_cc + el_cc_3,
                     alpha=0.5, color='#FF1A00', label=r'$N_{\rm CR}=10^3$')
    plt.plot(ls_all, sl_cc, 'k-')
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.ylim([6E-6, 0.5])
    plt.gca().tick_params(labelsize="large")
    plt.loglog()
    plt.legend(loc='upper right', fontsize=13, ncol=2, labelspacing=0.1)
    plt.xlabel(r'$\ell$', fontsize=15)
    plt.ylabel(r'${\cal S}^{\rm CR\,CR}_\ell$', fontsize=15)
    plt.savefig("figures/cl_el_cross.pdf", bbox_inches='tight')

if plot_sn:
    tracer_gal = g.get_tracer(csm, bias=1.5)
    tracer_crs = u.get_tracer(csm, '60', bias=1.5)
    bl = u.get_beam(ls_all)
    sl_cc = get_cl_all_ell(tracer_crs, tracer_crs) * bl**2
    sl_gc = get_cl_all_ell(tracer_gal, tracer_crs) * bl
    sl_gg = get_cl_all_ell(tracer_gal, tracer_gal)
    nl_cc_2 = np.ones_like(sl_cc) * 4 * np.pi / 1E2
    nl_cc_3 = np.ones_like(sl_cc) * 4 * np.pi / 1E3
    nl_gc = np.zeros_like(sl_gc)
    nl_gg = np.ones_like(sl_gg) * 4 * np.pi / 1E5
    cl_cc_2 = sl_cc + nl_cc_2
    cl_cc_3 = sl_cc + nl_cc_3
    cl_gc = sl_gc + nl_gc
    cl_gg = sl_gg + nl_gg

    dcl_A = np.array([2 * sl_cc])
    cov_A_2 = np.array([[2 * cl_cc_2**2 / (2*ls_all + 1)]])
    cov_A_3 = np.array([[2 * cl_cc_3**2 / (2*ls_all + 1)]])
    dcl_X = np.array([sl_gc])
    cov_X_2 = np.array([[(cl_cc_2 * cl_gg + cl_gc**2) / (2*ls_all + 1)]])
    cov_X_3 = np.array([[(cl_cc_3 * cl_gg + cl_gc**2) / (2*ls_all + 1)]])
    dcl_T = np.array([2 * sl_cc, sl_gc, 0 * sl_gg])
    cov_T_2 = np.array([[2*cl_cc_2**2 / (2*ls_all + 1),
                         2*cl_cc_2*cl_gc / (2*ls_all + 1),
                         2*cl_gc**2 / (2*ls_all + 1)],
                        [2*cl_cc_2*cl_gc / (2*ls_all + 1),
                         (cl_cc_2*cl_gg + cl_gc**2) / (2*ls_all + 1),
                         2*cl_gg*cl_gc / (2*ls_all + 1)],
                        [2*cl_gc**2 / (2*ls_all + 1),
                         2*cl_gg*cl_gc / (2*ls_all + 1),
                         2*cl_gg**2 / (2*ls_all + 1)]])
    cov_T_3 = np.array([[2*cl_cc_3**2 / (2*ls_all + 1),
                         2*cl_cc_3*cl_gc / (2*ls_all + 1),
                         2*cl_gc**2 / (2*ls_all + 1)],
                        [2*cl_cc_3*cl_gc / (2*ls_all + 1),
                         (cl_cc_3*cl_gg + cl_gc**2) / (2*ls_all + 1),
                         2*cl_gg*cl_gc / (2*ls_all + 1)],
                        [2*cl_gc**2 / (2*ls_all + 1),
                         2*cl_gg*cl_gc / (2*ls_all + 1),
                         2*cl_gg**2 / (2*ls_all + 1)]])

    def get_sn_l(d, c):
        cc = np.transpose(c, axes=[2,0,1])  # [Nl, Nt, Nt]
        dd = d.T  # [Nl, Nt]
        idd = np.linalg.solve(cc, dd) # [Nl, Nt]
        return np.sum(dd * idd, axis=1)
    sn2_A_2_l = get_sn_l(dcl_A, cov_A_2)
    sn2_A_3_l = get_sn_l(dcl_A, cov_A_3)
    sn2_X_2_l = get_sn_l(dcl_X, cov_X_2)
    sn2_X_3_l = get_sn_l(dcl_X, cov_X_3)
    sn2_T_2_l = get_sn_l(dcl_T, cov_T_2)
    sn2_T_3_l = get_sn_l(dcl_T, cov_T_3)

    sn_A_2_c = np.sqrt(np.cumsum(sn2_A_2_l))
    sn_A_3_c = np.sqrt(np.cumsum(sn2_A_3_l))
    sn_X_2_c = np.sqrt(np.cumsum(sn2_X_2_l))
    sn_X_3_c = np.sqrt(np.cumsum(sn2_X_3_l))
    sn_T_2_c = np.sqrt(np.cumsum(sn2_T_2_l))
    sn_T_3_c = np.sqrt(np.cumsum(sn2_T_3_l))
    
    plt.figure()
    plt.fill_between(ls_all, sn_A_2_c, sn_A_3_c, alpha=0.5, color='#FF1A00',
                     label='Auto-only')
    plt.fill_between(ls_all, sn_X_2_c, sn_X_3_c, alpha=0.5, color='#1A21FF',
                     label='Cross-only')
    plt.fill_between(ls_all, sn_T_2_c, sn_T_3_c, alpha=0.5, color='#AAAAAA',
                     label='All data')
    plt.plot(ls_all, 3.*np.ones_like(ls_all), 'k--', lw=1)
    plt.xlim([np.amin(ls_all), np.amax(ls_all)])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower right', fontsize=13)
    plt.xlabel(r'$\ell_{\rm max}$', fontsize=15)
    plt.ylabel(r'$(S / N)_{\ell \leq \ell_{\rm max}}$', fontsize=15)
    plt.savefig("figures/sn_lmax.pdf", bbox_inches='tight')

    sn_A_2_c = np.sqrt(np.cumsum(sn2_A_2_l[::-1]))[::-1]
    sn_A_3_c = np.sqrt(np.cumsum(sn2_A_3_l[::-1]))[::-1]
    sn_X_2_c = np.sqrt(np.cumsum(sn2_X_2_l[::-1]))[::-1]
    sn_X_3_c = np.sqrt(np.cumsum(sn2_X_3_l[::-1]))[::-1]
    sn_T_2_c = np.sqrt(np.cumsum(sn2_T_2_l[::-1]))[::-1]
    sn_T_3_c = np.sqrt(np.cumsum(sn2_T_3_l[::-1]))[::-1]
    
    plt.figure()
    plt.fill_between(ls_all, sn_A_2_c, sn_A_3_c, alpha=0.5, color='#FF1A00',
                     label='Auto-only')
    plt.fill_between(ls_all, sn_X_2_c, sn_X_3_c, alpha=0.5, color='#1A21FF',
                     label='Cross-only')
    plt.fill_between(ls_all, sn_T_2_c, sn_T_3_c, alpha=0.5, color='#AAAAAA',
                     label='All data')
    plt.plot(ls_all, 3.*np.ones_like(ls_all), 'k--', lw=1)
    plt.xlim([np.amin(ls_all), 300])
    plt.ylim([0.5, 36])
    plt.loglog()
    plt.gca().tick_params(labelsize="large")
    plt.legend(loc='lower left', fontsize=13)
    plt.xlabel(r'$\ell_{\rm min}$', fontsize=15)
    plt.ylabel(r'$(S / N)_{\ell \geq \ell_{\rm min}}$', fontsize=15)
    plt.savefig("figures/sn_lmin.pdf", bbox_inches='tight')

plt.show()
