import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from utils import UHECRs, Gals, get_cl
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

cols = ['#0000AA', '#AAAA00', '#AA0000']
us = []
us.append(UHECRs(100, "data/attenuation_A1_e10.0_E20.0_g2.6.dat", 30, 1.))
us.append(UHECRs(63, "data/attenuation_A1_e10.0_E20.0_g2.6.dat", 200, 1.))
us.append(UHECRs(39, "data/attenuation_A1_e10.0_E20.0_g2.6.dat", 1000, 1.))
gg = Gals("2MRS", 1E5)

save_cls = True
plot_fig1 = True
plot_fig2 = True
plot_fig3 = True
plot_fig4 = True
ls_all = np.arange(2, 1000)

cosmo = ccl.Cosmology(Omega_b=0.05,
                      Omega_c=0.25,
                      sigma8=0.8,
                      h=0.67,
                      n_s=0.96,
                      Omega_k=0)

if save_cls:
    for u in us:
        print(u.E_cut)
        cl1huu = get_cl(ls_all, cosmo, us[1], 1., t2=us[1], b2=1.,
                        use_hm=True, get_1h=True, get_2h=False)
        cl2huu = get_cl(ls_all, cosmo, us[1], 1., t2=us[1], b2=1.,
                        use_hm=True, get_1h=False, get_2h=True)
        cluu = cl1huu + cl2huu
        cl1hgu = get_cl(ls_all, cosmo, gg, 1., t2=us[1], b2=1.,
                        use_hm=True, get_1h=True, get_2h=False)
        cl2hgu = get_cl(ls_all, cosmo, gg, 1., t2=us[1], b2=1.,
                        use_hm=True, get_1h=False, get_2h=True)
        clgu = cl1hgu + cl2hgu
        np.savetxt("figures/cluu%d.txt" % (u.E_cut),
                   np.transpose([ls_all, cluu, cl1huu, cl2huu]),
                   header="[1]-ell [2]-1h+2h [3]-1h [4]-2h")
        np.savetxt("figures/clgu%d.txt" % (u.E_cut),
                   np.transpose([ls_all, clgu, cl1hgu, cl2hgu]),
                   header="[1]-ell [2]-1h+2h [3]-1h [4]-2h")
if plot_fig1:
    zz = np.linspace(0.000001, 0.15, 1024)
    plt.figure()
    nz = gg.get_nz(zz, cosmo)
    plt.plot(zz, nz/np.amax(nz), 'k-', label=r'$\phi_g,\,\,{\rm 2MRS}$')
    for u, c in zip(us, cols):
        nz = u.get_nz(zz, cosmo)
        plt.plot(zz, nz/np.amax(nz), '-', color=c,
                 label=r'$\phi_{\rm CR},\,\,E_{\rm cut}= %d\,{\rm EeV}$' % u.E_cut)
    plt.legend(loc='upper right', fontsize=13)
    plt.gca().tick_params(labelsize="large")
    plt.xlim([0, 0.15])
    plt.ylim([0, 1.1])
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$\phi(z)\,\,({\rm normalized})$', fontsize=15)
    plt.savefig("figures/phi.pdf", bbox_inches='tight')
if plot_fig2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    plt.subplots_adjust(wspace=0)

    for u, c in zip(us, cols):
        cl_1h_cc = get_cl(ls_all, cosmo, u, 1., t2=u, b2=1.,
                          use_hm=True, get_1h=True, get_2h=False)
        cl_1h_gc = get_cl(ls_all, cosmo, gg, 1., t2=u, b2=1.,
                          use_hm=True, get_1h=True, get_2h=False)
        cl_2h_cc = get_cl(ls_all, cosmo, u, 1., t2=u, b2=1.,
                          use_hm=True, get_1h=False, get_2h=True)
        cl_2h_gc = get_cl(ls_all, cosmo, gg, 1., t2=u, b2=1.,
                          use_hm=True, get_1h=False, get_2h=True)
        cl_cc = cl_1h_cc + cl_2h_cc
        cl_gc = cl_1h_gc + cl_2h_gc
        ax[0].plot(ls_all, cl_cc, '-', c=c)
        ax[0].plot(ls_all, cl_2h_cc, '--', c=c)
        ax[0].plot(ls_all, cl_1h_cc, ':', c=c)
        ax[1].plot(ls_all, cl_gc, '-', c=c)
        ax[1].plot(ls_all, cl_2h_gc, '--', c=c)
        ax[1].plot(ls_all, cl_1h_gc, ':', c=c)
    for x in ax:
        x.set_yscale('log')
        x.set_xscale('log')
        x.set_xlim([2, 1000])
        x.set_ylim([7E-6, 5])
        x.tick_params(labelsize="large")
    plt.savefig("figures/sl.pdf", bbox_inches='tight')
if plot_fig3:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    plt.subplots_adjust(wspace=0)

    gs = [Gals("2MRS", 1E5, t_other=u, cosmo=cosmo) for u in us]
    for u, g, c in zip(us, gs, cols):
        dl = 4
        bl = u.get_beam(ls_all)
        tl_cc = get_cl(ls_all, cosmo, u, 1., t2=u, b2=1.,
                       use_hm=True, get_1h=True, get_2h=True)
        tl_gc = get_cl(ls_all, cosmo, g, 1., t2=u, b2=1.,
                       use_hm=True, get_1h=True, get_2h=True)
        tl_gg = get_cl(ls_all, cosmo, g, 1., t2=g, b2=1.,
                       use_hm=True, get_1h=True, get_2h=True)
        tl_gcb = get_cl(ls_all, cosmo, gg, 1., t2=u, b2=1.,
                        use_hm=True, get_1h=True, get_2h=True)
        tl_ggb = get_cl(ls_all, cosmo, gg, 1., t2=gg, b2=1.,
                        use_hm=True, get_1h=True, get_2h=True)
        sl_cc = tl_cc * bl**2
        sl_gc = tl_gc * bl
        sl_gg = tl_gg
        sl_gcb = tl_gcb * bl
        sl_ggb = tl_ggb
        nl_cc = u.get_nl(n_ells=len(ls_all))
        nl_gc = np.zeros(len(ls_all))
        nl_gg = g.get_nl(n_ells=len(ls_all))
        nl_gcb = np.zeros(len(ls_all))
        nl_ggb = gg.get_nl(n_ells=len(ls_all))
        cl_cc = sl_cc + nl_cc
        cl_gc = sl_gc + nl_gc
        cl_gg = sl_gg + nl_gg
        cl_gcb = sl_gcb + nl_gcb
        cl_ggb = sl_ggb + nl_ggb
        err_prefac = 1/((2*ls_all+1)*dl)
        el_cc = np.sqrt(2*cl_cc**2*err_prefac)
        el_gc = np.sqrt((cl_cc*cl_gg + cl_gc**2)*err_prefac)
        el_gcb = np.sqrt((cl_cc*cl_ggb + cl_gcb**2)*err_prefac)
        ax[0].fill_between(ls_all, sl_cc-el_cc, sl_cc+el_cc,
                           alpha=0.5, color=c)
        ax[0].plot(ls_all, sl_cc, '-', c=c)
        ax[0].plot(ls_all, tl_cc, '--', c=c)
        ax[1].fill_between(ls_all, sl_gcb-el_gcb, sl_gcb+el_gcb,
                           alpha=0.5, color=c)
        ax[1].plot(ls_all, sl_gcb, '-', c=c)
        ax[1].plot(ls_all, tl_gcb, '--', c=c)
        ax[2].fill_between(ls_all, sl_gc-el_gc, sl_gc+el_gc,
                           alpha=0.5, color=c)
        ax[2].plot(ls_all, sl_gc, '-', c=c)
        ax[2].plot(ls_all, tl_gc, '--', c=c)
    for x in ax:
        x.set_yscale('log')
        x.set_xscale('log')
        x.set_xlim([2, 1000])
        x.set_ylim([7E-6, 5])
        x.tick_params(labelsize="large")
    plt.savefig("figures/cl_el.pdf", bbox_inches='tight')
if plot_fig4:
    def get_sn(d, c, reverse=False):
        cc = np.transpose(c, axes=[2, 0, 1])  # [Nl, Nt, Nt]
        dd = d.T  # [Nl, Nt]
        idd = np.linalg.solve(cc, dd)  # [Nl, Nt]
        sn_l = np.sum(dd * idd, axis=1)  # [Nl]
        if reverse:
            sn = np.sqrt(np.cumsum(sn_l[::-1]))[::-1]
        else:
            sn = np.sqrt(np.cumsum(sn_l))
        return sn

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    for x in ax.flatten():
        x.set_xscale('log')
        x.set_xlim([2, 1000])
        x.set_ylim([0, 35])
        x.tick_params(labelsize="large")

    gs = [Gals("2MRS", 1E5, t_other=u, cosmo=cosmo) for u in us]
    for u, g, c in zip(us, gs, cols):
        bl = u.get_beam(ls_all)
        tl_cc = get_cl(ls_all, cosmo, u, 1., t2=u, b2=1.,
                       use_hm=True, get_1h=True, get_2h=True)
        tl_gc = get_cl(ls_all, cosmo, g, 1., t2=u, b2=1.,
                       use_hm=True, get_1h=True, get_2h=True)
        tl_gg = get_cl(ls_all, cosmo, g, 1., t2=g, b2=1.,
                       use_hm=True, get_1h=True, get_2h=True)
        tl_gcb = get_cl(ls_all, cosmo, gg, 1., t2=u, b2=1.,
                        use_hm=True, get_1h=True, get_2h=True)
        tl_ggb = get_cl(ls_all, cosmo, gg, 1., t2=gg, b2=1.,
                        use_hm=True, get_1h=True, get_2h=True)
        sl_cc = tl_cc * bl**2
        sl_gc = tl_gc * bl
        sl_gg = tl_gg
        sl_gcb = tl_gcb * bl
        sl_ggb = tl_ggb
        nl_cc = u.get_nl(n_ells=len(ls_all))
        nl_gc = np.zeros(len(ls_all))
        nl_gg = g.get_nl(n_ells=len(ls_all))
        nl_gcb = np.zeros(len(ls_all))
        nl_ggb = gg.get_nl(n_ells=len(ls_all))
        cl_cc = sl_cc + nl_cc
        cl_gc = sl_gc + nl_gc
        cl_gg = sl_gg + nl_gg
        cl_gcb = sl_gcb + nl_gcb
        cl_ggb = sl_ggb + nl_ggb
        err_prefac = 1/((2*ls_all+1))
        dl_A = np.array([2 * sl_cc])
        cov_A = np.array([[2*cl_cc**2*err_prefac]])
        dl_X = np.array([sl_gc])
        cov_X = np.array([[(cl_cc*cl_gg + cl_gc**2)*err_prefac]])
        dl_Xb = np.array([sl_gcb])
        cov_Xb = np.array([[(cl_cc*cl_ggb + cl_gcb**2)*err_prefac]])
        dl_T = np.array([2*sl_cc, sl_gc, 0*sl_gg])
        cov_T = np.array([[2*cl_cc**2*err_prefac,
                           2*cl_cc*cl_gc*err_prefac,
                           2*cl_gc**2*err_prefac],
                          [2*cl_cc*cl_gc*err_prefac,
                           (cl_cc*cl_gg+cl_gc**2)*err_prefac,
                           2*cl_gg*cl_gc*err_prefac],
                          [2*cl_gc**2*err_prefac,
                           2*cl_gg*cl_gc*err_prefac,
                           2*cl_gg**2*err_prefac]])
        dl_Tb = np.array([2*sl_cc, sl_gcb, 0*sl_ggb])
        cov_Tb = np.array([[2*cl_cc**2*err_prefac,
                           2*cl_cc*cl_gcb*err_prefac,
                           2*cl_gcb**2*err_prefac],
                          [2*cl_cc*cl_gcb*err_prefac,
                           (cl_cc*cl_ggb+cl_gcb**2)*err_prefac,
                           2*cl_ggb*cl_gcb*err_prefac],
                          [2*cl_gcb**2*err_prefac,
                           2*cl_ggb*cl_gcb*err_prefac,
                           2*cl_ggb**2*err_prefac]])
        sn_A_lmax = get_sn(dl_A, cov_A, reverse=False)
        sn_A_lmin = get_sn(dl_A, cov_A, reverse=True)
        sn_X_lmax = get_sn(dl_X, cov_X, reverse=False)
        sn_X_lmin = get_sn(dl_X, cov_X, reverse=True)
        sn_T_lmax = get_sn(dl_T, cov_T, reverse=False)
        sn_T_lmin = get_sn(dl_T, cov_T, reverse=True)
        sn_Xb_lmax = get_sn(dl_Xb, cov_Xb, reverse=False)
        sn_Xb_lmin = get_sn(dl_Xb, cov_Xb, reverse=True)
        sn_Tb_lmax = get_sn(dl_Tb, cov_Tb, reverse=False)
        sn_Tb_lmin = get_sn(dl_Tb, cov_Tb, reverse=True)
        ax[1][0].plot(ls_all, sn_A_lmax, '-.', color=c)
        ax[1][0].plot(ls_all, sn_X_lmax, '--', color=c)
        ax[1][0].plot(ls_all, sn_T_lmax, '-', color=c)
        ax[1][1].plot(ls_all, sn_A_lmin, '-.', color=c)
        ax[1][1].plot(ls_all, sn_X_lmin, '--', color=c)
        ax[1][1].plot(ls_all, sn_T_lmin, '-', color=c)
        ax[0][0].plot(ls_all, sn_A_lmax, '-.', color=c)
        ax[0][0].plot(ls_all, sn_Xb_lmax, '--', color=c)
        ax[0][0].plot(ls_all, sn_Tb_lmax, '-', color=c)
        ax[0][1].plot(ls_all, sn_A_lmin, '-.', color=c)
        ax[0][1].plot(ls_all, sn_Xb_lmin, '--', color=c)
        ax[0][1].plot(ls_all, sn_Tb_lmin, '-', color=c)
    plt.savefig("figures/sn_cumul.pdf", bbox_inches='tight')
plt.show()
