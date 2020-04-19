import numpy as np
import pyccl as ccl
from HOD import HaloProfileHOD, Profile2ptHOD
from scipy.interpolate import interp1d


class DiscreteTracer(object):
    def __init__(self, n_obj):
        self.n_obj = n_obj

    def get_nz(self, z, cosmo):
        raise NotImplementedError("N(z) not implemented for this tracer")

    def get_tracer(self, cosmo, bias, z_max=0.15, nz=1024):
        zarr = np.linspace(0.000001, z_max, nz)
        bzarr = np.ones(nz) * bias
        nzarr = self.get_nz(zarr, cosmo)
        return ccl.NumberCountsTracer(cosmo, False,
                                      dndz=(zarr, nzarr),
                                      bias=(zarr, bzarr))

    def get_beam(self, ls):
        return np.exp(-0.5 * self.sigma_beam**2 * ls * (ls+1))

    def get_nl(self, n_ells=None):
        nl_val = 4*np.pi/self.n_obj
        if n_ells is not None:
            return nl_val * np.ones(n_ells)
        else:
            return nl_val


class UHECRs(DiscreteTracer):
    def __init__(self, E_cut, att_fname, n_crs, fwhm_deg,
                 z_max=0.15, nz=1024):
        self.E_cut = E_cut
        self.n_obj = n_crs
        self.sigma_beam = np.radians(fwhm_deg) / 2.355
        self.att_f = self._get_att_function(att_fname, E_cut,
                                            z_max=z_max, nz=nz)

    def get_nz(self, z, cosmo):
        att = self.att_f(z)
        # Add (1+z) * H(z) factor
        a = 1./(1+z)
        nz = att * a / ccl.h_over_h0(cosmo, a)
        return nz

    def _get_att_function(self, fname, E_cut, z_max=0.15, nz=1024):
        z, Ecut, att = np.loadtxt(fname, skiprows=1, unpack=True)
        Ecut = (Ecut*1E-18).astype(int)
        Ecut_list = np.unique(Ecut)
        if E_cut not in Ecut_list:
            raise ValueError("Input E_cut = %d EeV not in file" % E_cut)
        msk = Ecut == E_cut
        return interp1d(z[msk], att[msk], bounds_error=False, fill_value=0)


class Gals(DiscreteTracer):
    def __init__(self, nz_arr, n_gals,
                 z_max=0.15, nz=1024, t_other=None, cosmo=None):
        if nz_arr == '2MRS':
            self.nz_f = self._nz_2mrs
        else:
            self.nz_f = interp1d(nz_arr[0], nz_arr[1],
                                 bounds_error=False, fill_value=0)
        self.z_f = z_max
        self.nz = nz

        self.n_obj = n_gals
        self.sigma_beam = 0

        z_arr = np.linspace(0, z_max, nz)
        self.nz_cl = self.nz_f
        if t_other is not None:
            nz_mine = self.get_nz(z_arr, cosmo)
            nz_othr = t_other.get_nz(z_arr, cosmo)
            nzmax = np.amax(nz_mine)
            z_good = nz_mine > 1E-3 * nzmax
            wz = np.zeros(nz)
            wz[z_good] = nz_othr[z_good] / nz_mine[z_good]
            fill = 0
            self.nz_cl = interp1d(z_arr, nz_mine * wz,
                                  bounds_error=False, fill_value=0)
        else:
            wz = np.ones(nz)
            fill = 1
        self.wz_f = interp1d(z_arr, wz, bounds_error=False,
                             fill_value=fill)

    def get_nz(self, z, cosmo):
        return self.nz_cl(z)

    def get_wz(self, z, cosmo):
        return self.wz_f(z)

    def _nz_2mrs(self, z):
        # From 1706.05422
        m = 1.31
        beta = 1.64
        x = z / 0.0266
        return x**m * np.exp(-x**beta)

    def get_nl(self, n_ells=None, integ_quad=False):
        from scipy.integrate import quad, simps

        def integ_0(z):
            return self.nz_f(z)

        def integ_1(z):
            return self.wz_f(z) * self.nz_f(z)

        def integ_2(z):
            return self.wz_f(z)**2 * self.nz_f(z)

        def integrator_quad(fz):
            return quad(fz, 0., self.z_f, limit=1000)[0]

        def integrator_simps(fz):
            return np.sum(fz(z_arr)) * np.mean(np.diff(z_arr))
            print(fz(z_arr))
            return simps(z_arr, fz(z_arr))

        def integrator(fz):
            if integ_quad:
                return integrator_quad(fz)
            else:
                return integrator_simps(fz)

        if not integ_quad:
            z_arr = np.linspace(0., self.z_f, self.nz)

        i0 = integrator(integ_0)
        i1 = integrator(integ_1) / i0
        i2 = integrator(integ_2) / i0
        ndens = self.n_obj / (4*np.pi)
        nl_val = i2 / (i1**2 * ndens)
        if n_ells is not None:
            return nl_val * np.ones(n_ells)
        else:
            return nl_val


def get_cl(ell, cosmo, t1, b1, t2=None, b2=None, ell_pivots=None, use_hm=False,
           get_1h=True, get_2h=True):
    if ell_pivots is None:
        lmx = np.amax(ell)+1
        ell_pivots = np.unique(
            np.geomspace(2, lmx,
                         int(100 *
                             np.log10(lmx) / 3.)).astype(int)).astype(float)
    if b2 is None:
        b2 = b1

    if use_hm:
        b1_use = 1
        b2_use = 1
        hmd = ccl.halos.MassDef200c(c_m='Prada12')
        cM = ccl.halos.ConcentrationPrada12(hmd)
        nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd)
        bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd)
        prof = HaloProfileHOD(cM)
        p2pt = Profile2ptHOD()
        hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)
        k_arr = np.geomspace(1E-4, 1E4, 512)
        a_arr = np.linspace(0.8, 1, 32)
        pk = ccl.halos.halomod_Pk2D(cosmo, hmc, prof, prof_2pt=p2pt,
                                    normprof1=True, get_1h=get_1h,
                                    get_2h=get_2h, lk_arr=np.log(k_arr),
                                    a_arr=a_arr)
    else:
        pk = None
        b1_use = b1
        b2_use = b2

    ct1 = t1.get_tracer(cosmo, b1_use)
    if t2 is None:
        ct2 = ct1
    else:
        ct2 = t2.get_tracer(cosmo, b2_use)

    cl = ccl.angular_cl(cosmo, ct1, ct2, ell_pivots, p_of_k_a=pk)
    cli = interp1d(np.log(ell_pivots), ell_pivots * cl)
    return cli(np.log(ell))/ell
