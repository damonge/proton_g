import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
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
            raise ValueError("Input E_cut = %d EeV not in attenuation file" % E_cut)
        msk = Ecut == E_cut
        return interp1d(z[msk], att[msk], bounds_error=False, fill_value=0)

class Gals(DiscreteTracer):
    def __init__(self, nz_arr, n_gals,
                 z_max=0.15, nz=1024):
        if nz_arr=='2MRS':
            self.nz_f = self._nz_2mrs
        else:
            self.nz_f = interp1d(nz_arr[0], nz_arr[1],
                                 bounds_error=False, fill_value=0)
        self.n_obj = n_gals
        self.sigma_beam = 0

    def get_nz(self, z, cosmo):
        return self.nz_f(z)

    def _nz_2mrs(self, z):
        # From 1706.05422
        m = 1.31
        beta = 1.64
        x = z / 0.0266
        return x**m * np.exp(-x**beta)

def get_cl(ell, cosmo, t1, b1, t2=None, b2=None, ell_pivots=None):
    if ell_pivots is None:
        lmx = np.amax(ell)+1
        ell_pivots = np.unique(np.geomspace(2, lmx,
                                            int(100 * np.log10(lmx) / 3.)).astype(int)).astype(float)
    if b2 is None:
        b2 = b1

    ct1 = t1.get_tracer(cosmo, b1)
    if t2 is None:
        ct2 = ct1
    else:
        ct2 = t2.get_tracer(cosmo, b2)
        
    cl = ccl.angular_cl(cosmo, ct1, ct2, ell_pivots)
    cli = interp1d(np.log(ell_pivots), np.log(cl))
    return np.exp(cli(np.log(ell)))
