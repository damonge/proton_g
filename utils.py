import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from scipy.interpolate import interp1d

class UHECRs(object):
    def __init__(self, att_fname, n_crs, fwhm_deg):
        self.d_att = self._get_att_data(att_fname)
        self.n_crs = n_crs
        self.sigma_beam = np.radians(fwhm_deg) / 2.355

    def get_beam(self, ls):
        return np.exp(-0.5 * self.sigma_beam**2 * ls * (ls+1))

    def get_nz(self, z, E_cut, cosmo):
        att = self._get_att_extrapolated(E_cut, z)
        # Add (1+z) * H(z) factor
        a = 1./(1+z)
        nz = att * a / ccl.h_over_h0(cosmo, a)
        return nz
        
    def get_tracer(self, cosmo, E_cut, bias=1, z_max=0.15, nz=1024):
        zarr = np.linspace(0.000001, z_max, nz)
        bzarr = np.ones(nz) * bias
        nzarr = self.get_nz(zarr, E_cut, cosmo)
        return ccl.NumberCountsTracer(cosmo, False,
                                      dndz=(zarr, nzarr),
                                      bias=(zarr, bzarr))

    def _get_att_data(self, fname):
        chi, z, ecut, alpha = np.loadtxt(fname, unpack=True)
        ecut = ecut.astype(int)

        data = {}
        for e in np.unique(ecut):
            ids = ecut == e
            data['%d'%e] = {'chi': chi[ids], 'z': z[ids], 'att': alpha[ids]}
        return data

    def _get_att_extrapolated(self, E_cut, z_out):
        d_att = self.d_att[E_cut]
        att_f = interp1d(np.log(d_att['z']),
                         np.log(d_att['att']),
                         bounds_error=False,
                         fill_value=np.log(d_att['att'][0]))
        att_out = np.exp(att_f(np.log(z_out)))
        tilt = np.log(d_att['att'][-1]/d_att['att'][-2])/ \
               np.log(d_att['z'][-1]/d_att['z'][-2])
        id_out = np.where(z_out > d_att['z'][-1])[0]
        att_out[id_out]=d_att['att'][-1] * (z_out[id_out]/d_att['z'][-1])**tilt
        return att_out

class Gals(object):
    def __init__(self, nz, n_gals):
        if nz=='2MRS':
            self.nz = self._get_nz_2mrs()
        else:
            self.nz = nz
        self.n_gals = n_gals
        self.nzf = interp1d(self.nz[0], self.nz[1],
                            bounds_error=False,
                            fill_value=0)

    def get_nz(self, z):
        return self.nzf(z)

    def _get_nz_2mrs(self, z_max=0.15, nz=1024):
        # From 1706.05422
        z0 = 0.0266
        m = 1.31
        beta = 1.64
        zarr = np.linspace(0.000001, z_max, nz)
        xarr = zarr / z0
        nzarr = xarr**m * np.exp(-xarr**beta)
        return (zarr, nzarr)

    def get_tracer(self, cosmo, bias=1.):
        zarr = self.nz[0]
        bzarr = np.ones_like(zarr) * bias
        return ccl.NumberCountsTracer(cosmo, False,
                                      dndz=self.nz,
                                      bias=(zarr, bzarr))

    def get_beam(self, ls):
        return np.ones(len(ls))
