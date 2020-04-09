import numpy as np
import pyccl as ccl
from scipy.special import erf, sici


class HaloProfileHOD(ccl.halos.HaloProfileNFW):

    def __init__(self, c_M_relation,
                 lMmin=11.87, lM0=11.87,
                 lM1=11.97, bg=0.72, bmax=6.4,
                 sigmaLogM=0.15, alpha=0.855):
        self.Mmin = 10.**lMmin
        self.M0 = 10.**lM0
        self.M1 = 10.**lM1
        self.sigmaLogM = sigmaLogM
        self.alpha = alpha
        self.bg = bg
        self.bmax = bmax
        super(HaloProfileHOD, self).__init__(c_M_relation)
        self._fourier = self._fourier_analytic_hod

    def _fourier_analytic_sat(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M
        c_M *= self.bmax/self.bg

        x = k_use[None, :] * R_s[:, None] * self.bg
        Si1, Ci1 = sici((1 + c_M[:, None]) * x)
        Si2, Ci2 = sici(x)

        P1 = 1 / (np.log(1+c_M) - c_M/(1+c_M))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
        prof = P1[:, None] * (P2 - P3)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _Nc(self, M):
        # Number of centrals
        return 0.5 * (1 + erf(np.log10(M / self.Mmin) / self.sigmaLogM))

    def _Ns(self, M):
        # Number of satellites
        return np.heaviside(M-self.M0, 1) * \
            (np.fabs(M - self.M0) / self.M1)**self.alpha

    def _fourier_analytic_hod(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use)
        Ns = self._Ns(M_use)
        # NFW profile
        uk = self._fourier_analytic_sat(cosmo, k_use, M_use, a, mass_def)

        prof = Nc[:, None] * (1 + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use)
        Ns = self._Ns(M_use)
        # NFW profile
        uk = self._fourier_analytic_sat(cosmo, k_use, M_use, a, mass_def)

        prof = Ns[:, None] * uk
        prof = Nc[:, None] * (2 * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class Profile2ptHOD(ccl.halos.Profile2pt):
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        return prof._fourier_variance(cosmo, k, M, a, mass_def)
