import numpy as np
import netCDF4
import scipy.constants
import scipy.optimize
from .waves_reader_abc import waves_reader_abc
from .sx_nc_waves_reader import sx_nc_waves_reader
from pathlib import Path


__author__ = "Shalini Bhatt"
__copyright__ = (
    "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__maintainer__ = "Shalini Bhatt"
__email__ = "s.bhatt@mpie.de"
__date__ = " April 26, 2023"

HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]


# class of 1D Numerov
class numerov_1D:
    def __init__(self, V1, h, izend):
        self.V1 = V1
        self.h = h
        self.Nz = V1.shape[0]
        self.izend = izend

    def num_step(self, psi_1, psi_2, k1, k2, k3, h):
        """function to set up initial values of wavefunction as numerov method can be used to determine for
        n=2,3,4.... given two initial values
        Args:

         psi_1: ground wavefunction
         psi_2 : first wavefunction
         k1 : k_n at position n
         k2 : k_n-1 at position n-1
         k3 : k_n+1 at position n+1
         h : step size
        """
        m = 2 * (1 - 5 / 12.0 * h**2 * k2) * psi_2
        n = (1 + 1 / 12.0 * h**2 * k1) * psi_1
        o = 1 + 1 / 12.0 * h**2 * k3
        return (m - n) / o

    def compute(self, E):
        """function to set up the values of k,psi1 ,psi2
        args :
        E : energy eigen value kept to be constant"""

        k = 2 * (E - self.V1)

        psi1 = np.zeros(self.Nz)
        psi1[-1] = 0
        psi1[-2] = 0.01

        for j in range(3, self.Nz + 1 - self.izend):
            psi1[-j] = self.num_step(
                psi1[2 - j], psi1[1 - j], k[2 - j], k[1 - j], k[-j], self.h
            )

            if abs(psi1[-j]) > 1000:  # kx = 0
                psi1 = psi1 * (1 / psi1[-j])

        return psi1 / psi1[self.izend]


# Generalised Numerov
class Residual_numerov_gen:
    def __init__(self, E, h):
        self.E = E
        self.h = h

    def __call__(self, psi):
        lhs = psi + self.h**2 / 6 * (self.E * psi - self.apply_V_n2(psi))
        return lhs - self.rhs


def numerov_gen(psi_n, psi_n1, apply_V_n, apply_V_n1, apply_V_n2, E, h):
    """This performs one Numerov step

    psi_n      ... psi at position n
    psi_n1     ... psi at position n-1
    apply_V_n  ... applies V at position n
    apply_V_n1 ... applies V at position n-1
    apply_V_n2 ... applies V at position n+1
    E          ... energy value
    h          ... step size
    """

    rhs = 2 * (psi_n - 5 / 6 * h**2 * (E * psi_n - apply_V_n(psi_n))) - (
        psi_n1 + 1 / 6 * h**2 * (E * psi_n1 - apply_V_n1(psi_n1))
    )

    def residual(psi):
        lhs = psi + h**2 / 6 * (E * psi - apply_V_n2(psi))
        return lhs - rhs

    initial_guess = np.zeros_like(psi_n)

    sol = scipy.optimize.root(residual, initial_guess, method="krylov")
    return sol.x


class fourier_plane_V:
    def __init__(self, V_in, n_in, k_square):
        self.k_square = k_square
        self.Vn = V_in[:, :, n_in]

    def __call__(self, psi):
        psi_k = np.fft.fft2(psi)
        psi_k = psi_k * self.k_square
        return np.fft.ifft2(psi_k) + self.Vn * psi


class Residual_extra(numerov_1D):
    def __init__(self, total_V, h, izend, eps, psi_match):
        self.izstart = None
        self.psi_g_real = None
        if len(total_V.shape) != 3:
            raise ValueError(f"Dimension is {len(total_V.shape)}, should be 3")
        V1 = np.mean(total_V, axis=(0, 1))
        self.total_V = total_V
        super().__init__(V1, h, izend)
        self.psi_match = psi_match

        self.Nz_max = V1.shape[0]
        self.eps = eps

    def iso_contour(self, gk_1, gk_2, k_vec, cutoff, izstart_min, limit):
        self.Nx = gk_1.shape[0]
        self.Ny = gk_2.shape[0]

        self.istart_list = np.full(
            (self.Nx, self.Ny), fill_value=izstart_min, dtype=int
        )

        self.psi_1D = np.zeros((self.Nx, self.Ny, self.Nz_max))
        self.izstart = self.izend
        self.k_square = np.zeros([self.Nx, self.Ny])
        for ikx in range(0, self.Nx):
            for iky in range(0, self.Ny):
                gk_vec = gk_1[ikx] + gk_2[iky] + [k_vec[0], k_vec[1], 0]
                gk_sqr = np.sum(gk_vec**2)
                # computing k_square
                self.k_square[ikx, iky] = (gk_sqr) * 0.5
                if self.k_square[ikx, iky] > cutoff:
                    self.k_square[ikx, iky] = 0
                # computing 1D wavefunctions
                wave_1D = self.compute(E=self.eps - 0.5 * gk_sqr)
                self.psi_1D[ikx, iky, :] = wave_1D
                for i in range(self.izend, izstart_min + 1):
                    if abs(wave_1D[i]) < limit:
                        self.istart_list[ikx, iky] = i
                        if i > self.izstart:
                            self.izstart = i
                        break
                    else:
                        if i == izstart_min:
                            self.izstart = izstart_min

    def vac_to_match(self, psi_bottom):
        psi_rec_num = np.zeros((self.Nx, self.Ny, self.Nz_max), dtype=np.complex128)
        psi_real_num = np.zeros_like(psi_rec_num)

        for z in range(self.izstart, self.izend, -1):
            psi_rec_num[:, :, z] = np.fft.fft2(psi_real_num[:, :, z])

            for ikx in range(0, self.Nx):
                for iky in range(0, self.Ny):
                    if z > self.istart_list[ikx, iky]:
                        psi_rec_num[ikx, iky, z] = 0  # set to zero
                    if z == self.istart_list[ikx, iky]:
                        psi_rec_num[ikx, iky, z] = psi_bottom[ikx, iky]
                        psi_rec_num[ikx, iky, z + 1] = psi_bottom[ikx, iky] * (
                            self.psi_1D[ikx, iky, z + 1] / self.psi_1D[ikx, iky, z]
                        )  # rescaling from good coeff.

            # back transform rec to real
            psi_real_num[:, :, z] = np.fft.ifft2(psi_rec_num[:, :, z])
            psi_real_num[:, :, z + 1] = np.fft.ifft2(psi_rec_num[:, :, z + 1])
            psi_real_num[:, :, z - 1] = numerov_gen(
                psi_real_num[:, :, z],
                psi_real_num[:, :, z + 1],
                fourier_plane_V(self.total_V, z, self.k_square),
                fourier_plane_V(self.total_V, z + 1, self.k_square),
                fourier_plane_V(self.total_V, z - 1, self.k_square),
                E=self.eps,
                h=self.h,
            )

        psi_rec_num[:, :, self.izend] = np.fft.fft2(psi_real_num[:, :, self.izend])

        for ikx in range(0, self.Nx):
            for iky in range(0, self.Ny):
                iz0 = self.istart_list[ikx, iky]
                for z in range(iz0 + 1, self.Nz_max - 1):
                    psi_rec_num[ikx, iky, z] = psi_rec_num[ikx, iky, iz0] * (
                        self.psi_1D[ikx, iky, z] / self.psi_1D[ikx, iky, iz0]
                    )  # rescaling from good coeff.

        # go back to real space
        for z in range(self.izend, self.Nz_max - 1):
            psi_real_num[:, :, z] = np.fft.ifft2(psi_rec_num[:, :, z])

        return psi_rec_num, psi_real_num

    def __call__(self, psi_vac):
        psi_vac_rescaled = np.zeros_like(psi_vac)
        for ikx in range(0, self.Nx):
            for iky in range(0, self.Ny):
                psi_vac_rescaled[ikx, iky] = (
                    psi_vac[ikx, iky]
                    * self.psi_1D[ikx, iky, self.istart_list[ikx, iky]]
                    / self.psi_1D[ikx, iky, self.izend]
                )

        self.psi_g_rec, self.psi_g_real = self.vac_to_match(psi_vac_rescaled)
        return self.psi_match - self.psi_g_rec[:, :, self.izend]


class extra_waves:
    def __init__(self, inputDict, reader=None, pot=None):
        self.inputDict = inputDict
        if reader is None:
            self.dft_wv = sx_nc_waves_reader(
                Path(inputDict["working_directory"]) / "waves.sxb"
            )
        else:
            self.dft_wv = reader
            # check that reader has the required signature
            if not isinstance(reader, waves_reader_abc):
                missing = [
                    m
                    for m in waves_reader_abc.__abstractmethods__
                    if not hasattr(reader, m)
                ]
                if len(missing) > 0:
                    raise TypeError(
                        "Incomplete reader object\n"
                        + str("\n").join([f"Missing '{m}'" for m in missing])
                    )

        if pot is None:
            self.total_V, _, self.dz, cell = potential(inputDict).potential_cell()
        else:
            self.total_V = pot
            cell = self.dft_wv.cell
            _, self.dz = np.linspace(0, cell[2, 2], self.dft_wv.mesh[2], retstep=True)
        rec_cell = (
            np.linalg.inv(cell) * 2 * np.pi
        )  # get cell coordinates from potential file

        self.gk_1 = np.outer(np.fft.fftfreq(self.Nx, 1 / self.Nx), rec_cell[0])
        self.gk_2 = np.outer(np.fft.fftfreq(self.Ny, 1 / self.Ny), rec_cell[1])

        self.Nz = int(self.inputDict["z_max"] / self.dz)

    @property
    def Nx(self):
        return self.total_V.shape[0]

    @property
    def Ny(self):
        return self.total_V.shape[1]

    def get_psi(self, i, ispin, ik):
        """compute EXTRA wavefunctions of given i ,ispin,ik
        provides real space psi on the mesh of potential
        """
        psi_real = self.dft_wv.get_psi(i, ispin, ik)

        psi_match = np.fft.fft2(psi_real[:, :, self.inputDict["izend"]])
        # nrm_psi = np.linalg.norm(psi_match) # needs to be figured out
        nrm_psi = 1e4
        psi_match_1 = psi_match / nrm_psi

        residual = Residual_extra(
            self.total_V[:, :, : self.Nz, ispin],
            self.dz,
            self.inputDict["izend"],
            self.dft_wv.get_eps(i, ispin, ik) / HARTREE_TO_EV,
            psi_match_1,
        )

        istart_list = residual.iso_contour(
            self.gk_1,
            self.gk_2,
            self.dft_wv.get_kvec(ik),
            self.inputDict["cutoff"],
            self.inputDict["izstart_min"],
            self.inputDict["limit"],
        )

        sol = scipy.optimize.root(
            residual, np.zeros([self.Nx, self.Ny], dtype=np.complex128), method="Krylov"
        )
        residual.psi_g_real *= nrm_psi
        psi_extra = residual.psi_g_real
        return psi_real, psi_extra
