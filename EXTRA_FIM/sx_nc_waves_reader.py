__author__ = "Christoph Freysoldt"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__maintainer__ = "Christoph Freysoldt"
__email__ = "freysoldt@mpie.de"
__date__ = "February, 2024"

from .waves_reader_abc import waves_reader_abc
import netCDF4
import scipy
import numpy as np


class sx_nc_waves_reader(waves_reader_abc):
    """This is the wave function reader for netcdf wave function files from SPHInX"""

    HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]

    def __init__(self, waves_file):
        """Constructor

        waves_file ... name of file (waves.sxb netcdf format)
        """
        self.nc_wf = netCDF4.Dataset(waves_file)

        # read dimensions
        self._nspin = len(self.nc_wf.dimensions["nSpin"])

        # read small arrays
        self._nstates = int(self.nc_wf["nPerK"][0])
        self._mesh = np.asarray(self.nc_wf["meshDim"])
        self.k_weights = np.asarray(self.nc_wf["kWeights"])
        self.k_vec = np.asarray(self.nc_wf["kVec"])
        self._n_gk = np.asarray(self.nc_wf["nGk"])
        self._eps = np.asarray(self.nc_wf["eps"]).reshape(
            self.nk, self._nspin, self._nstates
        )

        # load mapping from compact storage to FFT mesh
        self._fft_idx = []
        off = 0
        for ngk in self._n_gk:
            self._fft_idx.append(self.nc_wf["fftIdx"][off : off + ngk])
            off += ngk

    def get_psi(self, i, ispin, ik):
        """Get wave function for state i, spin ispin, k-point ik

        Wave function is returned in real space on the (Nx,Ny,Nz) mesh
        """
        return np.fft.ifftn(self.get_psi_rec(i, ispin, ik))

    def get_psi_rec(self, i, ispin, ik, compact=False):
        """
        Loads a single wavefunction on full FFT mesh of shape mesh.
        params: i: state index (int)
               ispin: spin index (int)
               ik: k index (int)
               compact: (bool)
        returns:
            res: complex valued wavefunction indexed by (i,ispin,ik) loaded on to the FFT mesh.
            compact_wave: compact wavefunctions, without loading on FFT mesh.
        """
        # translate indices to pythonic style.
        i = np.arange(self._nstates)[i]
        ik = np.arange(self.nk)[ik]
        ispin = np.arange(self._nspin)[ispin]

        off = self._n_gk[ik] * (i + ispin * self._nstates)
        psire = self.nc_wf[f"psi-{ik+1}.re"][off : off + self._n_gk[ik]]
        psiim = self.nc_wf[f"psi-{ik+1}.im"][off : off + self._n_gk[ik]]
        compact_wave = psire + 1j * psiim
        if compact:
            return compact_wave
        res = np.zeros(shape=self._mesh, dtype=np.complex128)
        res.flat[self._fft_idx[ik]] = compact_wave
        return res

    def get_eps(self, i, ispin, ik):
        """Get eigenvalue (in eV) for state i, spin ispin, k-point ik"""
        return self._eps[ik, ispin, i] * self.HARTREE_TO_EV

    def kweight(self, ik):
        """Get integration weight for k-point ik"""
        return self.k_weights[ik]

    def get_kvec(self, ik):
        """Get k-vector for k-point ik

        Returns numpy 3-vector in inverse atomic units
        """
        return self.k_vec[ik, :]

    @property
    def nk(self):
        """Get number of k-points"""
        return len(self.k_weights)

    @property
    def n_spin(self):
        """Get number of spin channels"""
        return self._nspin

    @property
    def n_states(self):
        """Get number of states"""
        return self._nstates

    @property
    def mesh(self):
        """Get (Nx, Ny, Nz)"""
        return self._mesh[0], self._mesh[1], self._mesh[2]

    @property
    def cell(self):
        """Get simulation cell (in bohr units)"""
        return np.asarray(self.nc_wf["cell"])

    def get_fermi_energy(self):
        """Get Fermi energy (in eV)"""
        # get occupations and eigenvalues
        focc = np.asarray(self.nc_wf["focc"])
        eps = np.asarray(self.nc_wf["eps"])

        # get the value of half occupation
        foccHalf = 0.5 * (np.round(np.max(focc)) + np.round(np.min(focc)))
        if np.abs(foccHalf * self.n_spin - 1.0) > 1e-6:
            raise RuntimeError(
                f"Suspicious value of half-filled occupation: {foccHalf}, should be {1/self.n_spin}"
            )
        # find the "occupied" states
        aboveHalf = focc > foccHalf
        # occupied: high occupations
        occ = focc[aboveHalf], eps[aboveHalf]
        # empty: low occupations
        unocc = focc[~aboveHalf], eps[~aboveHalf]
        # find top of occupied and bottom of empty states (index)
        itop = np.argmax(occ[1])
        ibottom = np.argmin(unocc[1])
        # interpolate linearly between these states to find the Fermi energy
        x = (foccHalf - occ[0][itop]) / (unocc[0][ibottom] - occ[0][itop])
        eFermi = x * unocc[1][ibottom] + (1 - x) * occ[1][itop]
        return eFermi * self.HARTREE_TO_EV
