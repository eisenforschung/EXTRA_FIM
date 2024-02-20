__author__ = "Christoph Freysoldt"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__maintainer__ = "Christoph Freysoldt"
__email__ = "freysoldt@mpie.de"
__date__ = "February, 2024"

from abc import ABC, abstractmethod


class waves_reader_abc(ABC):
    """This is the abstract base class for wave function readers.

    It defines the interfaces that must be implemented.
    """

    @abstractmethod
    def get_psi(self, i, ispin, ik):
        """Get wave function for state i, spin ispin, k-point ik

        Wave function is returned in real space on the (Nx,Ny,Nz) mesh
        """
        pass

    @abstractmethod
    def get_eps(self, i, ispin, ik):
        """Get eigenvalue (in eV) for state i, spin ispin, k-point ik"""
        pass

    @abstractmethod
    def kweight(self, ik):
        """Get integration weight for k-point ik"""
        pass

    @abstractmethod
    def get_kvec(self, ik):
        """Get k-vector for k-point ik

        Returns numpy 3-vector in inverse atomic units
        """
        pass

    @property
    @abstractmethod
    def nk(self):
        """Get number of k-points"""
        pass

    @property
    @abstractmethod
    def n_spin(self):
        """Get number of spin channels"""
        pass

    @property
    @abstractmethod
    def n_states(self):
        """Get number of states"""
        pass

    @property
    @abstractmethod
    def mesh(self):
        """Get (Nx, Ny, Nz)"""
        pass

    @property
    @abstractmethod
    def cell(self):
        """Get simulation cell (in bohr units)"""
        pass
