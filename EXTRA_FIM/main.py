import numpy as np
import h5py
import netCDF4
import scipy.constants
import scipy.optimize
from pathlib import Path
from .extra import extra_waves


__author__ = "Shalini Bhatt"
__copyright__ = (
    "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__maintainer__ = "Shalini Bhatt"
__email__ = "s.bhatt@mpie.de"
__date__ = " April 26, 2023"

HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]


class potential:
    def __init__(self, inputDict):
        self.working_directory = inputDict["working_directory"]

    def potential_cell(self):
        v_file = netCDF4.Dataset(Path(self.working_directory) / "vElStat-eV.sxb")
        Potential = np.asarray(v_file["mesh"]).reshape(v_file["dim"])
        Potential = Potential / HARTREE_TO_EV
        vxc_file = netCDF4.Dataset(Path(self.working_directory) / "vXC.sxb")

        if "mesh" in vxc_file.variables:
            # non-spin calculations
            xc_Potential = np.asarray(vxc_file["mesh"]).reshape(vxc_file["dim"])
            total_V = np.zeros(list(Potential.shape) + [1], dtype=np.float64)
            total_V[:, :, :, 0] = np.add(Potential, xc_Potential)
        else:
            # spin calculations
            xc_Potential_0 = np.asarray(vxc_file["mesh-0"]).reshape(vxc_file["dim"])
            xc_Potential_1 = np.asarray(vxc_file["mesh-1"]).reshape(vxc_file["dim"])
            total_V = np.zeros(list(Potential.shape) + [2], dtype=np.float64)
            total_V[:, :, :, 0] = np.add(Potential, xc_Potential_0)
            total_V[:, :, :, 1] = np.add(Potential, xc_Potential_1)
        V1 = np.mean(total_V, axis=(0, 1))

        cell = np.asarray(v_file["cell"])
        rec_cell = (
            np.linalg.inv(cell) * 2 * np.pi
        )  # get cell coordinates from potential file

        # x_1D = np.linspace(0, cell[0, 0], Potential.shape[0])
        # dx = x_1D[1] - x_1D[0]
        # y_1D = np.linspace(0, cell[1, 1], Potential.shape[1])
        # dy = y_1D[1] - y_1D[0]
        _, dz = np.linspace(0, cell[2, 2], Potential.shape[2], retstep=True)

        return total_V, V1, dz, cell


class FIM_simulations:
    # Simulator = {
    #         'working_directory': 'working directory',
    #         'ik': 'ik',
    #         'z_max': 'zmax',
    #         'izstart_min': 'izstart_min',
    #         'izend': 'izend',
    #         'cutoff': 'cutoff',
    #         'limit': 'limit',
    #         'E_fermi': 'E_fermi',
    #         'E_max': 'E_max',
    #         'ionization_energies': 'ionization_energies'
    # }
    def __init__(self, inputDict, reader=None, V_total=None, V_elstat=None):
        """Parameters:
        inputDict   input dictionary with all settings
        reader      DFT wave function reader (must be derived from or
                    compatible to EXTRA_FIM.wave_reader_abc)
        V_total     effective potential for tail extrapolation (in Hartree)
        V_elstat    electrostatic potential for determining
                    the ionization position (in eV)
        """
        self.inputDict = inputDict
        self.extra = extra_waves(inputDict, reader=reader, pot=V_total)
        self.wf = self.extra.dft_wv

        if V_elstat is None:
            # note: potential for this class must be in eV
            self.V_elstat = self.extra.total_V * HARTREE_TO_EV
        else:
            self.V_elstat = V_elstat

    @property
    def Nx(self):
        return self.extra.Nx

    @property
    def Ny(self):
        return self.extra.Ny

    @property
    def Nz(self):
        return self.extra.Nz

    def search_V(self, V_target, V):
        if V_target < V[0]:
            raise ValueError(f"V_target={V_target} < V[0]={V[0]}")
        for iz in range(V.shape[0]):
            if V[iz] > V_target:
                lamda = (V_target - V[iz]) / (V[iz - 1] - V[iz])
                return iz - lamda
        raise ValueError("V_target=" + str(V_target) + "not found")

    def sum_all_states(self):
        """compute partial fim image for several ionization energies for all eigenstates between Efermi
        and Emax for all k points. Save the partial dos files"""
        for ik in range(0, self.wf.nk):
            sum_single_k(ik)

    def sum_single_k(self, ik):
        """compute partial fim image for several ionization energies for all eigenstates between Efermi
        and Emax for one k point (ik). Save the partial dos files"""

        # --- initialize the FIM sum arrays
        all_totals_dft = dict()
        all_totals_extra = dict()
        z_resolved = dict()
        for IE in self.inputDict["ionization_energies"]:
            # all_totals_dft[IE] = np.zeros((self.Nx, self.Ny), dtype=np.float64)
            all_totals_extra[IE] = np.zeros((self.Nx, self.Ny), dtype=np.float64)
            z_resolved[IE] = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)

        # --- loop over states/spins
        for ispin in range(self.wf.n_spin):
            # --- determine V1 for ionization location
            dim_elstat = len(self.V_elstat.shape)
            if dim_elstat == 1:
                V1 = self.V_elstat
            elif dim_elstat == 2:
                V1 = self.V_elstat[:, ispin]
            elif dim_elstat == 3:
                # TODO do search_V for each ix,iy
                V1 = np.mean(self.V_elstat, axis=(0, 1))
            elif dim_elstat == 4:
                # TODO do search_V for each ix,iy
                V1 = np.mean(self.V_elstat[:, :, :, ispin], axis=(0, 1))

            for i in range(0, self.wf.n_states):
                # --- select states in energy range E_fermi ... E_max
                if self.wf.get_eps(i, ispin, ik) < self.inputDict["E_fermi"]:
                    continue
                if self.wf.get_eps(i, ispin, ik) > self.inputDict["E_max"]:
                    continue

                # get wave function
                psi_dft, psi_extra = self.extra.get_psi(i, ispin, ik)

                # --- now sum the FIM signal
                for IE in self.inputDict["ionization_energies"]:
                    V_target = self.wf.get_eps(i, ispin, ik) + IE
                    z_plot = self.search_V(V_target, V1)  # floating point number
                    iz_plot = int(z_plot)  # getting integer part of it
                    lamda = z_plot - iz_plot

                    # linear interpolation of partial dos
                    # partial_dos_dft = (1 - lamda) * np.abs(psi_dft[:, :, iz_plot]) ** 2 \
                    #                 + lamda * np.abs(psi_dft[:, :, iz_plot + 1]) ** 2
                    partial_dos_extra = (1 - lamda) * np.abs(
                        psi_extra[:, :, iz_plot]
                    ) ** 2 + lamda * np.abs(psi_extra[:, :, iz_plot + 1]) ** 2

                    # all_totals_dft[IE]          += self.wf.kweight(ik) * partial_dos_dft
                    all_totals_extra[IE] += self.wf.kweight(ik) * partial_dos_extra
                    z_resolved[IE][:, :, iz_plot] += (
                        self.wf.kweight(ik) * partial_dos_extra
                    )

        # --- write output file
        filename = f"partial_dos{ik}.h5"
        with h5py.File(filename, "w") as handle:
            handle.create_dataset(
                "ionization_energies", data=self.inputDict["ionization_energies"]
            )
            for IE in all_totals_extra.keys():
                handle.create_dataset(f"IE={IE}", data=all_totals_extra[IE])
                handle.create_dataset(f"zIE={IE}", data=z_resolved[IE])
