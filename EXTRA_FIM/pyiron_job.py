"""Class to create pyiron jobs for ExtraFIM"""

from pathlib import Path
from pyiron_base.utils.error import ImportAlarm
from pyiron_base.jobs.master.parallel import ParallelMaster
from pyiron_base.jobs.job.jobtype import JobType
from pyiron_base import JobGenerator
from pyiron_base.jobs.job.template import TemplateJob
import h5py
import numpy as np
from pyiron_atomistics.sphinx.structure import read_atoms

try:
    import EXTRA_FIM.main as fim
    from EXTRA_FIM.potential import extend_potential, sx_el_potential3D_cell
    from EXTRA_FIM.datautils.pre_processing import suggest_input_dictionary
    from EXTRA_FIM.sx_nc_waves_reader import sx_nc_waves_reader
except ImportError:
    import_alarm = ImportAlarm("Unable to import EXTRA_FIM")

__author__ = "Christoph Freysoldt, Shyam Katnagallu"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1.0"
__maintainer__ = "Shyam Katnagallu"
__email__ = "s.katnagallu@mpie.de"
__status__ = "development"
__date__ = "March 5, 2024"


class ExtraFimSimulatorRefJob(TemplateJob):
    """Reference pyiron job for Extra FIM simulation."""

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input["waves_directory"] = None
        self.input["waves_reader"] = None
        self.input["kpoint"] = None
        self.input["ionization_energies"] = None
        self.input["extrapolate_potential"] = False
        self.input["total_kpoints"] = None

    def extrapolate_potential(self):
        """Extrapolate potential if needed, to extrapolate waves to higher distances"""

        elec_potential, _ = sx_el_potential3D_cell(
            self.input["simulator_dict"]["working_directory"]
        )
        pot, _, _, cell = fim.potential(self.input["simulator_dict"]).potential_cell()

        if self.input.extrapolate_potential:
            iz0 = self.input["simulator_dict"]["iz_ext_from"]
            new_z_max = self.input["simulator_dict"]["z_ext"]
            _, pot_ext = extend_potential(
                elec_potential / fim.HARTREE_TO_EV,
                iz0,
                pot,
                cell,
                z_max=new_z_max,
                izend=self.input["simulator_dict"]["izend"],
                dv_limit=1e-4,
                plotG=1,
            )
            # copy extension from pot to elec_potential
            elec_ext = pot_ext[:, :, :, 0] * fim.HARTREE_TO_EV
            elec_ext[:, :, 0:iz0] = elec_potential[:, :, :iz0]
        else:
            pot_ext = pot
            elec_ext = elec_potential
        return pot_ext, elec_potential
    
    def suggest_input_dict(self):
        """Suggests a input dictionary based on the electrostatic potential,
        Fermi and ionization energy"""
        waves_reader = sx_nc_waves_reader(
            Path(self.input["waves_directory"]) / "waves.sxb"
        )
        e_fermi = waves_reader.get_fermi_energy()
        _, sim = suggest_input_dictionary(
            self.input["waves_directory"],
            e_fermi,
            ionization_energies=self.input["ionization_energies"],
        )
        self.input["total_kpoints"] = waves_reader.nk
        self.input["simulator_dict"] = sim
        self.input["z_max"] = sim["z_max"]  # rename later
        self.input["izstart_min"] = sim["izstart_min"]  # rename later
        self.input["izend"] = sim["izend"]  # rename later
        self.input["limit"] = sim["limit"]  # rename later
        self.input["cutoff"] = sim["cutoff"]  # rename later
        self.input["E_fermi"] = sim["E_fermi"]  # rename later
        self.input["E_max"] = sim["E_max"]  # rename later
        return sim

    def run_static(self):

        self.project_hdf5.create_working_directory()
        # self.suggest_input_dict
        pot_ext, elec_ext = self.extrapolate_potential()
        waves_reader = sx_nc_waves_reader(self.input["waves_directory"] + "/waves.sxb")
        fimsim = fim.FIM_simulations(
            self.input["simulator_dict"],
            reader=waves_reader,
            V_total=pot_ext,
            V_elstat=elec_ext,
        )
        self.status.submitted = True
        fimsim.sum_single_k(self.input["kpoint"], path=self.working_directory)
        self.status.finished = True


class ExtraFimSimulatorJobGenerator(JobGenerator):
    """Job generator class for extra fim simulator pyiron jobs"""

    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        parameter_lst = []
        kpoints = self._master.input.get("kpoint")
        if kpoints is None:
            waves_reader = sx_nc_waves_reader(
                Path(self._master.input["waves_directory"]) / "waves.sxb"
            )
        for k in range(waves_reader.nk):
            parameter_lst.append(k)
        return parameter_lst

    def job_name(self, parameter):
        k_point = parameter
        return f"{self._master.job_name}_kpoint_{k_point}"

    def modify_job(self, job, parameter):
        job.structure = self._master.get_structure
        for k in self._master.input.keys():
            job.input[k] = self._master.input[k]
        job.input.kpoint = parameter
        return job


class ExtraFimSimulator(ParallelMaster):
    """ Pyiron Extra FIM simulator job class to make subjobs for each k point"""

    def __init__(self, project, job_name):
        super(ExtraFimSimulator, self).__init__(project, job_name)
        self.__version__ = "0.1.0"
        self.input["waves_directory"] = None
        self.input["waves_reader"] = None
        self.input["kpoint"] = None
        self.input["ionization_energies"] = None
        self.input["extrapolate_potential"] = False
        self.input["total_kpoints"] = None
        self._job_generator = ExtraFimSimulatorJobGenerator(self)
        self.ref_job = ExtraFimSimulatorRefJob(project=project, job_name=job_name)
        self.structure = None

    def extrapolate_potential(self):
        """returns extrapolated potential if true"""
        pot_ext, elec_potential = self.ref_job.extrapolate_potential()
        return pot_ext, elec_potential

    def suggest_input_dict(self):
        """Suggests a input dictionary based on the electrostatic potential,
        Fermi and ionization energy and populates input"""
        waves_reader = sx_nc_waves_reader(
            Path(self.input["waves_directory"]) / "waves.sxb"
        )
        e_fermi = waves_reader.get_fermi_energy()
        self.ref_job.iput["waves_directory"] = self.input["waves_directory"]
        self.ref_job.input["ionization_energies"] = self.input["ionization_energies"]
        self.ref_job.input["E_fermi"] = e_fermi
        _, sim = self.ref_job.suggest_input_dict()
        self.get_structure()
        return sim

    def get_structure(self):
        """Tries to get the relaxed sphinx structure if available"""
        if (Path(self.input["waves_directory"]) / "relaxedStr.sx").exists():
            self.structure = read_atoms(
                Path(self.input["waves_directory"]) / "relaxedStr.sx"
            )

    def collect_output(self):
        FIM_total = {}
        zFIM_total = {}
        for job_id in self.child_ids:
            subjob = self.project_hdf5.load(job_id)
            subjobdir = subjob.working_directory
            ik = subjob.input.kpoint
            IEs = subjob.input.ionization_energies
            with h5py.File(f"{subjobdir}/partial_dos{ik}.h5") as handle:
                for IE in IEs:
                    fim_k = np.asarray(handle[f"IE={IE}"])
                    zfim_k = np.asarray(handle[f"zIE={IE}"])
                    if IE not in FIM_total:
                        FIM_total[IE] = np.zeros_like(fim_k)
                        zFIM_total[IE] = np.zeros_like(zfim_k)
                    FIM_total[IE] += fim_k
                    zFIM_total[IE] += zfim_k

        for IE in IEs:
            self._output[f"total_FIM/{IE}"] = FIM_total[IE]
            self._output[f"z_resolved_FIM/{IE}"] = zFIM_total[IE]

        with self.project_hdf5.open("output") as hdf5_out:
            for key, val in self._output.items():
                hdf5_out[key] = val


JobType.register(ExtraFimSimulator)
