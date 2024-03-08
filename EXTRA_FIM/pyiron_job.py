from pyiron_base.utils.error import ImportAlarm
from pyiron_base.jobs.master.parallel import ParallelMaster
from pyiron_base.jobs.job.jobtype import JobType
from pyiron_base import JobGenerator
from pyiron_base.jobs.job.template import TemplateJob

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
        self.__version__ = "0.1.0"
        self.input["waves_directory"] = None
        self.input["waves_reader"] = None
        self.input["kpoint"] = None
        self.input["structure"] = None
        self.input["ionization_energies"] = None
        self.input["extrapolate_potential"] = False

    def extrpolate_potential(self):
        """Extrapolate potential if needed, to extrapolate waves to higher distances"""

        elec_potential, _ = sx_el_potential3D_cell(
            self.input.simulator_dict["working_directory"]
        )
        pot, _, _, cell = fim.potential(self.input.simulator_dict).potential_cell()

        if self.input.extrapolate_potential:
            iz0 = self.input.simulator_dict["iz_ext_from"]
            new_z_max = self.input.simulator_dict["z_ext"]
            _, pot_ext = extend_potential(
                elec_potential / fim.HARTREE_TO_EV,
                iz0,
                pot,
                cell,
                z_max=new_z_max,
                izend=self.input.simulator_dict["izend"],
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

    @property
    def suggest_input_dict(self):
        """Suggests a input dictionary based on the electrostatic potential, Fermi and ionization energy"""
        waves_reader = sx_nc_waves_reader(self.input["waves_directory"] + "/waves.sxb")
        e_fermi = waves_reader.get_fermi_energy()
        _, sim = suggest_input_dictionary(
            self.input.waves_directory,
            e_fermi,
            ionization_energies=self.input["ionization_energies"],
        )
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
        pot_ext, elec_ext = self.extrpolate_potential()
        waves_reader = sx_nc_waves_reader(self.input["waves_directory"] + "/waves.sxb")
        fimsim = fim.FIM_simulations(
            self.input["simulator_dict"],
            reader=waves_reader,
            V_total=pot_ext,
            V_elstat=elec_ext,
        )
        self.project_hdf5.create_working_directory()
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
        kpoints = self._master.input.get("kpoints")
        if kpoints is None:
            waves_reader = sx_nc_waves_reader(
                self._master.input["waves_directory"] + "/waves.sxb"
            )
        for k in waves_reader.nk:
            parameter_lst.append(k)
        return parameter_lst

    def job_name(self, parameter):
        k_point = parameter[0]
        return f"{self._master.job_name}_kpoint_{k_point}"

    def modify_job(self, job, parameter):
        job.input.kpoint = parameter[1]
        return job


class ExtraFimSimulator(ParallelMaster,ExtraFimSimulatorRefJob):
    """ "Pyiron Extra FIM simulator job class to make subjobs for each k point"""

    def __init__(self, project, job_name):
        super(ExtraFimSimulatorRefJob,self).__init__(project, job_name=job_name)
        self._job_generator = ExtraFimSimulatorJobGenerator(self)
    #TODO: collect_output
    # def collect_output(self):
    #     for job_id in self.child_ids:
    #         job = self.project_hdf5.inspect(job_id)
    #     return super().collect_output()


JobType.register(ExtraFimSimulator)
JobType.register(ExtraFimSimulatorRefJob)
