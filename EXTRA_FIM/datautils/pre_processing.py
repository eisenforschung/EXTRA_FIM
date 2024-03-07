__author__ = "Christoph Freysoldt, Shyam Katnagallu"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
# __maintainer__ = ""
# __email__ = ""
__date__ = "February, 2024"

import numpy as np
import matplotlib.pyplot as plt
from mendeleev import element
import scipy

from .plotting import potential_figure
from EXTRA_FIM.potential import sx_el_potential1D_cell


class PreProcessingFIM:
    """
    Class for giving a suggested set of parameters for FIM simulation

    Attributes:
        job (str): Job information.
        imaging_gas (mendeleev.Element): Imaging gas element.

    Methods:
        find_constant_slope_regions(x, y, slope_threshold, second_derivative_threshold):
            Identify constant slope regions in a given dataset.

        suggest_input_dictionary(slope_threshold=0.1, second_derivative_threshold=0.001):
            Suggest a dictionary of input parameters for FIM simulation.

    """

    def __init__(self, job=None, imaging_gas=None):
        """
        Initialize PreProcessingFIM object.

        Parameters:
            job (str): Job information.
            imaging_gas (str): Symbol of the imaging gas element.

        """
        self.job = job
        self.imaging_gas = element(imaging_gas)

    def suggest_input_dictionary(
        self, slope_threshold=0.1, second_derivative_threshold=0.001
    ):
        """
        Suggest a dictionary of input parameters for FIM simulation.

        Parameters:
            slope_threshold (float): Threshold for slope difference.
            second_derivative_threshold (float): Threshold for the second derivative.

        Returns:
            tuple: Figure object and suggested simulator parameters.

        """
        return __module__.suggest_input_dictionary(
            self.job.working_directory,
            E_fermi=self.job["output/generic/dft/bands_e_fermi"][-1],
            ionization_energies=[self.imaging_gas.ionenergies[1]],
            slope_threshold=slope_threshold,
            second_derivative_threshold=second_derivative_threshold,
        )


# --- auxiliary function
def find_constant_slope_regions(x, y, slope_threshold, second_derivative_threshold):
    """
    Identify constant slope regions in a given dataset.

    Parameters:
        x (numpy.ndarray): x-coordinates.
        y (numpy.ndarray): y-coordinates.
        slope_threshold (float): Threshold for slope difference.
        second_derivative_threshold (float): Threshold for the second derivative.

    Returns:
        tuple: Indices of start and end points of constant slope regions.

    """
    # Compute the first and second derivatives of y with respect to x
    dy_dx = np.gradient(y, x)
    d2y_dx2 = np.gradient(dy_dx, x)

    # Find indices where the absolute difference in consecutive derivatives is below the slope threshold

    constant_slope_indices = np.where(np.abs(np.diff(dy_dx)) < slope_threshold)[0]
    # print(constant_slope_indices)

    # Additional filter: Check if the second derivative is below the threshold
    constant_slope_indices = constant_slope_indices[
        d2y_dx2[constant_slope_indices] < second_derivative_threshold
    ]

    it = constant_slope_indices.__iter__()
    start_indices = []
    end_indices = []
    n = 1
    try:
        prev = it.__next__()
        while True:
            now = it.__next__()
            while now == prev + n:
                n += 1
                now = it.__next__()
            if n > 5:
                start_indices.append(prev)
                end_indices.append(prev + n)
            n = 1
            prev = now
    except StopIteration:
        if n > 5:
            start_indices.append(prev)
            end_indices.append(prev + n)
        pass

    start_indices = np.asarray(start_indices)
    end_indices = np.asarray(end_indices)

    return start_indices, end_indices


def suggest_input_dictionary(
    working_directory,
    E_fermi,
    ionization_energies,
    slope_threshold=0.1,
    second_derivative_threshold=0.001,
):
    """
    Suggest a dictionary of input parameters for FIM simulation.

    Parameters:
        slope_threshold (float): Threshold for slope difference.
        second_derivative_threshold (float): Threshold for the second derivative.

    Returns:
        tuple: Figure object and suggested simulator parameters.

    """
    Simulator = {
        "working_directory": working_directory,
        "z_max": None,
        "izstart_min": None,
        "izend": None,
        "cutoff": 10,
        "limit": 1e-6,
        "E_fermi": E_fermi,
        "E_max": E_fermi + 5,
        "ionization_energies": ionization_energies,
    }
    elec_potential, cell = sx_el_potential1D_cell(working_directory)
    z = np.linspace(cell[0, 2], cell[2, 2], elec_potential.shape[0])
    # Find start and end indices of constant slope regions with additional filter
    izend, izstart = find_constant_slope_regions(
        z, elec_potential, slope_threshold, second_derivative_threshold
    )
    if izend.size > 1:
        print("constant slope start vals(izend): ", izend)
        print("constant slope end vals(izstart): ", izstart)
        print(
            "constant slope potential drop: ",
            elec_potential[izstart] - elec_potential[izend],
        )
    for izs, ize in zip(izstart, izend):
        if elec_potential[ize] + 1.0 < elec_potential[izs]:
            Simulator["izstart_min"] = izs - 5
            Simulator["izend"] = ize
            Simulator["z_max"] = z[izs] / (
                scipy.constants.physical_constants["Bohr radius"][0] * 1e10
            )
            break

    fig = potential_figure(Simulator, z, elec_potential)
    ax = fig.get_axes()[0]
    # highlight the constant slope regions
    ax.scatter(z[izstart], elec_potential[izstart], color="green", label="$z_{start}$")
    ax.scatter(z[izend], elec_potential[izend], color="red", label="$z_{end}$")
    ax.legend()

    return fig, Simulator
