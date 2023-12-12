import numpy as np
from EXTRA_FIM import main as fim
import matplotlib.pyplot as plt
from mendeleev import element
import scipy

class PreProcessingFIM():
    '''
    Class for giving a suggested set of parameters for FIM simulation

    Attributes:
        job (str): Job information.
        imaging_gas (mendeleev.Element): Imaging gas element.

    Methods:
        find_constant_slope_regions(x, y, slope_threshold, second_derivative_threshold):
            Identify constant slope regions in a given dataset.

        suggest_input_dictionary(slope_threshold=0.1, second_derivative_threshold=0.001):
            Suggest a dictionary of input parameters for FIM simulation.

    '''
    
    def __init__(self, job=None,imaging_gas=None):
        '''
        Initialize PreProcessingFIM object.

        Parameters:
            job (str): Job information.
            imaging_gas (str): Symbol of the imaging gas element.

        '''
        self.job = job
        self.imaging_gas = element(imaging_gas)
   
    @staticmethod
    def find_constant_slope_regions(x, y, slope_threshold, second_derivative_threshold):
        '''
        Identify constant slope regions in a given dataset.

        Parameters:
            x (numpy.ndarray): x-coordinates.
            y (numpy.ndarray): y-coordinates.
            slope_threshold (float): Threshold for slope difference.
            second_derivative_threshold (float): Threshold for the second derivative.

        Returns:
            tuple: Indices of start and end points of constant slope regions.

        '''
        # Compute the first and second derivatives of y with respect to x
        dy_dx = np.gradient(y, x)
        d2y_dx2 = np.gradient(dy_dx, x)

        # Find indices where the absolute difference in consecutive derivatives is below the slope threshold
        constant_slope_indices = np.where(np.abs(np.diff(dy_dx)) < slope_threshold)[0]

        # Additional filter: Check if the second derivative is below the threshold
        constant_slope_indices = constant_slope_indices[d2y_dx2[constant_slope_indices] < second_derivative_threshold]

        # Identify start and end indices of constant slope regions
        start_indices = constant_slope_indices[0]+5
        end_indices = constant_slope_indices[np.where(np.diff(constant_slope_indices) > 1)[0]]-5

        return end_indices+5, start_indices, end_indices

    def suggest_input_dictionary(self, slope_threshold = 0.1 ,second_derivative_threshold = 0.001 ):
        '''
        Suggest a dictionary of input parameters for FIM simulation.

        Parameters:
            slope_threshold (float): Threshold for slope difference.
            second_derivative_threshold (float): Threshold for the second derivative.

        Returns:
            tuple: Figure object and suggested simulator parameters.

        '''
        Simulator = {
            'working_directory': self.job.working_directory,
            'z_max': None,
            'izstart_min': None,
            'izend': None,
            'cutoff': 10,
            'limit': 1e-6,
            'E_fermi': self.job['output/generic/dft/bands_e_fermi'][-1] ,
            'E_max': self.job['output/generic/dft/bands_e_fermi'][-1]+ 5,
            'ionization_energies': self.imaging_gas.ionenergies[1]
            }
        V_relax = self.job.get_electrostatic_potential()
        cell = self.job.get_structure().get_cell()
        elec_potential = V_relax.get_average_along_axis(ind=2)
        z = np.linspace(cell[0,2],cell[2,2],V_relax.total_data.shape[2])  
        # Find start and end indices of constant slope regions with additional filter
        zmax,izend, izstart = self.find_constant_slope_regions(elec_potential, z, slope_threshold, second_derivative_threshold)
        Simulator['izstart_min'] = izstart
        Simulator['izend'] = izend
        Simulator['zmax'] = z[zmax]/(scipy.constants.physical_constants["Bohr radius"][0] * 1e+10)
        # Plot the data and highlight the constant slope regions
        fig, ax = plt.subplots(figsize=[6.5, 4])
        ax.plot(z, elec_potential, label='Original Data')
        ax.scatter(x[Simulator['izstart_min']], y[Simulator['izstart_min']], color='green', label='Izstart')
        ax.scatter(x[Simulator['izend']], y[Simulator['izend']], color='red', label='Izend')
        ax.axhline(Simulator['E_fermi']+Simulator['ionization_energies'],ls='--')
        ax.axhline(Simulator['E_max']+Simulator['ionization_energies'],ls='--')
        ax.set_xlabel('z, ($\AA$)')
        ax.set_ylabel('Electrostatic potential, (eV)')
        ax.legend()

        return fig, Simulator
        

