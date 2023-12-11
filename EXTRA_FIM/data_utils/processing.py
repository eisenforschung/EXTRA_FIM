import numpy as np
from EXTRA_FIM import main as fim
import matplotlib.pyplot as plt

#functions for FIM images and 1D scan
def FIM_image(input_dict =None,path=None,repeat=None):
    '''returns a FIM image based on the input_dict, which has the simulation parameters used to do the actual FIM simulation job, path is the path to the FIM simulation job and repeat is the number of times xy plane should be repeated.'''
    all_totals = dict ()
    fim_simulation = fim.FIM_simulations(inputDict=input_dict)
    rec_cell = np.linalg.inv(fim_simulation.cell) * 2 * np.pi # get cell coordinates from potential file
    gk_1 = np.outer(np.fft.fftfreq(fim_simulation.Nx, 1 / fim_simulation.Nx), rec_cell[0])
    gk_2 = np.outer(np.fft.fftfreq(fim_simulation.Ny, 1 / fim_simulation.Ny), rec_cell[1])
    for ik in range(fim_simulation.wf.nk):
        for ispin in range(fim_simulation.wf.n_spin):
            with h5py.File(f'{path}/partial_dos{ik}.h5', 'r') as handle:
                for varname in handle.keys ():
                    if 'IE=' in varname:
                        IE = float(str(varname).replace('IE=',''))
                        if IE not in all_totals.keys ():
                            all_totals[IE] = np.zeros((fim_simulation.Nx, fim_simulation.Ny), dtype=np.float64)
                        all_totals[IE] += np.asarray(handle[varname])
    
    FIM_image_case =np.zeros([fim_simulation.Nx,fim_simulation.Ny],dtype=np.complex128)
    xp = np.linspace(0, fim_simulation.cell[0, 0]*repeat, fim_simulation.Nx)
    yp = np.linspace(0, fim_simulation.cell[1, 1]*repeat, fim_simulation.Ny)
    # FIM_line_1D_case= np.zeros_like(xp)
    prho_rec_case= np.fft.ifft2(all_totals[IE])
    for ix in range(xp.shape[0]):
        for iy in range(yp.shape[0]):
            x=xp[ix]
            y=yp[iy]
            phase_1=np.exp(-1j*(gk_1[:,0]*x+gk_1[:,1]*y))
            phase_2= np.exp(-1j*(gk_2[:,0]*x+gk_2[:,1]*y))
            phase=np.outer(phase_1,phase_2)
            FIM_image_case[ix,iy] =np.sum(prho_rec_case.flatten()*phase.flatten())
            # FIM_line_1D_case[ix] = np.sum(prho_rec_case.flatten() * phase.flatten()).real

    return FIM_image_case

def plot_fim_image(fim_image):
    '''fim_image output of the FIM_image function. Automatically creates levels and gives a matplotlib figure.'''
    vmax=np.max(fim_image.real)
    vmax_lev=np.power(10., np.trunc(np.log10(vmax))-1)
    vmax_lev=(np.trunc(vmax / vmax_lev * 10.)+1)*0.1 * vmax_lev
    xp = np.linspace(0, fim_simulation.cell[0, 0]*repeat, fim_simulation.Nx)
    yp = np.linspace(0, fim_simulation.cell[1, 1]*repeat, fim_simulation.Ny)
    fig = plt.figure(figsize=[6.5,4])
    plt.contourf(xp/1.89,yp/1.89,fim_image.real.T,vmin=0,vmax=vmax_lev,levels=np.linspace(0,vmax_lev,41))
    plt.rcParams['font.size'] = '16'
    plt.rcParams['font.family'] ='serif'
    plt.xlabel('x ($\AA$)')
    plt.ylabel('y ($\AA$)')
    plt.colorbar()
    return fig
