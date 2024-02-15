__author__ = "Christoph Freysoldt, Shyam Katnagallu"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__maintainer__ = "Christoph Freysoldt"
__email__ = "freysoldt@mpie.de"
__date__ = "February, 2024"

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.patches
import netCDF4
import scipy.constants


def extend_potential(
    elec_potential,
    iz0,
    pot,
    cell,
    Nz_ext=None,
    z_max=None,
    dv_limit=None,
    izend=None,
    plotG=1,
):
    """Extend the effective potential by extrapolation of the electrostatic part

    Parameters:
    elec_potential ... the 3D electrostatic potential
    iz0            ... z index where extrapolation starts
    pot            ... the 3D effective potential (3D + spin axis)
    cell           ... the 3D cell (in bohr units)

    Nz_ext         ... new number of z-points (this or z_max)
    z_max          ... new z_max (this or Nz_ext)

    # for in-plane extrapolation
    dv_limit       ... if given, extrapolate in-plane fluctuations below this
                       threshold for the Fourier component
    izend          ... earliest starting point for in-plane fluctuation
                       extrapolation
    plotG          ... plot dV components in Fourier space up to this |G|

    Returns:
    fig            ... the produced control figures
                       Plot 1: 1D plane-average profiles along z
                       Plot 2: rms of in-plane fluctuations on log scale
                       Plot 3: dV components in Fourier space (log plot)
    pot_ext        ... the 3D extended potential

    The average potential is extrapolated linearly from the electrostatic potential
    beyond z-index iz0.
    The xc potential contribution is truncated at iz0. It should be small
    compared to the electrostatic potential (both average and rms in-plane
    fluctuations) at this point.

    If dv_limit is set, the in-plane fluctuations of the electrostatic potential
    are extrapolated as well, as their decay in Fourier space is exponential when the
    charge density is zero. dv_limit defines the threshold below which extrapolation
    is used. A reasonable value seems 1e-3, but you may need to inspect plot 3.
    If dv_limit is too small, you may have reached the noise level. If too large,
    you may still sit in the non-exponential part. Use izend to define where the
    extrapolation sets in earliest.
    Extrapolating in-plane fluctuations needs a few sec (due to 2D FFTs).

    If neither z_max or Nz_ext is given, use number of z-points in elec_potential.
    """

    # --- get 1D potentials
    elec_potential1D = np.mean(elec_potential, axis=(0, 1))
    V1 = np.mean(pot, axis=(0, 1))

    # --- start plotting
    n_plot = 2 if dv_limit is None else 3
    fig, axs = plt.subplots(1, n_plot, figsize=[n_plot * 6.5, 4])
    #    fig, axs = plt.subplots (n_plot,1,figsize=[6.5, n_plot*4])
    ax = axs[0]
    z_pot, dz = np.linspace(0, cell[2, 2], pot.shape[2], retstep=True, endpoint=False)
    ax.plot(z_pot, elec_potential1D, color="royalblue", label="Electrostatic potential")
    ax.plot(z_pot, V1, label="Effective potential", color="k")

    # --- determine Nz_ext
    if Nz_ext is None:
        if z_max is None:
            Nz_ext = V1.shape[0]
        else:
            Nz_ext = int(z_max / dz)
    print(f"original Nz = {pot.shape[2]}")
    print(f"extended Nz = {Nz_ext}")

    # --- extrapolation of plane average
    slope = np.diff(elec_potential1D)[iz0]
    V1ext = np.zeros((Nz_ext, pot.shape[3]))
    V1ext[:iz0, :] = V1[:iz0, :]
    for iz in range(iz0, Nz_ext):
        V1ext[iz, :] = V1[iz0, :] + (iz - iz0) * slope

    # --- plot the plane-average of extended potential
    z_ext = np.linspace(0, Nz_ext * dz, Nz_ext, endpoint=False)
    ax.plot(z_ext, V1ext, ls="--", color="r", label="Extended potential")
    ax.set_xlabel(r"z (bohr)")
    ax.set_ylabel("pot (Hartree)")
    ax.axvline(z_pot[iz0], ls="--")
    ax.text(z_pot[iz0] + 0.5, ax.get_ylim()[1] - 0.05, r"$z_{0}$", va="top")
    ax.legend()
    ax.title.set_text("Plane-average potential")

    # --- plot 2: in-plane fluctations (in-plane rms)
    pot_var = np.mean((pot - V1) ** 2, axis=(0, 1))
    axs[1].plot(z_pot, np.log10(pot_var) / 2, label="effective potential", color="k")
    elpot_var = np.mean((elec_potential - elec_potential1D) ** 2, axis=(0, 1))
    axs[1].plot(
        z_pot,
        np.log10(elpot_var) / 2,
        label="electrostatic potential",
        color="royalblue",
    )
    xc_var = np.mean(
        ((pot.T - elec_potential.T).T - (V1.T - elec_potential1D.T).T) ** 2, axis=(0, 1)
    )
    axs[1].plot(z_pot, np.log10(xc_var) / 2, label="xc potential", color="green")

    axs[1].set_xlabel(r"z (bohr)")
    axs[1].set_ylabel("log$_{10}$ in-plane rms($\delta V$)/Hartree")
    axs[1].axvline(z_pot[iz0], ls="--")
    axs[1].text(z_pot[iz0] + 0.5, axs[1].get_ylim()[1] - 0.2, r"$z_{0}$", va="top")
    axs[1].title.set_text("In-plane fluctuations (rms)")

    # --- transfer extended potential from 1D to 3D
    pot_ext = np.zeros(list(pot.shape[0:2]) + [Nz_ext, pot.shape[3]])
    pot_ext[:, :, :iz0, :] = pot[:, :, :iz0, :]
    pot_ext[:, :, iz0:, :] = V1ext[iz0:, :]

    # --- now deal with in-plane potential fluctuations
    if dv_limit is not None:
        # --- extrapolate in-plane fluctuations from electrostatic potential
        #     via analytic exponential decay (as dictated by Poisson equation)

        # izend is the earliest extrapolation point
        if izend is None:
            izend = iz0 - 1
        # do 2D FFT parallel to surface
        rec_cell = np.linalg.inv(cell) * 2 * np.pi
        xy_shape = pot.shape[0:2]
        G1 = np.fft.fftfreq(xy_shape[0], 1.0 / xy_shape[0])
        G2 = np.fft.fftfreq(xy_shape[1], 1.0 / xy_shape[1])
        epot_fft = np.zeros(list(xy_shape) + [Nz_ext], dtype=np.complex128)
        for iz in range(izend, iz0):
            epot_fft[:, :, iz] = np.fft.fft2(elec_potential[:, :, iz])

        # --- extrapolation is G-dependent
        limit_v = dv_limit * np.prod(xy_shape)
        epot_fft[0, 0, :] = 0
        for kx, ky in np.ndindex(xy_shape):
            if kx == 0 and ky == 0:
                continue
            # --- find starting point for extrapolation: when it drops below
            #     dv_limit
            izm = izend
            while izm < iz0 - 1 and np.abs(epot_fft[kx, ky, izm]) > limit_v:
                izm += 1
            # --- extrapolate potential by appending exponential tail
            G_abs = np.linalg.norm(G1[kx] * rec_cell[0] + G2[ky] * rec_cell[1])
            Gdz = dz * G_abs
            fix_zrange = range(izm, Nz_ext)

            extrapol_pot = epot_fft[kx, ky, izm] * np.exp(
                -(np.asarray(fix_zrange) - izm) * Gdz
            )

            if G_abs <= plotG:
                axs[2].plot(
                    z_pot[izend:iz0],
                    np.log10(np.abs(epot_fft[kx, ky, izend:iz0]) / np.prod(xy_shape)),
                )
                axs[2].plot(
                    z_ext[izm:],
                    np.log10(np.abs(extrapol_pot) / np.prod(xy_shape)),
                    ls="--",
                    color=plt.gca().lines[-1].get_color(),
                )
            # --- correction for potential
            epot_fft[kx, ky, fix_zrange] -= extrapol_pot
            epot_fft[kx, ky, izend:izm] = 0

        # --- back to real space and apply correction
        for iz in range(izend, Nz_ext):
            dv_corr = np.fft.ifft2(epot_fft[:, :, iz]).real
            for ispin in range(pot_ext.shape[3]):
                pot_ext[:, :, iz, ispin] -= dv_corr

        # plot 3: set labels
        axs[2].set_xlabel(r"z (bohr)")
        axs[2].set_ylabel("log$_{10}$ |$\delta V(G)|$/Hartree")
        axs[2].set_ylim(-12, axs[2].get_ylim()[1])
        axs[2].axvline(z_pot[iz0], ls="--")
        axs[2].text(z_pot[iz0] + 0.5, axs[1].get_ylim()[1] - 1, r"$z_{0}$", va="top")
        axs[2].title.set_text("In-plane fluctuations (Fourier components)")

        # plot 2: in-plane variation of extrapolated potential, too
        ext_var = np.mean((pot_ext - V1ext) ** 2, axis=(0, 1))
        axs[1].plot(
            z_ext, np.log10(ext_var) / 2, label="extended potential", ls="--", color="r"
        )

    axs[1].legend()
    return fig, pot_ext


def sx_el_potential3D_cell(working_directory):
    """Load electrostatic potential in 3D from SPHInX directory
    Params:
        working_directory ... SPHInX directory
    Returns:
        (V,cell)
        V    ... electrostatic potential (in eV) on mesh
        cell ... cell (in Angstroem)
    """
    v_file = netCDF4.Dataset(working_directory + "/vElStat-eV.sxb")
    v_elstat = np.asarray(v_file["mesh"]).reshape(v_file["dim"])
    cell = np.asarray(v_file["cell"])
    return v_elstat, cell * (
        scipy.constants.physical_constants["Bohr radius"][0] * 1e10
    )


def sx_el_potential1D_cell(working_directory):
    """Load electrostatic potential in 1D from SPHInX directory
    Params:
        working_directory ... SPHInX directory
    Returns:
        (V,cell)
        V    ... plane averaged electrostatic potential (in eV) on z-mesh
        cell ... cell (in Angstroem)
    """
    v_elstat, cell = sx_el_potential3D_cell(working_directory)
    V1 = np.mean(v_elstat, axis=(0, 1))
    return V1, cell
