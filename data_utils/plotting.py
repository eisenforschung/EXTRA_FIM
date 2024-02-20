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
import matplotlib.patches
import scipy.constants

BOHR_TO_AA = scipy.constants.physical_constants["Bohr radius"][0] * 1e10


def potential_figure(Simulator, z, elec_potential):
    """Plot the potential figure

    Parameters:
    Simulator      ... The FIM simulation parameter dictionary
    z              ... z axis values (in Angstroem)
    elec_potential ... the potential to plot (in eV)

    Returns:
    fig            ... the produced figure

    This plot contains
    - the 1D electrostatic potential along z
    - marks for z_max
    - energy range from E_fermi ... E_max
    - the local ionization level in the range z_end ... z_max
      for each ionization energy in Simulation['']

    The shifted potential allows to read off the location where
    FIM ionization will take place. It must cross E_max
    before z_max.

    The function accesses the following Simulation parameters:
    izend, z_max, E_fermi, E_max, ionization_energies
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=[6.5, 4])
    ax.plot(z, elec_potential, label="potential")

    izL = Simulator["izend"]
    zR = Simulator["z_max"] * BOHR_TO_AA
    izR = int(zR / (z[1] - z[0]))
    ax.axvline(zR, ls="--")
    # ax.axvline(z[izL],ls='--')

    # --- plot energy range
    ax.axhline(Simulator["E_fermi"], ls="--")
    ax.text(0, Simulator["E_fermi"] + 0.5, r"$E_{Fermi}$")
    ax.axhline(Simulator["E_max"], ls="--")
    ax.text(0, Simulator["E_max"] + 0.5, r"$E_{max}$")
    plt.xlim(0, z[-1])
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (0, Simulator["E_fermi"]),
            zR,
            Simulator["E_max"] - Simulator["E_fermi"],
            color=(0, 0, 0, 0.1),
        )
    )
    # --- plot ionization levels
    colors = ["black", "red", "green", "blue"]
    nIE = len(Simulator["ionization_energies"])
    if nIE > 4:
        colors = colors + plt.cm.rainbow(np.linspace(0, 1, 4 - nIE))
    for IE, color in zip(Simulator["ionization_energies"], colors):
        ax.plot(z[izL:izR], elec_potential[izL:izR] - IE, ls="--", color=color)
        ax.text(z[izL] + 0.5, elec_potential[izL] - IE - 0.5, f"IE={IE}", color=color)

    ax.set_xlabel(r"z ($\AA$)")
    ax.set_ylabel("Electrostatic potential, (eV)")
    ax.legend()
    return fig


def waves_figure(
    Simulator,
    waves_reader,
    ik=0,
    ispin=0,
    istate=None,
    n_plot=2,
    compute_extra=False,
    pot=None,
):
    """Plot the potential figure

    Parameters:
    Simulator      ... The FIM simulation parameter dictionary
    waves_reader   ... a waves_reader object compatible to waves_reader_abc
    ik             ... k-point to plot
    ispin          ... spin channel to plot
    istate         ... state to plot, defaults to first state above Fermi level
    n_plot         ... number of plots
    compute_extra  ... if True, compute and plot extrapolated wave (takes a few 10 sec)
    pot            ... (optional) set potential for wave extrapolation

    Returns:
    fig            ... the produced figure

    Plot 1: Wave function decay on log scale
            - contains markers for matching plane (z_end) and z_max
    Plot 2: in-plane map at matching plane (izend)
            In this plot, the in-plane maximum is marked with a red circle.
    Plot 3(if n_plot=4): in-plane map at izstart_min for EXTRA wave
    Plot 3 or 4: in-plane map at zmax  (usually noise)

    The in-plane maps do not take cell shape into account. They will appear
    distorted, notably if the in-plane lattice vectors are non orthogonal.

    The function accesses the following Simulation parameters:
    izend, z_max, E_fermi, izstart_min (for plot 4)
    """

    z, dz = np.linspace(
        0,
        waves_reader.cell[2, 2] * BOHR_TO_AA,
        waves_reader.mesh[2],
        endpoint=False,
        retstep=True,
    )

    if n_plot < 4:
        fig, axs = plt.subplots(1, n_plot, figsize=[n_plot * 6.5, 4])
    else:
        fig, _ = plt.subplots(2, 2, figsize=[2 * 6.5, 2 * 4])
        axs = fig.get_axes()
    ax = axs[0] if n_plot > 1 else axs
    izL = Simulator["izend"]
    zR = Simulator["z_max"] * BOHR_TO_AA
    izR = int(zR / (z[1] - z[0]))

    if istate is None:
        for i in range(waves_reader.n_states):
            if waves_reader.get_eps(i, ispin, ik) > Simulator["E_fermi"]:
                istate = i
                break
    print(f"plotting i={istate} ispin={ispin} ik={ik}")
    psi_dft_abs = np.abs(waves_reader.get_psi(istate, ispin, ik))
    ix, iy = np.unravel_index(np.argmax(psi_dft_abs[:, :, izL]), psi_dft_abs.shape[0:2])
    print(f"max. at ix,iy = {ix}, {iy}")

    # Plot the data
    # wave function at in-plane maximum
    ax.plot(z, np.log10(psi_dft_abs[ix, iy, :]), label="@ in-plane max")
    # in-plane average
    ax.plot(
        z,
        np.log10(np.mean(psi_dft_abs**2, axis=(0, 1))) / 2,
        label="rms in plane",
    )

    # --- compute and plot extrapolated wave
    if compute_extra:
        print("Computing EXTRA wave...")
        from ..EXTRA_FIM.main import extra_waves

        waves_extra = extra_waves(Simulator, reader=waves_reader, pot=pot)
        _, psi_extra = waves_extra.get_psi(139, 0, 23)
        psi_extra_abs = np.abs(psi_extra)
        z_extra = np.linspace(
            0, dz * psi_extra_abs.shape[2], psi_extra_abs.shape[2], endpoint=False
        )
        ax.plot(
            z_extra[:-1],
            np.log10(psi_extra_abs[ix, iy, :-1]),
            label="EXTRA @ in-plane max",
            ls="--",
            color="lightgreen",
        )

    # --- mark z_end and z_max
    ax.axvline(zR, ls="--")
    ax.text(zR + 0.5, ax.get_ylim()[1] - 0.2, r"$z_{max}$", va="top")
    ax.axvline(z[izL], ls="--")
    ax.text(z[izL] + 0.5, ax.get_ylim()[0] + 0.2, r"$z_{end}$")

    ax.set_xlabel(r"z ($\AA$)")
    ax.set_ylabel("$log_{10} |\psi|$")
    ax.title.set_text("wave function amplitude")

    ax.legend()

    # --- additional plots
    if n_plot > 1:
        # Plot 2: in-plane map at izend (matching plane)
        axs[1].imshow(psi_dft_abs[:, :, izL].T)
        axs[1].add_patch(
            matplotlib.patches.Circle((ix, iy), 1.5, fill=False, ec="r", lw=1)
        )
        axs[1].text(
            ix + 2,
            iy,
            "max",
            color="r",
            va="center",
            bbox={
                "boxstyle": "Round, pad=0.1, rounding_size=0.5",
                "ec": None,
                "fc": (1, 1, 1, 0.5),
                "lw": 0,
            },
        )
        axs[1].title.set_text("$|\psi|$ in-plane at $z_{end}$")

    if n_plot >= 3:
        # Plot 3 or 4: in-plane map at z_max (typically noise)
        axs[-1].imshow(psi_dft_abs[:, :, izR].T)
        axs[-1].add_patch(
            matplotlib.patches.Circle((ix, iy), 1.5, fill=False, ec="r", lw=3)
        )
        axs[-1].title.set_text("raw $|\psi|$ in-plane at $z_{max}$" + f" (iz={izR})")

    if n_plot == 4 and compute_extra:
        # Plot 3(if 4 plots): in-plane map at izstart_min
        axs[2].imshow(psi_extra_abs[:, :, Simulator["izstart_min"]].T)
        axs[2].add_patch(
            matplotlib.patches.Circle((ix, iy), 1.5, fill=False, ec="r", lw=3)
        )
        axs[2].title.set_text(
            f"EXTRA $|\psi|$ in-plane at z={z[Simulator['izstart_min']]:.2f}$\AA$ (iz={Simulator['izstart_min']})"
        )

    for ax in axs[1:]:
        ax.set_xlabel("ix")
        ax.set_ylabel("iy")

    return fig
