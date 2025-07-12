"""
Visualize Normalized Blackbody Radiation with Rayleigh-Jeans Approximation using LogicTreeETC

This example generates a plot comparing normalized blackbody radiation spectra at different temperatures
using Planck's Law, alongside the Rayleigh-Jeans approximation. It showcases the capabilities of the
LogicTreeETC package for scientific visualization, particularly its annotation and custom arrow drawing features.

The plot highlights:
- The shift of peak wavelength with temperature (Wien's Displacement Law)
- The breakdown of the Rayleigh-Jeans approximation at short wavelengths
- Mathematical annotations for Planck's Law and Rayleigh-Jeans Law
- Custom arrows and labeled boxes rendered with LaTeX-style formatting

Features used from LogicTreeETC:
- `add_box` for labeled mathematical expressions
- `add_connection` for linking annotations to plot data
- `add_arrow_between` for illustrative double-headed arrows
- Custom figure styling, color modes, and legend formatting

The final plot is saved as a high-resolution PNG image fit for documentation or publication use.

Output:
    resources/logictree_examples/normalized_blackbody_spectrum-Example.png
"""

from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

# Compute absolute path to the parent directory of examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logictree import LogicTree  # noqa: E402

h = 6.62607015e-34  # Planck constant [J s]
c = 2.99792458e8  # Speed of light [m/s]
k = 1.380649e-23  # Boltzmann constant [J/K]


def planck_law(wavelength, T):
    a = 2 * h * c**2
    b = h * c / (wavelength * k * T)
    b = np.clip(b, None, 700)
    intensity = a / (wavelength**5 * np.expm1(b))
    return intensity


def rayleigh_jeans(wavelength, T):
    return (2 * c * k * T) / wavelength**4


def main():
    # wavelengths in meters
    wavelengths = np.linspace(1e-9, 3e-6, 1000)  # 1 nm to 3000 nm

    # temperatures in Kelvin
    temperatures = [3000, 4000, 5000, 6000, 7000]

    tree = LogicTree(
        fig_size=(12, 6), colormode="dark", xlims=(0, 3000), ylims=(0, 1.1)
    )

    # style axis
    tree.ax.axis("on")  # by default tree.ax is set to off

    # plot Planck curves (normalized)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(temperatures)))
    i = 0
    wavelength_hottest_peak = 0
    wavelength_coolest_peak = 0
    for T, color in zip(temperatures, colors):
        I = planck_law(wavelengths, T)
        I /= I.max()  # normalize for visual comparison
        if i == 0:
            peak_idx = np.argmax(I)
            wavelength_coolest_peak = wavelengths[peak_idx] * 1e9
        elif i == len(temperatures) - 1:
            peak_idx = np.argmax(I)
            wavelength_hottest_peak = wavelengths[peak_idx] * 1e9
        tree.ax.plot(wavelengths * 1e9, I, label=f"{T} K", color=color)

        i += 1

    # Rayleigh-Jeans at 5000 K
    T_rj = 5000
    RJ = rayleigh_jeans(wavelengths[500:], T_rj)
    RJ /= RJ.max()  # normalize to match Planck curve visually
    tree.ax.plot(
        wavelengths[500:] * 1e9,
        RJ,
        "--",
        color="gray",
        label=f"Rayleigh-Jeans (T={T_rj} K)",
        lw=3,
    )

    # formatting
    tree.ax.set_title(
        "Normalized Blackbody Spectra with Rayleigh-Jeans Approximation",
        fontsize=20,
        color="white",
    )
    tree.ax.set_xlabel("Wavelength (nm)", labelpad=10)
    tree.ax.set_ylabel(r"Normalized Intensity $B_\lambda(T)$", labelpad=10)
    tree.ax.set_xlim(0, 3000)
    tree.ax.set_ylim(0, 1.1)

    # setup legend
    legend = tree.ax.legend(title="Temperature", fontsize=14, title_fontsize=16)
    legend.get_frame().set_facecolor("#3d3d3d")  # fc
    legend.get_frame().set_edgecolor("white")
    for text in legend.get_texts():
        text.set_color("white")
    legend.get_title().set_color("white")

    tree.ax.grid(True, linestyle="--", alpha=0.4)

    # draw annotations
    plancks_law = r"$B_\lambda(T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{\frac{hc}{\lambda kT}} - 1}$"
    p_box = tree.add_box(
        xpos=750,
        ypos=0.2,
        text=plancks_law,
        box_name="planck law",
        bbox_fc=(0.2, 0.2, 0.2, 0.9),
        bbox_ec="white",
        use_tex_rendering=True,
        fs=34,
        ha="center",
    )

    rj_law = r"$B_\lambda^{\mathrm{RJ}}(T) = \frac{2ckT}{\lambda^4}$"
    rj_box = tree.add_box(
        xpos=2560,
        ypos=0.37,
        text=rj_law,
        box_name="rj law",
        bbox_fc=(0.2, 0.2, 0.2, 0.9),
        bbox_ec="white",
        use_tex_rendering=True,
        fs=24,
        ha="center",
    )

    # we can now add an arrow directly from a box to a point
    tree.add_connection(
        boxA=rj_box,
        boxB=(wavelengths[-340] / 1e-9, RJ[-340] / RJ.max()),
        shaft_width=9,
        sideA="left",
        fc="teal",
        ec="white",
        arrow_head=True,
    )

    # we can draw a double headed arrow between two points
    tree.add_arrow_between(
        start=(wavelength_hottest_peak, 1.05),
        end=(wavelength_coolest_peak, 1.05),
        shaft_width=12,
        arrow_head=True,
        arrow_head_at_tail=True,
        fc="#008400",
        ec="white",
        lw=1.5,
    )

    # one more annotation
    weins_law = r"$\lambda_{\text{peak}} \propto \tfrac{1}{T}$"
    tree.add_box(
        xpos=400,
        ypos=1.045,
        text=weins_law,
        box_name="weins law",
        bbox_fc=(0, 0, 0, 0),
        bbox_ec=(0, 0, 0, 0),
        va="center",
        ha="right",
        use_tex_rendering=True,
        fs=22,
    )

    save_path = (
        Path(__file__).resolve().parent.parent
        / "resources/logictree_examples/normalized_blackbody_spectrum-Example.png"
    )
    tree.save_as_png(save_path, content_padding=0.3, aspect="auto")


if __name__ == "__main__":
    main()
