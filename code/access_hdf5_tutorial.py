from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _read_dset(ds):
    if ds.shape == () or getattr(ds, "ndim", 0) == 0:
        return ds[()]
    return ds[...]


def print_tree(obj, prefix=""):
    if isinstance(obj, h5py.Dataset):
        print(f"{prefix} <Dataset shape={obj.shape} dtype={obj.dtype}>")
        return

    if prefix:
        print(f"{prefix}/ <Group>")
    for name in obj.keys():
        print_tree(obj[name], f"{prefix}/{name}" if prefix else f"/{name}")


def plot_sector_light_curve(g, title):
    lc_time = np.array(_read_dset(g["lc_time"]), dtype=float)
    lc_flux = np.array(_read_dset(g["lc_flux"]), dtype=float)
    flux_norm = lc_flux / np.nanmean(lc_flux)

    flare_mask = np.zeros_like(lc_time, dtype=bool)
    if "flare_mask" in g:
        flare_mask = np.array(_read_dset(g["flare_mask"]), dtype=bool)

    gp = None
    if "mu" in g:
        gp = np.array(_read_dset(g["mu"]), dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(lc_time, flux_norm, s=1.2, color="black", rasterized=True)

    if flare_mask.shape == lc_flux.shape and np.any(flare_mask):
        flare_times = lc_time[flare_mask]
        flare_flux = flux_norm[flare_mask]
        ax.scatter(flare_times, flare_flux, s=3, color="red", label="Flare")
        ax.scatter(
            flare_times,
            np.full(flare_times.shape, 0.98),
            transform=ax.get_xaxis_transform(),
            marker="v",
            s=24,
            color="limegreen",
            edgecolors="darkgreen",
            linewidths=0.4,
            alpha=0.9,
            zorder=6,
            rasterized=True,
        )

    if gp is not None and len(gp) == len(lc_time):
        ax.plot(lc_time, gp / np.nanmean(lc_flux), color="royalblue", linewidth=0.8, label="GP Model", rasterized=True)

    ax.text(
        0.575,
        0.85,
        title,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        zorder=7,
    )
    ax.set_xlabel("Time [BJD - 2457000]", fontsize=16)
    ax.set_ylabel("Normalized Flux", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13, length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.9)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    ax.minorticks_on()
    plt.tight_layout()
    plt.show()


def main():
    h5_path = Path("/Users/tobinw/Dropbox/Stellar_Flares_Project/CVZ/Cycle7_Fillin_Results/17_Candidates_Files/tic260506296_final_output_Cycle7_Fillin.h5")

    with h5py.File(h5_path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        print_tree(f)

        g = f["Sectors"]["sector_1_lc"]
        plot_sector_light_curve(g, "TIC 260506296 | Sector 1")


if __name__ == "__main__":
    main()