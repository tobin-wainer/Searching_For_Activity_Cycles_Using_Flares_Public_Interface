from __future__ import annotations

"""
Make six-panel diagnostic plots from Cycle 7 fill-in HDF5 flare-analysis products.

This script reads one or more per-star HDF5 files produced by the TESS CVZ flare
pipeline and generates the six-panel diagnostic plot used for visual inspection.

The six panels show:
    1. Sector-by-sector flare frequency distributions (FFDs), including raw points,
       completeness-corrected points, and fixed-slope power-law fits.
    2. Sector-by-sector FFD normalization beta.
    3. Sector-by-sector integrated flare power, L_fl / L_TESS.
    4. Number of detected flares per sector.
    5. Epoch-stitched and sub-epoch-stitched beta measurements.
    6. Epoch-stitched and sub-epoch-stitched integrated flare power.

Expected HDF5 structure:
    Summary/
    Sectors/
    Stitched/      optional

Usage from the command line:
    python make_diagnostic_plot.py tic167344043_final_output_Cycle7_Fillin.h5 \
        --output-dir ./figures

Usage for multiple files:
    python make_diagnostic_plot.py tic*_final_output_Cycle7_Fillin.h5 \
        --output-dir ./figures

Notes:
    * beta = -99 is treated as a sentinel value indicating an invalid FFD fit.
    * The time axis uses the same units as the lc_time arrays stored in the HDF5 file.
    * Downward arrows in the flare-power panels mark points whose lower uncertainty
      bound would extend below zero in linear space.
"""

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import numpy as np


# Minimum number of completeness-corrected flare measurements required for a
# sector to be shown in the diagnostic panels. This avoids plotting sectors where
# the FFD is too sparsely sampled above the 50% completeness threshold.
MIN_FLARES_ABOVE_50PCT = 3

# Default reduced-chi2 threshold used to flag sectors where the Gaussian-process
# light-curve fit may need visual inspection. The HDF5 file can override this
# value if it stores its own threshold in the Summary group.
GP_CHI2_WARN_THRESHOLD = 8.0

plt.rcParams.update(
    {
        "axes.labelsize": 26,
        "axes.titlesize": 14,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    }
)

def _read_dset(ds):
    """
    Read an HDF5 dataset and return either a scalar or an array.

    h5py represents scalar datasets differently from array datasets. This helper
    keeps the rest of the code simple by returning ds[()] for scalar datasets and
    ds[...] for array datasets.
    """
    if ds.shape == () or getattr(ds, "ndim", 0) == 0:
        return ds[()]
    return ds[...]


def _sector_number(key):
    """
    Extract the numerical sector number from an HDF5 sector key.

    Expected sector keys look like:
        sector_001_lc
        sector_045_lc

    If the sector number cannot be parsed, the function returns a very large
    value so that unparsable keys are sorted to the end.
    """
    match = re.search(r"sector_(\d+)_lc", str(key))
    return int(match.group(1)) if match else 10**9


def _count_flares_above_limit(corrected_eds):
    """
    Count the number of valid completeness-corrected flare equivalent durations.

    Zero values and non-finite values are excluded. The resulting count is used
    to decide whether a sector has enough flares above the completeness limit to
    appear in the diagnostic plot.
    """
    values = np.array(corrected_eds, dtype=float)
    return int(np.sum(np.isfinite(values) & (values != 0.0)))


def _beta_plot_yerr(beta_err, n_sectors, max_err=5.0):
    """
    Compute the plotted uncertainty on beta.

    The formal beta uncertainty can be unrealistically small for sparsely sampled
    FFDs. This helper adds a small floor that depends on how many sectors were
    combined. Single-sector measurements receive the largest added floor, while
    stitched measurements containing more sectors receive a smaller one.

    Parameters
    ----------
    beta_err : float
        Formal uncertainty on beta.
    n_sectors : int
        Number of sectors contributing to the measurement.
    max_err : float
        Maximum allowed plotted uncertainty.

    Returns
    -------
    float
        Error bar size used for plotting beta.
    """
    if np.isfinite(beta_err):
        base = abs(float(beta_err))
    else:
        base = 0.0

    n = int(n_sectors)
    if n <= 1:
        extra = 0.25
    elif n == 2:
        extra = 0.17
    elif n == 3:
        extra = 0.14
    elif n == 4:
        extra = 0.12
    else:
        extra = 0.05

    return float(np.clip(base + extra, 0.0, max_err))


def line(x, a, b):
    """
    Evaluate the fixed-slope FFD model.

    The FFDs are plotted in log space as:
        log(rate) = -a * log(ED) + b

    where a is the full-sample FFD slope and b is the sector-specific beta.
    """
    return -a * x + b


def _logyerr_with_truncated_lower(values, errors):
    """
    Convert linear-space values and errors into log-space values and errors.

    Integrated flare-power measurements are positive quantities plotted in log10
    space. Some lower uncertainty bounds can extend below zero in linear space,
    which cannot be plotted directly on a logarithmic scale. For those cases, the
    lower bound is truncated to the smallest positive value available, and the
    function returns a flag indicating which points were truncated.

    Returns
    -------
    y : ndarray
        log10(values)
    yerr : ndarray
        Asymmetric log-space uncertainties with shape (2, N).
    truncated : ndarray of bool
        True for points whose lower linear-space uncertainty crossed zero.
    ylo : ndarray
        log10 lower bounds used for plotting arrows.
    """
    vals = np.array(values, dtype=float)
    errs = np.abs(np.array(errors, dtype=float))
    errs = np.where(np.isfinite(errs), errs, 0.0)

    # The stored flare-power errors are interpreted here as full-width errors, so
    # the plotted upper/lower bounds use half of that value.
    half = 0.5 * errs
    lo_raw = vals - half
    hi_raw = vals + half

    # Choose a positive floor for lower bounds that would otherwise be <= 0.
    pos_lo = lo_raw[np.isfinite(lo_raw) & (lo_raw > 0)]
    if pos_lo.size > 0:
        lo_floor = float(np.nanmin(pos_lo))
    else:
        pos_vals = vals[np.isfinite(vals) & (vals > 0)]
        lo_floor = float(np.nanmin(pos_vals)) if pos_vals.size > 0 else 1e-6

    lo = np.where(lo_raw <= 0, lo_floor, lo_raw)
    lo = np.minimum(lo, vals)
    hi = np.maximum(hi_raw, vals)
    hi = np.maximum(hi, lo_floor)
    truncated = lo_raw <= 0

    y = np.log10(vals)
    ylo = np.log10(lo)
    yhi = np.log10(hi)
    yerr = np.vstack([y - ylo, yhi - y])
    yerr = np.maximum(yerr, 0.0)

    return y, yerr, truncated, ylo


def _load_epoch_group(grp):
    """
    Load the epoch-stitched measurements from the HDF5 Stitched/Epochs group.

    Each child group is expected to contain one stitched time bin. For each bin,
    the function reads the median time, beta, beta uncertainty, integrated flare
    power, flare-power uncertainty, number of flares, and number of sectors.
    """
    times = []
    betas = []
    beta_errs = []
    ffb = []
    ffb_err = []
    n_flares = []
    n_sectors = []

    if grp is None:
        return times, betas, beta_errs, ffb, ffb_err, n_flares, n_sectors

    for node_name in grp.keys():
        node = grp[node_name]

        # Skip non-group objects and stitched products that do not contain a
        # light-curve time array.
        if not isinstance(node, h5py.Group):
            continue
        if "lc_time" not in node:
            continue

        lc_time = np.array(_read_dset(node["lc_time"]), dtype=float)

        times.append(float(np.nanmedian(lc_time)) if lc_time.size else np.nan)
        betas.append(float(_read_dset(node["beta"])) if "beta" in node else np.nan)
        beta_errs.append(float(_read_dset(node["beta_err"])) if "beta_err" in node else np.nan)
        ffb.append(float(_read_dset(node["F_Flare_F_Bol"])) if "F_Flare_F_Bol" in node else np.nan)
        ffb_err.append(float(_read_dset(node["F_Flare_F_Bol_err"])) if "F_Flare_F_Bol_err" in node else np.nan)
        n_flares.append(int(len(np.array(_read_dset(node["FFD_eds"]), dtype=float))) if "FFD_eds" in node else 0)
        n_sectors.append(int(_read_dset(node["n_sectors"])) if "n_sectors" in node else 1)

    return times, betas, beta_errs, ffb, ffb_err, n_flares, n_sectors


def _load_subepoch_group(parent):
    """
    Load the sub-epoch-stitched measurements from the HDF5 Stitched/SubEpochs group.

    Sub-epoch products are stored one level deeper than epoch-stitched products:
        Stitched/SubEpochs/<epoch_name>/<subepoch_group_name>/

    The returned arrays have the same meaning as those returned by
    _load_epoch_group().
    """
    times = []
    betas = []
    beta_errs = []
    ffb = []
    ffb_err = []
    n_flares = []
    n_sectors = []

    if parent is None:
        return times, betas, beta_errs, ffb, ffb_err, n_flares, n_sectors

    for epoch_name in parent.keys():
        epoch_group = parent[epoch_name]
        if not isinstance(epoch_group, h5py.Group):
            continue

        for group_name in epoch_group.keys():
            node = epoch_group[group_name]
            if not isinstance(node, h5py.Group):
                continue
            if "lc_time" not in node:
                continue

            lc_time = np.array(_read_dset(node["lc_time"]), dtype=float)

            times.append(float(np.nanmedian(lc_time)) if lc_time.size else np.nan)
            betas.append(float(_read_dset(node["beta"])) if "beta" in node else np.nan)
            beta_errs.append(float(_read_dset(node["beta_err"])) if "beta_err" in node else np.nan)
            ffb.append(float(_read_dset(node["F_Flare_F_Bol"])) if "F_Flare_F_Bol" in node else np.nan)
            ffb_err.append(float(_read_dset(node["F_Flare_F_Bol_err"])) if "F_Flare_F_Bol_err" in node else np.nan)
            n_flares.append(int(len(np.array(_read_dset(node["FFD_eds"]), dtype=float))) if "FFD_eds" in node else 0)
            n_sectors.append(int(_read_dset(node["n_sectors"])) if "n_sectors" in node else 1)

    return times, betas, beta_errs, ffb, ffb_err, n_flares, n_sectors


def _build_scalar_mappable(times, cmap_name):
    """
    Build the colormap used to encode observing time.

    The default "cycle7" colormap reproduces the custom green-to-orange-to-red
    palette used in the paper. Any standard Matplotlib colormap name can also be
    supplied through the --cmap command-line option.
    """
    if cmap_name.lower() in {"cycle7", "custom"}:
        colors = [
            "#044A05",
            "#087804",
            "#089404",
            "#02AB2E",
            "#20B11D",
            "#D5AB09",
            "#FB7D07",
            "#F97306",
            "#FE420F",
            "#E50000",
            "#BF0C09",
            "#850E04",
        ]
        cmap = mcol.LinearSegmentedColormap.from_list("cycle7_custom", colors, N=256)
    else:
        cmap = plt.get_cmap(cmap_name)

    finite_times = np.array(times, dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]

    # If all times are missing, define a dummy range so the colormap still works.
    if finite_times.size == 0:
        finite_times = np.array([0.0, 1.0])

    norm = mcol.Normalize(
        vmin=float(np.nanmin(finite_times)),
        vmax=float(np.nanmax(finite_times)),
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return sm


def load_fillin_plot_inputs(h5_path):
    """
    Read all plotting inputs from a single Cycle 7 fill-in HDF5 file.

    The function collects sector-level quantities, epoch-stitched quantities,
    sub-epoch-stitched quantities, and summary metadata into a single dictionary.
    The returned dictionary is passed directly into make_diagnostic_plot().
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as h:
        summary = h["Summary"]
        sectors = h["Sectors"]
        stitched = h["Stitched"] if "Stitched" in h else None

        # Extract the TIC ID from the filename.
        tic_match = re.search(r"tic(\d+)", h5_path.name)
        tic_id = int(tic_match.group(1)) if tic_match else -1

        # Read full-sample FFD fit quantities from the Summary group.
        full_sample_slope = float(_read_dset(summary["full_sample_slope"]))
        full_sample_intercept = (
            float(_read_dset(summary["full_sample_intercept"]))
            if "full_sample_intercept" in summary
            else np.nan
        )

        # Read optional GP warning metadata. These warnings flag sectors whose GP
        # fits may require additional visual inspection.
        if "gp_chi2_warning_sectors" in summary:
            gp_warn_sectors = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in _read_dset(summary["gp_chi2_warning_sectors"])
            ]
        else:
            gp_warn_sectors = []

        gp_chi2_warn_threshold = (
            float(_read_dset(summary["gp_chi2_warning_threshold"]))
            if "gp_chi2_warning_threshold" in summary
            else GP_CHI2_WARN_THRESHOLD
        )

        # Read sector-level quantities. Each sector contributes one FFD, one beta
        # measurement, one flare-power measurement, and one flare count.
        sector_keys = sorted(sectors.keys(), key=_sector_number)
        sector_times = []
        betas = []
        beta_errs = []
        corrected_eds = []
        corrected_rates = []
        FFD_eds = []
        raw_rates = []
        F_flare_F_bols = []
        F_flare_F_bol_errs = []
        fifty_limits = []

        for sector_key in sector_keys:
            g = sectors[sector_key]

            lc_time = _read_dset(g["lc_time"]) if "lc_time" in g else np.array([])
            lc_time = np.array(lc_time, dtype=float)

            sector_times.append(float(np.nanmedian(lc_time)) if lc_time.size else np.nan)
            betas.append(float(_read_dset(g["beta"])) if "beta" in g else np.nan)
            beta_errs.append(float(_read_dset(g["beta_err"])) if "beta_err" in g else np.nan)
            corrected_eds.append(np.array(_read_dset(g["corrected_eds"])) if "corrected_eds" in g else np.array([]))
            corrected_rates.append(np.array(_read_dset(g["corrected_rate"])) if "corrected_rate" in g else np.array([]))
            FFD_eds.append(np.array(_read_dset(g["FFD_eds"])) if "FFD_eds" in g else np.array([]))
            raw_rates.append(np.array(_read_dset(g["raw_rate"])) if "raw_rate" in g else np.array([]))
            F_flare_F_bols.append(float(_read_dset(g["F_Flare_F_Bol"])) if "F_Flare_F_Bol" in g else np.nan)
            F_flare_F_bol_errs.append(float(_read_dset(g["F_Flare_F_Bol_err"])) if "F_Flare_F_Bol_err" in g else np.nan)
            fifty_limits.append(
                float(_read_dset(g["fifty_percent_completeness_limit"]))
                if "fifty_percent_completeness_limit" in g
                else np.nan
            )

        # Read stitched measurements if the HDF5 file contains them.
        epoch_times, epoch_betas, epoch_beta_errs, epoch_F_flare_F_bols, epoch_F_flare_F_bol_errs, epoch_n_flares, epoch_n_sectors = _load_epoch_group(
            stitched["Epochs"] if stitched is not None and "Epochs" in stitched else None
        )

        subepoch_times, subepoch_betas, subepoch_beta_errs, subepoch_F_flare_F_bols, subepoch_F_flare_F_bol_errs, subepoch_n_flares, subepoch_n_sectors = _load_subepoch_group(
            stitched["SubEpochs"] if stitched is not None and "SubEpochs" in stitched else None
        )

    # Store everything in a single dictionary so the plotting function can be
    # called either from the command line or from a Jupyter notebook.
    return dict(
        tic_id=tic_id,
        sector_times=sector_times,
        betas=betas,
        beta_errs=beta_errs,
        corrected_eds=corrected_eds,
        corrected_rates=corrected_rates,
        FFD_eds=FFD_eds,
        raw_rates=raw_rates,
        F_flare_F_bols=F_flare_F_bols,
        F_flare_F_bol_errs=F_flare_F_bol_errs,
        fifty_percent_completeness_limits=fifty_limits,
        epoch_times=epoch_times,
        epoch_betas=epoch_betas,
        epoch_beta_errs=epoch_beta_errs,
        epoch_F_flare_F_bols=epoch_F_flare_F_bols,
        epoch_F_flare_F_bol_errs=epoch_F_flare_F_bol_errs,
        epoch_n_flares=epoch_n_flares,
        epoch_n_sectors=epoch_n_sectors,
        subepoch_times=subepoch_times,
        subepoch_betas=subepoch_betas,
        subepoch_beta_errs=subepoch_beta_errs,
        subepoch_F_flare_F_bols=subepoch_F_flare_F_bols,
        subepoch_F_flare_F_bol_errs=subepoch_F_flare_F_bol_errs,
        subepoch_n_flares=subepoch_n_flares,
        subepoch_n_sectors=subepoch_n_sectors,
        gp_warn_sectors=gp_warn_sectors,
        gp_chi2_warn_threshold=gp_chi2_warn_threshold,
        full_sample_slope=full_sample_slope,

    )


def make_diagnostic_plot(
    tic_id,
    sector_times,
    betas,
    beta_errs,
    corrected_eds,
    corrected_rates,
    FFD_eds,
    raw_rates,
    F_flare_F_bols,
    F_flare_F_bol_errs,
    fifty_percent_completeness_limits,
    epoch_times,
    epoch_betas,
    epoch_beta_errs,
    epoch_F_flare_F_bols,
    epoch_F_flare_F_bol_errs,
    epoch_n_flares,
    epoch_n_sectors,
    subepoch_times,
    subepoch_betas,
    subepoch_beta_errs,
    subepoch_F_flare_F_bols,
    subepoch_F_flare_F_bol_errs,
    subepoch_n_flares,
    subepoch_n_sectors,
    gp_warn_sectors,
    gp_chi2_warn_threshold,
    full_sample_slope,
    output_path=None,
    cmap_name="cycle7",
    show=False,
):
    """
    Build and optionally save the six-panel diagnostic figure.

    The plotting logic follows the paper figure layout:
        upper left: sector FFDs and fixed-slope power-law fits
        upper middle: sector beta values
        upper right: sector integrated flare power
        lower left: flare counts per sector
        lower middle: stitched beta values
        lower right: stitched integrated flare power

    Parameters are passed explicitly rather than as a single dictionary so that
    the plotting function can also be called directly from a notebook.
    """
    # Keep only sectors with enough flares above the completeness threshold to
    # make a meaningful FFD measurement.
    plot_sector_times = list(sector_times)
    plot_sector_mask = [
        _count_flares_above_limit(corrected_eds[i]) >= MIN_FLARES_ABOVE_50PCT
        for i in range(len(plot_sector_times))
    ]

    # Build the time colormap using every available sector, epoch, and sub-epoch
    # time. Color encodes time from green to orange to red by default.
    all_times = np.array(plot_sector_times + list(epoch_times) + list(subepoch_times), dtype=float)
    all_times = all_times[np.isfinite(all_times)]
    if all_times.size == 0:
        all_times = np.array([0.0, 1.0])

    cpick = _build_scalar_mappable(all_times, cmap_name)
    legend_c = cpick.to_rgba(float(np.nanmedian(all_times)))

    # Create the six-panel figure.
    fig, axs = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    ax1, ax2, ax3 = axs[0]
    ax4, ax5, ax6 = axs[1]

    # -------------------------------------------------------------------------
    # Panel 1: sector-by-sector FFDs
    # -------------------------------------------------------------------------
    for i in range(len(plot_sector_times)):
        if not plot_sector_mask[i]:
            continue

        # beta = -99 is a sentinel value indicating that the sector did not
        # produce a valid FFD fit.
        if betas[i] == -99:
            continue

        c = cpick.to_rgba(plot_sector_times[i])

        # Raw FFD points are shown as faint circles.
        ax1.scatter(FFD_eds[i], raw_rates[i], color=c, s=15, alpha=0.1)

        # Completeness-corrected FFD points are shown as larger square markers.
        ax1.scatter(corrected_eds[i], corrected_rates[i], color=c, s=35, marker="s", alpha=0.5)

        # The sector FFD is fit with the full-sample slope and a sector-specific
        # intercept beta.
        ax1.plot(
            corrected_eds[i],
            line(corrected_eds[i], full_sample_slope, betas[i]),
            color=c,
            alpha=0.5,
        )

    # Add dummy artists for the panel legend.
    ylim = ax1.get_ylim()
    ax1.scatter(-99, -99, s=25, marker="s", alpha=0.75, label="Completeness\nCorrected", c="k")
    ax1.errorbar(-99, -99, yerr=0.01, lw=0.5, capsize=2, alpha=0.3, fmt="o", label="Raw", c="k")
    ax1.plot([0, 0], [0, 0], alpha=0.7, color="k", label="Powerlaw Fit")

    ax1.text(
        -0.6,
        0.08,
        f"Slope: {full_sample_slope:.2f}",
        transform=ax1.get_xaxis_transform(),
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        ha="left",
        va="bottom",
    )

    ax1.set_xlabel("log Equivalent Duration (s)")
    ax1.set_ylabel("log Flare Rate (per day)")
    ax1.set_title(f"TIC {tic_id}")

    # If the HDF5 file flagged any sectors with high GP reduced chi2, print that
    # warning in the figure title so the diagnostic plot calls attention to it.
    if len(gp_warn_sectors) > 0:
        warn_txt = (
            f"CHECK GP: reduced chi2 > {gp_chi2_warn_threshold:.1f} "
            f"in {len(gp_warn_sectors)} sector(s): "
            + ", ".join(gp_warn_sectors[:6])
            + (" ..." if len(gp_warn_sectors) > 6 else "")
        )
        fig.suptitle(warn_txt, color="crimson", fontsize=15, fontweight="bold", y=0.98)

    ax1.set_xlim(-0.8, 3.1)
    ax1.legend(fontsize=14, loc="best")
    ax1.set_ylim(ylim)

    # -------------------------------------------------------------------------
    # Panel 2: sector-by-sector beta
    # -------------------------------------------------------------------------
    valid_inds = (np.array(betas) > -3) & np.array(plot_sector_mask, dtype=bool)

    if np.any(valid_inds):
        times = np.array(plot_sector_times, dtype=float)[valid_inds]
        vbeta = np.array(betas, dtype=float)[valid_inds]
        vberr = np.array(beta_errs, dtype=float)[valid_inds]

        for t, b, be in zip(times, vbeta, vberr):
            yerr = _beta_plot_yerr(be, n_sectors=1)
            ax2.errorbar(
                t,
                b,
                yerr=yerr,
                fmt="o",
                color=cpick.to_rgba(t),
                lw=1.0,
                capsize=2.5,
                linestyle="none",
            )

        x_min = float(np.nanmin(times))
        x_max = float(np.nanmax(times))
    else:
        x_min = float(np.nanmin(all_times))
        x_max = float(np.nanmax(all_times))

    # Use a larger x-axis pad for stars with longer TESS baselines.
    x_pad = 100.0 if (x_max - x_min) > 1700.0 else 50.0
    panel2_xlim = (x_min - x_pad, x_max + x_pad)

    ax2.set_ylabel(r"$\beta$")
    ax2.set_xlabel("Time")
    ax2.set_xlim(panel2_xlim)

    # -------------------------------------------------------------------------
    # Panel 3: sector-by-sector integrated flare power
    # -------------------------------------------------------------------------
    sector_ff = np.array(F_flare_F_bols, dtype=float)
    sector_ff_err = np.array(F_flare_F_bol_errs, dtype=float)

    p3_mask = np.array(plot_sector_mask, dtype=bool)
    p3_times = np.array(plot_sector_times, dtype=float)[p3_mask]
    p3_ff = sector_ff[p3_mask]
    p3_ff_err = sector_ff_err[p3_mask]

    # Convert positive flare-power values to log10 space.
    p3_vals = np.full_like(p3_ff, np.nan, dtype=float)
    positive_ff = p3_ff > 0
    p3_vals[positive_ff] = np.log10(p3_ff[positive_ff])

    valid_p3 = np.isfinite(p3_vals) & np.isfinite(p3_ff_err) & (p3_ff > 0)

    if np.any(valid_p3):
        y, yerr, trunc, ylo = _logyerr_with_truncated_lower(p3_ff[valid_p3], p3_ff_err[valid_p3])
        vt = np.array(p3_times[valid_p3], dtype=float)

        for j in range(len(vt)):
            c = cpick.to_rgba(vt[j])

            ax3.errorbar(
                vt[j],
                y[j],
                yerr=np.array([[yerr[0, j]], [yerr[1, j]]]),
                fmt="none",
                ecolor=c,
                elinewidth=1.0,
                capsize=3.0,
                alpha=0.75,
            )

            # The lower uncertainty bound can cross zero in linear flare-power
            # space. Those cases are plotted with a downward arrow because the
            # corresponding lower error cannot be represented directly in log space.
            if trunc[j]:
                ymin, ymax = ax3.get_ylim()
                yr = max(ymax - ymin, 1e-6)
                y_tip = max(ylo[j] - 0.12, ymin + 0.03 * yr)
                y_tail = max(ylo[j] + 0.02, ymin + 0.10 * yr)

                ax3.annotate(
                    "",
                    xy=(vt[j], y_tip),
                    xytext=(vt[j], y_tail),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=2.0,
                        color=c,
                        mutation_scale=16,
                    ),
                )

    ax3.scatter(p3_times, p3_vals, c=cpick.to_rgba(np.array(p3_times)), s=90)
    ax3.set_ylabel(r"$log(\frac{L_{fl}}{L_{TESS}})$")
    ax3.set_xlabel("Time")
    ax3.set_xlim(panel2_xlim)

    # -------------------------------------------------------------------------
    # Panel 4: number of flares per sector
    # -------------------------------------------------------------------------
    flares_per_sector = [
        len(FFD_eds[i])
        for i in range(len(plot_sector_times))
        if plot_sector_mask[i]
    ]
    p4_times = [
        plot_sector_times[i]
        for i in range(len(plot_sector_times))
        if plot_sector_mask[i]
    ]

    if len(flares_per_sector) > 0:
        p4_n = np.array(flares_per_sector, dtype=float)
        p4_yerr = np.sqrt(np.maximum(p4_n, 0.0))
        p4_t = np.array(p4_times, dtype=float)

        # Use sqrt(N) Poisson uncertainties for flare counts.
        for t, nval, nerr in zip(p4_t, p4_n, p4_yerr):
            ax4.errorbar(
                t,
                nval,
                yerr=nerr,
                fmt="none",
                ecolor=cpick.to_rgba(t),
                elinewidth=1.0,
                capsize=2.5,
                alpha=0.75,
            )

    ax4.scatter(
        p4_times,
        flares_per_sector,
        c=cpick.to_rgba(np.array(p4_times, dtype=float)),
        s=90,
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("N Flares")
    ax4.set_xlim(panel2_xlim)

    # -------------------------------------------------------------------------
    # Panel 5: stitched beta measurements
    # -------------------------------------------------------------------------
    if len(epoch_times) > 0:
        et = np.array(epoch_times, dtype=float)
        eb = np.array(epoch_betas, dtype=float)
        ebe = np.array(epoch_beta_errs, dtype=float)
        ens = np.array(epoch_n_sectors, dtype=int)

        evalid = (eb > -3) & np.isfinite(et)

        # Filled triangles show epoch-stitched measurements.
        for t, b, be, nsec in zip(et[evalid], eb[evalid], ebe[evalid], ens[evalid]):
            yerr = _beta_plot_yerr(be, n_sectors=nsec)
            ax5.errorbar(
                t,
                b,
                yerr=yerr,
                fmt="^",
                color=cpick.to_rgba(t),
                mec=cpick.to_rgba(t),
                mew=0.8,
                ms=26,
                lw=1.2,
                capsize=3.0,
                linestyle="none",
                alpha=0.95,
            )

    if len(subepoch_times) > 0:
        st = np.array(subepoch_times, dtype=float)
        sb = np.array(subepoch_betas, dtype=float)
        sbe = np.array(subepoch_beta_errs, dtype=float)
        sns = np.array(subepoch_n_sectors, dtype=int)

        svalid = (sb > -3) & np.isfinite(st)

        # Open triangles show sub-epoch-stitched measurements.
        for t, b, be, nsec in zip(st[svalid], sb[svalid], sbe[svalid], sns[svalid]):
            yerr = _beta_plot_yerr(be, n_sectors=nsec)
            ax5.errorbar(
                t,
                b,
                yerr=yerr,
                fmt="^",
                color=cpick.to_rgba(t),
                mec=cpick.to_rgba(t),
                mfc="none",
                mew=1.4,
                ms=16,
                lw=1.2,
                capsize=3.0,
                linestyle="none",
                alpha=0.95,
            )

    # Add dummy artists for the stitched-beta legend.
    ax5.plot(
        [],
        [],
        "^",
        color="0.5",
        markerfacecolor="0.5",
        markeredgecolor="0.5",
        linestyle="none",
        label="Epoch stitched",
    )
    ax5.plot(
        [],
        [],
        "^",
        color="0.5",
        markerfacecolor="none",
        markeredgecolor="0.5",
        linestyle="none",
        label="Sub-epoch stitched",
    )

    ax5.set_ylabel(r"$\beta$")
    ax5.set_xlabel("Time")
    ax5.set_xlim(panel2_xlim)
    ax5.legend(fontsize=16, loc="upper right")

    # -------------------------------------------------------------------------
    # Panel 6: stitched integrated flare power
    # -------------------------------------------------------------------------
    if len(epoch_times) > 0:
        et = np.array(epoch_times, dtype=float)
        eff = np.array(epoch_F_flare_F_bols, dtype=float)
        eff_err = np.array(epoch_F_flare_F_bol_errs, dtype=float)

        eff_log = np.full_like(eff, np.nan, dtype=float)
        positive_eff = eff > 0
        eff_log[positive_eff] = np.log10(eff[positive_eff])

        valid_e = np.isfinite(eff_log) & np.isfinite(eff_err) & (eff > 0)

        if np.any(valid_e):
            y, yerr, trunc, ylo = _logyerr_with_truncated_lower(eff[valid_e], eff_err[valid_e])
            vet = et[valid_e]

            for j in range(len(vet)):
                c = cpick.to_rgba(vet[j])

                ax6.errorbar(
                    vet[j],
                    y[j],
                    yerr=np.array([[yerr[0, j]], [yerr[1, j]]]),
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.1,
                    capsize=3.0,
                    alpha=0.75,
                )

                # Add a downward arrow if the lower uncertainty bound extends
                # below zero in linear space.
                if trunc[j]:
                    ymin, ymax = ax6.get_ylim()
                    yr = max(ymax - ymin, 1e-6)
                    y_tip = max(ylo[j] - 0.12, ymin + 0.03 * yr)
                    y_tail = max(ylo[j] + 0.02, ymin + 0.10 * yr)

                    ax6.annotate(
                        "",
                        xy=(vet[j], y_tip),
                        xytext=(vet[j], y_tail),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            lw=2.0,
                            color=c,
                            mutation_scale=16,
                        ),
                    )

        # Filled triangles show epoch-stitched flare-power measurements.
        ax6.scatter(
            et,
            eff_log,
            c=cpick.to_rgba(et),
            marker="^",
            s=420,
            edgecolor=cpick.to_rgba(et),
            linewidth=0.8,
            alpha=0.95,
        )

    if len(subepoch_times) > 0:
        st = np.array(subepoch_times, dtype=float)
        sff = np.array(subepoch_F_flare_F_bols, dtype=float)
        sff_err = np.array(subepoch_F_flare_F_bol_errs, dtype=float)

        sff_log = np.full_like(sff, np.nan, dtype=float)
        positive_sff = sff > 0
        sff_log[positive_sff] = np.log10(sff[positive_sff])

        valid_s = np.isfinite(sff_log) & np.isfinite(sff_err) & (sff > 0)

        if np.any(valid_s):
            y, yerr, trunc, ylo = _logyerr_with_truncated_lower(sff[valid_s], sff_err[valid_s])
            vst = st[valid_s]

            for j in range(len(vst)):
                c = cpick.to_rgba(vst[j])

                ax6.errorbar(
                    vst[j],
                    y[j],
                    yerr=np.array([[yerr[0, j]], [yerr[1, j]]]),
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.1,
                    capsize=3.0,
                    alpha=0.75,
                )

                # Add a downward arrow if the lower uncertainty bound extends
                # below zero in linear space.
                if trunc[j]:
                    ymin, ymax = ax6.get_ylim()
                    yr = max(ymax - ymin, 1e-6)
                    y_tip = max(ylo[j] - 0.12, ymin + 0.03 * yr)
                    y_tail = max(ylo[j] + 0.02, ymin + 0.10 * yr)

                    ax6.annotate(
                        "",
                        xy=(vst[j], y_tip),
                        xytext=(vst[j], y_tail),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            lw=2.0,
                            color=c,
                            mutation_scale=16,
                        ),
                    )

        # Open triangles show sub-epoch-stitched flare-power measurements.
        ax6.scatter(
            st,
            sff_log,
            marker="^",
            s=240,
            facecolors="none",
            edgecolors=cpick.to_rgba(st),
            linewidth=1.5,
            alpha=0.95,
        )

    ax6.set_ylabel(r"$log(\frac{L_{fl}}{L_{TESS}})$")
    ax6.set_xlabel("Time")
    ax6.set_xlim(panel2_xlim)

    # Add dummy artists for the stitched-flare-power legend.
    ax6.plot(
        [],
        [],
        "^",
        color=legend_c,
        markerfacecolor=legend_c,
        markeredgecolor=legend_c,
        linestyle="none",
        label="Epoch stitched",
    )
    ax6.plot(
        [],
        [],
        "^",
        color=legend_c,
        markerfacecolor="none",
        markeredgecolor=legend_c,
        linestyle="none",
        label="Sub-epoch stitched",
    )

    # Apply consistent tick formatting across all six panels.
    for ax in axs.ravel():
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", labelsize=20, length=8, width=1.0)

    # Save the figure only if an output directory was supplied. Otherwise the
    # figure can be displayed interactively with --show or closed after generation.
    fig_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        fig_path = output_path / f"tic{tic_id}_Diagnostic_Figure_Cycle7_Fillin.pdf"
        fig.savefig(
            fig_path,
            dpi=250,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="w",
            format="pdf",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig_path


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Make the 6-panel Cycle 7 fill-in diagnostic plot."
    )

    parser.add_argument(
        "input_file",
        type=Path,
        nargs="+",
        help="One or more tic*_final_output_Cycle7_Fillin.h5 files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the PDFs will be written. "
            "If omitted, plots are only shown or generated in memory."
        ),
    )

    parser.add_argument(
        "--cmap",
        default="cycle7",
        help="Matplotlib colormap name, or 'cycle7' for the original custom palette.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each plot instead of only saving it.",
    )

    return parser.parse_args()


def main():
    """
    Command-line entry point.

    The script loops over every input HDF5 file, reads the plotting inputs, makes
    the diagnostic figure, and prints the saved path when an output directory is
    provided.
    """
    args = parse_args()
    h5_files = args.input_file

    for h5_path in h5_files:
        inputs = load_fillin_plot_inputs(h5_path)

        fig_path = make_diagnostic_plot(
            tic_id=inputs["tic_id"],
            sector_times=inputs["sector_times"],
            betas=inputs["betas"],
            beta_errs=inputs["beta_errs"],
            corrected_eds=inputs["corrected_eds"],
            corrected_rates=inputs["corrected_rates"],
            FFD_eds=inputs["FFD_eds"],
            raw_rates=inputs["raw_rates"],
            F_flare_F_bols=inputs["F_flare_F_bols"],
            F_flare_F_bol_errs=inputs["F_flare_F_bol_errs"],
            fifty_percent_completeness_limits=inputs["fifty_percent_completeness_limits"],
            epoch_times=inputs["epoch_times"],
            epoch_betas=inputs["epoch_betas"],
            epoch_beta_errs=inputs["epoch_beta_errs"],
            epoch_F_flare_F_bols=inputs["epoch_F_flare_F_bols"],
            epoch_F_flare_F_bol_errs=inputs["epoch_F_flare_F_bol_errs"],
            epoch_n_flares=inputs["epoch_n_flares"],
            epoch_n_sectors=inputs["epoch_n_sectors"],
            subepoch_times=inputs["subepoch_times"],
            subepoch_betas=inputs["subepoch_betas"],
            subepoch_beta_errs=inputs["subepoch_beta_errs"],
            subepoch_F_flare_F_bols=inputs["subepoch_F_flare_F_bols"],
            subepoch_F_flare_F_bol_errs=inputs["subepoch_F_flare_F_bol_errs"],
            subepoch_n_flares=inputs["subepoch_n_flares"],
            subepoch_n_sectors=inputs["subepoch_n_sectors"],
            gp_warn_sectors=inputs["gp_warn_sectors"],
            gp_chi2_warn_threshold=inputs["gp_chi2_warn_threshold"],
            full_sample_slope=inputs["full_sample_slope"],
            output_path=args.output_dir,
            cmap_name=args.cmap,
            show=args.show,
        )

        if fig_path is not None:
            print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()