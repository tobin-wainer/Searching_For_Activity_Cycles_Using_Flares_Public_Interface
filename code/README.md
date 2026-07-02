# HDF5 Structure and Access Example

The .h5 products are stored on Zenodo in HDF5 files named like `tic260506296_final_output_Cycle7_Fillin.h5`. Each file has three top-level groups:

- `Summary`
- `Sectors/sector_<n>_lc`
- `Stitched`

## Summary

This group stores global metadata for the TIC and the stitched activity-cycle products.

- `full_sample_slope`, `full_sample_intercept`: the full-sample power-law fit to the FFD, used in the diagnostic plots.
- `gp_chi2_warning_sectors`, `gp_chi2_warning_threshold`, `n_gp_chi2_warnings`: GP quality-control metadata.
- `n_sectors_current`, `n_sectors_filled_in`, `n_sectors_old_complete`: bookkeeping for the merged dataset.
- `old_final_used`: path to the upstream merged file that fed the fill-in product.
- `completeness_method_for_filled_sectors`: completeness handling used when sectors were filled.

## Sector Groups

Each `Sectors/sector_<n>_lc` group contains one TESS sector light curve and its flare products.

Main arrays:

- `sector_duration`: total observing duration in days.
- `lc_time`, `lc_flux`, `lc_flux_err`: the sector light curve.
- `mu`: GP model prediction for the light curve.
- `flare_mask`: boolean cadence mask marking flare cadences.
- `flare_starts`, `flare_ends`: indices in the sector light curve for each identified flare.
- `ed`, `ed_err`: raw flare equivalent durations.
- `FFD_eds`, `FFD_ed_err`: flare distribution values used in the rate fit.
- `raw_rate`, `raw_rate_err`: flare rates from the uncorrected equivalent durations.
- `corrected_eds`, `corrected_rate`: completeness-corrected FFD values above the 50% threshold.
- `F_Flare_F_Bol`, `F_Flare_F_Bol_err`: flare-to-bolometric flux fraction and uncertainty.
- `beta`, `beta_err`: sector FFD slope and uncertainty.
- `fifty_percent_completeness_limit`: ED completeness threshold.

Verification fields:

- `comp_function_params`: completeness function parameters.
- `gp_reduced_chi2`, `gp_check_required`: GP diagnostic information.
- `gp_params/log_*`: the stored GP parameterization.


## Stitched Groups

`Stitched` contains merged products across sectors.

- `Stitched/Epochs/epoch_<k>`: merged activity-cycle epochs.
- `Stitched/SubEpochs/epoch_<k>/group_<m>`: finer merged sub-epochs.
- `sector_keys`: sector IDs included in each merged product.
- `n_sectors`: number of sectors in the merged chunk.
- `flare_durations`: flare durations for the stitched products.
- `lc_time`, `lc_flux`, `lc_flux_err`: concatenated light curves.
- `ed`, `ed_err`, `FFD_eds`, `FFD_ed_err`: stitched flare measurements.
- `beta`, `beta_err`, `F_Flare_F_Bol`, `F_Flare_F_Bol_err`: stitched activity diagnostics.
- `sector_duration`, `fifty_percent_completeness_limit`: merged duration and completeness metadata.

At the top of `Stitched` you will also find:

- `ordered_sector_keys`, `ordered_sector_median_times`: the sector ordering used by the merge.
- `epoch_gap_days`: the gap threshold used to separate epochs.
- `preferred_subgroup_size`: the target subgroup size for the sub-epoch merge.
- `min_flares_above_50pct_per_sector`: minimum flare count required above the 50% completeness limit.

## Quick Access Example

```python
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

h5_path = Path("/path/to/tic260506296_final_output_Cycle7_Fillin.h5")

with h5py.File(h5_path, "r") as f:
    print(list(f.keys()))
    print(list(f["Sectors"].keys())[:5])

    g = f["Sectors"]["sector_1_lc"]
    lc_time = g["lc_time"][...]
    lc_flux = g["lc_flux"][...]
    flare_mask = g["flare_mask"][...].astype(bool) if "flare_mask" in g else np.zeros_like(lc_time, dtype=bool)
    gp = g["mu"][...] if "mu" in g else None

fig, ax = plt.subplots(figsize=(12, 4))
flux_norm = lc_flux / np.nanmean(lc_flux)
ax.scatter(lc_time, flux_norm, s=1.2, color="black", rasterized=True)

if flare_mask.shape == lc_flux.shape and np.any(flare_mask):
    flare_times = lc_time[flare_mask]
    flare_flux = flux_norm[flare_mask]
    ax.scatter(flare_times, flare_flux, s=3, color="red", label="Flare")

if gp is not None and len(gp) == len(lc_time):
    ax.plot(lc_time, gp / np.nanmean(lc_flux), color="royalblue", linewidth=0.8, label="GP Model")

ax.set_xlabel("Time [BJD - 2457000]")
ax.set_ylabel("Normalized Flux")
ax.legend(loc="upper left", frameon=False)
plt.tight_layout()
plt.show()
```

The reusable script in `code/access_hdf5_tutorial.py` prints the full tree before making a similar sector plot.