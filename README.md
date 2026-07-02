# Searching For Activity Cycles Using Flares

This repository is the public interface for the Cycle 7 fill-in candidate products. The large HDF5 data files are not stored in GitHub; they will be hosted on Zenodo and referenced here once the DOI is available. The figures in this repo are lightweight exports of the diagnostic plots for the best candidates.

## Data Access

The Zenodo record will contain the actual `.h5` files and should be cited directly in any downstream use. Replace the placeholder below with the final DOI after publication.

Zenodo DOI: **[to be added after release]**

Suggested citation:

> Author(s), Year, Cycle 7 fill-in candidate data archive, Zenodo, DOI: [to be added after release].

## What Is In This Repo

- `Figures/Best_Candidates/`: the candidate diagnostic PDFs renamed to `tic<ID>.pdf`.
- `code/access_hdf5_tutorial.py`: a small example that prints the HDF5 tree and plots one sector light curve with flare markers.
- `code/make_6_panel_diagnostic_plot.py`: a standalone version of the 6-panel diagnostic plot notebook, with a configurable colormap.
- `code/README.md`: the HDF5 structure summary and a short access tutorial.

## HDF5 Layout At A Glance

Each archive file is organized into three top-level groups:

- `Summary`: global metadata for the TIC and the stitched products.
- `Sectors/sector_<n>_lc`: one group per TESS sector.
- `Stitched`: combined products across sectors, split into `Epochs` and `SubEpochs`.

The sector-level groups contain the light curve, flare mask, flare equivalent durations, flare rates, GP model outputs, completeness limits, and the sector-specific `beta` measurements. The stitched groups hold the same style of products after sectors are merged into activity-cycle epochs and sub-epochs.

See `code/README.md` for a fuller field-by-field description and a quick access example.
