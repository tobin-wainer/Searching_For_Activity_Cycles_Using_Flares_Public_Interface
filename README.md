# Searching For Activity Cycles Using Flares

This repository is the public interface for data and Figures from. Wainer et al. 2026, "Searching for Stellar Activity Cycles using Flares II: The TESS CVZ". The HDF5 data products files are hosted on Zenodo and referenced here once the DOI is available. The figures in this repo are exports of the diagnostic plots for the 17 candidates from the paper.

## Data Access

The Zenodo record will contain the actual `.h5` files and should be cited directly in any downstream use. Replace the placeholder below with the final DOI after publication.

Zenodo DOI: 10.5281/zenodo.23271534



## What Is In This Repo

- `Figures/Best_Candidates/`: the diagnostic PDFs for the 17 candidates in the paper, named `tic<ID>.pdf`.
- `code/access_hdf5_tutorial.py`: a small example that prints the HDF5 tree and plots one sector light curve with flare markers.
- `code/make_6_panel_diagnostic_plot.py`: a standalone code to make 6-panel diagnostic plots, with a configurable colormap.
- `code/README.md`: the HDF5 structure summary and a short access tutorial.

## HDF5 Layout At A Glance

Each archive file is organized into three top-level groups:

- `Summary`: global metadata for the TIC and the stitched products.
- `Sectors/sector_<n>_lc`: one group per TESS sector.
- `Stitched`: combined products across sectors, split into `Epochs` and `SubEpochs`.

The sector-level groups contain the light curve, flare mask, flare equivalent durations, flare rates, GP model outputs, completeness limits, and the sector-specific `beta` measurements. The stitched groups hold the same style of products after sectors are merged into activity-cycle epochs and sub-epochs.

See `code/README.md` for a fuller field-by-field description and a quick access example.
