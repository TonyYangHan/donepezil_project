# Donepezil Project

Analyze metabolic and lipidomic changes in aging fruit flies (wild type) under different donepezil dosages using two-photon fluorescence, hyperspectral SRS clustering, and lipid droplet morphology/statistics. The repository provides reproducible scripts for molecular ratio computation, unsupervised clustering, correlation analyses, and droplet segmentation with statistics and plots.

## Environment and Installation

Conda is recommended.

Windows PowerShell (recommended):

```
conda env create -f hsi_analysis.yml -n hsi_analysis
conda activate hsi_analysis
```

Notes
- GPU-accelerated hyperspectral clustering uses RAPIDS cuML (UMAP, KMeans). Ensure a compatible CUDA setup and install RAPIDS into the same environment if it’s not already included in `hsi_analysis.yml`.
- If you don’t need hyperspectral clustering, the remaining scripts run on CPU with standard scientific Python packages.

## Data Organization Assumptions

Several scripts match files by an ROI/index number embedded in filenames (extracted by digits), requiring consistent naming across modalities for the same ROI. Typical expectations:
- Stokes beams used has wavelength of 1031nm
- Redox ratio: paired images per ROI labeled with `fad` and `nadh`.
- Unsaturation ratio: paired images per ROI at two Raman bands, often labeled with `787` (unsaturated lipid channel) and `794` (saturated lipid channel) in filenames.
- Turnover ratios: precomputed per-ROI images labeled `protein_turn` (841.8 nm/ 791.3 nm)and `lipid_turn` (844.6nm / 797.2nm).
- Correlation script expects precomputed ratio images named with `redox_ratio`, `protein_turn`, `lipid_turn` sharing the same ROI number.
- hyperspectral analysis expects directories containing stacks of `.tif/.tiff` files (one per channel), typically starting with a digit in the filename, plus a water baseline CSV (e.g., `water_HSI_76.csv`).

## Scripts and Usage

All commands below assume the conda environment is active and are shown for Windows PowerShell. Adjust input/output paths as needed.

### 1) Redox and Unsaturation Ratios: `calculate_ratios_gp_v2.py`
Purpose
- Compute redox ratios (from `fad`/`nadh`) and unsaturation ratios (e.g., `787`/`794`) across two conditions, optionally save ratio images, and run statistics (t-test, Mann–Whitney; optional block-median t-test). Also supports histogram/violin plotting.

Key arguments (subset)
- `dir_1 dir_2`: Input folders for condition 1 and 2.
- `--cond1/--cond2`: Labels for conditions in outputs.
- `--save`: Save ratio images per ROI.
- `--plot`: Save histograms/violin plots of distributions.
- `--out/-o`: Output directory.
- `--mean/-m`: Run tests on per-image means.
- `--block-ttest/-b`: Run block-median t-test.
- `--block-size/-bs`: Block size for block tests (default 32).
- `--block-min-nonzero/-bnz`: Min non-zero proportion per block to allow the block to be considered.

Example
```
python calculate_ratios_gp_v2.py D:\data\WT D:\data\D35 \
	--cond1 WT --cond2 D35 --save --plot \
	--out D:\results\ratios --block-ttest -bs 32 -bnz 0.25
```

Outputs
- Ratio images (if `--save`), histograms/violin plots, and printed stats (Welch’s t-test, Mann–Whitney). Filenames include labels and tests.

### 2) Lipid Droplet Segmentation and Stats (Accelerated): `lipid_analysis_accl.py`
Purpose
- Segment lipid droplets (threshold + size filtering), then compare droplet size distributions, pairwise centroid distances, and counts across two conditions. Default segmentation parameters are the best for this particular project. They could be specified easily by command line args. Uses parallel processing for speed and saves bar/violin plots with statistics.

Key arguments (subset)
- `dir_1 dir_2 cond_1 cond_2`: Folders and labels for two conditions.
- `--out/-o` (required): Output directory.
- `--threshold/-t`: Top intensity percentile to keep for segmentation (e.g., `1.0` keeps top 1%).
- `--fraction/-f`: Bottom fraction of components to discard by area.
- `--min-pixels/-m`: Minimum region area to keep.
- `--workers/-w`: Parallel workers.
- `--verbose/-v`: Per-image details.

Example
```
python lipid_analysis_accl.py D:\data\WT D:\data\D35 WT D35 \
	-o D:\results\lipid --threshold 1.0 --fraction 0.5 --min-pixels 9 -w 8 -v
```

Outputs
- Segmentation overlays/masks and statistical plots: droplet sizes, pairwise distances, and droplet counts (bar + violin) with t-test/Mann–Whitney annotations.

### 3) Redox vs. Turnover Correlation: `redox_turnover_corr.py`
Purpose
- Correlate per-pixel or superpixel values between precomputed `redox_ratio` images and `protein_turn`/`lipid_turn` images with matching ROI numbers. Produces hexbin plots with linear fit and correlation metrics.

Key arguments (subset)
- `path`: Root folder searched recursively for files.
- `--block-ttest/-b`: Use block medians (superpixels) for correlation inputs.
- `--block-size`: Superpixel size (default 32).
- `--block-min-nonzero/-bnz`: Min combined non-zero proportion per block to be considered.
- `-o`: Output directory for plots.
- `-p`: Filename prefix for saved plots.

Example
```
python redox_turnover_corr.py D:\data\ratios \
	-b --block-size 32 -bnz 0.5 -o D:\results\corr -p ROIset
```

Outputs
- Hexbin correlation plots for protein and lipid turnover vs. redox, with fitted line and r, p-values printed; PNGs saved if `-o` specified.

### 4) HSI Unsupervised Clustering (GPU): `hsi_unsup_kmeans_v2_gp_combined.py`
Purpose
- Combined analysis across one or more HSI directories: preprocess spectra (water baseline removal, normalization), GPU-UMAP embedding, GPU-KMeans clustering, cluster composition, UMAP/spectra plots, and per-directory cluster maps.

Key arguments (subset)
- `baseline_path`: Path to water baseline CSV (e.g., `water_HSI_76.csv`).
- `dirs...`: One or more directories with per-channel `.tif/.tiff` stacks.
- `--n_clusters/-n`: Number of clusters.
- `--spectra_start/--spectra_end`: Wavenumber range.
- `--expected_channels/-c`: Expected number of channels in stack.
- `--mask`: Use existing masks if present.
- `--sam`: Attempt SAM-based background masking.
- `--output/-o`: Output directory (required).

Example
```
python hsi_unsup_kmeans_v2_gp_combined.py D:\refs\water_HSI_76.csv \
	D:\hsi\sample1 D:\hsi\sample2 -n 6 -c 62 --sam -o D:\results\hsi
```

Outputs
- UMAP plot, cluster spectra, per-image cluster maps, and cluster composition plots saved under the output directory.

### 5) Notebooks
- `Masks_operations.ipynb`: Interactive utilities to segment lipid droplets, generate overlays on raw images, and build simple ratios/overlays for PRM/SRS outputs. Open in Jupyter/VS Code and run cells in order. Outputs include overlay images and example ratio `.tiff` files.

### 6) Helper Shell Scripts
- `run_ratio_all_v2.sh`, `run_lipid_all.sh`, `run_corr_all.sh`: Convenience wrappers to batch-run the above Python scripts.
	- On Windows, use Git Bash/WSL to execute `.sh`, or translate to PowerShell equivalents.

## Tips and Troubleshooting
- Ensure filenames for corresponding ROIs share the same digits so the scripts can match pairs across modalities.
- Remove or mask saturated pixels (4095) where applicable; several scripts already suppress these during ratio calculation.
- For GPU HSI: confirm RAPIDS/cuML installation matches your CUDA/driver versions.
- Use `--verbose` or print statements to inspect per-image filtering and block counts when tuning thresholds and `-bnz`.

## License
See `LICENSE` for details.
