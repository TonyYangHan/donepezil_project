import os, numpy as np, pandas as pd, cv2, argparse, matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from skimage import measure
from calculate_ratios_gp_v2 import get_number
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.spatial.distance import pdist



def filter_small_components(mask, fraction=0.1, min_pixels: int = 5, verbose: bool = False):
    labeled = measure.label(mask, connectivity=2)
    props_df = pd.DataFrame(
        measure.regionprops_table(
            labeled, properties=("label", "area", "centroid")
        )
    )
    if props_df.empty:
        return mask.astype(mask.dtype), props_df

    # Find cutoff size so that ~fraction of components are in the bottom group
    sizes = props_df["area"].values
    unique_sizes = np.unique(sizes)
    cutoff_index = int(np.floor(len(sizes) * fraction))
    # Determine area cutoff from fraction if applicable
    if cutoff_index > 0:
        sorted_sizes = np.sort(sizes)
        cutoff_size = sorted_sizes[cutoff_index]
        cutoff_size = max(unique_sizes[unique_sizes <= cutoff_size])
    else:
        cutoff_size = 0

    # Keep only components larger than cutoff_size
    # Apply BOTH filters: (1) minimum pixel area, (2) top (1 - fraction) by area
    keep_mask = (props_df["area"] >= int(min_pixels)) & (props_df["area"] > cutoff_size)
    keep_labels = props_df.loc[keep_mask, "label"].values
    filtered_mask = np.isin(labeled, keep_labels).astype(mask.dtype)
    filtered_props_df = props_df[props_df["label"].isin(keep_labels)].reset_index(drop=True)

    if verbose:
        print("Total regions detected:", len(props_df))
        print(f"Regions after filtering (min_pixels={min_pixels}, fraction={fraction}):", len(filtered_props_df))

    return filtered_mask, filtered_props_df.drop("label", axis = 1)



def seg_droplets(root_path, img_path, threshold: float = 1.0, fraction: float = 0.5, min_pixels: int = 5, verbose: bool = False):
    lipid = cv2.imread(os.path.join(root_path, img_path), cv2.IMREAD_UNCHANGED)
    # Threshold interpreted as "top percentile" (e.g., 1.0 means keep top 1% intensities)
    top = float(np.clip(threshold, 0.0, 100.0))
    thresh = np.percentile(lipid, [100 - top, 100])
    seg_lipid = np.where(lipid < thresh[0], 0, 255)
    mask, props_df = filter_small_components(seg_lipid, fraction=fraction, min_pixels=min_pixels, verbose=verbose)

    # Save mask image (keep output directory as root_path per requirements)
    cv2.imwrite(os.path.join(root_path, f"roi{get_number(img_path)}_seg_lipid.png"), (mask * 255).astype(np.uint8))

    # Overlay masked regions on original image with translucent red color
    overlay_path = os.path.join(root_path, f"roi{get_number(img_path)}_seg_lipid_overlay.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(lipid, cmap="gray")
    # Use a red colormap for the mask overlay
    plt.imshow(mask, cmap="Reds", alpha=0.6)
    plt.title("Lipid Droplet Segmentation Overlay")
    plt.axis("off")
    plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Return properties dataframe instead of saving it
    return props_df


def plot_bars_with_error(cond1_vals, cond2_vals, outdir, test_name, cond_1_label, cond_2_label, ylabel, p_val=None):
    cond1_vals = np.asarray(cond1_vals)
    cond2_vals = np.asarray(cond2_vals)
    means = [np.mean(cond1_vals) if cond1_vals.size else 0.0,
             np.mean(cond2_vals) if cond2_vals.size else 0.0]
    stds  = [np.std(cond1_vals, ddof=1) if cond1_vals.size > 1 else 0.0,
             np.std(cond2_vals, ddof=1) if cond2_vals.size > 1 else 0.0]

    # Taller and narrower figure to emphasize differences
    plt.figure(figsize=(4, 8))
    x = np.arange(2)
    plt.bar(x, means, yerr=stds, capsize=6, color=["tab:blue", "tab:orange"], alpha=0.8)
    plt.xticks(x, [cond_1_label, cond_2_label])
    plt.ylabel(ylabel)
    plt.title(f"{test_name}")
    # Zoom y-range near the bar tops to amplify visual difference
    top_candidates = [m + s for m, s in zip(means, stds)]
    y_top = max(top_candidates) if top_candidates else max(means)
    y_low_mean = min(means)
    span = max(y_top - y_low_mean, 1e-6)
    y_min = max(y_low_mean - 0.3 * span, 0)
    y_max = y_top + 0.15 * span
    if y_max <= y_min:
        y_max = y_min + 1.0
    plt.ylim(y_min, y_max)

    # Horizontal bar and asterisk for significance
    if p_val is not None:
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        y_bar = y_max - 0.05 * (y_max - y_min)
        plt.plot([x[0], x[1]], [y_bar, y_bar], color='black', linewidth=1.5)
        plt.text(np.mean(x), y_bar + 0.01 * (y_max - y_min), sig, ha='center', va='bottom', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"bar_{test_name}.png"), dpi=300)
    plt.close()


def plot_violins_with_stats(cond1, cond2, outdir, p_val_t=None, p_val_u=None, test_name=""):
    plt.figure(figsize=(6, 6))
    plt.violinplot([cond1, cond2], showmeans=True, showmedians=True, positions=[1, 2])
    plt.xticks([1, 2], [args.cond_1, args.cond_2])

    if test_name == "droplet_sizes":
        plt.ylabel(f"Droplet Size")
        plt.title(f"Droplet Size Distribution - {test_name}")
    elif test_name == "pairwise_distances":
        plt.ylabel(f"Pairwise Distance")
        plt.title(f"Pairwise Distance Distribution - {test_name}")
    elif test_name == "droplet_counts":
        plt.ylabel(f"Droplet Count")
        plt.title(f"Droplet Count Distribution - {test_name}")
    else:
        raise ValueError("Invalid plot type specified.")

    # Add horizontal bar and significance annotation
    y_max = max(np.max(cond1), np.max(cond2))
    y_min = min(np.min(cond1), np.min(cond2))
    y_range = y_max - y_min
    y_bar = y_max + 0.01 * y_range

    plt.plot([1, 2], [y_bar, y_bar], color='black', linewidth=1.5)

    if p_val_t is not None:
        if p_val_t < 0.001:
            sig = "***"
        elif p_val_t < 0.01:
            sig = "**"
        elif p_val_t < 0.05:
            sig = "*"
        else:
            sig = "n.s."
    else:
        sig = "n/a"

    plt.text(1.5, y_bar + 0.01 * y_range, sig, ha='center', va='bottom', fontsize=18, fontweight='bold')
    # Print both p-values near the top
    ptxt = []
    if p_val_t is not None:
        ptxt.append(f"t p={p_val_t:.2e}")
    if p_val_u is not None:
        ptxt.append(f"U p={p_val_u:.2e}")
    if ptxt:
        plt.text(1.5, y_bar + 0.06 * y_range, ", ".join(ptxt), ha='center', va='bottom', fontsize=10)

    plt.ylim(y_min - 0.07 * y_range, y_bar + 0.07 * y_range)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"violin_plot_{test_name}.png"), dpi=300)
    plt.close()



def tests_droplet_sizes(cond1_dfs, cond2_dfs):
    
    cond1_df = pd.concat(cond1_dfs, ignore_index=True)
    cond2_df = pd.concat(cond2_dfs, ignore_index=True)

    print(f"Total droplets used for sizes: {len(cond1_df)} ({args.cond_1}) vs {len(cond2_df)} ({args.cond_2})")

    t_stat, p_val_t = ttest_ind(cond1_df['area'], cond2_df['area'], equal_var=False)
    print(f"T-test for droplet sizes: t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4e}")

    u_stat, p_val_u = mannwhitneyu(cond1_df['area'], cond2_df['area'])
    print(f"Mann-Whitney U test for droplet sizes: U-statistic = {u_stat:.4f}, p-value = {p_val_u:.4e}")

    plot_bars_with_error(cond1_df['area'], cond2_df['area'], outdir=args.out,
                         test_name="droplet_sizes", cond_1_label=args.cond_1, cond_2_label=args.cond_2,
                         ylabel="Droplet Size", p_val=p_val_t)
    # Also save violin plot
    plot_violins_with_stats(cond1_df['area'], cond2_df['area'], outdir=args.out, p_val_t=p_val_t, p_val_u=p_val_u, test_name="droplet_sizes")



def test_centroid_distance(cond1_dfs, cond2_dfs):

    cond1_distances, cond2_distances = [], []
    # Prevent different lengths of dfs between conditions
    for cond1_df in cond1_dfs: 
        cond1_centroids = cond1_df[["centroid-0", "centroid-1"]].values
        cond1_distances += pdist(cond1_centroids).tolist()
    for cond2_df in cond2_dfs:
        cond2_centroids = cond2_df[["centroid-0", "centroid-1"]].values
        cond2_distances += pdist(cond2_centroids).tolist()

    total_d1 = sum(df.shape[0] for df in cond1_dfs)
    total_d2 = sum(df.shape[0] for df in cond2_dfs)
    print(f"Total droplets used for centroid distances: {total_d1} ({args.cond_1}) vs {total_d2} ({args.cond_2})")

    t, p_val_t = ttest_ind(cond1_distances, cond2_distances, equal_var=False)
    print(f"T-test for centroid distances: t-statistic = {t:.4f}, p-value = {p_val_t:.4e}")

    u_stat, p_val_u = mannwhitneyu(cond1_distances, cond2_distances)
    print(f"Mann-Whitney U test: U-statistic = {u_stat:.4f}, p-value = {p_val_u:.4e}")

    plot_bars_with_error(cond1_distances, cond2_distances, outdir=args.out,
                         test_name="pairwise_distances", cond_1_label=args.cond_1, cond_2_label=args.cond_2,
                         ylabel="Pairwise Distance", p_val=p_val_t)
    # Also save violin plot
    plot_violins_with_stats(cond1_distances, cond2_distances, outdir=args.out, p_val_t=p_val_t, p_val_u=p_val_u, test_name="pairwise_distances")



def test_droplet_counts(cond1_dfs, cond2_dfs):

    cond1_counts = [df.shape[0] for df in cond1_dfs]
    cond2_counts = [df.shape[0] for df in cond2_dfs]

    print(f"Total droplets (sum over images): {sum(cond1_counts)} ({args.cond_1}) vs {sum(cond2_counts)} ({args.cond_2})")

    t_stat, p_val_t = ttest_ind(cond1_counts, cond2_counts, equal_var=False)
    print(f"T-test for droplet counts: t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4e}")

    u_stat, p_val_u = mannwhitneyu(cond1_counts, cond2_counts)
    print(f"Mann-Whitney U test: U-statistic = {u_stat:.4f}, p-value = {p_val_u:.4e}")

    plot_bars_with_error(cond1_counts, cond2_counts, outdir=args.out,
                         test_name="droplet_counts", cond_1_label=args.cond_1, cond_2_label=args.cond_2,
                         ylabel="Droplet Count", p_val=p_val_t)
    # Also save violin plot
    plot_violins_with_stats(cond1_counts, cond2_counts, outdir=args.out, p_val_t=p_val_t, p_val_u=p_val_u, test_name="droplet_counts")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze lipid droplets in images (accelerated with parallel processing).")
    parser.add_argument("dir_1", type=str, help="Input directory for condition 1 images")
    parser.add_argument("dir_2", type=str, help="Input directory for condition 2 images")
    parser.add_argument("cond_1", type=str, help="Condition 1 name")
    parser.add_argument("cond_2", type=str, help="Condition 2 name")
    parser.add_argument("--out", "-o", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--threshold", "-t", type=float, default=1.0, help="Top intensity percentile to keep (e.g., 1.0 keeps top 1%)")
    parser.add_argument("--fraction", "-f", type=float, default=0.5, help="Fraction for filtering small components (bottom fraction cutoff)")
    parser.add_argument("--min-pixels", "-m", type=int, default=9, help="Minimum region size (in pixels) to keep in the mask")
    parser.add_argument("--workers", "-w", type=int, default=(os.cpu_count() or 1), help="Number of parallel worker processes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-image region counts during segmentation")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cond1_dfs, cond2_dfs = [], []

    # Process dir_1 images with '797' in filename
    if not os.path.isdir(args.dir_1):
        raise FileNotFoundError(f"Directory not found: {args.dir_1}")
    files1 = [fname for fname in sorted(os.listdir(args.dir_1)) if "797" in fname and fname.lower().endswith((".tif", ".tiff"))]
    # Parallel processing with progress bar
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures1 = {ex.submit(seg_droplets, args.dir_1, fname, args.threshold, args.fraction, args.min_pixels, args.verbose): fname for fname in files1}
        with tqdm(total=len(futures1), desc=f"Segmenting {args.cond_1}") as pbar:
            for fut in as_completed(futures1):
                fname = futures1[fut]
                try:
                    props_df = fut.result()
                    if props_df is not None and not props_df.empty:
                        cond1_dfs.append(props_df)
                except Exception as e:
                    print(f"Error processing {args.cond_1} file {fname}: {e}")
                pbar.update(1)

    # Process dir_2 images with '797' in filename
    if not os.path.isdir(args.dir_2):
        raise FileNotFoundError(f"Directory not found: {args.dir_2}")
    files2 = [fname for fname in sorted(os.listdir(args.dir_2)) if "797" in fname and fname.lower().endswith((".tif", ".tiff"))]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures2 = {ex.submit(seg_droplets, args.dir_2, fname, args.threshold, args.fraction, args.min_pixels, args.verbose): fname for fname in files2}
        with tqdm(total=len(futures2), desc=f"Segmenting {args.cond_2}") as pbar:
            for fut in as_completed(futures2):
                fname = futures2[fut]
                try:
                    props_df = fut.result()
                    if props_df is not None and not props_df.empty:
                        cond2_dfs.append(props_df)
                except Exception as e:
                    print(f"Error processing {args.cond_2} file {fname}: {e}")
                pbar.update(1)

    if not cond1_dfs or not cond2_dfs:
        raise RuntimeError("Insufficient data: one or both conditions have no detected droplets.")

    # Run tests
    tests_droplet_sizes(cond1_dfs, cond2_dfs)
    test_centroid_distance(cond1_dfs, cond2_dfs)
    test_droplet_counts(cond1_dfs, cond2_dfs)
    

