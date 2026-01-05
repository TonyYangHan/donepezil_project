import os, numpy as np, pandas as pd, cv2, argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from skimage import measure
from utils import get_number
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.spatial.distance import pdist
from matplotlib.backends.backend_pdf import PdfPages
from visualizations import (
    plot_bars_all,
    plot_violins_all,
    save_mask_overlay_gray,
)



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
    roi_id = get_number(img_path)
    roi_dir = os.path.join(root_path, f"roi_{roi_id}")
    os.makedirs(roi_dir, exist_ok=True)
    # Threshold interpreted as "top percentile" (e.g., 1.0 means keep top 1% intensities)
    top = float(np.clip(threshold, 0.0, 100.0))
    thresh = np.percentile(lipid, [100 - top, 100])
    seg_lipid = np.where(lipid < thresh[0], 0, 255)
    mask, props_df = filter_small_components(seg_lipid, fraction=fraction, min_pixels=min_pixels, verbose=verbose)

    # Save mask image at root (not inside ROI subdirectory)
    cv2.imwrite(os.path.join(root_path, f"{roi_id}_seg_lipid.png"), (mask * 255).astype(np.uint8))

    # Overlay masked regions on original image with translucent red color
    overlay_path = os.path.join(roi_dir, f"{roi_id}_seg_lipid_overlay.png")
    save_mask_overlay_gray(lipid, mask, overlay_path, title="Lipid Droplet Segmentation Overlay", mask_cmap="Reds")

    mask_binary = (mask > 0).astype(np.uint8)
    total_pixels = mask_binary.size
    droplet_pixels = int(mask_binary.sum())
    area_fraction = (droplet_pixels / total_pixels) if total_pixels else 0.0

    # Return properties dataframe and per-image area fraction
    return props_df, area_fraction


def pairwise_tests(metric_values):
    results = {}
    conds = list(metric_values.keys())
    for c1, c2 in itertools.combinations(conds, 2):
        v1 = np.asarray(metric_values[c1])
        v2 = np.asarray(metric_values[c2])
        if v1.size == 0 or v2.size == 0:
            p_t, p_u = np.nan, np.nan
        else:
            _, p_t = ttest_ind(v1, v2, equal_var=False)
            _, p_u = mannwhitneyu(v1, v2)
        results[(c1, c2)] = (p_t, p_u)
    return results


def print_pairwise(results, label):
    print(f"\nPairwise stats for {label}:")
    for (c1, c2), (p_t, p_u) in results.items():
        t_txt = f"{p_t:.4e}" if p_t is not None and not np.isnan(p_t) else "nan"
        u_txt = f"{p_u:.4e}" if p_u is not None and not np.isnan(p_u) else "nan"
        print(f"{c1} vs {c2} -> t p={t_txt}, U p={u_txt}")


def tests_droplet_sizes(cond_dfs_map, cond_order, outdir, hide_ns=False, pdf_pages=None):
    sizes = {cond: pd.concat(dfs, ignore_index=True)["area"].values for cond, dfs in cond_dfs_map.items()}
    pairwise = {(c1, c2): p[0] for (c1, c2), p in pairwise_tests(sizes).items()}
    print_pairwise({k: (p, None) for k, p in pairwise.items()}, "droplet sizes (t-test shown)")
    plot_bars_all(sizes, cond_order, pairwise, outdir, "droplet_sizes", "Droplet Size", hide_ns=hide_ns, pdf_pages=pdf_pages)
    plot_violins_all(sizes, cond_order, pairwise, outdir, "droplet_sizes", "Droplet Size", hide_ns=hide_ns, pdf_pages=pdf_pages)


def test_centroid_distance(cond_dfs_map, cond_order, outdir, hide_ns=False, pdf_pages=None):
    distances = {}
    for cond, dfs in cond_dfs_map.items():
        dists = []
        for df in dfs:
            centroids = df[["centroid-0", "centroid-1"]].values
            if len(centroids) > 1:
                dists += pdist(centroids).tolist()
        distances[cond] = np.asarray(dists)
    pairwise = {(c1, c2): p[0] for (c1, c2), p in pairwise_tests(distances).items()}
    print_pairwise({k: (p, None) for k, p in pairwise.items()}, "centroid distances (t-test shown)")
    plot_bars_all(distances, cond_order, pairwise, outdir, "pairwise_distances", "Pairwise Distance", hide_ns=hide_ns, pdf_pages=pdf_pages)
    plot_violins_all(distances, cond_order, pairwise, outdir, "pairwise_distances", "Pairwise Distance", hide_ns=hide_ns, pdf_pages=pdf_pages)


def test_droplet_counts(cond_dfs_map, cond_order, outdir, hide_ns=False, pdf_pages=None):
    counts = {cond: np.asarray([df.shape[0] for df in dfs]) for cond, dfs in cond_dfs_map.items()}
    pairwise = {(c1, c2): p[0] for (c1, c2), p in pairwise_tests(counts).items()}
    print_pairwise({k: (p, None) for k, p in pairwise.items()}, "droplet counts (t-test shown)")
    plot_bars_all(counts, cond_order, pairwise, outdir, "droplet_counts", "Droplet Count", hide_ns=hide_ns, pdf_pages=pdf_pages)
    plot_violins_all(counts, cond_order, pairwise, outdir, "droplet_counts", "Droplet Count", hide_ns=hide_ns, pdf_pages=pdf_pages)


def test_area_fraction(cond_frac_map, cond_order, outdir, hide_ns=False, pdf_pages=None):
    pairwise = {(c1, c2): p[0] for (c1, c2), p in pairwise_tests(cond_frac_map).items()}
    print_pairwise({k: (p, None) for k, p in pairwise.items()}, "area fraction (t-test shown)")
    plot_bars_all(cond_frac_map, cond_order, pairwise, outdir, "area_fraction", "Droplet Area Fraction", hide_ns=hide_ns, pdf_pages=pdf_pages)
    plot_violins_all(cond_frac_map, cond_order, pairwise, outdir, "area_fraction", "Droplet Area Fraction", hide_ns=hide_ns, pdf_pages=pdf_pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze lipid droplets in images (accelerated with parallel processing).")
    parser.add_argument("dirs", nargs="+", type=str, help="Input directories for each condition")
    parser.add_argument("--conds", "-c", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--out", "-o", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--threshold", "-t", type=float, default=1.0, help="Top intensity percentile to keep (e.g., 1.0 keeps top 1%)")
    parser.add_argument("--fraction", "-f", type=float, default=0.5, help="Fraction for filtering small components (bottom fraction cutoff)")
    parser.add_argument("--min-pixels", "-m", type=int, default=9, help="Minimum region size (in pixels) to keep in the mask")
    parser.add_argument("--workers", "-w", type=int, default=(os.cpu_count() or 1), help="Number of parallel worker processes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-image region counts during segmentation")
    parser.add_argument("--hide-ns", action="store_true", help="Hide non-significant pairwise bars/labels (p >= 0.05)")
    parser.add_argument("--pdf-out", "-p", type=str, default=None, help="Optional path to save all plots into a single multi-page PDF")
    args = parser.parse_args()

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    os.makedirs(args.out, exist_ok=True)

    cond_dfs_map = {cond: [] for cond in args.conds}
    cond_frac_map = {cond: [] for cond in args.conds}

    for cond, dir_path in zip(args.conds, args.dirs):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        files = [fname for fname in sorted(os.listdir(dir_path)) if "797" in fname and fname.lower().endswith((".tif", ".tiff"))]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(seg_droplets, dir_path, fname, args.threshold, args.fraction, args.min_pixels, args.verbose): fname for fname in files}
            with tqdm(total=len(futures), desc=f"Segmenting {cond}") as pbar:
                for fut in as_completed(futures):
                    fname = futures[fut]
                    try:
                        props_df, area_fraction = fut.result()
                        # Record fraction even if no droplets found for that image
                        cond_frac_map[cond].append(area_fraction)
                        if props_df is not None and not props_df.empty:
                            cond_dfs_map[cond].append(props_df)
                    except Exception as e:
                        print(f"Error processing {cond} file {fname}: {e}")
                    pbar.update(1)

    missing = [cond for cond, dfs in cond_dfs_map.items() if not dfs]
    if missing:
        raise RuntimeError(f"Insufficient data: no detected droplets for {', '.join(missing)}")

    pdf_pages = PdfPages(args.pdf_out) if args.pdf_out else None
    try:
        tests_droplet_sizes(cond_dfs_map, args.conds, args.out, args.hide_ns, pdf_pages)
        test_centroid_distance(cond_dfs_map, args.conds, args.out, args.hide_ns, pdf_pages)
        test_droplet_counts(cond_dfs_map, args.conds, args.out, args.hide_ns, pdf_pages)
        test_area_fraction(cond_frac_map, args.conds, args.out, args.hide_ns, pdf_pages)
    finally:
        if pdf_pages is not None:
            pdf_pages.close()
    

