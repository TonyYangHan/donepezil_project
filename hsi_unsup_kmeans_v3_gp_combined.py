import argparse, os, numpy as np, pandas as pd, rampy as rp, tifffile, cv2
from matplotlib.backends.backend_pdf import PdfPages
from cuml import UMAP as GPU_UMAP
from cuml.cluster import KMeans as GPU_KMeans
from scipy.interpolate import interp1d
from visualizations import (
    plot_cluster_umap,
    plot_cluster_spectra,
    map_clusters,
    plot_cluster_composition,
    plot_cluster_composition_by_condition,
)

# Shortened code, accelerated, and segmenting out background using SAM (use other code to pre-generate masks)
# Further accelerated, add stats testing

def mask_generation(img):

    img8 = cv2.convertScaleAbs(img, alpha=255.0 / float(img.max()) if float(img.max()) > 0 else 1.0)
    img8 = np.clip(img8, 0, 255).astype(np.uint8)
    _, mask8 = cv2.threshold(img8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask8

def analyze_cluster_composition(cluster_labels, n_clusters):
    counts = np.bincount(cluster_labels, minlength=n_clusters) if cluster_labels.size else np.zeros(n_clusters, dtype=int)
    ratios = counts / cluster_labels.size if cluster_labels.size > 0 else np.zeros(n_clusters)
    return pd.DataFrame({"cluster_id": np.arange(n_clusters), "count": counts, "ratio": ratios})


def smooth_spectrum(spectrum, lamda=0.2):
    return rp.smooth(np.arange(len(spectrum)), spectrum, method="whittaker", Lambda=lamda)

def load_water_baseline(spectra_start, spectra_end, n_channels, water_csv_path=None):
    if water_csv_path is None:
        water_csv_path = "water_HSI_76.csv"
    baseline = pd.read_csv(water_csv_path, header=None).values.flatten()[::-1]
    x = np.linspace(spectra_start, spectra_end, len(baseline))
    return interp1d(x, baseline, kind="cubic", fill_value=(baseline[0], baseline[-1]), bounds_error=False), np.linspace(spectra_start, spectra_end, n_channels)

def preprocess_spectra(spectra, wavenumbers, baseline_interpolator):
    water_baseline = np.clip(baseline_interpolator(wavenumbers), 0, 1)
    scale = np.mean(spectra[:5, :], axis=0)
    spectra = np.maximum(spectra, 0) - water_baseline[:, None] * scale
    spectra -= np.mean(spectra[:5, :], axis=0)
    spectra = np.maximum(spectra, 0)
    max_vals = np.max(spectra, axis=0)
    valid = max_vals > 0
    spectra[:, valid] /= max_vals[valid]
    for i in np.where(valid)[0]: spectra[:, i] = smooth_spectrum(spectra[:, i])
    return np.nan_to_num(spectra, nan=0.0, posinf=1.0, neginf=0.0)

def resample_stack_to_expected_channels(stack, expected_channels, spectra_start, spectra_end):
    if stack.shape[0] == expected_channels:
        return stack.astype(np.float32)

    orig_channels = stack.shape[0]
    old_axis = np.linspace(spectra_start, spectra_end, orig_channels)
    new_axis = np.linspace(spectra_start, spectra_end, expected_channels)
    flat = stack.reshape(orig_channels, -1).astype(np.float32)
    f = interp1d(old_axis, flat, axis=0, kind="linear", fill_value="extrapolate", bounds_error=False)
    resampled = f(new_axis).reshape((expected_channels,) + stack.shape[1:])
    print(f"Interpolated stack from {orig_channels} to {expected_channels} channels.")
    return resampled.astype(np.float32)


def load_hyperstacks_from_dirs(input_dirs, expected_channels, spectra_start, spectra_end):
    stacks, valid_dirs = [], []
    for d in input_dirs:
        stack_path = os.path.join(d, "masked_stack.tif")
        if not os.path.isfile(stack_path):
            continue

        stack = tifffile.imread(stack_path)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        if stack.shape[0] != expected_channels:
            stack = resample_stack_to_expected_channels(stack, expected_channels, spectra_start, spectra_end)
        else:
            stack = stack.astype(np.float32)

        stacks.append(stack)
        valid_dirs.append(d)

    return stacks, valid_dirs

def preprocess_all_stacks(stacks, baseline_interpolator, wavenumbers,
                          drop_invalid=True, deduplicate=True):
    """
    Returns:
      combined_spectra_f : (C, P_kept) float32  — filtered & deduplicated spectra
      all_img_shapes     : list[(C, H, W)]      — unchanged per directory
      all_indices_f      : list[(rows_kept, cols_kept)] — filtered pixel indices
    """
    all_img_shapes, all_indices, all_spectra = [], [], []

    # 1) Per-directory preprocessing
    for stack in stacks:
        image = np.flip(stack, axis=0)               # (C, H, W)
        C, H, W = image.shape
        mask = np.sum(image, axis=0) > 0             # keep non-empty pixels
        r, c = np.where(mask)
        spectra = preprocess_spectra(
            image[:, mask].reshape(C, -1),
            wavenumbers,
            baseline_interpolator
        ).astype(np.float32)

        all_spectra.append(spectra)                  # (C, P_dir_kept_initial)
        all_img_shapes.append((C, H, W))
        all_indices.append((r, c))

    # 2) Concatenate and build global keep mask
    combined = np.concatenate(all_spectra, axis=1)   # (C, P_total)
    P_total = combined.shape[1]

    sizes  = [idx[0].size for idx in all_indices]
    starts = np.cumsum([0] + sizes[:-1])

    keep_mask = np.ones(P_total, dtype=bool)
    if drop_invalid:
        is_finite   = np.isfinite(combined).all(axis=0)
        has_var     = combined.var(axis=0) > 1e-12
        nonzero_max = combined.max(axis=0) > 0
        keep_mask &= (is_finite & has_var & nonzero_max)

    if deduplicate:
        kept_cols = np.flatnonzero(keep_mask)
        Xt = np.ascontiguousarray(combined[:, keep_mask].T)   # (P_keep, C)
        Xt_unique, unique_idx = np.unique(Xt, axis=0, return_index=True)
        final_keep = np.zeros(P_total, dtype=bool)
        final_keep[kept_cols[unique_idx]] = True
        keep_mask = final_keep

    if not keep_mask.any():
        raise ValueError("No pixels left after filtering/deduplication.")

    # 3) Apply mask to spectra and per-directory indices
    combined_f = combined[:, keep_mask]                        # (C, P_kept)

    all_indices_f = []
    for i, (r, c) in enumerate(all_indices):
        start, end = starts[i], starts[i] + sizes[i]
        dir_keep = keep_mask[start:end]
        all_indices_f.append((r[dir_keep], c[dir_keep]))

    kept_after = int(keep_mask.sum())
    print(f"Combined spectra shape (filtered): {combined_f.shape} | "
          f"removed {P_total - kept_after} pixels (invalid/duplicates)")

    return combined_f, all_img_shapes, all_indices_f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperspectral SRS Image Analysis for multiple directories (combined analysis).")
    parser.add_argument("baseline_path", type=str, help="Path to water baseline CSV")
    parser.add_argument("dirs", nargs="+", help="One or more directories containing stacks")
    parser.add_argument("--conds", "-C", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--n_clusters", "-n", type=int, default=6)
    parser.add_argument("--spectra_start", type=float, default=2700)
    parser.add_argument("--spectra_end", type=float, default=3100)
    parser.add_argument("--expected_channels", "-c", type=int, default=62)
    parser.add_argument("--mask", "-m", action = "store_true", help = "Use existing masks")
    parser.add_argument("--output", "-o", type=str, required = True, help="Output directory for plots")
    parser.add_argument("--pdf", action="store_true", help="Also save all generated plots into a single PDF in the output directory")
    parser.add_argument(
        "--drop-lowest-ptp",
        "-d",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help=(
            "Drop the lowest-K clusters by mean per-pixel PTP (max-min) from the initial clustering, "
            "then recluster the remaining spectra. Use '-d' for K=1 or '-d K' for K>1. Default: 0 (disabled)."
        ),
    )
    args = parser.parse_args()

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    input_dirs, dir_metadata = [], []

    def find_masked_stack_dirs(base: str):
        """Recursively search for all directories containing masked_stack.tif under base."""
        found = []
        for dir_path, _, files in os.walk(base):
            if "masked_stack.tif" in files:
                found.append(dir_path)
        return sorted(found)

    for base, cond in zip(args.dirs, args.conds):
        if not os.path.isdir(base):
            raise ValueError(f"Directory not found: {base}")

        stack_dirs = find_masked_stack_dirs(base)
        if not stack_dirs:
            raise FileNotFoundError(f"masked_stack.tif not found under {base}")

        for stack_dir in stack_dirs:
            stack_path = os.path.join(stack_dir, "masked_stack.tif")
            try:
                with tifffile.TiffFile(stack_path) as tf:
                    if len(tf.pages) == 0:
                        raise FileNotFoundError(f"masked_stack.tif unreadable in {stack_dir}")
            except Exception as exc:
                raise FileNotFoundError(f"masked_stack.tif unreadable in {stack_dir}") from exc

            print(f"Found stack: {stack_path} (condition: {cond})")
            input_dirs.append(stack_dir)
            dir_metadata.append({"dir": stack_dir, "condition": cond})

    print(f"Found {len(input_dirs)} valid stack directories across {len(args.conds)} conditions.")
    stacks, _ = load_hyperstacks_from_dirs(
        input_dirs,
        args.expected_channels,
        args.spectra_start,
        args.spectra_end,
    )

    if args.mask:
        for idx, stack in enumerate(stacks):
            n_channels = stack.shape[0]
            
            protein_img = stack[int(0.6 * n_channels)]
            mask8 = mask_generation(protein_img)
            
            if mask8.max() > 1:
                mask8 = (mask8 > 128).astype(np.uint8)
            mask8 = mask8.astype(stack.dtype) # Ensure same dtype as stack
            stack *= mask8

    if not stacks:
        print("No valid directories found. Exiting."); exit(1)

    baseline_interpolator, wavenumbers = load_water_baseline(args.spectra_start, args.spectra_end, args.expected_channels, args.baseline_path)
    combined_spectra, all_img_shapes, all_indices = preprocess_all_stacks(stacks, baseline_interpolator, wavenumbers)
    print(f"Combined spectra shape: {combined_spectra.shape}")

    # Initial DR + clustering
    embedding = GPU_UMAP(n_components=2, random_state=42).fit_transform(combined_spectra.T)
    cluster_labels = GPU_KMeans(n_clusters=args.n_clusters, random_state=42).fit_predict(combined_spectra.T)

    # Optionally drop the cluster with lowest mean PTP and recluster
    final_spectra = combined_spectra
    final_embedding = embedding
    final_labels = cluster_labels
    final_n_clusters = args.n_clusters
    drop_mask_global = np.ones(combined_spectra.shape[1], dtype=bool)

    if args.drop_lowest_ptp:
        if args.n_clusters <= 1:
            print("--drop-lowest-ptp requested but n_clusters <= 1; skipping drop.")
        else:
            k = int(args.drop_lowest_ptp)
            if k < 0:
                raise ValueError("--drop-lowest-ptp/-d must be >= 0")

            # Cannot drop all clusters; keep at least 1 cluster for reclustering
            k = min(k, args.n_clusters - 1)
            if k == 0:
                print("--drop-lowest-ptp specified as 0; skipping drop.")
            else:
                ptp_vals = final_spectra.max(axis=0) - final_spectra.min(axis=0)  # per-pixel PTP
                means = []
                for i in range(args.n_clusters):
                    idx = (final_labels == i)
                    if not np.any(idx):
                        means.append(np.inf)  # ignore empty clusters
                    else:
                        means.append(float(np.mean(ptp_vals[idx])))

                ranked = np.argsort(np.asarray(means))
                drop_clusters = [int(x) for x in ranked[:k]]
                drop_means = [means[i] for i in drop_clusters]
                print(
                    "Dropping clusters (lowest mean PTP): "
                    + ", ".join([f"{cid} (mean={m:.6f})" for cid, m in zip(drop_clusters, drop_means)])
                )

                drop_mask_global = ~np.isin(final_labels, np.asarray(drop_clusters))
                kept_count = int(drop_mask_global.sum())
                if kept_count <= 0:
                    raise ValueError("All pixels removed by PTP drop; cannot recluster.")

                final_spectra = final_spectra[:, drop_mask_global]
                final_n_clusters = max(1, args.n_clusters - k)
                final_embedding = GPU_UMAP(n_components=2, random_state=42).fit_transform(final_spectra.T)
                final_labels = GPU_KMeans(n_clusters=final_n_clusters, random_state=42).fit_predict(final_spectra.T)

    # # Original color from Zhi
    # colors = np.array([
    #     [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    #     [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    #     [1.0, 0.5, 0.0], [0.5, 0.5, 0.5],
    # ])

    # Alternative color palette
    colors = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [0.470588, 0.368627, 0.941176], [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0], [0.5, 0.5, 0.5],
    ])
    # cmap = plt.get_cmap('tab10')  # or 'tab20', 'Set1', etc.
    # colors = np.array([cmap(i)[:3] for i in range(args.n_clusters)])

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    pdf_pages = None
    pdf_path = None
    if args.pdf:
        pdf_path = os.path.join(out_dir, "plots.pdf")
        pdf_pages = PdfPages(pdf_path)

    try:
        # Use spectra/labels after optional drop so lengths match
        plot_cluster_umap(final_embedding, final_labels, final_n_clusters, colors, out_dir, pdf_pages=pdf_pages)
        plot_cluster_spectra(final_spectra, final_labels, final_n_clusters, wavenumbers, colors, out_dir, pdf_pages=pdf_pages)

        pixel_offset = 0
        condition_counts = {meta["condition"]: np.zeros(final_n_clusters, dtype=int) for meta in dir_metadata}
        # For mapping back per directory when dropping, we need per-directory keep mask derived from drop_mask_global
        # Build starts from per-directory sizes (counts before optional drop)
        dir_sizes = [len(idx_pair[0]) for idx_pair in all_indices]
        dir_starts = np.cumsum([0] + dir_sizes[:-1])
        reduced_offset = 0  # offset into final_labels sequence

        for i, meta in enumerate(dir_metadata):
            # filtered pixel indices for this directory
            r, c = all_indices[i]
            n_pixels = r.size
            start, end = dir_starts[i], dir_starts[i] + n_pixels
            if args.drop_lowest_ptp:
                local_keep = drop_mask_global[start:end]
                n_keep = int(local_keep.sum())
                dir_labels = final_labels[reduced_offset : reduced_offset + n_keep]
                reduced_offset += n_keep
                r_kept, c_kept = r[local_keep], c[local_keep]
            else:
                dir_labels = final_labels[pixel_offset : pixel_offset + n_pixels]
                pixel_offset += n_pixels
                r_kept, c_kept = r, c

            stats = analyze_cluster_composition(dir_labels, final_n_clusters)
            condition_counts[meta["condition"]] += stats["count"].to_numpy()
            stats.to_csv(os.path.join(meta["dir"], "cluster_stats_combined.csv"), index=False)
            plot_cluster_composition(stats, final_n_clusters, meta["dir"], tag="combined")
            map_clusters(dir_labels, all_img_shapes[i], (r_kept, c_kept), final_n_clusters, colors, meta["dir"], tag="combined")
            print(f"Processed {meta['dir']}.")

        condition_ratios = {}
        for cond, counts in condition_counts.items():
            total = float(counts.sum())
            condition_ratios[cond] = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

        plot_cluster_composition_by_condition(condition_ratios, out_dir, tag="by_condition", colors=colors, pdf_pages=pdf_pages)

    finally:
        if pdf_pages is not None:
            pdf_pages.close()
            print(f"Saved combined PDF: {pdf_path}")

    # Condition-level comparison removed for condition-agnostic combined analysis
    
    print("All tasks completed.")
