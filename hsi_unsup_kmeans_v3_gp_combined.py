import argparse, os, numpy as np, pandas as pd, rampy as rp, tifffile, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cuml import UMAP as GPU_UMAP
from cuml.cluster import KMeans as GPU_KMeans
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency  # (Imported earlier; safe_chi2 retained though not used)

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

def plot_cluster_umap(embedding, cluster_labels, n_clusters, colors, output_folder):
    # Filter out data points beyond 0.01 and 99.99 percentile range
    if embedding.shape[0] == 0:
        return
    low_pct, high_pct = 0.2, 99.8
    x_min, x_max = np.percentile(embedding[:, 0], [low_pct, high_pct])
    y_min, y_max = np.percentile(embedding[:, 1], [low_pct, high_pct])
    mask_range = (embedding[:, 0] >= x_min) & (embedding[:, 0] <= x_max) & \
                 (embedding[:, 1] >= y_min) & (embedding[:, 1] <= y_max)
    filtered_embedding = embedding[mask_range]
    filtered_labels = cluster_labels[mask_range]

    plt.figure(figsize=(8, 8))
    for i in range(n_clusters):
        mask = filtered_labels == i
        if np.any(mask):
            plt.scatter(filtered_embedding[mask, 0], filtered_embedding[mask, 1], c=[colors[i]], label=f"Cluster {i+1}", alpha=0.6, s=5)
            plt.text(np.mean(filtered_embedding[mask, 0]), np.mean(filtered_embedding[mask, 1]), f"{i+1}", fontsize=12, ha="center", va="center")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2"); plt.xticks([]); plt.yticks([])
    plt.legend(prop={"size": 10}, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "clusters_umap.tiff"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_cluster_spectra(spectra, cluster_labels, n_clusters, wavenumbers, colors, output_folder):
    if spectra.shape[1] == 0:
        return
    plt.figure(figsize=(6, 10))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            mean = np.mean(spectra[:, mask], axis=1)
            std = np.std(spectra[:, mask], axis=1)
            plt.plot(wavenumbers, mean + i * 0.5, color=colors[i], label=f"Cluster {i+1}", alpha=0.5)
            plt.fill_between(wavenumbers, mean + i * 0.5 - std, mean + i * 0.5 + std, color=colors[i], alpha=0.2)
    plt.xlabel(r"Wavenumber (cm$^{-1}$)"); plt.ylabel("Normalized Intensity")
    plt.legend(prop={"size": 10}); plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "cluster_spectra.tiff"), dpi=300)
    plt.close()

def map_clusters(cluster_labels, img_shape, indices, n_clusters, colors, output_folder, tag: str = "combined"):
    label_map = np.full(img_shape[1:], -1, dtype=np.int32)  # -1 = background
    if cluster_labels.size:
        label_map[indices] = cluster_labels
    rgb_map = np.zeros((*img_shape[1:], 3), dtype=np.float32)
    for i in range(n_clusters): rgb_map[label_map == i] = colors[i]
    fname = f"clusters_map_{tag}.tiff"
    tifffile.imwrite(os.path.join(output_folder, fname), (rgb_map * 255).astype(np.uint8))

def plot_cluster_composition(cluster_stats, n_clusters, output_folder, tag: str = "combined"):
    plt.figure(figsize=(8,6))
    x = np.arange(n_clusters)
    plt.bar(x, cluster_stats["ratio"], color="royalblue", alpha=0.7)
    plt.xticks(x, [f"Cluster {i+1}" for i in x], fontsize=10, rotation=45)
    plt.ylabel("Fraction of Pixels in Cluster")
    plt.tight_layout()
    fname = f"cluster_composition_{tag}.tiff"
    plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches="tight")
    plt.close()

def safe_chi2(contingency: np.ndarray):
    # 1) remove columns with zero total and rows with zero total
    keep_cols = contingency.sum(axis=0) > 0
    keep_rows = contingency.sum(axis=1) > 0
    cleaned = contingency[np.ix_(keep_rows, keep_cols)]

    # 2) must be at least 2x2 to test independence
    if cleaned.shape[0] < 2 or cleaned.shape[1] < 2:
        print("Warning: Contingency table too small after cleaning, skipping chi-square test.")
        return None, None, None, None  # or print("Skipped: <2x2 after cleaning")

    # 3) run chi-square (no Yates correction for RxC)
    chi2, p, dof, expected = chi2_contingency(cleaned, correction=False)
    return chi2, p, dof, expected

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

def load_hyperstacks_from_dirs(input_dirs, use_sam):
    stacks, valid_dirs = [], []
    masks = list()
    for d in input_dirs:
        files = sorted([f for f in os.listdir(d) if f[0].isdigit() and f.lower().endswith(('.tif', '.tiff'))])
        stacks.append(np.stack([tifffile.imread(os.path.join(d, f)) for f in files], axis=0))
        valid_dirs.append(d)

        if use_sam:
            fl = [f for f in os.listdir(d) if "mask" in f]

            if len(fl) > 0:
                masks.append(cv2.imread(os.path.join(d, fl[0]), cv2.IMREAD_GRAYSCALE))
            else:
                raise FileNotFoundError(f"No SAM mask found in {d}. Please generate first")

    return stacks, valid_dirs, masks

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
    parser.add_argument("dirs", nargs="+", help="One or more directories containing stacks or subdirectories with stacks")
    parser.add_argument("--n_clusters", "-n", type=int, default=6)
    parser.add_argument("--spectra_start", type=float, default=2700)
    parser.add_argument("--spectra_end", type=float, default=3100)
    parser.add_argument("--expected_channels", "-c", type=int, default=62)
    parser.add_argument("--mask", "-m", action = "store_true", help = "Use existing masks")
    parser.add_argument("--sam", "-s", action = "store_true", help = "Use SAM masks if available")
    parser.add_argument("--output", "-o", type=str, required = True, help="Output directory for plots")
    parser.add_argument("--drop-lowest-ptp", "-d", action="store_true", help="Remove the cluster with the lowest mean PTP (max-min) and recluster the remaining spectra")
    args = parser.parse_args()

    input_dirs, dir_metadata = [], []
    def looks_valid_stack(d: str) -> bool:
        files = [f for f in os.listdir(d) if f[0].isdigit() and f.endswith((".tif", ".tiff"))]
        return len(files) == args.expected_channels

    for base in args.dirs:
        if not os.path.isdir(base):
            continue
        # If base itself is a stack dir
        if looks_valid_stack(base) and "out" not in os.path.basename(base).lower():
            print(f"Found dir: {base}")
            input_dirs.append(base)
            dir_metadata.append({"dir": base})
            continue
        # Otherwise scan subdirectories
        for dir_path, _, _ in os.walk(base):
            folder = dir_path.lower()
            if "out" in folder:
                continue
            try:
                if looks_valid_stack(dir_path):
                    print(f"Found dir: {dir_path}")
                    input_dirs.append(dir_path)
                    dir_metadata.append({"dir": dir_path})
            except PermissionError:
                continue

    print(f"Found {len(input_dirs)} valid directories.")
    stacks, valid_dirs, masks = load_hyperstacks_from_dirs(input_dirs, args.sam)

    if args.mask:
        for idx, stack in enumerate(stacks):
            n_channels = stack.shape[0]
            
            protein_img = stack[int(0.6 * n_channels)]
            if args.sam and np.sum(masks[idx]) > 0.6 * masks[idx].size:
                mask8 = masks[idx]
            else:
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
            ptp_vals = final_spectra.max(axis=0) - final_spectra.min(axis=0)  # per-pixel PTP
            means = []
            for i in range(args.n_clusters):
                idx = (final_labels == i)
                if not np.any(idx):
                    means.append(np.inf)  # ignore empty clusters
                else:
                    means.append(float(np.mean(ptp_vals[idx])))
            drop_idx = int(np.argmin(means))
            print(f"Dropping cluster {drop_idx} with lowest mean PTP = {means[drop_idx]:.6f}")

            drop_mask_global = (final_labels != drop_idx)
            kept_count = int(drop_mask_global.sum())
            if kept_count <= 0:
                raise ValueError("All pixels removed by PTP drop; cannot recluster.")

            final_spectra = final_spectra[:, drop_mask_global]
            final_n_clusters = max(1, args.n_clusters - 1)
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
    
    # Use spectra/labels after optional drop so lengths match
    plot_cluster_umap(final_embedding, final_labels, final_n_clusters, colors, out_dir)
    plot_cluster_spectra(final_spectra, final_labels, final_n_clusters, wavenumbers, colors, out_dir)

    pixel_offset = 0
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
        stats.to_csv(os.path.join(meta["dir"], "cluster_stats_combined.csv"), index=False)
        plot_cluster_composition(stats, final_n_clusters, meta["dir"], tag="combined")
        map_clusters(dir_labels, all_img_shapes[i], (r_kept, c_kept), final_n_clusters, colors, meta["dir"], tag="combined")
        print(f"Processed {meta['dir']}.")

    # Condition-level comparison removed for condition-agnostic combined analysis
    
    print("All tasks completed.")
