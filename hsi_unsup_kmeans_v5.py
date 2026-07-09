import argparse, os, numpy as np, pandas as pd, tifffile
import re
from matplotlib.backends.backend_pdf import PdfPages
try:
    from cuml import UMAP as GPU_UMAP
except Exception:
    GPU_UMAP = None
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

STACK_FILENAMES = ("masked_stack_sub_water.tif", "masked_stack_sub_water.tiff")
GPU_UMAP_ENABLED = GPU_UMAP is not None
SPECTRA_START = 2700.0
SPECTRA_END = 3100.0
VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def find_stack_path(base_dir: str):
    for name in STACK_FILENAMES:
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return name.strip("._") or "unnamed"

def analyze_cluster_composition(cluster_labels, n_clusters):
    counts = np.bincount(cluster_labels, minlength=n_clusters) if cluster_labels.size else np.zeros(n_clusters, dtype=int)
    ratios = counts / cluster_labels.size if cluster_labels.size > 0 else np.zeros(n_clusters)
    return pd.DataFrame({"cluster_id": np.arange(n_clusters), "count": counts, "ratio": ratios})


def select_low_auc_clusters(
    spectra,
    labels,
    n_clusters,
    wavenumbers,
    drop_n,
    auc_start=2830.0,
    auc_end=3050.0,
):
    if drop_n <= 0:
        return np.array([], dtype=int), np.zeros(n_clusters, dtype=float)

    if auc_start >= auc_end:
        raise ValueError("auc_start must be < auc_end")

    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if wavenumbers.min() > auc_start or wavenumbers.max() < auc_end:
        raise ValueError("Wavenumber range does not cover requested AUC window.")

    range_mask = (wavenumbers >= auc_start) & (wavenumbers <= auc_end)
    if not np.any(range_mask):
        raise ValueError("No wavenumbers found in the requested AUC window.")
    auc_wavenumbers = wavenumbers[range_mask]

    aucs = np.full(n_clusters, -np.inf, dtype=float)
    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue
        mean_spectrum = spectra[:, mask].mean(axis=1)

        start_int = np.interp(auc_start, wavenumbers, mean_spectrum)
        end_int = np.interp(auc_end, wavenumbers, mean_spectrum)
        baseline = np.interp(auc_wavenumbers, [auc_start, auc_end], [start_int, end_int])
        aucs[cid] = np.trapezoid(mean_spectrum[range_mask] - baseline, x=auc_wavenumbers)

    drop_ids = np.argsort(aucs)[:drop_n]
    return drop_ids.astype(int), aucs


def compute_umap_embedding(features, random_state=42):
    global GPU_UMAP_ENABLED

    if GPU_UMAP_ENABLED:
        try:
            return GPU_UMAP(n_components=2, random_state=random_state).fit_transform(features)
        except Exception as exc:
            GPU_UMAP_ENABLED = False
            vprint(f"GPU UMAP failed ({exc}); falling back to CPU UMAP.")
    else:
        vprint("GPU UMAP unavailable; using CPU UMAP.")

    try:
        from umap import UMAP as CPU_UMAP
    except Exception as exc:
        raise RuntimeError(
            "CPU fallback requires umap-learn. Install it with: pip install umap-learn"
        ) from exc

    return CPU_UMAP(n_components=2).fit_transform(np.asarray(features))

def preprocess_spectra(spectra, eps=1e-12):
    head = min(10, spectra.shape[0])
    baseline = np.max(spectra[:head, :], axis=0)
    spectra = spectra - baseline
    spectra = np.maximum(spectra, 0)

    denom = np.max(spectra, axis=0)
    denom = np.where(denom > eps, denom, np.nan)
    spectra = spectra / denom

    return np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)

def resample_stack_to_expected_channels(stack, expected_channels, spectra_start, spectra_end):
    if stack.shape[0] == expected_channels:
        return stack.astype(np.float32)

    orig_channels = stack.shape[0]
    old_axis = np.linspace(spectra_start, spectra_end, orig_channels)
    new_axis = np.linspace(spectra_start, spectra_end, expected_channels)
    flat = stack.reshape(orig_channels, -1).astype(np.float32)
    f = interp1d(old_axis, flat, axis=0, kind="linear", fill_value="extrapolate", bounds_error=False)
    resampled = f(new_axis).reshape((expected_channels,) + stack.shape[1:])
    vprint(f"Interpolated stack from {orig_channels} to {expected_channels} channels.")
    return resampled.astype(np.float32)


def load_hyperstacks_from_dirs(input_dirs, expected_channels, spectra_start, spectra_end):
    stacks, valid_dirs = [], []
    for d in input_dirs:
        stack_path = find_stack_path(d)
        if stack_path is None:
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

def preprocess_all_stacks(stacks, wavenumbers, drop_invalid=True, deduplicate=True):
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
    print(f"Combined spectra shape (filtered): {combined_f.shape} | removed {P_total - kept_after} pixels (invalid/duplicates)")

    return combined_f, all_img_shapes, all_indices_f


def filter_indices_by_mask(all_indices, keep_mask):
    sizes = [r.size for r, _ in all_indices]
    total = int(np.sum(sizes))
    if keep_mask.size != total:
        raise ValueError("keep_mask size does not match total pixel count.")

    starts = np.cumsum([0] + sizes[:-1])
    filtered = []
    for i, (r, c) in enumerate(all_indices):
        start, end = starts[i], starts[i] + sizes[i]
        dir_keep = keep_mask[start:end]
        filtered.append((r[dir_keep], c[dir_keep]))

    return filtered


def print_silhouette_score(features, labels):
    def report_unavailable(message):
        if VERBOSE:
            print(message)
        else:
            print("Silhouette score (final clustering): unavailable")

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        report_unavailable("Silhouette score skipped: need at least 2 clusters.")
        return
    if features.shape[0] <= unique_labels.size:
        report_unavailable("Silhouette score skipped: not enough samples for silhouette computation.")
        return

    try:
        from sklearn.metrics import silhouette_score
    except Exception as exc:
        report_unavailable(f"Silhouette score skipped (scikit-learn unavailable: {exc}).")
        return

    try:
        score = silhouette_score(features, labels, sample_size=5000, random_state=42)
    except Exception as exc:
        report_unavailable(f"Silhouette score failed: {exc}")
        return

    print(f"Silhouette score (final clustering): {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperspectral SRS Image Analysis for multiple directories (combined analysis).")
    parser.add_argument("dirs", nargs="+", help="One or more directories containing stacks")
    parser.add_argument("--conds", "-C", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--n_clusters", "-n", type=int, default=6)
    parser.add_argument("--expected_channels", "-c", type=int, default=62)
    parser.add_argument("--split", "-s", type=int, default=0, help="Further split the largest cluster into this many subclusters (replace it). Use 0 to disable.")
    parser.add_argument("--drop", "-d", type=int, default=0, help="Drop N clusters with lowest spectral AUC and recluster remaining pixels into n_clusters - N.")
    parser.add_argument("--output", "-o", type=str, required = True, help="Output directory for plots")
    parser.add_argument("--pdf", "-p", action="store_true", help="Also save all generated plots into a single PDF in the output directory")
    parser.add_argument("--umap", "-u", action="store_true", help="Run UMAP and plot the embedding")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress output")
    args = parser.parse_args()

    VERBOSE = args.verbose

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    input_dirs, dir_metadata = [], []

    def find_masked_stack_dirs(base: str):
        """Recursively search for all directories containing a masked_stack.tif(f) under base."""
        found = []
        for dir_path, _, files in os.walk(base):
            if any(name in files for name in STACK_FILENAMES):
                found.append(dir_path)
        return sorted(found)

    for base, cond in zip(args.dirs, args.conds):
        if not os.path.isdir(base):
            raise ValueError(f"Directory not found: {base}")

        stack_dirs = find_masked_stack_dirs(base)
        if not stack_dirs:
            raise FileNotFoundError(f"masked_stack.tif or masked_stack.tiff not found under {base}")

        for stack_dir in stack_dirs:
            stack_path = find_stack_path(stack_dir)
            if stack_path is None:
                raise FileNotFoundError(f"masked_stack.tif(f) not found in {stack_dir}")
            try:
                with tifffile.TiffFile(stack_path) as tf:
                    if len(tf.pages) == 0:
                        raise FileNotFoundError(f"masked_stack unreadable in {stack_dir}")
            except Exception as exc:
                raise FileNotFoundError(f"masked_stack unreadable in {stack_dir}") from exc

            vprint(f"Found stack: {stack_path} (condition: {cond})")
            input_dirs.append(stack_dir)
            dir_metadata.append({"dir": stack_dir, "condition": cond})

    print(f"Found {len(input_dirs)} valid stack directories across {len(args.conds)} conditions.")
    spectra_start = SPECTRA_START
    spectra_end = SPECTRA_END

    stacks, _ = load_hyperstacks_from_dirs(
        input_dirs,
        args.expected_channels,
        spectra_start,
        spectra_end,
    )

    if not stacks:
        vprint("No valid directories found. Exiting.")
        raise SystemExit(1)

    wavenumbers = np.linspace(spectra_start, spectra_end, args.expected_channels)
    combined_spectra, all_img_shapes, all_indices = preprocess_all_stacks(stacks, wavenumbers)
    vprint(f"Combined spectra shape: {combined_spectra.shape}")

    # Feature preparation (no PCA)
    raw_spectra = combined_spectra
    clustering_features = raw_spectra.T.astype(np.float32)

    # Initial DR + clustering
    embedding = None
    if args.umap:
        embedding = compute_umap_embedding(clustering_features, random_state=42)
    cluster_labels = GPU_KMeans(n_clusters=args.n_clusters, random_state=42).fit_predict(clustering_features)

    final_spectra = raw_spectra
    final_features = clustering_features
    final_embedding = embedding
    final_labels = cluster_labels
    final_n_clusters = args.n_clusters

    if args.drop:
        if args.drop < 0:
            raise ValueError("--drop must be >= 0")
        if args.drop >= final_n_clusters:
            raise ValueError("--drop must be less than n_clusters")

        drop_ids, _ = select_low_auc_clusters(
            final_spectra,
            final_labels,
            final_n_clusters,
            wavenumbers,
            args.drop,
        )
        if drop_ids.size:
            keep_mask = ~np.isin(final_labels, drop_ids)
            if not keep_mask.any():
                raise ValueError("No pixels left after dropping clusters.")

            vprint(f"Dropping clusters with lowest AUC: {drop_ids.tolist()}")
            final_spectra = final_spectra[:, keep_mask]
            final_features = final_features[keep_mask]
            if final_embedding is not None:
                final_embedding = final_embedding[keep_mask]
            all_indices = filter_indices_by_mask(all_indices, keep_mask)

            final_n_clusters = final_n_clusters - args.drop
            final_labels = GPU_KMeans(n_clusters=final_n_clusters, random_state=42).fit_predict(final_features)

    # Optional split of largest cluster into additional subclusters
    if args.split:
        if args.split < 2:
            vprint("--split provided but <2; skipping split.")
        else:
            counts = np.bincount(final_labels, minlength=final_n_clusters)
            largest_id = int(np.argmax(counts))
            largest_mask = final_labels == largest_id
            largest_count = int(largest_mask.sum())

            if largest_count < args.split:
                vprint(f"--split={args.split} but largest cluster size is {largest_count}; skipping split.")
            else:
                sub_features = final_features[largest_mask]
                sub_labels = GPU_KMeans(n_clusters=args.split, random_state=42).fit_predict(sub_features)

                new_total = final_n_clusters - 1 + args.split

                # Remap old clusters (except the largest) to contiguous ids, then append new subcluster ids
                map_arr = np.full(final_n_clusters, -1, dtype=int)
                next_id = 0
                for old in range(final_n_clusters):
                    if old == largest_id:
                        continue
                    map_arr[old] = next_id
                    next_id += 1

                sub_ids = np.arange(next_id, next_id + args.split, dtype=int)

                new_labels = np.empty_like(final_labels)
                new_labels[~largest_mask] = map_arr[final_labels[~largest_mask]]
                new_labels[largest_mask] = sub_ids[sub_labels]

                final_labels = new_labels
                final_n_clusters = new_total
                vprint(f"Split largest cluster {largest_id} (size={largest_count}) into {args.split} subclusters; total clusters now {final_n_clusters}.")

    print_silhouette_score(final_features, final_labels)

    # Alternative color palette
    colors = np.array([
        [0.121, 0.466, 0.705],  # blue
        [1.000, 0.498, 0.054],  # orange
        [0.172, 0.627, 0.172],  # green
        [0.839, 0.153, 0.157],  # red
        [0.580, 0.404, 0.741],  # purple
        [0.549, 0.337, 0.294],  # brown
        [0.890, 0.466, 0.760],  # pink
        [0.498, 0.498, 0.498],  # gray
        [0.737, 0.741, 0.133],  # olive
        [0.090, 0.745, 0.811],  # cyan
        [0.737, 0.502, 0.741],  # lilac
        [0.996, 0.894, 0.000],  # yellow
        [0.996, 0.643, 0.376],  # apricot
        [0.000, 0.000, 0.000],  # black
        [0.254, 0.713, 0.768],  # teal
    ])
    
    if final_n_clusters > colors.shape[0]:
        raise ValueError(f"Requested total clusters ({final_n_clusters}) exceeds available colors ({colors.shape[0]}).")


    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    pdf_pages = None
    pdf_path = None
    if args.pdf:
        pdf_path = os.path.join(out_dir, "plots.pdf")
        pdf_pages = PdfPages(pdf_path)

    save_combined_images = not args.pdf

    mean_spectra = np.empty((wavenumbers.size, final_n_clusters), dtype=np.float32)
    for cid in range(final_n_clusters):
        mask = final_labels == cid
        if np.any(mask):
            mean_spectra[:, cid] = final_spectra[:, mask].mean(axis=1)
        else:
            mean_spectra[:, cid] = np.nan

    mean_df = pd.DataFrame(mean_spectra, columns=[f"cluster_{i}" for i in range(final_n_clusters)])
    mean_df.insert(0, "raman_shift", wavenumbers)
    mean_df.to_csv(os.path.join(out_dir, "cluster_mean_spectra.csv"), index=False)

    try:
        if args.umap:
            plot_cluster_umap(final_embedding, final_labels, final_n_clusters, colors, out_dir, pdf_pages=None)
        plot_mask = (wavenumbers >= 2800.0) & (wavenumbers <= 3100.0)
        plot_wavenumbers = wavenumbers[plot_mask]
        plot_spectra = final_spectra[plot_mask, :]
        plot_cluster_spectra(
            plot_spectra,
            final_labels,
            final_n_clusters,
            plot_wavenumbers,
            colors,
            out_dir,
            pdf_pages=pdf_pages,
            save_image=save_combined_images,
        )

        pixel_offset = 0
        cond_order = list(dict.fromkeys(args.conds))
        condition_counts = {cond: np.zeros(final_n_clusters, dtype=int) for cond in cond_order}
        condition_cluster_ratios = {cond: [[] for _ in range(final_n_clusters)] for cond in cond_order}
        per_stack_rows = []
        cluster_map_tag_counts = {}
        for i, meta in enumerate(dir_metadata):
            # filtered pixel indices for this directory
            r, c = all_indices[i]
            n_pixels = r.size
            dir_labels = final_labels[pixel_offset : pixel_offset + n_pixels]
            pixel_offset += n_pixels
            r_kept, c_kept = r, c

            stats = analyze_cluster_composition(dir_labels, final_n_clusters)
            condition_counts[meta["condition"]] += stats["count"].to_numpy()
            ratio_vals = stats["ratio"].to_numpy()
            ratio_sum = float(ratio_vals.sum())
            if ratio_sum > 0:
                ratio_vals = ratio_vals / ratio_sum
                row = {"condition": meta["condition"]}
                row.update({f"cluster_{k}": float(ratio_vals[k]) for k in range(final_n_clusters)})
                per_stack_rows.append(row)
            ratio_list = condition_cluster_ratios.get(meta["condition"])
            if ratio_list is not None:
                for k in range(final_n_clusters):
                    ratio_list[k].append(float(ratio_vals[k]))
            stats.to_csv(os.path.join(meta["dir"], "cluster_stats_combined.csv"), index=False)
            plot_cluster_composition(stats, final_n_clusters, meta["dir"], tag="combined")
            map_output_dir = os.path.join(out_dir, safe_name(meta["condition"]))
            os.makedirs(map_output_dir, exist_ok=True)
            map_tag = safe_name(os.path.basename(meta["dir"]))
            tag_key = (map_output_dir, map_tag)
            cluster_map_tag_counts[tag_key] = cluster_map_tag_counts.get(tag_key, 0) + 1
            if cluster_map_tag_counts[tag_key] > 1:
                map_tag = f"{map_tag}_{cluster_map_tag_counts[tag_key]}"
            map_clusters(
                dir_labels,
                all_img_shapes[i],
                (r_kept, c_kept),
                final_n_clusters,
                colors,
                map_output_dir,
                tag=map_tag,
            )
            vprint(f"Processed {meta['dir']}.")

        if per_stack_rows:
            per_stack_df = pd.DataFrame(per_stack_rows)
            per_stack_df.to_csv(os.path.join(out_dir, "cluster_proportions_per_image.csv"), index=False)

        condition_ratios = {}
        for cond in cond_order:
            counts = condition_counts.get(cond)
            if counts is None:
                continue
            total = float(counts.sum())
            condition_ratios[cond] = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

        plot_cluster_composition_by_condition(
            condition_ratios,
            out_dir,
            tag="by_condition",
            colors=colors,
            pdf_pages=pdf_pages,
            save_image=save_combined_images,
            condition_cluster_ratios=condition_cluster_ratios,
        )

    finally:
        if pdf_pages is not None:
            pdf_pages.close()
            vprint(f"Saved combined PDF: {pdf_path}")

    # Condition-level comparison removed for condition-agnostic combined analysis
    
    vprint("All tasks completed.")
