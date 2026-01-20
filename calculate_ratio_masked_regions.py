import argparse
import concurrent.futures
import itertools
import os

import cv2
import numpy as np
import tifffile as tiff
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu, ttest_ind

from utils import get_number
from visualizations import plot_bars_all, plot_region_scatter_3d, plot_violins_all, save_ratio_image

# Default parameters (kept lean for debugging)
REGION_NAMES = ("SEZ", "AL", "LP", "MB")
BLOCK_SIZE = 32
BLOCK_MIN_NONZERO = 0.25
GAMMA_DEFAULT = 1.0
CBAR_DEFAULT = "turbo"
LOW_PCT_DEFAULT = 1.0
HIGH_PCT_DEFAULT = 99.0


def load_region_masks(root_path):
    region_masks = {}
    for fname in os.listdir(root_path):
        if not fname.lower().endswith(".png"):
            continue
        roi_id = get_number(fname)
        region_name = os.path.splitext(fname)[0].split("_")[-1].upper()
        mask = cv2.imread(os.path.join(root_path, fname), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask file unreadable: {fname}")
        if region_name not in REGION_NAMES:
            continue
        region_masks.setdefault(roi_id, {})[region_name] = (mask > 0).astype(np.float32)

    for roi_id, regions in region_masks.items():
        missing = [r for r in REGION_NAMES if r not in regions]
        if missing:
            raise ValueError(f"Missing region masks {missing} for ROI {roi_id} in {root_path}")
    return region_masks


def _index_tiffs(file_names, labels):
    idx = {lab: {} for lab in labels}
    for fname in file_names:
        low = fname.lower()
        if not (low.endswith(".tif") or low.endswith(".tiff")):
            continue
        for lab in labels:
            if lab in low:
                idx[lab][get_number(fname)] = fname
    return idx


def calculate_ratio(root_path, fad_path, nadh_path, suffix, save,
                    region_mask=None, region_name=None,
                    save_dir=None, tiff_dir=None, **kwargs):
    gamma = kwargs.get("gamma", 1.0)
    cbar = kwargs.get("cbar", "turbo")

    fad = cv2.imread(os.path.join(root_path, fad_path), cv2.IMREAD_UNCHANGED)
    nadh = cv2.imread(os.path.join(root_path, nadh_path), cv2.IMREAD_UNCHANGED)

    sat_mask = (fad == 4095) | (nadh == 4095)
    fad[sat_mask] = 0
    nadh[sat_mask] = 0

    if region_mask is not None:
        fad = (fad.astype(np.float32) * region_mask).astype(fad.dtype)
        nadh = (nadh.astype(np.float32) * region_mask).astype(nadh.dtype)

    f = fad.astype(np.float32)
    n = nadh.astype(np.float32)

    if suffix in ["redox", "unsat"]:
        den = f + n
        den[den == 0] = np.finfo(np.float32).eps
        ratio = f / den
    else:
        n[n == 0] = np.finfo(np.float32).eps
        ratio = (f / n) / 2

    valid = ratio > 0
    if suffix in ["protein_turn", "lipid_turn"] and np.any(valid):
        vmin, vmax = ratio[valid].min(), ratio[valid].max()
        ratio[valid] = ((ratio[valid] - vmin) / (vmax - vmin + 1e-8)) * 0.25

    if region_mask is not None:
        ratio = ratio * region_mask

    if save:
        base = get_number(fad_path) + f"_{suffix}_ratio" + (f"_{region_name}" if region_name else "")

        # Save PNG into ROI-specific directory when provided; otherwise root
        png_dir = save_dir or root_path
        os.makedirs(png_dir, exist_ok=True)
        save_ratio_image(ratio, png_dir, base + ".png", LOW_PCT_DEFAULT, HIGH_PCT_DEFAULT, gamma, cbar)

        # Save TIFF to requested directory (default to root_path)
        tiff_out_dir = tiff_dir or root_path
        os.makedirs(tiff_out_dir, exist_ok=True)
        tiff.imwrite(os.path.join(tiff_out_dir, base + ".tiff"), ratio.astype(np.float32))
    return ratio


def _region_pixel_median(arr):
    if arr is None:
        return None
    vals = arr[arr > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def process_condition(root, enable_redox=True, enable_unsat=True, enable_turnover=False,
                     save=False, **kwargs):
    files = os.listdir(root)
    label_list = ["fad", "nadh", "787", "794", "841", "844", "791", "797"]
    idx = _index_tiffs(files, label_list)

    region_masks = load_region_masks(root)

    metrics = []
    if enable_redox:
        metrics.append("redox")
    if enable_unsat:
        metrics.append("unsat")
    if enable_turnover:
        metrics.extend(["protein_turn", "lipid_turn"])

    data_ratios = {(region, metric): [] for region in REGION_NAMES for metric in metrics}
    data_medians = {(region, metric): [] for region in REGION_NAMES for metric in metrics}
    region_sizes = {region: [] for region in REGION_NAMES}

    roi_ids = sorted(set().union(*(set(idx.get(lab, {})) for lab in label_list)))

    for roi_id in roi_ids:
        roi_dir = os.path.join(root, f"roi_{roi_id}")
        os.makedirs(roi_dir, exist_ok=True)

        roi_regions = region_masks.get(roi_id, {})
        if not roi_regions:
            continue

        # Precompute channel paths per ROI once
        fad_path = idx["fad"].get(roi_id)
        nadh_path = idx["nadh"].get(roi_id)
        unsat_path = idx["787"].get(roi_id)
        sat_path = idx["794"].get(roi_id)
        d_pro_path = idx["841"].get(roi_id)
        d_lip_path = idx["844"].get(roi_id)
        pro_path = idx["791"].get(roi_id)
        lip_path = idx["797"].get(roi_id)

        # Union mask across all regions in this ROI
        combined_mask = np.zeros_like(next(iter(roi_regions.values())))
        for m in roi_regions.values():
            combined_mask = np.maximum(combined_mask, m)

        for region_name, region_mask in roi_regions.items():
            region_sizes[region_name].append(float(np.sum(region_mask > 0)))

            if enable_redox and fad_path and nadh_path:
                ratio = calculate_ratio(root, fad_path, nadh_path, "redox", save,
                                        region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                data_ratios[(region_name, "redox")].append(ratio)
                data_medians[(region_name, "redox")].append(_region_pixel_median(ratio))

            if enable_unsat and unsat_path and sat_path:
                ratio = calculate_ratio(root, unsat_path, sat_path, "unsat", save,
                                        region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                data_ratios[(region_name, "unsat")].append(ratio)
                data_medians[(region_name, "unsat")].append(_region_pixel_median(ratio))

            if enable_turnover:
                if d_pro_path and pro_path:
                    ratio = calculate_ratio(root, d_pro_path, pro_path, "protein_turn", save,
                                            region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                    data_ratios[(region_name, "protein_turn")].append(ratio)
                    data_medians[(region_name, "protein_turn")].append(_region_pixel_median(ratio))
                if d_lip_path and lip_path:
                    ratio = calculate_ratio(root, d_lip_path, lip_path, "lipid_turn", save,
                                            region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                    data_ratios[(region_name, "lipid_turn")].append(ratio)
                    data_medians[(region_name, "lipid_turn")].append(_region_pixel_median(ratio))

        # Combined-region outputs saved in ROI folder when data channels exist
        if enable_redox and fad_path and nadh_path:
            calculate_ratio(root, fad_path, nadh_path, "redox", save,
                            region_mask=combined_mask, region_name="ALL", save_dir=roi_dir, tiff_dir=roi_dir, **kwargs)
        if enable_unsat and unsat_path and sat_path:
            calculate_ratio(root, unsat_path, sat_path, "unsat", save,
                            region_mask=combined_mask, region_name="ALL", save_dir=roi_dir, tiff_dir=roi_dir, **kwargs)
        if enable_turnover:
            if d_pro_path and pro_path:
                calculate_ratio(root, d_pro_path, pro_path, "protein_turn", save,
                                region_mask=combined_mask, region_name="ALL", save_dir=roi_dir, tiff_dir=roi_dir, **kwargs)
            if d_lip_path and lip_path:
                calculate_ratio(root, d_lip_path, lip_path, "lipid_turn", save,
                                region_mask=combined_mask, region_name="ALL", save_dir=roi_dir, tiff_dir=roi_dir, **kwargs)

    return data_ratios, data_medians, region_sizes


def _block_medians_from_image(arr: np.ndarray, block_size: int, min_nonzero_prop: float = 0.5) -> np.ndarray:
    """Compute per-block medians for non-overlapping blocks."""
    if arr is None:
        raise ValueError("Input array is None.")
    a = np.asarray(arr, dtype=float)
    h, w = a.shape[:2]
    if h < block_size or w < block_size:
        raise ValueError("Image is smaller than block size.")
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        raise ValueError("Image is smaller than block size.")
    a = a[: bh * block_size, : bw * block_size]
    a4 = a.reshape(bh, block_size, bw, block_size)
    block_area = float(block_size * block_size)
    nonzero_counts = (a4 != 0).sum(axis=(1, 3))
    mask = (nonzero_counts.astype(float) / block_area) >= float(min_nonzero_prop)
    if not np.any(mask):
        return np.array([], dtype=float)
    a4t = a4.transpose(0, 2, 1, 3)
    kept_blocks = a4t[mask]
    kept_blocks_nz = np.where(kept_blocks == 0, np.nan, kept_blocks)
    block_meds_kept = np.nanmedian(kept_blocks_nz, axis=(1, 2))
    return block_meds_kept


def collect_block_medians(cond_ratios, block_size: int = BLOCK_SIZE, min_nonzero_prop: float = 0.5) -> np.ndarray:
    vals = []
    for arr in cond_ratios:
        if arr is None:
            continue
        vals.append(_block_medians_from_image(arr, int(block_size), float(min_nonzero_prop)))
    if len(vals) == 0:
        return np.array([], dtype=float)
    return np.concatenate(vals) if len(vals) else np.array([], dtype=float)


def collect_pixel_medians(cond_ratios) -> np.ndarray:
    vals = []
    for m in cond_ratios:
        if m is None:
            continue
        vals.append(m)
    return np.asarray(vals, dtype=float) if vals else np.array([], dtype=float)


def pairwise_tests(metric_values):
    results = {}
    conds = list(metric_values.keys())
    for c1, c2 in itertools.combinations(conds, 2):
        v1 = np.asarray(metric_values[c1])
        v2 = np.asarray(metric_values[c2])
        n1, n2 = v1.size, v2.size
        if n1 == 0 or n2 == 0:
            p_t, p_u = np.nan, np.nan
        else:
            _, p_t = ttest_ind(v1, v2, equal_var=False)
            _, p_u = mannwhitneyu(v1, v2)
        results[(c1, c2)] = (p_t, p_u, n1, n2)
    return results


def print_pairwise(results, label, sample_unit="samples"):
    print(f"\nPairwise stats for {label}:")
    for (c1, c2), (p_t, p_u, n1, n2) in results.items():
        t_txt = f"{p_t:.4e}" if p_t is not None and not np.isnan(p_t) else "nan"
        u_txt = f"{p_u:.4e}" if p_u is not None and not np.isnan(p_u) else "nan"
        print(f"{c1} ({sample_unit}={n1}) vs {c2} ({sample_unit}={n2}) -> t p={t_txt}, U p={u_txt}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate ratios across multiple conditions and plot group comparisons.")
    parser.add_argument("dirs", nargs="+", type=str, help="Input directories for each condition")
    parser.add_argument("--conds", "-c", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--out", "-o", type=str, default=".", help="Output directory for plots")
    parser.add_argument("-d", "--deuterated", action="store_true", help="Include deuterated turnover channels")
    parser.add_argument("--save", "-s", action="store_true", help="Save per-image ratio maps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--workers", "-w", type=int, default=os.cpu_count(), help="Number of parallel workers (default: os.cpu_count())")
    parser.add_argument("--pdf-out", "-p", action="store_true", help="Save all plots into a single multi-page PDF in the output directory")
    parser.add_argument("--hide-ns", action="store_true", help="Hide non-significant comparisons in plots")
    args = parser.parse_args()

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    save = args.save
    hide_ns = args.hide_ns
    os.makedirs(args.out, exist_ok=True)
    pdf_pages = PdfPages(os.path.join(args.out, "plots.pdf")) if args.pdf_out else None
    violin_pdf_pages = None  # Exclude violin plots from PDF output
    save_png = not args.pdf_out
    if not args.verbose:
        print("Input conditions and directories:")
        for cond, dir_path in zip(args.conds, args.dirs):
            print(f"  {cond}: {dir_path}")
    kwargs = {
        "gamma": GAMMA_DEFAULT,
        "cbar": CBAR_DEFAULT,
    }

    metrics = ["redox", "unsat"]
    if args.deuterated:
        metrics.extend(["protein_turn", "lipid_turn"])

    cond_region_ratios = {(region, metric): {cond: [] for cond in args.conds} for region in REGION_NAMES for metric in metrics}
    cond_region_medians = {(region, metric): {cond: [] for cond in args.conds} for region in REGION_NAMES for metric in metrics}
    cond_region_sizes = {region: {cond: [] for cond in args.conds} for region in REGION_NAMES}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_map = {}
        for cond, dir_path in zip(args.conds, args.dirs):
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            fut = ex.submit(process_condition, dir_path, True, True, args.deuterated, save, **kwargs)
            future_map[fut] = cond

        for fut in concurrent.futures.as_completed(future_map):
            cond = future_map[fut]
            region_ratios, region_medians, region_sizes = fut.result()

            for key in cond_region_ratios:
                cond_region_ratios[key][cond] = region_ratios.get(key, [])
                med_list = [m for m in region_medians.get(key, []) if m is not None]
                cond_region_medians[key][cond] = med_list
            for region in REGION_NAMES:
                cond_region_sizes[region][cond] = region_sizes.get(region, [])

            if args.verbose:
                summary_bits = []
                for metric in metrics:
                    count = sum(len(region_ratios.get((r, metric), [])) for r in REGION_NAMES)
                    summary_bits.append(f"{metric}:{count}")
                print(f"Processed {cond} -> " + ", ".join(summary_bits))

    if args.verbose:
        print("Processing complete.")
        print("Running block-median tests and plotting group comparisons...")

    def block_map(metric_map):
        return {cond: collect_block_medians(ratios, block_size=BLOCK_SIZE, min_nonzero_prop=BLOCK_MIN_NONZERO)
                for cond, ratios in metric_map.items()}

    ylabels = {
        "redox": "Redox ratio",
        "unsat": "Unsaturation ratio",
        "protein_turn": "Protein turnover ratio",
        "lipid_turn": "Lipid turnover ratio",
    }

    for region in REGION_NAMES:
        for metric in metrics:
            key = (region, metric)
            blocks = block_map(cond_region_ratios[key])
            if all(len(v) == 0 for v in blocks.values()):
                continue
            pairwise = pairwise_tests(blocks)
            if args.verbose:
                print_pairwise(pairwise, f"{metric} (block medians, {region})", sample_unit="blocks")
            p_map = {k: v[0] for k, v in pairwise.items()}
            label_prefix = f"{metric}_{region}"
            plot_violins_all(blocks, args.conds, p_map, args.out, "Block-median t-test", ylabels.get(metric, metric), label_prefix, hide_ns, violin_pdf_pages, save_png)
            plot_bars_all(blocks, args.conds, p_map, args.out, "Block-median t-test", ylabels.get(metric, metric), label_prefix, hide_ns, pdf_pages, save_png)

    if args.verbose:
        print("\nRunning region size comparisons...")
    size_ylabel = "Region size (pixels)"
    for region in REGION_NAMES:
        size_map = cond_region_sizes[region]
        if all(len(v) == 0 for v in size_map.values()):
            continue
        size_pairwise = pairwise_tests(size_map)
        if args.verbose:
            print_pairwise(size_pairwise, f"Region size ({region})", sample_unit="samples")
        size_p_map = {k: v[0] for k, v in size_pairwise.items()}
        label_prefix = f"region_size_{region}"
        plot_violins_all(size_map, args.conds, size_p_map, args.out, "Region size t-test", size_ylabel, label_prefix, hide_ns, violin_pdf_pages, save_png)
        plot_bars_all(size_map, args.conds, size_p_map, args.out, "Region size t-test", size_ylabel, label_prefix, hide_ns, pdf_pages, save_png)
        if args.verbose:
            mean_sizes = {cond: (float(np.mean(vals)) if len(vals) else float('nan')) for cond, vals in size_map.items()}
            mean_txt = ", ".join(
                f"{cond}={mean_sizes[cond]:.1f}" if not np.isnan(mean_sizes[cond]) else f"{cond}=nan"
                for cond in args.conds
            )
            print(f"Mean sizes for {region}: {mean_txt}")

    if args.deuterated:
        for region in REGION_NAMES:
            scatter_vals = {}
            for cond in args.conds:
                redox_vals = cond_region_medians.get((region, "redox"), {}).get(cond, [])
                protein_vals = cond_region_medians.get((region, "protein_turn"), {}).get(cond, [])
                lipid_vals = cond_region_medians.get((region, "lipid_turn"), {}).get(cond, [])
                if not redox_vals or not protein_vals or not lipid_vals:
                    continue
                scatter_vals[cond] = (
                    float(np.median(redox_vals)),
                    float(np.median(protein_vals)),
                    float(np.median(lipid_vals)),
                )
            if scatter_vals:
                plot_region_scatter_3d(scatter_vals, args.out, region)

    if pdf_pages is not None:
        pdf_pages.close()

