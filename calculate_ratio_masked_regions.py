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
# REGION_NAMES = ("SEZ", "AL", "LP", "MB")
REGION_NAMES = ("SEZ","AL")
GAMMA_DEFAULT = 1.0
CBAR_DEFAULT = "turbo"
LOW_PCT_DEFAULT = 1.0
HIGH_PCT_DEFAULT = 99.0
D2O_PCT = 0.2


def _canonical_roi_id(raw: str) -> str:
    try:
        return str(int(raw))
    except Exception:
        return str(raw)


def _trim_extremes(arr: np.ndarray, drop_n: int = 1) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size <= 2 * drop_n:
        return a
    order = np.argsort(a)
    keep = order[drop_n:-drop_n]
    return a[keep]


def load_region_masks(root_path):
    region_masks = {}
    for fname in os.listdir(root_path):
        if not fname.lower().endswith(".png"):
            continue
        roi_id = _canonical_roi_id(get_number(fname))
        region_name = os.path.splitext(fname)[0].split("_")[-1].upper()
        mask = cv2.imread(os.path.join(root_path, fname), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask file unreadable: {fname}")
        if region_name not in REGION_NAMES:
            continue
        region_masks.setdefault(roi_id, {})[region_name] = (mask > 0).astype(np.float32)
    return region_masks


def _index_tiffs(file_names, labels):
    idx = {lab: {} for lab in labels}
    for fname in file_names:
        low = fname.lower()
        if not (low.endswith(".tif") or low.endswith(".tiff")):
            continue
        for lab in labels:
            if lab in low:
                roi_id = _canonical_roi_id(get_number(fname))
                idx[lab][roi_id] = fname
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
        vals = ratio[valid]
        p_low, p_high = np.percentile(vals, [0, 99])
        if p_high > p_low:
            ratio_clipped = np.clip(ratio, p_low, p_high)
            scaled = (ratio_clipped - p_low) / (p_high - p_low)
            ratio[valid] = scaled[valid] * D2O_PCT
        else:
            ratio[valid] = 0.0

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


def _region_pixel_mean(arr):
    if arr is None:
        return None
    vals = arr[arr > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def process_condition(root, enable_redox=True, enable_unsat=True, enable_turnover=False,
                     save=False, verbose=False, **kwargs):
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

    data_means = {(region, metric): [] for region in REGION_NAMES for metric in metrics}
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
                data_means[(region_name, "redox")].append(_region_pixel_mean(ratio))

            if enable_unsat and unsat_path and sat_path:
                ratio = calculate_ratio(root, unsat_path, sat_path, "unsat", save,
                                        region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                data_means[(region_name, "unsat")].append(_region_pixel_mean(ratio))

            if enable_turnover:
                if d_pro_path and pro_path:
                    ratio = calculate_ratio(root, d_pro_path, pro_path, "protein_turn", save,
                                            region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                    data_means[(region_name, "protein_turn")].append(_region_pixel_mean(ratio))
                if d_lip_path and lip_path:
                    ratio = calculate_ratio(root, d_lip_path, lip_path, "lipid_turn", save,
                                            region_mask=region_mask, region_name=region_name, save_dir=roi_dir, **kwargs)
                    data_means[(region_name, "lipid_turn")].append(_region_pixel_mean(ratio))

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

    return data_means, region_sizes


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


def _trim_metric_values(metric_values, drop_n: int = 1, enabled: bool = False):
    drop_n = max(0, int(drop_n))
    if not enabled or drop_n == 0:
        return metric_values
    trimmed = {}
    for cond, vals in metric_values.items():
        original = list(vals)
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        if n < 3:
            trimmed_vals = arr
        else:
            # Ensure at least 3 points remain after trimming both ends
            max_drop_each_side = max(0, (n - 3) // 2)
            eff_drop = min(drop_n, max_drop_each_side)
            trimmed_vals = _trim_extremes(arr, eff_drop)
        # If trimming (or nan filtering) dropped everything but we had data, fall back to original
        if trimmed_vals.size == 0 and len(original) > 0:
            trimmed_vals = np.asarray(original, dtype=float)
        trimmed[cond] = trimmed_vals.tolist()
    return trimmed


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
    parser.add_argument("--trim-extremes", "-t", type=int, default=0, help="Drop N lowest and N highest values per condition before stats/plots (0=disabled)")
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

    cond_region_means = {(region, metric): {cond: [] for cond in args.conds} for region in REGION_NAMES for metric in metrics}
    cond_region_sizes = {region: {cond: [] for cond in args.conds} for region in REGION_NAMES}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_map = {}
        for cond, dir_path in zip(args.conds, args.dirs):
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            fut = ex.submit(process_condition, dir_path, True, True, args.deuterated, save, args.verbose, **kwargs)
            future_map[fut] = cond

        for fut in concurrent.futures.as_completed(future_map):
            cond = future_map[fut]
            region_means, region_sizes = fut.result()

            for key in cond_region_means:
                cond_region_means[key][cond] = [m for m in region_means.get(key, []) if m is not None]
            for region in REGION_NAMES:
                cond_region_sizes[region][cond] = region_sizes.get(region, [])

            if args.verbose:
                summary_bits = []
                for metric in metrics:
                    count = sum(len(region_means.get((r, metric), [])) for r in REGION_NAMES)
                    summary_bits.append(f"{metric}:{count}")
                print(f"Processed {cond} -> " + ", ".join(summary_bits))

    if args.verbose:
        print("Processing complete.")
        print("Running region-level mean tests and plotting group comparisons...")

    ylabels = {
        "redox": "Redox ratio",
        "unsat": "Unsaturation ratio",
        "protein_turn": "Protein turnover ratio",
        "lipid_turn": "Lipid turnover ratio",
    }

    for region in REGION_NAMES:
        for metric in metrics:
            key = (region, metric)
            mean_map = _trim_metric_values(cond_region_means[key], drop_n=args.trim_extremes, enabled=bool(args.trim_extremes))
            if all(len(v) == 0 for v in mean_map.values()):
                continue
            pairwise = pairwise_tests(mean_map)
            if args.verbose:
                print_pairwise(pairwise, f"{metric} (region means, {region})", sample_unit="regions")
            p_map = {k: v[0] for k, v in pairwise.items()}
            label_prefix = f"{metric}_{region}"
            plot_violins_all(mean_map, args.conds, p_map, args.out, "Region-mean t-test", ylabels.get(metric, metric), label_prefix, hide_ns, violin_pdf_pages, save_png)
            plot_bars_all(mean_map, args.conds, p_map, args.out, "Region-mean t-test", ylabels.get(metric, metric), label_prefix, hide_ns, pdf_pages, save_png)

    if args.verbose:
        print("\nRunning region size comparisons...")
    size_ylabel = "Region size (pixels)"
    for region in REGION_NAMES:
        size_map = cond_region_sizes[region]
        size_map = _trim_metric_values(size_map, drop_n=args.trim_extremes, enabled=bool(args.trim_extremes))
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
                redox_vals = cond_region_means.get((region, "redox"), {}).get(cond, [])
                protein_vals = cond_region_means.get((region, "protein_turn"), {}).get(cond, [])
                lipid_vals = cond_region_means.get((region, "lipid_turn"), {}).get(cond, [])
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

