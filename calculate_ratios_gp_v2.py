import os, numpy as np, cv2, argparse, tifffile as tiff
import itertools, re
from scipy.stats import ttest_ind, mannwhitneyu
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from matplotlib.backends.backend_pdf import PdfPages
from utils import get_number, match_files, generate_model_masks
from visualizations import save_ratio_image, plot_bars_all, plot_violins_all

# Default parameters (no external config)
HIDE_NON_SIGNIFICANT = False
GAMMA_DEFAULT = 1.0
CBAR_DEFAULT = "turbo"
LOW_PCT_DEFAULT = 1.0
HIGH_PCT_DEFAULT = 99.0
MASK_RX = re.compile(r"^(\d+)_mask$", re.IGNORECASE)
D2O_PCT = 0.2


def _trim_extremes(arr: np.ndarray, drop_n: int = 1) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    drop_n = max(0, int(drop_n))
    if drop_n == 0 or a.size <= 2 * drop_n:
        return a
    order = np.argsort(a)
    keep = order[drop_n:-drop_n]
    return a[keep]


def _channel_stats(arr: np.ndarray):
    a = np.asarray(arr, dtype=np.float32)
    a = np.where(a == 0, np.nan, a)
    mean = np.nanmean(a)
    med = np.nanmedian(a)
    mean = mean if np.isfinite(mean) else np.nan
    med = med if np.isfinite(med) else np.nan
    return mean, med

def calculate_ratio(root_path, fad_path, nadh_path, suffix, save,
                    mask_map=None, **kwargs):
    gamma = kwargs.get("gamma", 1.0)
    cbar = kwargs.get("cbar", 'turbo')
    
    fad = cv2.imread(os.path.join(root_path, fad_path), cv2.IMREAD_UNCHANGED)
    nadh = cv2.imread(os.path.join(root_path, nadh_path), cv2.IMREAD_UNCHANGED)

    sat_mask = (fad == 4095) | (nadh == 4095)  # union of saturated pixels
    fad[sat_mask] = 0
    nadh[sat_mask] = 0

    roi_id = get_number(fad_path)
    if mask_map is not None and roi_id not in mask_map:
        print("WARNING: no model mask for roi_id=", roi_id, "file=", fad_path)

    if mask_map is not None and roi_id in mask_map:
        roi_mask = mask_map[roi_id]
        if roi_mask.shape != fad.shape:
            roi_mask = cv2.resize(roi_mask, (fad.shape[1], fad.shape[0]), interpolation=cv2.INTER_NEAREST)
        fad = (fad.astype(np.float32) * roi_mask).astype(fad.dtype)
        nadh = (nadh.astype(np.float32) * roi_mask).astype(nadh.dtype)

    f = fad.astype(np.float32)
    n = nadh.astype(np.float32)
    
    if suffix in ["redox", "unsat"]:
        den = f + n
        den[den == 0] = np.finfo(np.float32).eps
        ratio = f / den
    elif suffix in ["protein_turn", "lipid_turn"]:
        n[n == 0] = np.finfo(np.float32).eps # Avoid division by zero
        ratio = (f / n) / 2 # CD channel use twice the power compared to CH channel
    else:
        raise ValueError("Suffix must be one of 'redox', 'unsat', 'protein_turn', or 'lipid_turn'.")
    
    valid = ratio > 0
    if suffix in ["protein_turn", "lipid_turn"] and np.any(valid):
        vals = ratio[valid]
        p_low, p_high = np.percentile(vals, [0, 100])
        if p_high > p_low:
            ratio_clipped = np.clip(ratio, p_low, p_high)
            scaled = (ratio_clipped - p_low) / (p_high - p_low)
            ratio[valid] = scaled[valid] * D2O_PCT
        else:
            ratio[valid] = 0.0


    if save:
        roi_dir = os.path.join(root_path, f"roi_{roi_id}") if suffix in ["protein_turn", "lipid_turn"] else root_path
        os.makedirs(roi_dir, exist_ok=True)
        save_name = get_number(fad_path) + f'_{suffix}_ratio.png'
        save_ratio_image(ratio, roi_dir, save_name, LOW_PCT_DEFAULT, HIGH_PCT_DEFAULT, gamma, cbar)
        tiff.imwrite(os.path.join(root_path, get_number(fad_path) + f'_{suffix}_ratio.tiff'), ratio.astype(np.float32))
    return ratio


def load_existing_masks(root_path: str):
    masks = {}
    for fname in os.listdir(root_path):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in {".jpg", ".png"}:
            continue
        m = MASK_RX.match(stem)
        if not m:
            continue
        roi = m.group(1)
        mask_img = cv2.imread(os.path.join(root_path, fname), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        masks[roi] = (mask_img > 0).astype(np.float32)
    return masks


def expected_rois_for_masks(root_path: str, enable_redox: bool, enable_unsat: bool, enable_turnover: bool):
    files = os.listdir(root_path)
    fad_dict = match_files(files, "fad")
    nadh_dict = match_files(files, "nadh")
    unsat_dict = match_files(files, "787")
    sat_dict = match_files(files, "794")
    d_pro_dict = match_files(files, "841")
    d_lip_dict = match_files(files, "844")
    pro_dict = match_files(files, "791")
    lip_dict = match_files(files, "797")

    expected = set()
    if enable_redox:
        expected.update({k for k in fad_dict if k in nadh_dict})
    if enable_unsat:
        expected.update({k for k in unsat_dict if k in sat_dict})
    if enable_turnover:
        expected.update({k for k in d_pro_dict if k in pro_dict})
        expected.update({k for k in d_lip_dict if k in lip_dict})
    return expected


def process_condition(root, enable_redox=True, enable_unsat=True, enable_turnover=False,
                     save=False, workers=1, mask_map=None, **kwargs):
    files = os.listdir(root)
    fad_dict = match_files(files, "fad")
    nadh_dict = match_files(files, "nadh")
    unsat_dict = match_files(files, "787")
    sat_dict = match_files(files, "794")
    d_pro_dict = match_files(files, "841")
    d_lip_dict = match_files(files, "844")
    pro_dict = match_files(files, "791")
    lip_dict = match_files(files, "797")

    data = {
        "redox": [] if enable_redox else None,
        "unsat": [] if enable_unsat else None,
        "protein_turn": [] if enable_turnover else None,
        "lipid_turn": [] if enable_turnover else None,
        "d_ratio": [] if enable_turnover else None,
    }

    tasks = []
    if enable_redox:
        for num, fad_path in sorted(fad_dict.items()):
            nadh_path = nadh_dict.get(num)
            if nadh_path:
                tasks.append(("redox", fad_path, nadh_path))

    if enable_unsat:
        for num, unsat_path in sorted(unsat_dict.items()):
            sat_path = sat_dict.get(num)
            if sat_path:
                tasks.append(("unsat", unsat_path, sat_path))

    if enable_turnover:
        for num, d_pro_path in sorted(d_pro_dict.items()):
            pro_path = pro_dict.get(num)
            if pro_path:
                tasks.append(("protein_turn", d_pro_path, pro_path))
        for num, d_lip_path in sorted(d_lip_dict.items()):
            lip_path = lip_dict.get(num)
            if lip_path:
                tasks.append(("lipid_turn", d_lip_path, lip_path))

        # Compute per-ROI d_protein/d_lipid channel stats for downstream ratio comparison
        shared_ids = sorted(set(d_pro_dict.keys()) & set(d_lip_dict.keys()))
        for roi_id in shared_ids:
            d_pro_path = d_pro_dict.get(roi_id)
            d_lip_path = d_lip_dict.get(roi_id)
            if not (d_pro_path and d_lip_path):
                continue
            pro_img = cv2.imread(os.path.join(root, d_pro_path), cv2.IMREAD_UNCHANGED)
            lip_img = cv2.imread(os.path.join(root, d_lip_path), cv2.IMREAD_UNCHANGED)
            if pro_img is None or lip_img is None:
                continue
            sat_mask = (pro_img == 4095) | (lip_img == 4095)
            pro_img[sat_mask] = 0
            lip_img[sat_mask] = 0

            if mask_map is not None and roi_id in mask_map:
                roi_mask = mask_map[roi_id]
                if roi_mask.shape != pro_img.shape:
                    roi_mask = cv2.resize(roi_mask, (pro_img.shape[1], pro_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                pro_img = (pro_img.astype(np.float32) * roi_mask).astype(pro_img.dtype)
                lip_img = (lip_img.astype(np.float32) * roi_mask).astype(lip_img.dtype)

            pro_mean, pro_med = _channel_stats(pro_img)
            lip_mean, lip_med = _channel_stats(lip_img)
            data["d_ratio"].append((pro_mean, pro_med, lip_mean, lip_med))

            # Save per-pixel dProtein/dLipid ratio map similar to turnover outputs
            if save:
                pro_float = pro_img.astype(np.float32)
                lip_float = lip_img.astype(np.float32)
                denom = np.where(lip_float == 0, np.finfo(np.float32).eps, lip_float)
                ratio_map = pro_float / denom
                ratio_map[~np.isfinite(ratio_map)] = 0  # Guard against NaN/inf from unexpected values
                ratio_map = np.clip(ratio_map, 0.0, 2.0)

                roi_dir = os.path.join(root, f"roi_{roi_id}")
                os.makedirs(roi_dir, exist_ok=True)
                img_stem = f"{roi_id}_d_pro_over_lip_ratio"
                save_ratio_image(ratio_map, roi_dir, img_stem + ".png", LOW_PCT_DEFAULT, HIGH_PCT_DEFAULT, kwargs.get("gamma", GAMMA_DEFAULT), kwargs.get("cbar", CBAR_DEFAULT))
                tiff.imwrite(os.path.join(root, img_stem + ".tiff"), ratio_map.astype(np.float32))

    max_workers = max(1, int(workers) if workers is not None else 1)
    if max_workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(calculate_ratio, root, path_a, path_b, suffix, save, mask_map=mask_map, **kwargs): suffix
                for suffix, path_a, path_b in tasks
            }
            for fut in as_completed(future_map):
                suffix = future_map[fut]
                data[suffix].append(fut.result())
    else:
        for suffix, path_a, path_b in tasks:
            data[suffix].append(calculate_ratio(root, path_a, path_b, suffix, save, mask_map=mask_map, **kwargs))

    return data


def collect_image_medians(cond_ratios) -> np.ndarray:
    vals = []
    for arr in cond_ratios:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = np.where(a == 0, np.nan, a)
        med = np.nanmedian(a)
        if np.isfinite(med):
            vals.append(med)
    return np.asarray(vals, dtype=float)


def collect_image_means(cond_ratios) -> np.ndarray:
    vals = []
    for arr in cond_ratios:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = np.where(a == 0, np.nan, a)
        mean = np.nanmean(a)
        if np.isfinite(mean):
            vals.append(mean)
    return np.asarray(vals, dtype=float)


def pairwise_tests(metric_values):
    results = {}
    conds = list(metric_values.keys())
    for c1, c2 in itertools.combinations(conds, 2):
        v1 = np.asarray(metric_values[c1])
        v2 = np.asarray(metric_values[c2])
        n1, n2 = v1.size, v2.size
        if v1.size == 0 or v2.size == 0:
            p_t, p_u = np.nan, np.nan
        else:
            _, p_t = ttest_ind(v1, v2, equal_var=False)
            _, p_u = mannwhitneyu(v1, v2)
        results[(c1, c2)] = (p_t, p_u, n1, n2)
    return results


def print_pairwise(results, label):
    print(f"\nPairwise stats for {label}:")
    for (c1, c2), (p_t, p_u, n1, n2) in results.items():
        t_txt = f"{p_t:.4e}" if p_t is not None and not np.isnan(p_t) else "nan"
        u_txt = f"{p_u:.4e}" if p_u is not None and not np.isnan(p_u) else "nan"
        print(f"{c1} vs {c2} -> t p={t_txt}, U p={u_txt} | n1={n1}, n2={n2}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate ratios across multiple conditions and plot group comparisons.")
    parser.add_argument("dirs", nargs="+", type=str, help="Input directories for each condition")
    parser.add_argument("--conds", "-c", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--out", "-o", type=str, default=".", help="Output directory for plots")
    parser.add_argument("-d", "--deuterated", action="store_true", help="Include deuterated turnover channels")
    parser.add_argument("--save", "-s", action="store_true", help="Save per-image ratio maps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("-r", "--skip-redox", action="store_true", help="Skip redox analysis")
    parser.add_argument("-u", "--skip-unsat", action="store_true", help="Skip unsaturation analysis")
    parser.add_argument("--workers", "-w", type=int, default=(os.cpu_count() or 1), help="Number of parallel workers (default: os.cpu_count())")
    parser.add_argument("--use-model-mask", "-m", action="store_true", help="Use ROI masks; prefer existing '<roi>_mask.jpg' files, otherwise generate with model weights")
    parser.add_argument("--mask-weights", "-mw", type=str, default=None, help="Path to the MultiScaleUNet mask weights (.pth)")
    parser.add_argument("--mask-threshold", "-mt", type=float, default=0.5, help="Sigmoid threshold for binarizing the predicted mask")
    parser.add_argument("--image-median", "-i", action="store_true", help="Run stats on per-image medians instead of per-image means")
    parser.add_argument("--pdf-out", "-p", action="store_true", help="Save all plots into a single multi-page PDF in the output directory")
    parser.add_argument("--hide-ns", action="store_true", help="Hide non-significant comparisons in plots")
    parser.add_argument("--trim-extremes", "-t", type=int, default=0, help="Drop N lowest and N highest values per condition before stats/plots (0=disabled)")
    args = parser.parse_args()

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    save = args.save
    hide_ns = args.hide_ns
    use_image_median = args.image_median
    os.makedirs(args.out, exist_ok=True)
    pdf_pages = PdfPages(os.path.join(args.out, "plots.pdf")) if args.pdf_out else None
    pdf_pages_violin = None  # Intentionally skip writing violin plots to the PDF
    save_png = not args.pdf_out
    kwargs = {
        "gamma": GAMMA_DEFAULT,
        "cbar": CBAR_DEFAULT,
    }

    # Collect ratios per condition
    enable_redox = not args.skip_redox
    enable_unsat = not args.skip_unsat

    cond_redox = {cond: [] for cond in args.conds} if enable_redox else None
    cond_unsat = {cond: [] for cond in args.conds} if enable_unsat else None
    cond_turn_protein = {cond: [] for cond in args.conds} if args.deuterated else None
    cond_turn_lipid = {cond: [] for cond in args.conds} if args.deuterated else None
    cond_d_ratio = {cond: [] for cond in args.conds} if args.deuterated else None

    for cond, dir_path in zip(args.conds, args.dirs):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        mask_map = None
        if args.use_model_mask:
            expected_rois = expected_rois_for_masks(dir_path, enable_redox, enable_unsat, args.deuterated)
            mask_map = load_existing_masks(dir_path)
            missing_rois = expected_rois - set(mask_map.keys())

            if missing_rois:
                print(f"Model masks will be generated for missing ROI IDs {sorted(missing_rois)} in {dir_path}")
                if not args.mask_weights:
                    raise ValueError(
                        "Mask weights must be provided when --use-model-mask is set and masks are missing "
                        f"for {dir_path}. Missing ROI IDs: {sorted(missing_rois)}"
                    )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generated_masks = generate_model_masks(
                    dir_path, args.mask_weights, device=device, threshold=args.mask_threshold
                )
                for roi, mask in generated_masks.items():
                    if roi not in mask_map:
                        mask_map[roi] = mask

                still_missing = expected_rois - set(mask_map.keys())
                if still_missing:
                    print(
                        f"WARNING: masks still missing for ROI IDs {sorted(still_missing)} in {dir_path}"
                    )
        data = process_condition(dir_path, enable_redox=enable_redox, enable_unsat=enable_unsat,
                     enable_turnover=args.deuterated, save=save, workers=args.workers, mask_map=mask_map, **kwargs)
        if cond_redox is not None:
            cond_redox[cond] = data.get("redox", [])
        if cond_unsat is not None:
            cond_unsat[cond] = data.get("unsat", [])
        if cond_turn_protein is not None:
            cond_turn_protein[cond] = data.get("protein_turn", [])
        if cond_turn_lipid is not None:
            cond_turn_lipid[cond] = data.get("lipid_turn", [])
        if cond_d_ratio is not None:
            cond_d_ratio[cond] = data.get("d_ratio", [])

        if args.verbose:
            print(f"Processed {cond}: {len(data.get('redox') or [])} redox, {len(data.get('unsat') or [])} unsat, "
                  f"{len(data.get('protein_turn') or [])} protein turn, {len(data.get('lipid_turn') or [])} lipid turn")

    print("Processing complete.")

    print("Running statistical tests and plotting group comparisons...")

    if use_image_median:
        def agg_fn(ratios):
            return collect_image_medians(ratios)
        stat_label = "Image-median t-test"
    else:
        def agg_fn(ratios):
            return collect_image_means(ratios)
        stat_label = "Image-mean t-test"

    def metric_map(metric_map):
        return {cond: agg_fn(ratios) for cond, ratios in metric_map.items()}

    def trim_metric_values(metric_values):
        drop_n = max(0, int(args.trim_extremes or 0))
        if drop_n == 0:
            return metric_values
        return {cond: _trim_extremes(vals, drop_n=drop_n) for cond, vals in metric_values.items()}

    def d_ratio_values(stat_map):
        vals = {}
        for cond, stats in stat_map.items():
            res = []
            for pro_mean, pro_med, lip_mean, lip_med in stats:
                num = pro_med if use_image_median else pro_mean
                den = lip_med if use_image_median else lip_mean
                if den is None or not np.isfinite(den) or den <= 0 or num is None or not np.isfinite(num):
                    continue
                res.append(num / den)
            vals[cond] = np.asarray(res, dtype=float)
        return vals

    if cond_redox is not None:
        redox_blocks = trim_metric_values(metric_map(cond_redox))
        redox_pairwise = pairwise_tests(redox_blocks)
        print_pairwise(redox_pairwise, "redox (" + stat_label + ")")
        redox_p = {k: v[0] for k, v in redox_pairwise.items()}
        plot_violins_all(redox_blocks, args.conds, redox_p, args.out, stat_label, "Redox ratio", "redox", hide_ns, pdf_pages_violin, save_png)
        plot_bars_all(redox_blocks, args.conds, redox_p, args.out, stat_label, "Redox ratio", "redox", hide_ns, pdf_pages, save_png)

    if cond_unsat is not None:
        unsat_blocks = trim_metric_values(metric_map(cond_unsat))
        unsat_pairwise = pairwise_tests(unsat_blocks)
        print_pairwise(unsat_pairwise, "unsaturation (" + stat_label + ")")
        unsat_p = {k: v[0] for k, v in unsat_pairwise.items()}
        plot_violins_all(unsat_blocks, args.conds, unsat_p, args.out, stat_label, "Unsaturation ratio", "unsat", hide_ns, pdf_pages_violin, save_png)
        plot_bars_all(unsat_blocks, args.conds, unsat_p, args.out, stat_label, "Unsaturation ratio", "unsat", hide_ns, pdf_pages, save_png)

    if args.deuterated:
        d_ratio_blocks = trim_metric_values(d_ratio_values(cond_d_ratio))
        d_ratio_pairwise = pairwise_tests(d_ratio_blocks)
        print_pairwise(d_ratio_pairwise, "d_protein/d_lipid (" + stat_label + ")")
        d_ratio_p = {k: v[0] for k, v in d_ratio_pairwise.items()}
        plot_violins_all(d_ratio_blocks, args.conds, d_ratio_p, args.out, stat_label, "dProtein/dLipid channel ratio", "d_pro_over_lip", hide_ns, pdf_pages_violin, save_png)
        plot_bars_all(d_ratio_blocks, args.conds, d_ratio_p, args.out, stat_label, "dProtein/dLipid channel ratio", "d_pro_over_lip", hide_ns, pdf_pages, save_png)

        pt_blocks = trim_metric_values(metric_map(cond_turn_protein))
        lt_blocks = trim_metric_values(metric_map(cond_turn_lipid))
        pt_pairwise = pairwise_tests(pt_blocks)
        lt_pairwise = pairwise_tests(lt_blocks)
        print_pairwise(pt_pairwise, "protein turnover (" + stat_label + ")")
        print_pairwise(lt_pairwise, "lipid turnover (" + stat_label + ")")
        pt_p = {k: v[0] for k, v in pt_pairwise.items()}
        lt_p = {k: v[0] for k, v in lt_pairwise.items()}
        plot_violins_all(pt_blocks, args.conds, pt_p, args.out, stat_label, "Protein turnover ratio", "protein_turn", hide_ns, pdf_pages_violin, save_png)
        plot_bars_all(pt_blocks, args.conds, pt_p, args.out, stat_label, "Protein turnover ratio", "protein_turn", hide_ns, pdf_pages, save_png)
        plot_violins_all(lt_blocks, args.conds, lt_p, args.out, stat_label, "Lipid turnover ratio", "lipid_turn", hide_ns, pdf_pages_violin, save_png)
        plot_bars_all(lt_blocks, args.conds, lt_p, args.out, stat_label, "Lipid turnover ratio", "lipid_turn", hide_ns, pdf_pages, save_png)

    if pdf_pages is not None:
        pdf_pages.close()

