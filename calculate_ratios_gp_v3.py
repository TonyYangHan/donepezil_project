import argparse
import csv
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import tifffile as tiff
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import f_oneway, studentized_range, ttest_ind

from utils import get_number
from visualizations import plot_bars_all

D2O_PCT = 0.25
TIFF_EXTENSIONS = {".tif", ".tiff"}


def _is_fad_or_nadh(path: str) -> bool:
    name = os.path.basename(path).lower()
    return "fad" in name or "nadh" in name


def _is_power_corrected_channel(path: str) -> bool:
    name = os.path.basename(path).lower()
    return "841" in name or "844" in name


def _uses_power_corrected_background(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(channel in name for channel in ("787", "791", "794", "797"))


def match_tiff_files(files, label):
    label = label.lower()
    matched = {}
    for fname in files:
        _, ext = os.path.splitext(fname)
        if ext.lower() not in TIFF_EXTENSIONS or label not in fname.lower():
            continue
        matched[get_number(fname)] = fname
    return matched


def _warn_missing(root_path: str, roi_id: str, missing: list[str]):
    print(f"WARNING: missing {', '.join(missing)} for folder '{root_path}', ROI {roi_id}")


def _warn_load(root_path: str, roi_id: str, path: str, label: str):
    print(f"WARNING: failed to load {label} image '{path}' for folder '{root_path}', ROI {roi_id}")


def _read_image(root_path: str, path: str, roi_id: str, label: str):
    try:
        img = tiff.imread(os.path.join(root_path, path))
    except Exception:
        _warn_load(root_path, roi_id, path, label)
        return None
    return img.astype(np.float32)


def load_masks(root_path: str):
    masks = {}
    for fname in os.listdir(root_path):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in TIFF_EXTENSIONS or "mask" not in stem.lower():
            continue
        try:
            roi_id = get_number(fname)
        except ValueError:
            print(f"WARNING: could not extract ROI number from mask '{fname}' in folder '{root_path}'")
            continue

        mask_img = _read_image(root_path, fname, roi_id, "mask")
        if mask_img is None:
            continue
        if mask_img.ndim > 2:
            mask_img = mask_img[..., 0]
        masks[roi_id] = np.clip(mask_img.astype(np.float32) / 255.0, 0.0, 1.0)
    return masks


def _prepare_channel(
    root_path: str,
    image_path: str,
    background_path: str | None,
    roi_id: str,
    label: str,
    power_correction: float = 1.0,
):
    raw = _read_image(root_path, image_path, roi_id, label)
    if raw is None:
        return None, None

    adjusted = raw.copy()
    if not _is_fad_or_nadh(image_path):
        if background_path is None:
            _warn_missing(root_path, roi_id, ["862"])
            return None, None
        background = _read_image(root_path, background_path, roi_id, "862")
        if background is None:
            return None, None
        if background.shape != raw.shape:
            print(
                f"WARNING: 862 image shape {background.shape} does not match {label} shape {raw.shape} "
                f"for folder '{root_path}', ROI {roi_id}"
            )
            return None, None
        if _is_power_corrected_channel(image_path):
            adjusted = (raw - background) / power_correction
        elif _uses_power_corrected_background(image_path):
            adjusted = raw - (background / power_correction)
        else:
            adjusted = raw - background

    return raw, adjusted


def _get_mask(mask_map, root_path: str, roi_id: str, shape):
    if mask_map is None:
        return None
    if roi_id not in mask_map:
        _warn_missing(root_path, roi_id, ["mask"])
        return None
    mask = mask_map[roi_id]
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.clip(mask.astype(np.float32), 0.0, 1.0)


def calculate_ratio(
    root_path,
    numerator_path,
    denominator_path,
    background_path,
    suffix,
    save,
    ratio_method="frac",
    mask_map=None,
    power_correction=1.0,
    protein_reference_path=None,
    lipid_reference_path=None,
):
    roi_id = get_number(numerator_path)
    if suffix == "d_ratio":
        if protein_reference_path is None or lipid_reference_path is None:
            raise ValueError("d_ratio requires corresponding 791 and 797 image paths")
        protein_result = calculate_ratio(
            root_path,
            numerator_path,
            protein_reference_path,
            background_path,
            "protein_turn",
            False,
            ratio_method,
            mask_map,
            power_correction,
        )
        lipid_result = calculate_ratio(
            root_path,
            denominator_path,
            lipid_reference_path,
            background_path,
            "lipid_turn",
            False,
            ratio_method,
            mask_map,
            power_correction,
        )
        if protein_result is None or lipid_result is None:
            return None
        protein_turn = protein_result[1]
        lipid_turn = lipid_result[1]
        if protein_turn.shape != lipid_turn.shape:
            print(
                f"WARNING: protein turnover shape {protein_turn.shape} does not match "
                f"lipid turnover shape {lipid_turn.shape} for folder '{root_path}', ROI {roi_id}"
            )
            return None
        valid = (
            np.isfinite(protein_turn)
            & np.isfinite(lipid_turn)
            & (protein_turn != 0)
            & (lipid_turn != 0)
        )
        ratio = np.zeros(protein_turn.shape, dtype=np.float32)
        ratio[valid] = protein_turn[valid] / lipid_turn[valid]
        ratio[~np.isfinite(ratio)] = 0
        if save:
            tiff.imwrite(os.path.join(root_path, f"{roi_id}_d_ratio_ratio.tiff"), ratio)
        return roi_id, ratio

    num_raw, numerator = _prepare_channel(
        root_path, numerator_path, background_path, roi_id, "numerator", power_correction
    )
    den_raw, denominator = _prepare_channel(
        root_path, denominator_path, background_path, roi_id, "denominator", power_correction
    )
    if numerator is None or denominator is None:
        return None
    if numerator.shape != denominator.shape:
        print(
            f"WARNING: numerator shape {numerator.shape} does not match denominator shape {denominator.shape} "
            f"for folder '{root_path}', ROI {roi_id}"
        )
        return None

    mask = _get_mask(mask_map, root_path, roi_id, numerator.shape)
    if mask_map is not None and mask is None:
        return None
    if mask is not None:
        numerator = numerator * mask
        denominator = denominator * mask

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & np.isfinite(num_raw)
        & np.isfinite(den_raw)
        & (num_raw != 0)
        & (den_raw != 0)
        & (num_raw != 4095)
        & (den_raw != 4095)
        & (numerator != 0)
        & (denominator != 0)
    )
    if mask is not None:
        valid &= mask > 0

    ratio = np.zeros(numerator.shape, dtype=np.float32)
    if suffix in {"redox", "unsat"}:
        if ratio_method == "div":
            ratio[valid] = numerator[valid] / denominator[valid]
        elif ratio_method == "frac":
            denom_sum = numerator + denominator
            frac_valid = valid & np.isfinite(denom_sum) & (denom_sum != 0)
            ratio[frac_valid] = numerator[frac_valid] / denom_sum[frac_valid]
        else:
            raise ValueError("ratio_method must be 'div' or 'frac'")
    elif suffix in {"protein_turn", "lipid_turn"}:
        turnover = np.zeros_like(ratio)
        turnover[valid] = numerator[valid] / denominator[valid]
        keep = valid & np.isfinite(turnover) & (turnover >= 0) & (turnover <= D2O_PCT)
        ratio[keep] = turnover[keep]
    elif suffix == "lipid_over_protein_ratio":
        ratio[valid] = numerator[valid] / denominator[valid]
    else:
        raise ValueError("suffix must be redox, unsat, protein_turn, lipid_turn, d_ratio, or lipid_over_protein_ratio")

    ratio[~np.isfinite(ratio)] = 0
    if save:
        save_name = f"{roi_id}_{suffix}.tiff" if suffix == "lipid_over_protein_ratio" else f"{roi_id}_{suffix}_ratio.tiff"
        tiff.imwrite(os.path.join(root_path, save_name), ratio.astype(np.float32))
    return roi_id, ratio


def process_condition(
    root,
    enable_turnover=False,
    save=False,
    workers=1,
    ratio_method="frac",
    mask_map=None,
    power_correction=1.0,
):
    files = os.listdir(root)
    fad_dict = match_tiff_files(files, "fad")
    nadh_dict = match_tiff_files(files, "nadh")
    background_dict = match_tiff_files(files, "862")
    unsat_dict = match_tiff_files(files, "787")
    sat_dict = match_tiff_files(files, "794")
    d_pro_dict = match_tiff_files(files, "841")
    d_lip_dict = match_tiff_files(files, "844")
    pro_dict = match_tiff_files(files, "791")
    lip_dict = match_tiff_files(files, "797")

    data = {
        "redox": [],
        "unsat": [],
        "protein_turn": [] if enable_turnover else None,
        "lipid_turn": [] if enable_turnover else None,
        "d_ratio": [] if enable_turnover else None,
        "lipid_over_protein_ratio": [] if enable_turnover else None,
    }

    tasks = []
    for roi_id in sorted(set(fad_dict) | set(nadh_dict)):
        missing = []
        if roi_id not in fad_dict:
            missing.append("fad")
        if roi_id not in nadh_dict:
            missing.append("nadh")
        if missing:
            _warn_missing(root, roi_id, missing)
            continue
        tasks.append((roi_id, "redox", fad_dict[roi_id], nadh_dict[roi_id], background_dict.get(roi_id), None, None))

    for roi_id in sorted(set(unsat_dict) | set(sat_dict) | set(background_dict)):
        missing = []
        if roi_id not in unsat_dict:
            missing.append("787")
        if roi_id not in sat_dict:
            missing.append("794")
        if roi_id not in background_dict:
            missing.append("862")
        if missing:
            _warn_missing(root, roi_id, missing)
            continue
        tasks.append((roi_id, "unsat", unsat_dict[roi_id], sat_dict[roi_id], background_dict[roi_id], None, None))

    if enable_turnover:
        for roi_id in sorted(set(d_pro_dict) | set(pro_dict) | set(background_dict)):
            missing = []
            if roi_id not in d_pro_dict:
                missing.append("841")
            if roi_id not in pro_dict:
                missing.append("791")
            if roi_id not in background_dict:
                missing.append("862")
            if missing:
                _warn_missing(root, roi_id, missing)
                continue
            tasks.append(
                (roi_id, "protein_turn", d_pro_dict[roi_id], pro_dict[roi_id], background_dict[roi_id], None, None)
            )

        for roi_id in sorted(set(d_lip_dict) | set(lip_dict) | set(background_dict)):
            missing = []
            if roi_id not in d_lip_dict:
                missing.append("844")
            if roi_id not in lip_dict:
                missing.append("797")
            if roi_id not in background_dict:
                missing.append("862")
            if missing:
                _warn_missing(root, roi_id, missing)
                continue
            tasks.append(
                (roi_id, "lipid_turn", d_lip_dict[roi_id], lip_dict[roi_id], background_dict[roi_id], None, None)
            )

        for roi_id in sorted(
            set(d_pro_dict) | set(pro_dict) | set(d_lip_dict) | set(lip_dict) | set(background_dict)
        ):
            missing = []
            if roi_id not in d_pro_dict:
                missing.append("841")
            if roi_id not in pro_dict:
                missing.append("791")
            if roi_id not in d_lip_dict:
                missing.append("844")
            if roi_id not in lip_dict:
                missing.append("797")
            if roi_id not in background_dict:
                missing.append("862")
            if missing:
                _warn_missing(root, roi_id, missing)
                continue
            tasks.append(
                (
                    roi_id,
                    "d_ratio",
                    d_pro_dict[roi_id],
                    d_lip_dict[roi_id],
                    background_dict[roi_id],
                    pro_dict[roi_id],
                    lip_dict[roi_id],
                )
            )

        for roi_id in sorted(set(lip_dict) | set(pro_dict) | set(background_dict)):
            missing = []
            if roi_id not in lip_dict:
                missing.append("797")
            if roi_id not in pro_dict:
                missing.append("791")
            if roi_id not in background_dict:
                missing.append("862")
            if missing:
                _warn_missing(root, roi_id, missing)
                continue
            tasks.append(
                (
                    roi_id,
                    "lipid_over_protein_ratio",
                    lip_dict[roi_id],
                    pro_dict[roi_id],
                    background_dict[roi_id],
                    None,
                    None,
                )
            )

    def _append_result(result, suffix):
        if result is not None:
            data[suffix].append(result)

    max_workers = max(1, int(workers) if workers is not None else 1)
    if max_workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(
                    calculate_ratio,
                    root,
                    numerator_path,
                    denominator_path,
                    background_path,
                    suffix,
                    save,
                    ratio_method,
                    mask_map,
                    power_correction,
                    protein_reference_path,
                    lipid_reference_path,
                ): (roi_id, suffix)
                for (
                    roi_id,
                    suffix,
                    numerator_path,
                    denominator_path,
                    background_path,
                    protein_reference_path,
                    lipid_reference_path,
                ) in tasks
            }
            for fut in as_completed(future_map):
                roi_id, suffix = future_map[fut]
                try:
                    _append_result(fut.result(), suffix)
                except Exception as exc:
                    print(f"WARNING: failed processing folder '{root}', ROI {roi_id}, {suffix}: {exc}")
    else:
        for (
            _,
            suffix,
            numerator_path,
            denominator_path,
            background_path,
            protein_reference_path,
            lipid_reference_path,
        ) in tasks:
            result = calculate_ratio(
                root,
                numerator_path,
                denominator_path,
                background_path,
                suffix,
                save,
                ratio_method,
                mask_map,
                power_correction,
                protein_reference_path,
                lipid_reference_path,
            )
            _append_result(result, suffix)

    return data


def image_summary_value(ratio, use_median=False):
    arr = np.asarray(ratio, dtype=float)
    arr = np.where(arr == 0, np.nan, arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        value = np.nanmedian(arr) if use_median else np.nanmean(arr)
    return value if np.isfinite(value) else np.nan


def collect_image_values(records, use_median=False) -> np.ndarray:
    vals = []
    for _, ratio in records:
        value = image_summary_value(ratio, use_median=use_median)
        if np.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=float)


def write_image_mean_csv(metric_records, cond_order, outdir, metric_name):
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{metric_name}_image_means.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ROI number", "value", "condition name"])
        for cond in cond_order:
            for roi_id, ratio in metric_records.get(cond, []):
                value = image_summary_value(ratio, use_median=False)
                if np.isfinite(value):
                    writer.writerow([roi_id, value, cond])


def write_statistics_csv(statistics, outdir):
    os.makedirs(outdir, exist_ok=True)
    fieldnames = [
        "metric",
        "test",
        "comparison",
        "statistic_name",
        "statistic",
        "p_value",
        "degrees_of_freedom",
        "n1",
        "n2",
    ]
    with open(os.path.join(outdir, "statistics.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(statistics)


def t_test(metric_values, cond_order=None, equal_var=False):
    results = {}
    conds = cond_order if cond_order is not None else list(metric_values.keys())
    for c1, c2 in itertools.combinations(conds, 2):
        v1 = np.asarray(metric_values[c1], dtype=float)
        v2 = np.asarray(metric_values[c2], dtype=float)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        n1, n2 = v1.size, v2.size
        if n1 == 0 or n2 == 0:
            t_stat, p_t, df = np.nan, np.nan, np.nan
        elif n1 > 1 and n2 > 1 and np.var(v1, ddof=1) == 0 and np.var(v2, ddof=1) == 0:
            same_mean = np.mean(v1) == np.mean(v2)
            t_stat = 0.0 if same_mean else np.copysign(np.inf, np.mean(v1) - np.mean(v2))
            p_t = 1.0 if same_mean else 0.0
            df = (n1 + n2 - 2) if equal_var else np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                test_result = ttest_ind(v1, v2, equal_var=equal_var)
            t_stat, p_t = test_result.statistic, test_result.pvalue
            df = getattr(test_result, "df", np.nan)
        results[(c1, c2)] = (t_stat, p_t, df, n1, n2)
    return results


def print_t_test(results, label, test_name="Welch t-test"):
    print(f"\n{test_name} stats for {label}:")
    for (c1, c2), (t_stat, p_t, _, n1, n2) in results.items():
        stat_txt = f"{t_stat:.4e}" if np.isfinite(t_stat) else ("inf" if np.isinf(t_stat) else "nan")
        t_txt = f"{p_t:.4e}" if p_t is not None and not np.isnan(p_t) else "nan"
        print(f"{c1} vs {c2} -> t={stat_txt}, p={t_txt} | n1={n1}, n2={n2}")


def t_test_rows(metric_values, cond_order, metric_name, test_name, equal_var=False):
    results = t_test(metric_values, cond_order=cond_order, equal_var=equal_var)
    rows = [
        {
            "metric": metric_name,
            "test": test_name,
            "comparison": f"{c1} vs {c2}",
            "statistic_name": "t",
            "statistic": t_stat,
            "p_value": p_val,
            "degrees_of_freedom": df,
            "n1": n1,
            "n2": n2,
        }
        for (c1, c2), (t_stat, p_val, df, n1, n2) in results.items()
    ]
    return results, rows


def one_way_anova(metric_values, cond_order):
    groups = [np.asarray(metric_values[cond], dtype=float) for cond in cond_order]
    groups = [group[np.isfinite(group)] for group in groups if group.size]
    if len(groups) < 2:
        return np.nan, np.nan

    means = np.asarray([np.mean(group) for group in groups], dtype=float)
    variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
    if np.all(variances == 0):
        return (0.0, 1.0) if np.all(means == means[0]) else (np.inf, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            f_stat, p_val = f_oneway(*groups)
            return f_stat, p_val
        except Exception:
            return np.nan, np.nan


def games_howell_tests(metric_values, cond_order):
    results = {}
    k = len(cond_order)
    for c1, c2 in itertools.combinations(cond_order, 2):
        v1 = np.asarray(metric_values[c1], dtype=float)
        v2 = np.asarray(metric_values[c2], dtype=float)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        if v1.size < 2 or v2.size < 2:
            results[(c1, c2)] = (np.nan, np.nan, np.nan, v1.size, v2.size)
            continue

        mean_diff = abs(np.mean(v1) - np.mean(v2))
        var1 = np.var(v1, ddof=1)
        var2 = np.var(v2, ddof=1)
        term1 = var1 / v1.size
        term2 = var2 / v2.size
        se = np.sqrt(term1 + term2)
        if se == 0:
            q_stat = 0.0 if mean_diff == 0 else np.inf
            p_val = 1.0 if mean_diff == 0 else 0.0
            results[(c1, c2)] = (q_stat, p_val, np.inf, v1.size, v2.size)
            continue

        df_num = (term1 + term2) ** 2
        df_den = (term1**2 / (v1.size - 1)) + (term2**2 / (v2.size - 1))
        df = df_num / df_den if df_den > 0 else np.inf
        q_stat = np.sqrt(2.0) * (mean_diff / se)
        p_val = studentized_range.sf(q_stat, k, df)
        results[(c1, c2)] = (q_stat, p_val, df, v1.size, v2.size)
    return results


def comparison_pvalues(metric_values, cond_order, metric_name, label, verbose=False):
    if len(cond_order) == 2:
        results, rows = t_test_rows(metric_values, cond_order, metric_name, "Welch t-test", equal_var=False)
        if verbose:
            print_t_test(results, label, "Welch t-test")
        return {k: v[1] for k, v in results.items()}, "Welch t-test", rows
    if len(cond_order) < 2:
        if verbose:
            print(f"\nSkipping statistical tests for {label}: at least two conditions are required.")
        return {}, "No stats", []

    f_stat, p_val = one_way_anova(metric_values, cond_order)
    finite_groups = [
        np.asarray(metric_values[cond], dtype=float)[np.isfinite(metric_values[cond])]
        for cond in cond_order
    ]
    finite_groups = [group for group in finite_groups if group.size]
    anova_df = (
        f"{len(finite_groups) - 1},{sum(group.size for group in finite_groups) - len(finite_groups)}"
        if len(finite_groups) >= 2
        else ""
    )
    rows = [
        {
            "metric": metric_name,
            "test": "One-way ANOVA",
            "comparison": "all conditions",
            "statistic_name": "F",
            "statistic": f_stat,
            "p_value": p_val,
            "degrees_of_freedom": anova_df,
            "n1": "",
            "n2": "",
        }
    ]
    if len(cond_order) > 2:
        _, welch_rows = t_test_rows(metric_values, cond_order, metric_name, "Pairwise Welch t-test", equal_var=False)
        _, student_rows = t_test_rows(metric_values, cond_order, metric_name, "Pairwise Student's t-test", equal_var=True)
        rows.extend(welch_rows)
        rows.extend(student_rows)

    f_txt = f"{f_stat:.4e}" if np.isfinite(f_stat) else ("inf" if np.isinf(f_stat) else "nan")
    p_txt = f"{p_val:.4e}" if np.isfinite(p_val) else "nan"
    if verbose:
        print(f"\nOne-way ANOVA for {label}: F={f_txt}, p={p_txt}")
    if np.isfinite(p_val) and p_val < 0.05:
        gh = games_howell_tests(metric_values, cond_order)
        if verbose:
            print(f"Games-Howell post hoc for {label}:")
        for (c1, c2), (q_stat, gh_p, df, n1, n2) in gh.items():
            if verbose:
                p_out = f"{gh_p:.4e}" if np.isfinite(gh_p) else "nan"
                print(f"{c1} vs {c2} -> q={q_stat:.4e}, p={p_out}")
            rows.append(
                {
                    "metric": metric_name,
                    "test": "Games-Howell",
                    "comparison": f"{c1} vs {c2}",
                    "statistic_name": "q",
                    "statistic": q_stat,
                    "p_value": gh_p,
                    "degrees_of_freedom": df,
                    "n1": n1,
                    "n2": n2,
                }
            )
        return {k: v[1] for k, v in gh.items()}, "One-way ANOVA-Games-Howell", rows
    if verbose:
        print(f"One-way ANOVA was not significant for {label}; skipping Games-Howell post hoc tests.")
    return {}, "One-way ANOVA", rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ratios across multiple conditions and plot group comparisons.")
    parser.add_argument("dirs", nargs="+", type=str, help="Input directories for each condition")
    parser.add_argument("--conds", "-c", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--out", "-o", type=str, default=".", help="Output directory for plots and CSV files")
    parser.add_argument("-d", "--deuterated", action="store_true", help="Include deuterated turnover channels")
    parser.add_argument("--save", "-s", action="store_true", help="Save per-image ratio TIFF maps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--workers", "-w", type=int, default=(os.cpu_count() or 1), help="Number of parallel workers")
    parser.add_argument("--mask", "-m", action="store_true", help="Use human-generated masks from each image folder")
    parser.add_argument(
        "--ratio-method",
        choices=("frac", "div"),
        default="div",
        help="Redox/unsat calculation: frac = numerator/(numerator+denominator), div = numerator/denominator",
    )
    parser.add_argument(
        "--power-correction",
        "-pc",
        type=float,
        default=1.0,
        help=(
            "Divide 841/844 values by this factor; divide 862 by it only when "
            "subtracting background from 787/791/794/797"
        ),
    )
    parser.add_argument("--image-median", "-i", action="store_true", help="Run stats on per-image medians instead of means")
    parser.add_argument("--hide-ns", action="store_true", help="Hide non-significant comparisons in plots")
    args = parser.parse_args()

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")
    if not np.isfinite(args.power_correction) or args.power_correction <= 0:
        parser.error("--power-correction/-pc must be a finite number greater than 0")

    os.makedirs(args.out, exist_ok=True)
    pdf_pages = PdfPages(os.path.join(args.out, "plots.pdf"))

    cond_redox = {cond: [] for cond in args.conds}
    cond_unsat = {cond: [] for cond in args.conds}
    cond_turn_protein = {cond: [] for cond in args.conds} if args.deuterated else None
    cond_turn_lipid = {cond: [] for cond in args.conds} if args.deuterated else None
    cond_d_ratio = {cond: [] for cond in args.conds} if args.deuterated else None
    cond_lipid_over_protein = {cond: [] for cond in args.conds} if args.deuterated else None

    for cond, dir_path in zip(args.conds, args.dirs):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        mask_map = load_masks(dir_path) if args.mask else None
        data = process_condition(
            dir_path,
            enable_turnover=args.deuterated,
            save=args.save,
            workers=args.workers,
            ratio_method=args.ratio_method,
            mask_map=mask_map,
            power_correction=args.power_correction,
        )
        cond_redox[cond] = data.get("redox", [])
        cond_unsat[cond] = data.get("unsat", [])
        if cond_turn_protein is not None:
            cond_turn_protein[cond] = data.get("protein_turn", [])
        if cond_turn_lipid is not None:
            cond_turn_lipid[cond] = data.get("lipid_turn", [])
        if cond_d_ratio is not None:
            cond_d_ratio[cond] = data.get("d_ratio", [])
        if cond_lipid_over_protein is not None:
            cond_lipid_over_protein[cond] = data.get("lipid_over_protein_ratio", [])

        if args.verbose:
            print(
                f"Processed {cond}: {len(data.get('redox') or [])} redox, "
                f"{len(data.get('unsat') or [])} unsat, "
                f"{len(data.get('protein_turn') or [])} protein turn, "
                f"{len(data.get('lipid_turn') or [])} lipid turn, "
                f"{len(data.get('lipid_over_protein_ratio') or [])} lipid/protein"
            )

    print("Processing complete.")
    print("Running statistical tests and plotting group comparisons...")

    use_image_median = args.image_median
    stat_prefix = "Image-median" if use_image_median else "Image-mean"
    statistics = []

    def metric_map(metric_records):
        return {cond: collect_image_values(records, use_median=use_image_median) for cond, records in metric_records.items()}

    def analyze_metric(metric_records, metric_name, ylabel):
        write_image_mean_csv(metric_records, args.conds, args.out, metric_name)
        blocks = metric_map(metric_records)
        pairwise_p, test_name, metric_statistics = comparison_pvalues(
            blocks,
            args.conds,
            metric_name,
            f"{metric_name} ({stat_prefix})",
            verbose=args.verbose,
        )
        statistics.extend(metric_statistics)
        plot_label = f"{stat_prefix} {test_name}"
        plot_bars_all(
            blocks,
            args.conds,
            pairwise_p,
            args.out,
            plot_label,
            ylabel,
            metric_name,
            args.hide_ns,
            pdf_pages,
            False,
            bar_width=0.45,
        )

    analyze_metric(cond_redox, "redox", "Redox ratio")
    analyze_metric(cond_unsat, "unsat", "Unsaturation ratio")

    if args.deuterated:
        analyze_metric(cond_d_ratio, "d_pro_over_lip", "dProtein/dLipid channel ratio")
        analyze_metric(cond_lipid_over_protein, "lipid_over_protein_ratio", "Lipid/Protein channel ratio")
        analyze_metric(cond_turn_protein, "protein_turn", "Protein turnover ratio")
        analyze_metric(cond_turn_lipid, "lipid_turn", "Lipid turnover ratio")

    write_statistics_csv(statistics, args.out)
    pdf_pages.close()
