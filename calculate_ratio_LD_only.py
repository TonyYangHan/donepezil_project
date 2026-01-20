import argparse
import itertools
import os
import numpy as np
import cv2
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind, mannwhitneyu

from utils import get_number
from visualizations import significance_label, annotate_pairs


def _find_ratio_files(root: str, suffix: str):
    matches = {}
    for fname in os.listdir(root):
        if fname.lower().endswith(f"_{suffix}_turn_ratio.tiff"):
            try:
                roi = get_number(fname)
            except ValueError:
                continue
            matches[roi] = os.path.join(root, fname)
    return matches


def _load_mask(root: str, roi: str):
    path = os.path.join(root, f"{roi}_mask.png")
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask.astype(np.float32) / 255.0


def _masked_median(ratio_path: str, mask: np.ndarray):
    arr = tiff.imread(ratio_path).astype(np.float32)
    if mask is None:
        return None
    if mask.shape != arr.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match ratio image {arr.shape} for {ratio_path}"
        )
    arr = arr * mask
    valid = arr > 0
    if not np.any(valid):
        return None
    return float(np.median(arr[valid]))


def _collect_condition_medians(root: str):
    medians = {"protein": [], "lipid": []}
    for suffix in ("protein", "lipid"):
        ratio_files = _find_ratio_files(root, suffix)
        for roi, path in ratio_files.items():
            mask = _load_mask(root, roi)
            if mask is None:
                print(f"WARNING: missing mask for ROI {roi} in {root}")
                continue
            med = _masked_median(path, mask)
            if med is not None:
                medians[suffix].append(med)
    return medians


def _pairwise_tests(metric_map):
    results = {}
    conds = list(metric_map.keys())
    for c1, c2 in itertools.combinations(conds, 2):
        v1 = np.asarray(metric_map[c1], dtype=float)
        v2 = np.asarray(metric_map[c2], dtype=float)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        n1, n2 = v1.size, v2.size
        if n1 == 0 or n2 == 0:
            results[(c1, c2)] = (None, None, n1, n2)
            continue
        _, p_t = ttest_ind(v1, v2, equal_var=False)
        _, p_u = mannwhitneyu(v1, v2)
        results[(c1, c2)] = (p_t, p_u, n1, n2)
    return results


def _print_pairwise(results, label: str):
    print(f"\nPairwise stats for {label}:")
    for (c1, c2), (p_t, p_u, n1, n2) in results.items():
        t_txt = f"{p_t:.4e}" if p_t is not None and not np.isnan(p_t) else "nan"
        u_txt = f"{p_u:.4e}" if p_u is not None and not np.isnan(p_u) else "nan"
        sig = significance_label(p_t) if p_t is not None else "n/a"
        print(f"{c1} vs {c2} -> t p={t_txt}, U p={u_txt} | n1={n1}, n2={n2} | sig={sig}")


def _plot_bars(metric_map, title: str, ylabel: str, pdf_pages=None, pairwise_p=None, hide_ns: bool = False, outdir: str = None):
    conds = list(metric_map.keys())
    n_cond = len(conds)
    medians = []
    ses = []
    counts = []
    for cond in conds:
        vals = np.asarray(metric_map[cond], dtype=float)
        vals = vals[np.isfinite(vals)]
        counts.append(len(vals))
        medians.append(float(np.median(vals)) if len(vals) else 0.0)
        if len(vals) > 1:
            ses.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
        else:
            ses.append(0.0)

    fig, ax = plt.subplots(figsize=(max(5.0, 2.2 * n_cond), 6.0))
    x = np.arange(n_cond)
    ax.bar(x, medians, yerr=ses, capsize=6, alpha=0.9, width=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(title)
    denom = max(medians + ses) if medians else 1.0
    for xi, y, se, n in zip(x, medians, ses, counts):
        ax.text(xi, y + se + 0.02 * denom, f"n={n}", ha="center", va="bottom", fontsize=9)

    if pairwise_p:
        span = max(medians) - min(medians) if medians else 1.0
        span = max(span, 1e-6)
        y_top = max(medians) + max(ses or [0])
        x_pos_map = {cond: pos for cond, pos in zip(conds, x)}
        annotate_pairs(ax, x_pos_map, y_top + 0.02 * span, pairwise_p, 0.04 * span, hide_ns)

    fig.tight_layout()
    if pdf_pages is not None and outdir is not None:
        base = title.lower().replace(" ", "_").replace("/", "-")
        svg_path = os.path.join(outdir, f"{base}.svg")
        fig.savefig(svg_path, format="svg")
        pdf_pages.savefig(fig, bbox_inches="tight")
    elif pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Median turnover comparison across conditions (lipid/protein)."
    )
    parser.add_argument("dirs", nargs="+", help="Input directories (one per condition)")
    parser.add_argument("--conds", "-c", nargs="+", required=True,
                        help="Condition names matching order of dirs")
    parser.add_argument(
        "--out", "-o", required=True,
        help="Full output PDF path (e.g., /path/to/turnover_medians.pdf)."
    )
    args = parser.parse_args(argv)

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pdf_pages = PdfPages(args.out)

    cond_protein = {cond: [] for cond in args.conds}
    cond_lipid = {cond: [] for cond in args.conds}

    for cond, root in zip(args.conds, args.dirs):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Directory not found: {root}")
        medians = _collect_condition_medians(root)
        cond_protein[cond].extend(medians["protein"])
        cond_lipid[cond].extend(medians["lipid"])
        print(f"{cond}: protein n={len(medians['protein'])}, lipid n={len(medians['lipid'])}")

    protein_res = _pairwise_tests(cond_protein)
    lipid_res = _pairwise_tests(cond_lipid)

    _print_pairwise(protein_res, "protein turnover")
    _print_pairwise(lipid_res, "lipid turnover")

    protein_p = {k: v[0] for k, v in protein_res.items()}
    lipid_p = {k: v[0] for k, v in lipid_res.items()}

    outdir = os.path.dirname(args.out) or "."
    _plot_bars(cond_protein, "Protein turnover medians", "Median ratio",
               pdf_pages=pdf_pages, pairwise_p=protein_p, outdir=outdir)
    _plot_bars(cond_lipid, "Lipid turnover medians", "Median ratio",
               pdf_pages=pdf_pages, pairwise_p=lipid_p, outdir=outdir)

    if pdf_pages is not None:
        pdf_pages.close()


if __name__ == "__main__":
    main()
