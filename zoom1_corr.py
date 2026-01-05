import argparse
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from visualizations import plot_condition_correlation
from utils import get_number

REGION_NAMES = ("SEZ", "AL", "LP", "MB")
METRICS = ["lipid_turn", "protein_turn", "redox", "unsat"]


def center_nonzero(arr, use_mean: bool = False):
    flat = np.asarray(arr).ravel()
    flat = flat[(flat > 0) & np.isfinite(flat)]
    if flat.size == 0:
        return np.nan
    return float(np.mean(flat)) if use_mean else float(np.median(flat))


def load_roi_metrics(cond_dir, regions, use_mean: bool):
    """Load per-ROI metric medians/means for each region."""
    files = [f for f in os.listdir(cond_dir) if f.lower().endswith(".tiff")]
    roi_ids = sorted({get_number(f) for f in files if get_number(f) is not None})
    records = []
    for roi in roi_ids:
        for region in regions:
            entry = {"roi": roi, "region": region, "cond": None}
            has_any = False
            for metric in METRICS:
                fname = f"{roi}_{metric}_ratio_{region}.tiff"
                fpath = os.path.join(cond_dir, fname)
                if not os.path.isfile(fpath):
                    entry[metric] = np.nan
                    continue
                img = tifffile.imread(fpath)
                entry[metric] = center_nonzero(img, use_mean=use_mean)
                has_any = True
            if has_any:
                records.append(entry)
    return records


def aggregate_by_condition(records, cond_name, regions, use_mean: bool):
    """Aggregate ROI-level records to one value per condition per region."""
    agg_fn = np.nanmean if use_mean else np.nanmedian
    agg_records = []
    for region in regions:
        region_rows = [r for r in records if r["region"] == region]
        if not region_rows:
            continue
        agg_entry = {"cond": cond_name, "region": region}
        for metric in METRICS:
            vals = [r.get(metric, np.nan) for r in region_rows]
            vals = np.asarray(vals, dtype=float)
            agg_entry[metric] = agg_fn(vals) if vals.size else np.nan
        agg_records.append(agg_entry)
    return agg_records


def plt_colors(n):
    base = np.linspace(0, 1, max(n, 1), endpoint=False)
    return [list(plt.cm.tab10(b % 1.0))[:3] for b in base]


def main():
    parser = argparse.ArgumentParser(description="Zoom1: correlate turnover and redox/unsat metrics by region across conditions.")
    parser.add_argument("dirs", nargs="+", help="Input directories, one per condition")
    parser.add_argument("--conds", nargs="+", required=True, help="Condition names (match order of dirs)")
    parser.add_argument("--out", default="corr_zoom1", help="Directory to save plots")
    parser.add_argument("--pdf-out", default=None, help="Optional path to write all plots into a PDF")
    parser.add_argument("--stat", choices=["median", "mean"], default="median", help="Central tendency for pixel aggregation (default: median)")
    parser.add_argument("--per-cond", action="store_true", help="Use one aggregated point per condition instead of per-ROI points")
    parser.add_argument("--min-value", type=float, default=0.0, help="Drop points with x or y below this threshold when plotting/correlating")
    args = parser.parse_args()

    if len(args.dirs) != len(args.conds):
        raise ValueError("dirs and conds must have the same length")

    use_mean = args.stat == "mean"
    min_value = float(args.min_value)

    all_roi_records = []
    per_cond_records = []

    for cond, d in zip(args.conds, args.dirs):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
        roi_recs = load_roi_metrics(d, REGION_NAMES, use_mean=use_mean)
        for rec in roi_recs:
            rec["cond"] = cond
        all_roi_records.extend(roi_recs)
        per_cond_records.extend(aggregate_by_condition(roi_recs, cond, REGION_NAMES, use_mean=use_mean))
        print(f"{cond}: loaded {len(roi_recs)} ROI-region records")

    # metric pairs: (x_key, y_key, title, xlabel, ylabel)
    stat_label = "mean" if use_mean else "median"
    metric_pairs = [
        ("lipid_turn", "redox", f"{stat_label.capitalize()} lipid turnover vs redox", f"Redox ratio ({stat_label})", f"Lipid turnover ({stat_label})"),
        ("protein_turn", "redox", f"{stat_label.capitalize()} protein turnover vs redox", f"Redox ratio ({stat_label})", f"Protein turnover ({stat_label})"),
        ("lipid_turn", "protein_turn", f"{stat_label.capitalize()} lipid vs protein turnover", f"Protein turnover ({stat_label})", f"Lipid turnover ({stat_label})"),
        ("lipid_turn", "unsat", f"{stat_label.capitalize()} lipid turnover vs unsaturation", f"Unsaturation ratio ({stat_label})", f"Lipid turnover ({stat_label})"),
    ]

    cond_colors = {cond: color for cond, color in zip(args.conds, plt_colors(len(args.conds)))}

    os.makedirs(args.out, exist_ok=True)
    pdf_pages = PdfPages(args.pdf_out) if args.pdf_out else None

    try:
        for region in REGION_NAMES:
            region_dir = os.path.join(args.out, region)
            os.makedirs(region_dir, exist_ok=True)
            for x_key, y_key, title, xlabel, ylabel in metric_pairs:
                if args.per_cond:
                    rows = [r for r in per_cond_records if r["region"] == region]
                else:
                    rows = [r for r in all_roi_records if r["region"] == region]
                if not rows:
                    continue
                xs = [r.get(x_key, np.nan) for r in rows]
                ys = [r.get(y_key, np.nan) for r in rows]
                conds = [r["cond"] for r in rows]
                colors = [cond_colors[c] for c in conds]
                plot_condition_correlation(
                    xs=xs,
                    ys=ys,
                    cond_names=conds,
                    colors=colors,
                    outdir=region_dir,
                    title=f"{title} ({region})",
                    xlabel=xlabel,
                    ylabel=ylabel,
                    pdf_pages=pdf_pages,
                    dedup_labels=not args.per_cond,
                    stat_label=stat_label,
                    min_value=min_value,
                )
    finally:
        if pdf_pages is not None:
            pdf_pages.close()


if __name__ == "__main__":
    main()
