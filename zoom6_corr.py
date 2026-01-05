import argparse, os, numpy as np, tifffile, cv2, matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial.distance import pdist
from matplotlib.backends.backend_pdf import PdfPages
from visualizations import plot_condition_correlation
from utils import get_number


def center_nonzero(arr, use_mean: bool = False):
	flat = np.asarray(arr).ravel()
	flat = flat[(flat > 0) & np.isfinite(flat)]
	if flat.size == 0:
		return np.nan
	return float(np.mean(flat)) if use_mean else float(np.median(flat))


def collect_condition_stats(cond_name, cond_dir, use_mean: bool = False):
	lipid_files = [f for f in os.listdir(cond_dir) if f.lower().endswith("lipid_turn_ratio.tiff")]
	lipid_vals, protein_vals = [], []
	size_vals, count_vals, frac_vals, dist_vals = [], [], [], []
	roi_rows = []
	total_rois = 0
	kept_rois = 0

	for lipid_fname in sorted(lipid_files):
		roi = get_number(lipid_fname)
		if not roi:
			continue
		total_rois += 1

		lipid_path = os.path.join(cond_dir, lipid_fname)
		protein_path = os.path.join(cond_dir, f"{roi}_protein_turn_ratio.tiff")
		mask_candidates = [
			os.path.join(cond_dir, f"{roi}_seg_lipid.png"),
			os.path.join(cond_dir, f"roi{roi}_seg_lipid.png"),
			os.path.join(cond_dir, f"roi_{roi}", f"{roi}_seg_lipid.png"),
		]
		mask_path = next((p for p in mask_candidates if os.path.isfile(p)), None)
		if not os.path.isfile(protein_path) or mask_path is None:
			continue

		lipid_img = tifffile.imread(lipid_path)
		protein_img = tifffile.imread(protein_path)
		lipid_val = center_nonzero(lipid_img, use_mean)
		protein_val = center_nonzero(protein_img, use_mean)

		mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
		if mask_img is None:
			continue
		mask_gray = mask_img[..., 0] if mask_img.ndim == 3 else mask_img
		mask_bool = mask_gray > 0
		if mask_bool.size == 0:
			continue

		frac_val = float(mask_bool.mean() * 100.0)
		labeled = measure.label(mask_bool, connectivity=2)
		props = measure.regionprops(labeled)
		areas = np.array([p.area for p in props], dtype=float)
		centroids = np.array([p.centroid for p in props], dtype=float)
		roi_size = center_nonzero(areas, use_mean) if areas.size else np.nan
		roi_count = float(len(props))
		roi_dist = float(np.mean(pdist(centroids)) if use_mean else np.median(pdist(centroids))) if centroids.shape[0] >= 2 else np.nan

		metrics = [lipid_val, protein_val, roi_size, roi_count, frac_val, roi_dist]
		# Previously filtered by min_value; now keep all finite/nonzero-derived metrics

		lipid_vals.append(lipid_val)
		protein_vals.append(protein_val)
		frac_vals.append(frac_val)
		size_vals.append(roi_size)
		count_vals.append(roi_count)
		dist_vals.append(roi_dist)

		roi_rows.append({
			"cond": cond_name,
			"roi": roi,
			"lipid": lipid_val,
			"protein": protein_val,
			"size": roi_size,
			"count": roi_count,
			"fraction": frac_val,
			"distance": roi_dist,
		})
		kept_rois += 1

	agg = np.nanmean if use_mean else np.nanmedian
	summary = {
		"cond": cond_name,
		"lipid": agg(lipid_vals) if lipid_vals else np.nan,
		"protein": agg(protein_vals) if protein_vals else np.nan,
		"size": agg(size_vals) if size_vals else np.nan,
		"count": agg(count_vals) if count_vals else np.nan,
		"fraction": agg(frac_vals) if frac_vals else np.nan,
		"distance": agg(dist_vals) if dist_vals else np.nan,
	}
	return summary, roi_rows, total_rois, kept_rois


def main():
	parser = argparse.ArgumentParser(description="Plot correlations between turnover metrics and droplet properties across conditions.")
	parser.add_argument("dirs", nargs="+", help="Input directories, one per condition")
	parser.add_argument("--conds", nargs="+", required=True, help="Condition names (match order of dirs)")
	parser.add_argument("--out", default="corr_plots", help="Directory to save plots")
	parser.add_argument("--pdf-out", default=None, help="Optional path to write all plots into a PDF")
	parser.add_argument("--per-roi", "-pr", action="store_true", help="Plot each ROI median as a point (colored by condition) instead of one point per condition")
	parser.add_argument("--stat", choices=["median", "mean"], default="median", help="Central tendency for pixel/ROI aggregation (default: median)")
	parser.add_argument("--min-value", type=float, default=0.0, help="Drop points with x or y below this threshold when plotting/correlating")
	args = parser.parse_args()

	if len(args.dirs) != len(args.conds):
		raise ValueError("dirs and conds must have the same length")

	stats = []
	roi_records = []
	use_mean = args.stat == "mean"
	min_value = float(args.min_value)
	for cond, d in zip(args.conds, args.dirs):
		if not os.path.isdir(d):
			raise FileNotFoundError(f"Directory not found: {d}")
		summary, roi_rows, total_rois, kept_rois = collect_condition_stats(cond, d, use_mean=use_mean)
		stats.append(summary)
		roi_records.extend(roi_rows)
		print(f"{cond}: kept {kept_rois}/{total_rois} ROIs")

	cond_names = [s["cond"] for s in stats]
	cond_colors = {cond: color for cond, color in zip(cond_names, plt_colors(len(cond_names)))}

	stat_label = "mean" if use_mean else "median"
	metric_pairs = [
		("size", f"{stat_label.capitalize()} lipid turnover vs droplet size", "Droplet size", f"Lipid turnover ({stat_label})"),
		("count", f"{stat_label.capitalize()} lipid turnover vs droplet count", "Droplet count", f"Lipid turnover ({stat_label})"),
		("fraction", f"{stat_label.capitalize()} lipid turnover vs LD area fraction", "LD area fraction", f"Lipid turnover ({stat_label})"),
		("distance", f"{stat_label.capitalize()} lipid turnover vs droplet pairwise distance", "Pairwise distance", f"Lipid turnover ({stat_label})"),
		("protein", f"{stat_label.capitalize()} lipid turnover vs protein turnover", f"Protein turnover ({stat_label})", f"Lipid turnover ({stat_label})"),
	]

	pdf_pages = PdfPages(args.pdf_out) if args.pdf_out else None
	try:
		for key, title, xlabel, ylabel in metric_pairs:
			if args.per_roi:
				xs = [r[key] for r in roi_records]
				ys = [r["lipid"] for r in roi_records]
				cond_list = [r["cond"] for r in roi_records]
				color_list = [cond_colors[c] for c in cond_list]
				plot_condition_correlation(xs=xs, ys=ys, cond_names=cond_list, colors=color_list, outdir=args.out, title=title, xlabel=xlabel, ylabel=ylabel, pdf_pages=pdf_pages, stat_label=stat_label, min_value=min_value)
			else:
				xs = [s[key] for s in stats]
				ys = [s["lipid"] for s in stats]
				colors = [cond_colors[c] for c in cond_names]
				plot_condition_correlation(xs=xs, ys=ys, cond_names=cond_names, colors=colors, outdir=args.out, title=title, xlabel=xlabel, ylabel=ylabel, pdf_pages=pdf_pages, stat_label=stat_label, min_value=min_value)
	finally:
		if pdf_pages is not None:
			pdf_pages.close()


def plt_colors(n):
	# Simple palette aligned to number of conditions
	base = np.linspace(0, 1, max(n, 1), endpoint=False)
	return [list(plt.cm.tab10(b % 1.0))[:3] for b in base]


if __name__ == "__main__":
	main()
