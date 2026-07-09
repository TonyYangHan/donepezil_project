from __future__ import annotations

import argparse
import itertools
import os, sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind

from visualizations import plot_bars_all


TIFF_EXTS = {".tif", ".tiff"}
BASE_RX = re.compile(r"^(\d+)$")
ESTER_RX = re.compile(r"^(\d+)_ester$")

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


def ratio_images_for_dir(folder: Path, recursive: bool = False):
	files = [p for p in (folder.rglob("*") if recursive else folder.iterdir()) if p.is_file() and p.suffix.lower() in TIFF_EXTS]
	base = {}
	ester = {}
	for p in files:
		stem = p.stem
		m0 = BASE_RX.match(stem)
		m1 = ESTER_RX.match(stem)
		if m0:
			base[m0.group(1)] = p
		elif m1:
			ester[m1.group(1)] = p

	ratios = []
	for k in sorted(set(base) & set(ester), key=lambda x: int(x)):
		a = tifffile.imread(base[k]).astype(np.float32)
		b = tifffile.imread(ester[k]).astype(np.float32)
		if a.shape != b.shape:
			raise ValueError(f"Shape mismatch for pair {k} in {folder}: {a.shape} vs {b.shape}")
		r = np.zeros_like(a, dtype=np.float32)
		mask = (a != 0) & (b != 0)
		np.divide(a, b, out=r, where=mask)
		ratios.append(r)
	return ratios


def pairwise_ttests(metric_values):
	out = {}
	for c1, c2 in itertools.combinations(metric_values.keys(), 2):
		v1 = np.asarray(metric_values[c1], dtype=float)
		v2 = np.asarray(metric_values[c2], dtype=float)
		if v1.size == 0 or v2.size == 0:
			t_stat, p_val = np.nan, np.nan
		else:
			t_stat, p_val = ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
		out[(c1, c2)] = (t_stat, p_val, v1.size, v2.size)
	return out


def main() -> None:
	p = argparse.ArgumentParser(description="Compare cholesterol ratios (<num>.tiff / <num>_ester.tiff) across conditions.")
	p.add_argument("dirs", nargs="+", type=Path, help="Input directories")
	p.add_argument("--conds", "-c", nargs="+", required=True, help="Condition labels in the same order as dirs")
	p.add_argument("--out", "-o", type=Path, default=Path("."), help="Output directory")
	p.add_argument("--recursive", "-r", action="store_true", help="Search for files recursively")
	p.add_argument("--hide-ns", action="store_true", help="Hide non-significant comparisons in the bar plot")
	p.add_argument("--pdf-out", action="store_true", help="Also write plots.pdf")
	p.add_argument("--image-median", "-i", action="store_true", help="Use image-level medians instead of means")
	args = p.parse_args()

	if len(args.dirs) != len(args.conds):
		raise ValueError("dirs and --conds must have the same length")
	for d in args.dirs:
		if not d.is_dir():
			raise NotADirectoryError(f"Directory not found: {d}")

	os.makedirs(args.out, exist_ok=True)
	pdf_pages = PdfPages(args.out / "plots.pdf") if args.pdf_out else None
	agg_fn = collect_image_medians if args.image_median else collect_image_means
	metric_name = "median" if args.image_median else "mean"

	cond_means = {}
	for cond, d in zip(args.conds, args.dirs):
		ratios = ratio_images_for_dir(d, recursive=args.recursive)
		if not ratios:
			raise FileNotFoundError(f"No matched <num> and <num>_ester pairs found in {d}")
		means = agg_fn(ratios)
		if means.size == 0:
			raise ValueError(f"No valid image {metric_name}s in {d}")
		cond_means[cond] = means

	print(f"Per-condition image-{metric_name} summary:")
	for c in args.conds:
		v = cond_means[c]
		std = np.std(v, ddof=1) if v.size > 1 else 0.0
		print(f"{c}: n={v.size}, mean={np.mean(v):.6f}, std={std:.6f}")

	stats = pairwise_ttests(cond_means)
	print(f"\nPairwise Welch t-tests (image {metric_name}s):")
	for (c1, c2), (t_stat, p_val, n1, n2) in stats.items():
		t_txt = f"{t_stat:.5g}" if np.isfinite(t_stat) else "nan"
		p_txt = f"{p_val:.5g}" if np.isfinite(p_val) else "nan"
		print(f"{c1} vs {c2} -> t={t_txt}, p={p_txt} | n1={n1}, n2={n2}")

	plot_bars_all(
		metric_values=cond_means,
		cond_order=args.conds,
		pairwise_p={k: v[1] for k, v in stats.items()},
		outdir=str(args.out),
		test_name=f"Welch t-test (image {metric_name})",
		ylabel=f"Cholesterol ratio (image {metric_name})",
		label_prefix="cholesterol_ratio",
		hide_ns=args.hide_ns,
		pdf_pages=pdf_pages,
		save_png=True,
	)

	if pdf_pages is not None:
		pdf_pages.close()


if __name__ == "__main__":
	main()
