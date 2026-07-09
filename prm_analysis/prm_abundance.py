"""Compute PRM abundance summaries across multiple directories.

For each input directory, the script loads TIFF images, computes per-image means
using collect_image_means, runs pairwise Welch t-tests across directories, and
plots a grouped bar chart with significance annotations.
"""

from __future__ import annotations

import argparse, sys
import itertools
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind

from calculate_ratios_gp_v2 import collect_image_means
from visualizations import plot_bars_all


TIFF_EXTS = {".tif", ".tiff"}


def find_tiff_files(folder: Path, recursive: bool = False) -> List[Path]:
	"""Return sorted TIFF files in a directory."""

	if recursive:
		candidates = [p for p in folder.rglob("*") if p.is_file()]
	else:
		candidates = [p for p in folder.iterdir() if p.is_file()]
	return sorted([p for p in candidates if p.suffix.lower() in TIFF_EXTS])


def load_images(files: Sequence[Path]) -> List[np.ndarray]:
	"""Load TIFF files into float arrays."""

	images: List[np.ndarray] = []
	for file_path in files:
		arr = tifffile.imread(file_path)
		images.append(np.asarray(arr, dtype=float))
	return images


def default_labels(dirs: Sequence[Path]) -> List[str]:
	"""Create readable labels from folder names, disambiguating duplicates."""

	counts: Dict[str, int] = {}
	labels: List[str] = []
	for path in dirs:
		base = path.name or str(path)
		counts[base] = counts.get(base, 0) + 1
		label = base if counts[base] == 1 else f"{base}_{counts[base]}"
		labels.append(label)
	return labels


def pairwise_ttests(metric_values: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], Tuple[float, float, int, int]]:
	"""Run Welch t-tests for every unique pair of conditions."""

	results: Dict[Tuple[str, str], Tuple[float, float, int, int]] = {}
	conds = list(metric_values.keys())
	for c1, c2 in itertools.combinations(conds, 2):
		v1 = np.asarray(metric_values[c1], dtype=float)
		v2 = np.asarray(metric_values[c2], dtype=float)
		n1, n2 = int(v1.size), int(v2.size)
		if n1 == 0 or n2 == 0:
			t_stat, p_val = np.nan, np.nan
		else:
			t_stat, p_val = ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
		results[(c1, c2)] = (float(t_stat), float(p_val), n1, n2)
	return results


def print_summary(label: str, values: np.ndarray) -> None:
	"""Print per-directory descriptive statistics."""

	vals = np.asarray(values, dtype=float)
	if vals.size == 0:
		print(f"{label}: n=0")
		return
	print(
		f"{label}: n={vals.size}, mean={np.mean(vals):.6f}, "
		f"std={np.std(vals, ddof=1) if vals.size > 1 else 0.0:.6f}"
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute per-image abundance means, pairwise t-tests, and bar plots across directories."
	)
	parser.add_argument("dirs", nargs="+", type=Path, help="Input directories containing TIFF images.")
	parser.add_argument(
		"--conds", "-c",
		nargs="+",
		default=None,
		help="Optional condition labels, in the same order as input directories.",
	)
	parser.add_argument("--out", "-o", type=Path, default=Path("."), help="Output directory for plots.")
	parser.add_argument("--recursive", "-r", action="store_true", help="Search TIFF files recursively in each directory.")
	parser.add_argument("--hide-ns", action="store_true", help="Hide non-significant pairwise labels in the bar plot.")
	parser.add_argument("--pdf-out", "-p", action="store_true", help="Also save the bar plot into plots.pdf.")
	parser.add_argument("--verbose", "-v", action="store_true", help="Print loaded TIFF file names for each directory.")
	args = parser.parse_args()

	if len(args.dirs) < 2:
		raise ValueError("Provide at least two input directories for pairwise tests.")

	for dir_path in args.dirs:
		if not dir_path.is_dir():
			raise NotADirectoryError(f"Directory does not exist: {dir_path}")

	labels = args.conds if args.conds is not None else default_labels(args.dirs)
	if len(labels) != len(args.dirs):
		raise ValueError("--labels/--conds must have the same number of items as input directories.")

	if len(set(labels)) != len(labels):
		raise ValueError("Labels must be unique.")

	os.makedirs(args.out, exist_ok=True)
	pdf_pages = PdfPages(args.out / "plots.pdf") if args.pdf_out else None

	means_by_label: Dict[str, np.ndarray] = {}
	for label, folder in zip(labels, args.dirs):
		tiff_files = find_tiff_files(folder, recursive=args.recursive)
		if not tiff_files:
			raise FileNotFoundError(f"No TIFF files found in {folder}")

		if args.verbose:
			print(f"\n{label} ({folder}):")
			for p in tiff_files:
				print(f"  - {p}")

		image_arrays = load_images(tiff_files)
		image_means = collect_image_means(image_arrays)
		if image_means.size == 0:
			raise ValueError(f"No valid per-image means could be computed for {folder}")
		means_by_label[label] = image_means

	print("Per-directory image-mean summary:")
	for label in labels:
		print_summary(label, means_by_label[label])

	results = pairwise_ttests(means_by_label)
	print("\nPairwise Welch t-tests (image means):")
	for (c1, c2), (t_stat, p_val, n1, n2) in results.items():
		t_txt = f"{t_stat:.5g}" if np.isfinite(t_stat) else "nan"
		p_txt = f"{p_val:.5g}" if np.isfinite(p_val) else "nan"
		print(f"{c1} vs {c2} -> t={t_txt}, p={p_txt} | n1={n1}, n2={n2}")

	pairwise_p = {pair: vals[1] for pair, vals in results.items()}
	plot_bars_all(
		metric_values=means_by_label,
		cond_order=labels,
		pairwise_p=pairwise_p,
		outdir=str(args.out),
		test_name="Welch t-test",
		ylabel="PRM abundance (image mean)",
		label_prefix="prm_abundance",
		hide_ns=args.hide_ns,
		pdf_pages=pdf_pages,
		save_png=True,
	)

	if pdf_pages is not None:
		pdf_pages.close()

	print(f"\nSaved bar plot to: {args.out}")


if __name__ == "__main__":
	main()