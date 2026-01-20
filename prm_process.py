"""Normalize a folder of TIFF images to a reference image.

The script searches for a reference image whose filename *contains* a
user-provided substring and *does not contain* another substring. All images
in the folder are then normalized by pixel-wise division against that
reference, and the outputs are written to a ``normalized_images`` subfolder.

Example:
	python prm_process.py \
		--input-dir /path/to/folder \
		--include-key baseline \
		--exclude-key tmp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile


def find_tiff_files(folder: Path) -> List[Path]:
	"""Return all .tif/.tiff files in the folder (non-recursive), sorted."""

	candidates = sorted(
		[p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"}]
	)
	return candidates


def pick_reference(files: Iterable[Path], include_key: str, exclude_key: str) -> Path:
	"""Pick the first file containing include_key and not containing exclude_key."""

	for file in files:
		name = file.name
		if include_key in name and (not exclude_key or exclude_key not in name):
			return file
	raise FileNotFoundError(
		f"No TIFF file found matching include_key='{include_key}' and "
		f"exclude_key='{exclude_key}'."
	)


def normalize_images(
	input_dir: Path,
	include_key: str,
	exclude_key: str,
	output_dir: Path,
	thresh: int = 64,
	clip: float = 2.0,
	verbose: bool = False,
) -> None:
	"""Normalize all TIFF images to the reference image by pixel-wise division."""

	tiff_files = find_tiff_files(input_dir)
	if not tiff_files:
		raise FileNotFoundError(f"No .tif/.tiff files found in {input_dir}")

	ref_file = pick_reference(tiff_files, include_key, exclude_key)
	ref_image = tifffile.imread(ref_file).astype(np.float32)

	output_dir.mkdir(exist_ok=True)

	for tif_path in tiff_files:
		image = tifffile.imread(tif_path).astype(np.float32)

		if image.shape != ref_image.shape:
			raise ValueError(
				f"Shape mismatch between {tif_path.name} {image.shape} and "
				f"reference {ref_file.name} {ref_image.shape}."
			)

		zero_mask = (ref_image < thresh) | (image < thresh)
		normalized = np.zeros_like(image, dtype=np.float32)
		np.divide(image, ref_image, out=normalized, where=~zero_mask)
		normalized[zero_mask] = 0.0
		np.clip(normalized, a_min=None, a_max=clip, out=normalized)

		out_path = output_dir / tif_path.name
		tifffile.imwrite(out_path, normalized, dtype=np.float32)

		if verbose:
			print(f"Saved: {out_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input-dir", "-d",
		type=Path,
		required=True,
		help="Folder containing .tif/.tiff images (non-recursive).",
	)
	parser.add_argument(
		"--include-key", "-i",
		required=True,
		help="Substring that must be present in the reference filename.",
	)
	parser.add_argument(
		"--exclude-key", "-e",
		type=str,
		default="",
		help="Substring that must NOT be present in the reference filename.",
	)
	parser.add_argument(
		"--verbose", "-v",
		action="store_true",
		help="Print a line for every saved file.",
	)
	parser.add_argument(
		"--thresh", "-t",
		type=int,
		default=64,
		help="Threshold below which pixels are considered zero for normalization.",
	)
	parser.add_argument(
		"--clip",
		type=float,
		default=2.0,
		help="Clip normalized pixel intensities to this maximum value.",
	)

	args = parser.parse_args()

	input_dir: Path = args.input_dir
	if not input_dir.is_dir():
		raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

	output_dir = input_dir / "normalized_images"
	print(f"Processing folder: {input_dir}")

	normalize_images(
		input_dir=input_dir,
		include_key=args.include_key,
		exclude_key=args.exclude_key,
		output_dir=output_dir,
		thresh=args.thresh,
		clip=args.clip,
		verbose=args.verbose,
	)
