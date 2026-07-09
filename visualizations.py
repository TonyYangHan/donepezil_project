import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
from matplotlib import cm
from scipy.stats import pearsonr

PALETTE5 = [
	"#1f77b4",
	"#D55E00",
	"#009E73",
	"#E69F00",
	"#CC79A7",
]

MARKERS5 = ["o", "^", "s", "p", "*"]

# Functions for hyperspectral image clustering plots


def save_ratio_image(ratio, root_path, filename, low_pct=2, high_pct=98, gamma=1.0, cbar='turbo'):
	ratio_pos = ratio[ratio > 0]
	if ratio_pos.size == 0:
		return
	p_low, p_high = np.percentile(ratio_pos, (low_pct, high_pct))
	norm = mcolors.PowerNorm(gamma=gamma, vmin=p_low, vmax=p_high, clip=True)
	plt.figure(figsize=(7, 7))
	plt.imshow(ratio, cmap=cbar, norm=norm)
	cb = plt.colorbar(label='Ratio')
	cb.ax.tick_params(labelsize=20)
	cb.set_label('Ratio', fontsize=24)
	cb.set_ticks([p_low, (p_low + p_high) / 2, p_high])
	plt.title(f'Ratio ({low_pct}-{high_pct}th percent)')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(os.path.join(root_path, filename), dpi=300)
	plt.close()


def save_mask_overlay_gray(image, mask, out_path, title="Mask Overlay", mask_cmap="Reds"):
	plt.figure(figsize=(7, 7))
	plt.imshow(image, cmap="gray")
	plt.imshow(mask, cmap=mask_cmap, alpha=0.6)
	plt.title(title)
	plt.axis('off')
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()


def significance_label(p_val):
	if p_val is None or np.isnan(p_val):
		return "n/a"
	if p_val < 1e-3:
		return "***"
	if p_val < 1e-2:
		return "**"
	if p_val < 5e-2:
		return "*"
	return "n.s."


def standard_error(values):
	values = np.asarray(values)
	if values.size <= 1:
		return 0.0
	return np.std(values, ddof=1) / np.sqrt(values.size)


def annotate_pairs(ax, x_positions, top_y, pvals, step, hide_ns=False):
	for idx, ((c1, c2), p_val) in enumerate(pvals.items()):
		if hide_ns and (p_val is None or np.isnan(p_val) or p_val >= 0.05):
			continue
		y = top_y + step * idx
		x1, x2 = x_positions[c1], x_positions[c2]
		ax.plot([x1, x2], [y, y], color="black", linewidth=1.2)
		ax.text((x1 + x2) / 2, y + step * 0.15, significance_label(p_val), ha="center", va="bottom", fontsize=10, fontweight="bold")


def plot_bars_all(
	metric_values,
	cond_order,
	pairwise_p,
	outdir,
	test_name,
	ylabel,
	label_prefix=None,
	hide_ns=False,
	pdf_pages=None,
	save_png=True,
	scatter_values=None,
	bar_width=0.9,
):
	means, ses, all_vals_list = [], [], []
	scatter_source = scatter_values if scatter_values is not None else metric_values
	for cond in cond_order:
		vals = np.asarray(metric_values[cond])
		scatter_vals = np.asarray(scatter_source.get(cond, []))
		if scatter_vals.size:
			all_vals_list.append(scatter_vals)
		means.append(np.mean(vals) if vals.size else 0.0)
		ses.append(standard_error(vals))

	all_vals = np.concatenate(all_vals_list) if all_vals_list else np.array([])

	color_map = {cond: PALETTE5[idx % len(PALETTE5)] for idx, cond in enumerate(cond_order)}

	x = np.arange(len(cond_order))
	width = max(6.0, len(cond_order) * 2.4)
	fig, ax = plt.subplots(figsize=(width, 8))
	bar_colors = [color_map[c] for c in cond_order]
	ax.bar(x, means, yerr=ses, capsize=6, alpha=0.85, width=bar_width, color=bar_colors)

	for idx, cond in enumerate(cond_order):
		vals = np.asarray(scatter_source.get(cond, []))
		if not vals.size:
			continue
		n_points = vals.size
		if n_points == 1:
			offsets = np.array([0.0])
		else:
			offsets = np.linspace(-0.12, 0.12, n_points)
		x_positions = x[idx] + offsets
		marker = MARKERS5[idx % len(MARKERS5)]
		ax.scatter(x_positions, vals, color="black", marker=marker, edgecolors="black", linewidths=0.5, zorder=3)
	ax.set_xticks(x)
	ax.set_xticklabels(cond_order, rotation=30, ha="right", fontsize=20)
	ax.set_ylabel(ylabel, fontsize=24)
	ax.tick_params(axis="both", labelsize=20)
	label_txt = label_prefix or test_name
	ax.set_title(f"{label_txt} ({test_name})")

	data_min = all_vals.min() if all_vals.size else (min(means) if means else 0.0)
	y_top = max(max(means) + max(ses, default=0), all_vals.max() if all_vals.size else max(means))
	span = max(y_top - data_min, 1e-6)
	y_min = data_min - 0.05 * span
	y_max = y_top + 0.08 * span + 0.04 * span * len(pairwise_p)
	ax.set_ylim(y_min, y_max)

	x_pos_map = {cond: pos for cond, pos in zip(cond_order, x)}
	annotate_pairs(ax, x_pos_map, y_top + 0.02 * span, pairwise_p, 0.04 * span, hide_ns)

	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fname = f"bar_all_{label_txt}_{test_name}.png".replace(" ", "_")
	png_path = os.path.join(outdir, fname)
	svg_path = os.path.join(outdir, fname.rsplit(".", 1)[0] + ".svg")
	if save_png:
		fig.savefig(png_path, dpi=300)
	if pdf_pages is not None:
		fig.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)


def plot_violins_all(metric_values, cond_order, pairwise_p, outdir, test_name, ylabel, label_prefix=None, hide_ns=False, pdf_pages=None, save_png=True):
	data = []
	for c in cond_order:
		vals = np.asarray(metric_values[c])
		data.append(vals if vals.size else np.array([0.0]))
	positions = np.arange(len(cond_order))

	width = max(6.0, len(cond_order) * 2.4)
	fig, ax = plt.subplots(figsize=(width, 8))
	ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
	ax.set_xticks(positions)
	ax.set_xticklabels(cond_order, rotation=30, ha="right", fontsize=20)
	ax.set_ylabel(ylabel, fontsize=24)
	ax.tick_params(axis="both", labelsize=20)
	label_txt = label_prefix or test_name
	ax.set_title(f"{label_txt} ({test_name})")

	flat_all = np.concatenate(data) if any(arr.size for arr in data) else np.array([0.0])
	y_min = flat_all.min() if flat_all.size else 0.0
	y_max = flat_all.max() if flat_all.size else 1.0
	span = max(y_max - y_min, 1e-6)
	y_max = y_max + 0.12 * span + 0.04 * span * len(pairwise_p)
	ax.set_ylim(y_min - 0.02 * span, y_max)

	x_pos_map = {cond: pos for cond, pos in zip(cond_order, positions)}
	annotate_pairs(ax, x_pos_map, flat_all.max() + 0.02 * span, pairwise_p, 0.04 * span, hide_ns)

	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fname = f"violin_all_{label_txt}_{test_name}.png".replace(" ", "_")
	png_path = os.path.join(outdir, fname)
	svg_path = os.path.join(outdir, fname.rsplit(".", 1)[0] + ".svg")
	if save_png:
		fig.savefig(png_path, dpi=300)
	if pdf_pages is not None:
		fig.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)


def plot_region_scatter_3d(scatter_map, outdir, region):
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection="3d")
	colors = plt.cm.tab10(np.linspace(0, 1, len(scatter_map)))
	vals_arr = np.array(list(scatter_map.values())) if scatter_map else np.zeros((1, 3))
	z_span = vals_arr[:, 2].max() - vals_arr[:, 2].min() if vals_arr.size else 0.0
	z_offset = max(0.02 * z_span, 0.02)
	for (cond, vals), color in zip(scatter_map.items(), colors):
		redox, protein_turn, lipid_turn = vals
		ax.scatter(redox, protein_turn, lipid_turn, color=color, label=cond, s=60)
		ax.text(redox, protein_turn, lipid_turn + 0.2 * z_offset, cond, fontsize=10)
	ax.set_xlabel("Median redox ratio", fontsize=24)
	ax.set_ylabel("Median protein turnover ratio", fontsize=24)
	ax.set_zlabel("Median lipid turnover ratio", fontsize=24)
	ax.tick_params(axis="both", labelsize=20)
	ax.set_title(f"3D medians ({region})")
	ax.legend(loc="upper left", bbox_to_anchor=(0.9, 1), prop={"size": 16})
	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fig.savefig(os.path.join(outdir, f"scatter_3d_{region}.png"), dpi=300)
	plt.close(fig)


def plot_cluster_umap(embedding, cluster_labels, n_clusters, colors, output_folder, pdf_pages=None, save_image=True):
	# Filter out data points beyond 0.01 and 99.99 percentile range
	if embedding.shape[0] == 0:
		return
	low_pct, high_pct = 0.2, 99.8
	x_min, x_max = np.percentile(embedding[:, 0], [low_pct, high_pct])
	y_min, y_max = np.percentile(embedding[:, 1], [low_pct, high_pct])
	mask_range = (embedding[:, 0] >= x_min) & (embedding[:, 0] <= x_max) & \
				 (embedding[:, 1] >= y_min) & (embedding[:, 1] <= y_max)
	filtered_embedding = embedding[mask_range]
	filtered_labels = cluster_labels[mask_range]

	fig = plt.figure(figsize=(10, 8))
	for i in range(n_clusters):
		mask = filtered_labels == i
		if np.any(mask):
			plt.scatter(filtered_embedding[mask, 0], filtered_embedding[mask, 1], c=[colors[i]], label=f"Cluster {i+1}", alpha=0.6, s=5)
			plt.text(np.mean(filtered_embedding[mask, 0]), np.mean(filtered_embedding[mask, 1]), f"{i+1}", fontsize=12, ha="center", va="center")
	plt.xlabel("UMAP 1", fontsize=24); plt.ylabel("UMAP 2", fontsize=24); plt.xticks([]); plt.yticks([])
	plt.legend(prop={"size": 16}, loc="best")
	plt.tight_layout()
	if save_image:
		plt.savefig(os.path.join(output_folder, "clusters_umap.tiff"), dpi=300, bbox_inches="tight")
	if pdf_pages is not None:
		svg_path = os.path.join(output_folder, "clusters_umap.svg")
		plt.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)


def plot_cluster_spectra(spectra, cluster_labels, n_clusters, wavenumbers, colors, output_folder, pdf_pages=None, save_image=True):
	if spectra.shape[1] == 0:
		return
	fig = plt.figure(figsize=(7, 12))
	for i in range(n_clusters):
		mask = cluster_labels == i
		if np.any(mask):
			mean = np.mean(spectra[:, mask], axis=1)
			std = np.std(spectra[:, mask], axis=1)
			plt.plot(wavenumbers, mean + i * 0.5, color=colors[i], label=f"Cluster {i+1}", alpha=0.5)
			plt.fill_between(wavenumbers, mean + i * 0.5 - std, mean + i * 0.5 + std, color=colors[i], alpha=0.2)
	plt.xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=24); plt.ylabel("Normalized Intensity", fontsize=24)
	plt.tick_params(axis="both", labelsize=20)
	plt.legend(prop={"size": 16}); plt.tight_layout()
	plt.grid()
	if save_image:
		plt.savefig(os.path.join(output_folder, "cluster_spectra.tiff"), dpi=300)
	if pdf_pages is not None:
		svg_path = os.path.join(output_folder, "cluster_spectra.svg")
		plt.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)


def map_clusters(cluster_labels, img_shape, indices, n_clusters, colors, output_folder, tag: str = "combined", pdf_pages=None):
	os.makedirs(output_folder, exist_ok=True)
	label_map = np.full(img_shape[1:], -1, dtype=np.int32)  # -1 = background
	if cluster_labels.size:
		label_map[indices] = cluster_labels
	rgb_map = np.zeros((*img_shape[1:], 3), dtype=np.float32)
	for i in range(n_clusters):
		rgb_map[label_map == i] = colors[i]
	fname = f"clusters_map_{tag}.tiff"
	tifffile.imwrite(os.path.join(output_folder, fname), (rgb_map * 255).astype(np.uint8))

	# Optional PDF export for quick review
	if pdf_pages is not None:
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.imshow(rgb_map)
		ax.set_title(f"Cluster map ({tag})")
		ax.axis("off")
		fig.tight_layout()
		svg_path = os.path.join(output_folder, f"clusters_map_{tag}.svg")
		fig.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
		plt.close(fig)


def plot_cluster_composition(cluster_stats, n_clusters, output_folder, tag: str = "combined", pdf_pages=None):
	fig = plt.figure(figsize=(10,7))
	x = np.arange(n_clusters)
	plt.bar(x, cluster_stats["ratio"], color="royalblue", alpha=0.7, width=0.9)
	plt.xticks(x, [f"Cluster {i+1}" for i in x], fontsize=20, rotation=45)
	plt.ylabel("Fraction of Pixels in Cluster", fontsize=24)
	plt.tick_params(axis="both", labelsize=20)
	plt.tight_layout()
	fname = f"cluster_composition_{tag}.tiff"
	plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches="tight")
	if pdf_pages is not None:
		svg_path = os.path.join(output_folder, f"cluster_composition_{tag}.svg")
		fig.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)


def plot_cluster_composition_by_condition(condition_ratios, output_folder, tag: str = "by_condition", colors=None, pdf_pages=None, save_image=True, condition_cluster_ratios=None):
	if not condition_ratios:
		return
	conditions = list(condition_ratios.keys())
	n_clusters = len(next(iter(condition_ratios.values())))
	x = np.arange(len(conditions))
	# Use provided color palette when available to stay consistent with UMAP/spectra plots
	palette = colors if colors is not None else cm.get_cmap("tab10")(np.linspace(0, 1, max(n_clusters, 1)))
	fig = plt.figure(figsize=(12, 7))
	bar_width = 0.9 / max(n_clusters, 1)
	shift = (n_clusters - 1) * bar_width / 2.0
	for k in range(n_clusters):
		color = palette[k % len(palette)] if len(np.atleast_1d(palette)) else None
		if condition_cluster_ratios is not None:
			means = []
			ses = []
			for cond in conditions:
				values = np.asarray(condition_cluster_ratios.get(cond, [[]] * n_clusters)[k], dtype=float)
				values = values[np.isfinite(values)]
				means.append(float(values.mean()) if values.size else 0.0)
				ses.append(standard_error(values) if values.size else 0.0)
			heights = np.asarray(means, dtype=float)
		else:
			heights = np.array([np.asarray(condition_ratios[c])[k] if len(condition_ratios[c]) > k else 0.0 for c in conditions], dtype=float)
		plt.bar(x + k * bar_width - shift, heights, width=bar_width, label=f"Cluster {k+1}", color=color, alpha=0.85)

		if condition_cluster_ratios is not None:
			for idx, cond in enumerate(conditions):
				values = np.asarray(condition_cluster_ratios.get(cond, [[]] * n_clusters)[k], dtype=float)
				values = values[np.isfinite(values)]
				n_points = values.size
				x_center = x[idx] + k * bar_width - shift
				jitter = bar_width * 0.35
				if n_points <= 1:
					offsets = np.array([0.0])
				else:
					offsets = np.linspace(-jitter, jitter, n_points)
				x_positions = x_center + offsets
				ax = plt.gca()
				ax.scatter(x_positions, values, color=color, edgecolors="black", linewidths=0.4, s=45, alpha=0.9, zorder=3)
				ax.errorbar(x_center, heights[idx], yerr=ses[idx], color="black", linewidth=1.2, capsize=4, zorder=4)
	plt.xticks(x, conditions, rotation=45, ha="right", fontsize=20)
	plt.ylabel("Fraction of spectra per condition", fontsize=24)
	plt.tick_params(axis="both", labelsize=20)
	plt.ylim(0, 1)
	plt.legend(title="Cluster", fontsize=16, title_fontsize=16)
	plt.tight_layout()
	fname = f"cluster_composition_{tag}.tiff"
	svg_path = os.path.join(output_folder, f"cluster_composition_{tag}.svg")
	if save_image:
		plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches="tight")
	if pdf_pages is not None:
		fig.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)


def plot_cluster_proportions_by_condition(condition_cluster_ratios, cond_order, output_folder, tag: str = "by_condition_scatter", colors=None, pdf_pages=None, save_image=True):
	if not condition_cluster_ratios or not cond_order:
		return

	n_clusters = len(condition_cluster_ratios[cond_order[0]])
	palette = colors if colors is not None else cm.get_cmap("tab10")(np.linspace(0, 1, max(n_clusters, 1)))
	x = np.arange(len(cond_order))
	width = max(6.0, len(cond_order) * 2.4)

	for k in range(n_clusters):
		fig, ax = plt.subplots(figsize=(width, 7))
		color = palette[k % len(palette)] if len(np.atleast_1d(palette)) else "royalblue"
		for idx, cond in enumerate(cond_order):
			values = np.asarray(condition_cluster_ratios.get(cond, [[]] * n_clusters)[k], dtype=float)
			values = values[np.isfinite(values)]
			n_points = values.size
			if n_points == 0:
				offsets = np.array([0.0])
				mean = 0.0
				se = 0.0
			else:
				offsets = np.array([0.0]) if n_points == 1 else np.linspace(-0.12, 0.12, n_points)
				mean = float(values.mean())
				se = standard_error(values)
			x_positions = x[idx] + offsets
			ax.scatter(x_positions, values, color=color, edgecolors="black", linewidths=0.4, s=55, alpha=0.9, zorder=3)
			ax.errorbar(x[idx], mean, yerr=se, color="black", linewidth=1.4, capsize=5, zorder=4)

		ax.set_xticks(x)
		ax.set_xticklabels(cond_order, rotation=30, ha="right", fontsize=20)
		ax.set_ylabel("Fraction of pixels in cluster", fontsize=24)
		ax.set_ylim(0, 1)
		ax.tick_params(axis="both", labelsize=20)
		ax.set_title(f"Cluster {k+1} proportions by condition")
		ax.grid(alpha=0.25, axis="y")

		os.makedirs(output_folder, exist_ok=True)
		fig.tight_layout()
		fname = f"cluster_{k+1}_proportions_{tag}.png"
		png_path = os.path.join(output_folder, fname)
		svg_path = os.path.join(output_folder, fname.rsplit(".", 1)[0] + ".svg")
		if save_image:
			fig.savefig(png_path, dpi=300, bbox_inches="tight")
		if pdf_pages is not None:
			fig.savefig(svg_path, format="svg")
			pdf_pages.savefig(fig, bbox_inches="tight")
		plt.close(fig)


# Correlation plots for turnover vs droplet metrics (used by zoom6_corr.py and zoom1_corr.py)
def plot_condition_correlation(xs, ys, cond_names, colors, outdir, title, xlabel, ylabel, pdf_pages=None, dedup_labels=True, stat_label: str = "", min_value: float = 0.0):
	xs = np.asarray(xs, dtype=float)
	ys = np.asarray(ys, dtype=float)
	mask = (np.isfinite(xs) & np.isfinite(ys) & (xs >= min_value) & (ys >= min_value))
	os.makedirs(outdir, exist_ok=True)

	if mask.sum() == 0:
		return  # nothing to plot

	xs = xs[mask]
	ys = ys[mask]
	cond_names = np.asarray(cond_names)[mask]
	colors = np.asarray(colors, dtype=object)[mask]

	fig, ax = plt.subplots(figsize=(8, 6))
	seen = set()
	for x_val, y_val, cond, color in zip(xs, ys, cond_names, colors):
		label = None
		if not dedup_labels or cond not in seen:
			label = cond
			seen.add(cond)
		ax.scatter(x_val, y_val, color=color, s=65, alpha=0.9, label=label)

	r_text = "r=n/a"
	p_text = "p=n/a"
	if xs.size >= 2:
		slope, intercept = np.polyfit(xs, ys, 1)
		x_line = np.array([xs.min(), xs.max()])
		y_line = slope * x_line + intercept
		ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.1, label="Linear fit")
		try:
			r_val, p_val = pearsonr(xs, ys)
			r_text = f"r={r_val:.3f}"
			p_text = f"p={p_val:.3g}"
		except Exception:
			pass

	x_lab = xlabel
	y_lab = ylabel
	ax.set_xlabel(x_lab, fontsize=24)
	ax.set_ylabel(y_lab, fontsize=24)
	ax.tick_params(axis="both", labelsize=20)
	ax.set_title(f"{title} ({r_text}, {p_text})")

	# Only draw legend if we actually have labeled artists
	handles, labels = ax.get_legend_handles_labels()
	if labels:
		ax.legend(fontsize=16)

	ax.grid(alpha=0.25)
	fig.tight_layout()
	fname = title.lower().replace(" ", "_").replace("/", "-") + ".png"
	png_path = os.path.join(outdir, fname)
	svg_path = os.path.join(outdir, fname.rsplit(".", 1)[0] + ".svg")
	fig.savefig(png_path, dpi=300)
	if pdf_pages is not None:
		fig.savefig(svg_path, format="svg")
		pdf_pages.savefig(fig, bbox_inches="tight")
	plt.close(fig)
