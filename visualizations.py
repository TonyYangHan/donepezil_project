import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
from matplotlib import cm

# Functions for hyperspectral image clustering plots


def save_ratio_image(ratio, root_path, filename, low_pct=2, high_pct=98, gamma=1.0, cbar='turbo'):
	ratio_pos = ratio[ratio > 0]
	if ratio_pos.size == 0:
		return
	p_low, p_high = np.percentile(ratio_pos, (low_pct, high_pct))
	norm = mcolors.PowerNorm(gamma=gamma, vmin=p_low, vmax=p_high, clip=True)
	plt.figure(figsize=(6, 6))
	plt.imshow(ratio, cmap=cbar, norm=norm)
	cb = plt.colorbar(label='Ratio')
	cb.set_ticks([p_low, (p_low + p_high) / 2, p_high])
	plt.title(f'Ratio ({low_pct}-{high_pct}th percent)')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(os.path.join(root_path, filename), dpi=300)
	plt.close()


def save_mask_overlay_gray(image, mask, out_path, title="Mask Overlay", mask_cmap="Reds"):
	plt.figure(figsize=(6, 6))
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


def plot_bars_all(metric_values, cond_order, pairwise_p, outdir, test_name, ylabel, label_prefix=None, hide_ns=False, pdf_pages=None):
	means, ses = [], []
	for cond in cond_order:
		vals = np.asarray(metric_values[cond])
		means.append(np.mean(vals) if vals.size else 0.0)
		ses.append(standard_error(vals))

	x = np.arange(len(cond_order))
	width = max(3, max(6, len(cond_order) * 1.5) - 3)
	fig, ax = plt.subplots(figsize=(width, 8))
	ax.bar(x, means, yerr=ses, capsize=6, alpha=0.85)
	ax.set_xticks(x)
	ax.set_xticklabels(cond_order, rotation=30, ha="right")
	ax.set_ylabel(ylabel)
	label_txt = label_prefix or test_name
	ax.set_title(f"{label_txt} ({test_name})")

	span = max(max(means) + max(ses, default=0) - min(means), 1e-6)
	y_top = max(means) + max(ses, default=0)
	y_min = max(min(means) - 0.05 * span, 0)
	y_max = y_top + 0.12 * span + 0.05 * span * len(pairwise_p)
	ax.set_ylim(y_min, y_max)

	x_pos_map = {cond: pos for cond, pos in zip(cond_order, x)}
	annotate_pairs(ax, x_pos_map, y_top + 0.02 * span, pairwise_p, 0.05 * span, hide_ns)

	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fname = f"bar_all_{label_txt}_{test_name}.png".replace(" ", "_")
	fig.savefig(os.path.join(outdir, fname), dpi=300)
	if pdf_pages is not None:
		pdf_pages.savefig(fig)
	plt.close(fig)


def plot_violins_all(metric_values, cond_order, pairwise_p, outdir, test_name, ylabel, label_prefix=None, hide_ns=False, pdf_pages=None):
	data = []
	for c in cond_order:
		vals = np.asarray(metric_values[c])
		data.append(vals if vals.size else np.array([0.0]))
	positions = np.arange(len(cond_order))

	width = max(3, max(6, len(cond_order) * 1.5) - 3)
	fig, ax = plt.subplots(figsize=(width, 8))
	ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
	ax.set_xticks(positions)
	ax.set_xticklabels(cond_order, rotation=30, ha="right")
	ax.set_ylabel(ylabel)
	label_txt = label_prefix or test_name
	ax.set_title(f"{label_txt} ({test_name})")

	flat_all = np.concatenate(data) if any(arr.size for arr in data) else np.array([0.0])
	y_min = flat_all.min() if flat_all.size else 0.0
	y_max = flat_all.max() if flat_all.size else 1.0
	span = max(y_max - y_min, 1e-6)
	y_max = y_max + 0.18 * span + 0.05 * span * len(pairwise_p)
	ax.set_ylim(y_min - 0.05 * span, y_max)

	x_pos_map = {cond: pos for cond, pos in zip(cond_order, positions)}
	annotate_pairs(ax, x_pos_map, flat_all.max() + 0.02 * span, pairwise_p, 0.05 * span, hide_ns)

	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fname = f"violin_all_{label_txt}_{test_name}.png".replace(" ", "_")
	fig.savefig(os.path.join(outdir, fname), dpi=300)
	if pdf_pages is not None:
		pdf_pages.savefig(fig)
	plt.close(fig)


def plot_region_scatter_3d(scatter_map, outdir, region):
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection="3d")
	colors = plt.cm.tab10(np.linspace(0, 1, len(scatter_map)))
	vals_arr = np.array(list(scatter_map.values())) if scatter_map else np.zeros((1, 3))
	z_span = vals_arr[:, 2].max() - vals_arr[:, 2].min() if vals_arr.size else 0.0
	z_offset = max(0.02 * z_span, 0.02)
	for (cond, vals), color in zip(scatter_map.items(), colors):
		redox, protein_turn, lipid_turn = vals
		ax.scatter(redox, protein_turn, lipid_turn, color=color, label=cond, s=60)
		ax.text(redox, protein_turn, lipid_turn + 0.2 * z_offset, cond, fontsize=10)
	ax.set_xlabel("Median redox ratio")
	ax.set_ylabel("Median protein turnover ratio")
	ax.set_zlabel("Median lipid turnover ratio")
	ax.set_title(f"3D medians ({region})")
	ax.legend(loc="upper left", bbox_to_anchor=(0.9, 1))
	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fig.savefig(os.path.join(outdir, f"scatter_3d_{region}.png"), dpi=300)
	plt.close(fig)


def plot_cluster_umap(embedding, cluster_labels, n_clusters, colors, output_folder):
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

	plt.figure(figsize=(8, 8))
	for i in range(n_clusters):
		mask = filtered_labels == i
		if np.any(mask):
			plt.scatter(filtered_embedding[mask, 0], filtered_embedding[mask, 1], c=[colors[i]], label=f"Cluster {i+1}", alpha=0.6, s=5)
			plt.text(np.mean(filtered_embedding[mask, 0]), np.mean(filtered_embedding[mask, 1]), f"{i+1}", fontsize=12, ha="center", va="center")
	plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2"); plt.xticks([]); plt.yticks([])
	plt.legend(prop={"size": 10}, loc="best")
	plt.tight_layout()
	plt.savefig(os.path.join(output_folder, "clusters_umap.tiff"), dpi=300, bbox_inches="tight")
	plt.close()


def plot_cluster_spectra(spectra, cluster_labels, n_clusters, wavenumbers, colors, output_folder):
	if spectra.shape[1] == 0:
		return
	plt.figure(figsize=(6, 10))
	for i in range(n_clusters):
		mask = cluster_labels == i
		if np.any(mask):
			mean = np.mean(spectra[:, mask], axis=1)
			std = np.std(spectra[:, mask], axis=1)
			plt.plot(wavenumbers, mean + i * 0.5, color=colors[i], label=f"Cluster {i+1}", alpha=0.5)
			plt.fill_between(wavenumbers, mean + i * 0.5 - std, mean + i * 0.5 + std, color=colors[i], alpha=0.2)
	plt.xlabel(r"Wavenumber (cm$^{-1}$)"); plt.ylabel("Normalized Intensity")
	plt.legend(prop={"size": 10}); plt.tight_layout()
	plt.grid()
	plt.savefig(os.path.join(output_folder, "cluster_spectra.tiff"), dpi=300)
	plt.close()


def map_clusters(cluster_labels, img_shape, indices, n_clusters, colors, output_folder, tag: str = "combined"):
	label_map = np.full(img_shape[1:], -1, dtype=np.int32)  # -1 = background
	if cluster_labels.size:
		label_map[indices] = cluster_labels
	rgb_map = np.zeros((*img_shape[1:], 3), dtype=np.float32)
	for i in range(n_clusters):
		rgb_map[label_map == i] = colors[i]
	fname = f"clusters_map_{tag}.tiff"
	tifffile.imwrite(os.path.join(output_folder, fname), (rgb_map * 255).astype(np.uint8))


def plot_cluster_composition(cluster_stats, n_clusters, output_folder, tag: str = "combined"):
	plt.figure(figsize=(8,6))
	x = np.arange(n_clusters)
	plt.bar(x, cluster_stats["ratio"], color="royalblue", alpha=0.7)
	plt.xticks(x, [f"Cluster {i+1}" for i in x], fontsize=10, rotation=45)
	plt.ylabel("Fraction of Pixels in Cluster")
	plt.tight_layout()
	fname = f"cluster_composition_{tag}.tiff"
	plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches="tight")
	plt.close()


def plot_cluster_composition_by_condition(condition_ratios, output_folder, tag: str = "by_condition"):
	if not condition_ratios:
		return
	conditions = list(condition_ratios.keys())
	n_clusters = len(next(iter(condition_ratios.values())))
	x = np.arange(n_clusters)
	n_cond = len(conditions)
	bar_width = 0.8 / max(n_cond, 1)
	shift = (n_cond - 1) * bar_width / 2.0
	cmap = cm.get_cmap("tab10")
	plt.figure(figsize=(10, 6))
	for j, cond in enumerate(conditions):
		ratios = np.asarray(condition_ratios[cond])
		plt.bar(x + j * bar_width - shift, ratios, width=bar_width, label=cond, color=cmap(j % 10), alpha=0.85)
	plt.xticks(x, [f"Cluster {i+1}" for i in x], rotation=45, ha="right")
	plt.ylabel("Fraction of Pixels in Cluster")
	plt.ylim(0, 1)
	plt.legend()
	plt.tight_layout()
	fname = f"cluster_composition_{tag}.tiff"
	plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches="tight")
	plt.close()
