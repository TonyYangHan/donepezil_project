import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
	y_min = max(min(means) - 0.2 * span, 0)
	y_max = y_top + 0.3 * span + 0.08 * span * len(pairwise_p)
	ax.set_ylim(y_min, y_max)

	x_pos_map = {cond: pos for cond, pos in zip(cond_order, x)}
	annotate_pairs(ax, x_pos_map, y_top + 0.05 * span, pairwise_p, 0.07 * span, hide_ns)

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
	y_max = y_max + 0.3 * span + 0.08 * span * len(pairwise_p)
	ax.set_ylim(y_min - 0.1 * span, y_max)

	x_pos_map = {cond: pos for cond, pos in zip(cond_order, positions)}
	annotate_pairs(ax, x_pos_map, flat_all.max() + 0.05 * span, pairwise_p, 0.07 * span, hide_ns)

	os.makedirs(outdir, exist_ok=True)
	fig.tight_layout()
	fname = f"violin_all_{label_txt}_{test_name}.png".replace(" ", "_")
	fig.savefig(os.path.join(outdir, fname), dpi=300)
	if pdf_pages is not None:
		pdf_pages.savefig(fig)
	plt.close(fig)
