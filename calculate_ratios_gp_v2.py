import os, numpy as np, cv2, argparse, tifffile as tiff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import re
from scipy.stats import ttest_ind, mannwhitneyu



def save_ratio_image(ratio, root_path, filename, low_pct = 2, high_pct = 98, gamma = 1.0, cbar = 'turbo'):
    p_low, p_high = np.percentile(ratio[ratio > 0], (low_pct, high_pct))
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



def calculate_ratio(root_path, fad_path, nadh_path, suffix, save,
                    **kwargs):
    
    low_pct = kwargs.get("low_pct", 2)
    high_pct = kwargs.get("high_pct", 98)
    gamma = kwargs.get("gamma", 1.0)
    cbar = kwargs.get("cbar", 'turbo')
    use_mask = kwargs.get("use_mask", True)
    
    fad = cv2.imread(os.path.join(root_path, fad_path), cv2.IMREAD_UNCHANGED)
    nadh = cv2.imread(os.path.join(root_path, nadh_path), cv2.IMREAD_UNCHANGED)
    fad = np.where(fad == 4095, 0, fad)  # Remove saturated pixels
    nadh = np.where(nadh == 4095, 0, nadh)  # Remove saturated pixels

    sat_mask = (fad == 4095) | (nadh == 4095)  # union of saturated pixels
    fad[sat_mask] = 0
    nadh[sat_mask] = 0

    mask8 = np.ones_like(fad, dtype=np.float32)
    if use_mask:
        fad8 = cv2.convertScaleAbs(fad, alpha=255.0 / float(fad.max()))
        _, mask8 = cv2.threshold(fad8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    f = fad.astype(np.float32)
    n = nadh.astype(np.float32)
    
    if suffix in ["redox", "unsat"]:
        den = f + n
        den[den == 0] = np.finfo(np.float32).eps
        ratio = (f / den) * mask8
    elif suffix in ["protein_turn", "lipid_turn"]:
        n[n == 0] = np.finfo(np.float32).eps # Avoid division by zero
        ratio = (f / n) * mask8 / 2 # CD channel use twice the power compared to CH channel
    else:
        raise ValueError("Suffix must be one of 'redox', 'unsat', 'protein_turn', or 'lipid_turn'.")
    
    low, high = 0, 0.9
    ratio[(ratio < low) | (ratio > high)] = 0


    if save:
        save_name = get_number(fad_path) + f'_{suffix}_ratio.png'
        save_ratio_image(ratio, root_path, save_name, low_pct, high_pct, gamma, cbar)
        tiff.imwrite(os.path.join(root_path, get_number(fad_path) + f'_{suffix}_ratio.tiff'), ratio.astype(np.float32))
    return ratio



def get_number(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def match_files(files, label):
    return {get_number(f): f for f in files if label in f and (f.endswith('.tiff') or f.endswith('.tif'))}



def process_group(root, files, redox_list, unsat_list, protein_turn_list, lipid_turn_list, save=False, **kwargs):
    fad_dict = match_files(files, "fad")
    nadh_dict = match_files(files, "nadh")
    unsat_dict = match_files(files, "787")
    sat_dict = match_files(files, "794")

    # Redox ratios
    for num in fad_dict:
        redox_list.append(calculate_ratio(root, fad_dict[num], nadh_dict[num], "redox", save, **kwargs))
    # Unsat ratios
    for num in unsat_dict:
        unsat_list.append(calculate_ratio(root, unsat_dict[num], sat_dict[num], "unsat", save, **kwargs))

    if protein_turn_list is not None and lipid_turn_list is not None:
        d_pro_dict = match_files(files, "841")
        d_lip_dict = match_files(files, "844")
        pro_dict = match_files(files, "791")
        lip_dict = match_files(files, "797")

        for num in d_pro_dict:
            protein_turn_list.append(calculate_ratio(root, d_pro_dict[num], pro_dict[num], "protein_turn", save, **kwargs))
        for num in d_lip_dict:
            lipid_turn_list.append(calculate_ratio(root, d_lip_dict[num], lip_dict[num], "lipid_turn", save, **kwargs))



def flatten_and_filter(ratio_list):
    # Flatten list of arrays and exclude zeros
    return np.concatenate([r.ravel() for r in ratio_list if r is not None])



def run_test_image_mean(cond1_ratios, cond2_ratios, label="redox"):

    cond1_means = [np.mean(r) for r in cond1_ratios if r is not None]
    cond2_means = [np.mean(r) for r in cond2_ratios if r is not None]

    t_stat, p_val = ttest_ind(cond1_means, cond2_means, equal_var=False)
    u_stat, p_val_mw = mannwhitneyu(cond1_means, cond2_means)

    print(f"Student's t-test for {label} (mean pixel values): t={t_stat:.4f}, p={p_val}")
    print()
    print(f"Mann-Whitney U test for {label} (mean pixel values): U={u_stat:.4f}, p={p_val_mw}")

    return t_stat, p_val


def _block_medians_from_image(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Compute per-block medians for non-overlapping blocks.
    Discard blocks whose sum is 0. Returns a 1D array of medians.
    """
    if arr is None:
        raise ValueError("Input array is None.")
    a = np.asarray(arr, dtype=float)
    h, w = a.shape[:2]
    if h < block_size or w < block_size:
        raise ValueError("Image is smaller than block size.")
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        raise ValueError("Image is smaller than block size.")
    a = a[: bh * block_size, : bw * block_size]
    # reshape to (bh, block_size, bw, block_size)
    a4 = a.reshape(bh, block_size, bw, block_size)
    # sums and medians per block
    block_sums = a4.sum(axis=(1, 3))            # shape (bh, bw)
    block_meds = np.median(a4, axis=(1, 3))     # shape (bh, bw)
    mask = block_sums != 0
    return block_meds[mask].ravel()


def run_block_median_ttest(cond1_ratios, cond2_ratios, block_size: int = 16, label: str = "redox"):
    """Block-median t-test between two conditions.

    For each image, partition into non-overlapping block_size x block_size blocks,
    discard blocks whose sum is 0, and use the median within each block as its value.
    Concatenate all valid block medians per condition and run Welch's t-test.

    Returns (t_stat, p_val, n_blocks_cond1, n_blocks_cond2). If insufficient data,
    returns (None, None, 0, 0).
    """
    bs = int(block_size)
    c1_vals = []
    c2_vals = []

    for arr in cond1_ratios:
        if arr is None:
            continue
        c1_vals.append(_block_medians_from_image(arr, bs))
    for arr in cond2_ratios:
        if arr is None:
            continue
        c2_vals.append(_block_medians_from_image(arr, bs))

    c1 = np.concatenate(c1_vals) if len(c1_vals) else np.array([], dtype=float)
    c2 = np.concatenate(c2_vals) if len(c2_vals) else np.array([], dtype=float)

    n1, n2 = c1.size, c2.size
    if n1 == 0 or n2 == 0:
        print(f"Block-median t-test for {label}: insufficient data (n1={n1}, n2={n2}).")
        return None, None, n1, n2

    t_stat, p_val = ttest_ind(c1, c2, equal_var=False)
    u_stat, p_val_mw = mannwhitneyu(c1, c2)
    print(f"Block-median t-test for {label} (block {bs}x{bs}): t={t_stat:.4f}, p={p_val} | n1={n1}, n2={n2}")
    print(f"Block-median Mann-Whitney U test for {label} (block {bs}x{bs}): U={u_stat:.4f}, p={p_val_mw}")
    return t_stat, p_val, n1, n2



def plot_histograms(cond1_ratios, cond2_ratios, 
                    cond1 = "cond1", cond2 = "cond2", label="redox", bins=100, outdir="."):
    cond1_vals = flatten_and_filter(cond1_ratios)
    cond2_vals = flatten_and_filter(cond2_ratios)
    cond1_vals = cond1_vals[cond1_vals != 0]
    cond2_vals = cond2_vals[cond2_vals != 0]
    plt.figure(figsize=(8, 5))
    plt.hist(cond1_vals, bins=bins, alpha=0.6, label=cond1, color="tab:blue", density=True)
    plt.hist(cond2_vals, bins=bins, alpha=0.6, label=cond2, color="tab:orange", density=True)
    plt.xlabel(f"{label.capitalize()} ratio")
    plt.ylabel("Density")
    plt.title(f"Distribution of {label} ratios: {cond1} vs {cond2}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_{cond1}_vs_{cond2}_hist.png"), dpi = 300)
    plt.close()



def plot_violins_with_stats(cond1_ratios, cond2_ratios, 
                            cond1="cond1", cond2="cond2", label="redox", outdir=".", p_val=None, test_name=""):
    cond1_vals = flatten_and_filter(cond1_ratios)
    cond2_vals = flatten_and_filter(cond2_ratios)
    cond1_vals = cond1_vals[cond1_vals != 0]
    cond2_vals = cond2_vals[cond2_vals != 0]
    data = [cond1_vals, cond2_vals]

    plt.figure(figsize=(6, 10))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks([1, 2], [cond1, cond2])
    plt.ylabel(f"{label.capitalize()} ratio")
    plt.title(f"Violin plot of {label} ratios: {cond1} vs {cond2}")

    # Add connecting bar and asterisk(s) for significance
    y_max = max(np.max(cond1_vals), np.max(cond2_vals))
    y_min = min(np.min(cond1_vals), np.min(cond2_vals))
    y_range = y_max - y_min
    y_bar = y_max + 0.01 * y_range  # Move bar closer to data

    plt.plot([1, 2], [y_bar, y_bar], color='black', linewidth=1.5)

    # Determine significance stars
    if p_val is not None:
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."
    else:
        sig = "n.s."

    plt.text(1.5, y_bar + 0.01 * y_range, sig,
             ha='center', va='bottom', fontsize=18, fontweight='bold')

    # Optionally add p-value annotation, closer to the bar
    if p_val is not None:
        plt.text(1.5, y_bar + 0.03 * y_range,
                 f"{test_name} p={p_val:.2e}", ha='center', va='bottom', fontsize=15)

    # Leave some room below the lowest data point
    plt.ylim(y_min - 0.07 * y_range, y_bar + 0.07 * y_range)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_{cond1}_vs_{cond2}_violin_stats_{test_name}.png"), dpi=300)
    plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate redox and unsaturation ratios from TIFF files.")
    parser.add_argument("dir_1", type=str, help="condition 1 folder")
    parser.add_argument("dir_2", type=str, help="condition 2 folder")
    parser.add_argument("-d", action="store_true", help="Deuterated samples")
    parser.add_argument("--cond1", type=str, default="cond1", help="Label for condition 1")
    parser.add_argument("--cond2", type=str, default="cond2", help="Label for condition 2")
    parser.add_argument("--mean", "-m", action="store_true", help="Run t-test on image means")
    parser.add_argument("--save", "-s", action="store_true", help="Save ratio images")
    parser.add_argument("--plot", "-p", help="Plot histograms", action="store_true")
    parser.add_argument("--out", "-o", type=str, default=".", help="Root directory for saving outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--block-ttest", "-b", action="store_true", help="Run block-median t-test")
    parser.add_argument("--block-size", type=int, default=16, help="Block size for block-median t-test (default: 16)")
    args = parser.parse_args()

    save = args.save
    os.makedirs(args.out, exist_ok=True)
    cond1, cond2 = args.cond1, args.cond2
    kwargs = {"low_pct": 0, "high_pct": 100, "gamma": 1.0, "cbar": 'turbo', "use_mask": False}
    
    # Collect ratios
    cond1_redox, cond2_redox, cond1_unsat, cond2_unsat = [], [], [], []
    if args.d:
        cond1_turn_protein, cond2_turn_protein, cond1_turn_lipid, cond2_turn_lipid = [], [], [], []
    else:
        cond1_turn_protein, cond2_turn_protein, cond1_turn_lipid, cond2_turn_lipid = None, None, None, None
    dir1_files = os.listdir(args.dir_1)
    dir2_files = os.listdir(args.dir_2)
    process_group(args.dir_1, dir1_files, cond1_redox, cond1_unsat, cond1_turn_protein, cond1_turn_lipid,
                  save, **kwargs)
    process_group(args.dir_2, dir2_files, cond2_redox, cond2_unsat, cond2_turn_protein, cond2_turn_lipid,
                  save, **kwargs)
    print("Processing complete.")

    if args.verbose:
        print(f"Found {len(cond1_redox)} cond1 redox images and {len(cond2_redox)} cond2 redox images.")
        print(f"Found {len(cond1_unsat)} cond1 unsat images and {len(cond2_unsat)} cond2 unsat images.")
        print(f"Found {len(cond1_turn_protein) if cond1_turn_protein else 0} cond1 protein images and {len(cond2_turn_protein) if cond2_turn_protein else 0} cond2 protein images.")
        print(f"Found {len(cond1_turn_lipid) if cond1_turn_lipid else 0} cond1 lipid images and {len(cond2_turn_lipid) if cond2_turn_lipid else 0} cond2 lipid images.")

    if args.mean:
        print("Running t-tests on image means...")

        run_test_image_mean(cond1_redox, cond2_redox, label="redox")
        print()
        run_test_image_mean(cond1_unsat, cond2_unsat, label="unsat")
        print()

        if args.d:
            run_test_image_mean(cond1_turn_protein, cond2_turn_protein, label="protein_turn")
            print()
            run_test_image_mean(cond1_turn_lipid, cond2_turn_lipid, label="lipid_turn")
            print()

    if args.block_ttest:
        print("Running block-median t-tests...")
        _, p_val_redox, _, _ = run_block_median_ttest(cond1_redox, cond2_redox, block_size=args.block_size, label="redox")
        print()
        _, p_val_unsat, _, _ = run_block_median_ttest(cond1_unsat, cond2_unsat, block_size=args.block_size, label="unsat")
        print()

        plot_violins_with_stats(cond1_redox, cond2_redox, cond1=cond1, cond2=cond2,
                                label="redox", outdir=args.out, p_val=p_val_redox, test_name="Block-median t-test")
        plot_violins_with_stats(cond1_unsat, cond2_unsat, cond1=cond1, cond2=cond2,
                                label="unsat", outdir=args.out, p_val=p_val_unsat, test_name="Block-median t-test")

        if args.d:
            _, p_val_pt, _, _ = run_block_median_ttest(cond1_turn_protein, cond2_turn_protein, block_size=args.block_size, label="protein_turn")
            print()
            _, p_val_lt, _, _ = run_block_median_ttest(cond1_turn_lipid, cond2_turn_lipid, block_size=args.block_size, label="lipid_turn")
            print()

            plot_violins_with_stats(cond1_turn_protein, cond2_turn_protein, cond1=cond1, cond2=cond2,
                                    label="protein_turn", outdir=args.out, p_val=p_val_pt, test_name="Block-median t-test")
            plot_violins_with_stats(cond1_turn_lipid, cond2_turn_lipid, cond1=cond1, cond2=cond2,
                                    label="lipid_turn", outdir=args.out, p_val=p_val_lt, test_name="Block-median t-test")

    # Plot histograms
    if args.plot:
        print("Plotting histograms...")
        plot_histograms(cond1_redox, cond2_redox, cond1=cond1, cond2=cond2,
                        label="redox", bins=100, outdir=args.out)
        plot_histograms(cond1_unsat, cond2_unsat, cond1=cond1, cond2=cond2,
                        label="unsat", bins=100, outdir=args.out)

        if args.d:
            plot_histograms(cond1_turn_protein, cond2_turn_protein, cond1=cond1, cond2=cond2,
                            label="protein_turn", bins=100, outdir=args.out)
            plot_histograms(cond1_turn_lipid, cond2_turn_lipid, cond1=cond1, cond2=cond2,
                           label="lipid_turn", bins=100, outdir=args.out)

