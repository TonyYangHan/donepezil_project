import os, numpy as np, cv2, argparse, math, itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import re
from scipy.stats import ttest_ind, mannwhitneyu
from concurrent.futures import ThreadPoolExecutor



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
    fad8 = cv2.convertScaleAbs(fad, alpha=255.0 / float(fad.max()))
    _, mask8 = cv2.threshold(fad8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f = fad.astype(np.float32)
    n = nadh.astype(np.float32)

    if use_mask == False:
        mask8 = np.ones_like(f, dtype=np.float32)
    
    if suffix in ["redox", "unsat"]:
        den = f + n
        den[den == 0] = np.finfo(np.float32).eps
        ratio = (f / den) * mask8
    elif suffix in ["protein_turn", "lipid_turn"]:
        n[n == 0] = np.finfo(np.float32).eps # Avoid division by zero
        ratio = (f / n) * mask8
    else:
        raise ValueError("Suffix must be one of 'redox', 'unsat', 'protein_turn', or 'lipid_turn'.")
    
    ratio = np.clip(ratio, 0, 5)  # Ensure no negative values

    if save:
        save_name = get_number(fad_path) + f'_{suffix}_ratio.png'
        save_ratio_image(ratio, root_path, save_name, low_pct, high_pct, gamma, cbar)

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



def run_mannwhitney_pixels(cond1_ratios, cond2_ratios, label="redox"):
    cond1_vals = flatten_and_filter(cond1_ratios)
    cond2_vals = flatten_and_filter(cond2_ratios)
    # Exclude zeros
    cond1_vals = cond1_vals[cond1_vals != 0]
    cond2_vals = cond2_vals[cond2_vals != 0]
    u_stat, p_val = mannwhitneyu(cond1_vals, cond2_vals, alternative='two-sided')
    print(f"Mann-Whitney U test for {label} (all pixels): U={u_stat:.4f}, p={p_val}")
    print(f"cond1 median: {np.median(cond1_vals):.4f}, cond2 median: {np.median(cond2_vals):.4f}")
    print()
    return u_stat, p_val



def run_test_image_mean(cond1_ratios, cond2_ratios, label="redox"):

    cond1_means = [np.mean(r) for r in cond1_ratios if r is not None]
    cond2_means = [np.mean(r) for r in cond2_ratios if r is not None]

    t_stat, p_val = ttest_ind(cond1_means, cond2_means, equal_var=False)
    u_stat, p_val_mw = mannwhitneyu(cond1_means, cond2_means)

    print(f"Student's t-test for {label} (mean pixel values): t={t_stat:.4f}, p={p_val}")
    print()
    print(f"Mann-Whitney U test for {label} (mean pixel values): U={u_stat:.4f}, p={p_val_mw}")

    return t_stat, p_val



def sample_pixels_for_ttest(ratio_list):
    # For each image, sample img_size_per_image pixels (excluding zeros)
    n_images = len(ratio_list)
    samples = []
    for arr in ratio_list:
        vals = arr.ravel()
        vals = vals[vals != 0]
        # img_size_per_image = int(len(vals) / n_images)
        img_size_per_image = 500
        np.random.seed(42)  # For reproducibility
        samples.append(np.random.choice(vals, img_size_per_image, replace=False))

    return np.concatenate(samples)

def repeated_ttest_sampling(cond1_ratios, cond2_ratios, n_repeat=50, label="redox"):
    # Each combined sample will have ~number of pixels in one image
    def single_run(_):
        cond1_sample = sample_pixels_for_ttest(cond1_ratios)
        cond2_sample = sample_pixels_for_ttest(cond2_ratios)
        t_stat, p_val = ttest_ind(cond1_sample, cond2_sample, equal_var=False)
        return p_val
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        p_values = list(executor.map(single_run, range(n_repeat)))
    avg_p = np.mean(p_values)
    print(f"Average p-value over {n_repeat} runs for {label} : {avg_p}")
    return p_values, avg_p



def quantile_perm_energy(cond1, cond2, qs=np.arange(0.05, 1.00, 0.05)):
    """Two-sample test on per-sample quantile vectors using energy distance + exact permutations.
       cond1/cond2: lists of arrays (pixels per image/sample)."""
    def qvec(arr):
        v = np.asarray(arr, float).ravel()
        v = v[v > 0]          # drop zeros
        v = np.log(v)         # log-scale for ratios
        return np.quantile(v, qs)

    A = np.vstack([qvec(a) for a in cond1])
    B = np.vstack([qvec(b) for b in cond2])
    X = np.vstack([A, B])
    labels = np.array([1]*len(A) + [0]*len(B), int)
    nA, N = int(labels.sum()), len(labels)

    def mean_pairwise_dist(U, V):
        D = U[:, None, :] - V[None, :, :]
        return np.sqrt((D*D).sum(-1)).mean()

    def energy_stat(lab):
        Xa, Xb = X[lab == 1], X[lab == 0]
        return 2*mean_pairwise_dist(Xa, Xb) - mean_pairwise_dist(Xa, Xa) - mean_pairwise_dist(Xb, Xb)

    T_obs = energy_stat(labels)

    # exact permutations over sample labels
    b = 0
    Btot = math.comb(N, nA)
    for Aidx in itertools.combinations(range(N), nA):
        lab = np.zeros(N, int); lab[list(Aidx)] = 1
        if energy_stat(lab) >= T_obs:  # one-sided on separation
            b += 1

    p = (b + 1) / (Btot + 1)  # small-sample correction
    return {"p": p, "E": T_obs, "qs": qs, "meanA": A.mean(0), "meanB": B.mean(0)}



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
    parser.add_argument("--ranksum", "-r", help="Run rank-sum (Mann-Whitney U) test", action="store_true")
    parser.add_argument("--ttest", "-t", help="Run repeated t-test", action="store_true")
    parser.add_argument("--q_test", "-q", help="Run quantile test", action="store_true")
    parser.add_argument("--plot", "-p", help="Plot histograms", action="store_true")
    parser.add_argument("--out", "-o", type=str, default=".", help="Root directory for saving outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
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

    # Run Mann-Whitney U tests
    if args.ranksum:
        print("Running Mann-Whitney U tests...")
        u_redox, p_redox = run_mannwhitney_pixels(cond1_redox, cond2_redox, label="redox")
        print()
        u_unsat, p_unsat = run_mannwhitney_pixels(cond1_unsat, cond2_unsat, label="unsat")
        print()
        # Plot violin plots with stats
        plot_violins_with_stats(cond1_redox, cond2_redox, label="redox", cond1=cond1, cond2=cond2,
                                outdir=args.out, p_val=p_redox, test_name="MWU")
        plot_violins_with_stats(cond1_unsat, cond2_unsat, label="unsat", cond1=cond1, cond2=cond2,
                                outdir=args.out, p_val=p_unsat, test_name="MWU")

        if args.d:
            u_protein, p_protein = run_mannwhitney_pixels(cond1_turn_protein, cond2_turn_protein, label="protein_turn")
            print()
            u_lipid, p_lipid = run_mannwhitney_pixels(cond1_turn_lipid, cond2_turn_lipid, label="lipid_turn")
            print()
            plot_violins_with_stats(cond1_turn_protein, cond2_turn_protein, label="protein_turn",
                                    cond1=cond1, cond2=cond2, outdir=args.out, p_val=p_protein, test_name="MWU")
            plot_violins_with_stats(cond1_turn_lipid, cond2_turn_lipid, label="lipid_turn",
                                    cond1=cond1, cond2=cond2, outdir=args.out, p_val=p_lipid, test_name="MWU")

    # Run repeated t-tests
    if args.ttest:
        print("Running repeated t-tests...")

        ttest_pvals_redox, avg_p_redox = repeated_ttest_sampling(cond1_redox, cond2_redox, n_repeat=500, label="redox")
        ttest_pvals_unsat, avg_p_unsat = repeated_ttest_sampling(cond1_unsat, cond2_unsat, n_repeat=500, label="unsat")
        print()

        # Plot violin plots with stats (using average p-value)
        plot_violins_with_stats(cond1_redox, cond2_redox, label="redox", cond1=cond1, cond2=cond2,
                                outdir=args.out, p_val=avg_p_redox, test_name="t-test")
        plot_violins_with_stats(cond1_unsat, cond2_unsat, label="unsat", cond1=cond1, cond2=cond2,
                                outdir=args.out, p_val=avg_p_unsat, test_name="t-test")

        if args.d:
            ttest_pvals_protein, avg_p_protein = repeated_ttest_sampling(cond1_turn_protein, cond2_turn_protein, n_repeat=500, label="protein_turn")
            ttest_pvals_lipid, avg_p_lipid = repeated_ttest_sampling(cond1_turn_lipid, cond2_turn_lipid, n_repeat=500, label="lipid_turn")
            print()
            plot_violins_with_stats(cond1_turn_protein, cond2_turn_protein, cond1=cond1, cond2=cond2,
                                    label="protein_turn", outdir=args.out, p_val=avg_p_protein, test_name="t-test")
            plot_violins_with_stats(cond1_turn_lipid, cond2_turn_lipid, cond1=cond1, cond2=cond2,
                                    label="lipid_turn", outdir=args.out, p_val=avg_p_lipid, test_name="t-test")

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
    
    if args.q_test:
        print("Running quantile tests...")
        res = quantile_perm_energy(cond1_redox, cond2_redox)
        print("p =", res["p"], "Energy =", res["E"])
        print()

        q_unsat = quantile_perm_energy(cond1_unsat, cond2_unsat)
        print(f"Quantile test for unsat: p={q_unsat['p']}, Energy={q_unsat['E']}")
        print()

        if args.d:
            q_protein = quantile_perm_energy(cond1_turn_protein, cond2_turn_protein)
            print(f"Quantile test for protein_turn: p={q_protein['p']}, Energy={q_protein['E']}")
            print()

            q_lipid = quantile_perm_energy(cond1_turn_lipid, cond2_turn_lipid)
            print(f"Quantile test for lipid_turn: p={q_lipid['p']}, Energy={q_lipid['E']}")
            print()

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

