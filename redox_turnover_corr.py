import os, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt, argparse, re, tifffile
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import linregress

def get_number(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def match_files(files, label):
    return {get_number(f): f for f in files if label in f and (f.endswith('.tiff') or f.endswith('.tif'))}

def process_folder(root, files, redox_list: list, protein_turn_list: list, lipid_turn_list: list):
    redox_dict = match_files(files, 'redox_ratio')
    protein_turn_dict = match_files(files, 'protein_turn')
    lipid_turn_dict = match_files(files, 'lipid_turn')

    common_keys = set(redox_dict.keys()).intersection(protein_turn_dict.keys(), lipid_turn_dict.keys())
    for key in common_keys:
        redox_path = os.path.join(root, redox_dict[key])
        protein_turn_path = os.path.join(root, protein_turn_dict[key])
        lipid_turn_path = os.path.join(root, lipid_turn_dict[key])
        redox_img = tifffile.imread(redox_path)
        protein_turn_img = tifffile.imread(protein_turn_path)
        lipid_turn_img = tifffile.imread(lipid_turn_path)
        if redox_img.shape != protein_turn_img.shape or redox_img.shape != lipid_turn_img.shape:
            print(f"Warning: Shape mismatch for {key}, skipping.")
            continue
        redox_list.append(redox_img)
        protein_turn_list.append(protein_turn_img)
        lipid_turn_list.append(lipid_turn_img)


def img_correlation(A_list, B_list, order="K", dtype=np.float64, superpixel=None, min_nonzero_prop: float = 0.5):
    if len(A_list) != len(B_list):
        raise ValueError("A_list and B_list must have the same length.")
    H, W = np.asarray(A_list[0]).shape

    if superpixel is None or int(superpixel) <= 1:
        n_total = len(A_list) * H * W
        x = np.empty(n_total, dtype=dtype)
        y = np.empty(n_total, dtype=dtype)
        off = 0
        for Ai, Bi in zip(A_list, B_list):
            Ai = np.asarray(Ai, dtype=dtype)
            Bi = np.asarray(Bi, dtype=dtype)
            if Ai.shape != (H, W) or Bi.shape != (H, W):
                raise ValueError("All images must share the same 2D shape.")
            xi = np.nan_to_num(Ai.ravel(order=order), nan=0.0, posinf=0.0, neginf=0.0)
            yi = np.nan_to_num(Bi.ravel(order=order), nan=0.0, posinf=0.0, neginf=0.0)
            n = xi.size
            x[off:off+n] = xi
            y[off:off+n] = yi
            off += n
        
        m = np.isfinite(x) & np.isfinite(y)
        m &= ~((x == 0) | (y == 0))  # remove either 0 or both 0
        xr, yr = x[m], y[m]
        lr = linregress(xr, yr)
        print(f"total pixels: {n_total}, valid pixels: {off}, invalid pixels: {n_total - off}")
        return lr, xr, yr

    s = int(superpixel)
    Hc, Wc = (H // s) * s, (W // s) * s
    total_sp = len(A_list) * (Hc // s) * (Wc // s)
    x_list, y_list = [], []
    valid_sp = 0
    dropped_sp = 0

    for Ai, Bi in zip(A_list, B_list):
        Ai = np.asarray(Ai, dtype=dtype)
        Bi = np.asarray(Bi, dtype=dtype)
        if Ai.shape != (H, W) or Bi.shape != (H, W):
            raise ValueError("All images must share the same 2D shape.")
        Ai = np.nan_to_num(Ai[:Hc, :Wc], nan=0.0, posinf=0.0, neginf=0.0)
        Bi = np.nan_to_num(Bi[:Hc, :Wc], nan=0.0, posinf=0.0, neginf=0.0)
        Ablk = Ai.reshape(Hc//s, s, Wc//s, s)
        Bblk = Bi.reshape(Hc//s, s, Wc//s, s)
        Am = np.median(Ablk, axis=(1,3))
        Bm = np.median(Bblk, axis=(1,3))
        # Non-zero proportion filter (like -bnz)
        nonzero_counts = (Ablk != 0).sum(axis=(1,3)) + (Bblk != 0).sum(axis=(1,3))
        block_area = float(2 * s * s)  # counting both A and B contributions
        nz_prop = nonzero_counts.astype(float) / block_area
        # Keep blocks where combined proportion >= threshold
        keep_blocks = nz_prop >= float(min_nonzero_prop)
        xm = Am[keep_blocks].ravel()
        ym = Bm[keep_blocks].ravel()
        valid_sp += xm.size
        dropped_sp += (~keep_blocks).sum()
        x_list.append(xm)
        y_list.append(ym)

    x = np.concatenate(x_list) if x_list else np.array([], dtype=dtype)
    y = np.concatenate(y_list) if y_list else np.array([], dtype=dtype)


    m = np.isfinite(x) & np.isfinite(y)
    m &= ~((x == 0) | (y == 0))  # remove either 0 or both 0
    xr, yr = x[m], y[m]

    p = [1, 99]
    qx1, qx99 = np.percentile(xr, p)
    qy1, qy99 = np.percentile(yr, p)
    xr = np.clip(xr, qx1, qx99)
    yr = np.clip(yr, qy1, qy99)


    lr = linregress(xr, yr) if xr.size >= 2 else linregress([0,1], [0,1])
    print(f"total superpixels: {total_sp}, valid superpixels: {valid_sp}, dropped by nz filter: {dropped_sp}")
    return lr, xr, yr

def plot_hex_with_fit(x, y, a, b, r, gridsize=200, prefix = None, suffix=None, save_dir=None):

    plt.figure()
    hb = plt.hexbin(x, y, gridsize=gridsize, bins="log")
    plt.colorbar(hb, label="log10(count)")
    xlim = np.array(plt.gca().get_xlim())
    plt.plot(xlim, a + b * xlim, linewidth=2)
    plt.text(0.05, 0.95, f"y = {b:.4f}x + {a:.4f}\nr = {r:.4f}", transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.xlabel("Intensity in A"); plt.ylabel("Intensity in B")
    plt.title(f"Global pixel–pixel pairs with linear fit for {prefix + "_" + suffix}")

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_redox_turnover_correlation_{suffix}.png"), dpi=300)
        return plt.close()
    else:
        plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate redox and unsaturation ratios from TIFF files.")
    parser.add_argument("path", type=str, help="condition 1 folder")
    parser.add_argument("--block-ttest", "-b", action="store_true", help="Run block-median t-test")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for block-median correlation (default: 16)")
    parser.add_argument("--block-min-nonzero", "-bnz", type=float, default=0.5, help="Minimum combined non-zero pixel proportion per block (default: 0.5)")
    parser.add_argument("-o", type=str, default=None, help="Directory to save plots")
    parser.add_argument("-p", type=str, default=None, help="Prefix for saved plots")
    args = parser.parse_args()

    redox_list, protein_turn_list, lipid_turn_list = [], [], []
    for root, _, files in os.walk(args.path):
        process_folder(root, files, redox_list, protein_turn_list, lipid_turn_list)
    
    lrp, xp, yp = img_correlation(redox_list, protein_turn_list, superpixel=args.block_size if args.block_ttest else None, min_nonzero_prop=args.block_min_nonzero)
    lrl, xl, yl = img_correlation(redox_list, lipid_turn_list, superpixel=args.block_size if args.block_ttest else None, min_nonzero_prop=args.block_min_nonzero)

    print(f"Corr. protein w/ redox: slope={lrp.slope:.4f}, intercept={lrp.intercept:.4f}, r={lrp.rvalue:.4f}, p={lrp.pvalue:.4e}")
    print(f"Corr. lipid w/ redox: slope={lrl.slope:.4f}, intercept={lrl.intercept:.4f}, r={lrl.rvalue:.4f}, p={lrl.pvalue:.4e}")

    # Decide output directory: if -o provided, save there; else show interactively
    out_dir = args.o
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if args.p == None:
            print("Warning: No prefix provided for saved plots.")
    plot_hex_with_fit(xp, yp, lrp.intercept, lrp.slope, lrp.rvalue, prefix=args.p, suffix = "protein", save_dir = out_dir)
    plot_hex_with_fit(xl, yl, lrl.intercept, lrl.slope, lrl.rvalue, prefix=args.p, suffix = "lipid", save_dir = out_dir)
