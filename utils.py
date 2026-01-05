import os, re, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, tifffile as tiff
from pathlib import Path
from torch.utils.checkpoint import checkpoint
from scipy.stats import chi2_contingency

_RX = re.compile(r'^(\d+)[_-]')

def get_number(filename: str) -> str:
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)

    m = _RX.match(stem)
    if not m:
        raise ValueError(
            f"get_number(): expected filename like '<digits>-...ext' (digits before '-'), got '{base}'"
        )
    return m.group(1)


def match_files(files, label):
    return {get_number(f): f for f in files if label in f and (f.endswith('.tiff') or f.endswith('.tif'))}


def safe_chi2(contingency: np.ndarray):
    """Run chi-square on a contingency table after dropping empty rows/cols."""
    keep_cols = contingency.sum(axis=0) > 0
    keep_rows = contingency.sum(axis=1) > 0
    cleaned = contingency[np.ix_(keep_rows, keep_cols)]
    if cleaned.shape[0] < 2 or cleaned.shape[1] < 2:
        return None, None, None, None
    chi2, p, dof, expected = chi2_contingency(cleaned, correction=False)
    return chi2, p, dof, expected


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 1)
        self.c3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c5 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
        self.c7 = nn.Conv2d(in_ch, out_ch, 7, padding=3)
        self.fuse = nn.Conv2d(out_ch * 4, out_ch, 1)

    def forward(self, x):
        y = torch.cat([
            F.elu(self.c1(x)),
            F.elu(self.c3(x)),
            F.elu(self.c5(x)),
            F.elu(self.c7(x)),
        ], dim=1)
        return F.elu(self.fuse(y))


class MultiScaleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = MultiScaleBlock(3, 64)
        self.e2 = MultiScaleBlock(64, 128)
        self.e3 = MultiScaleBlock(128, 256)
        self.e4 = MultiScaleBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3 = MultiScaleBlock(256 + 512, 256)
        self.d2 = MultiScaleBlock(128 + 256, 128)
        self.d1 = MultiScaleBlock(64 + 128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = checkpoint(self.e3, self.pool(x2), use_reentrant=False)
        x4 = checkpoint(self.e4, self.pool(x3), use_reentrant=False)
        y = self.up(x4)
        y = checkpoint(self.d3, torch.cat([y, x3], 1), use_reentrant=False)
        y = self.up(y)
        y = checkpoint(self.d2, torch.cat([y, x2], 1), use_reentrant=False)
        y = self.up(y)
        y = checkpoint(self.d1, torch.cat([y, x1], 1), use_reentrant=False)
        return self.out(y)


def generate_model_masks(root_path: str, weights_path: str, device: str = "cpu", threshold: float = 0.5):
    """
    Assumptions:
      - All images are 1-channel uint16 TIFF with 12-bit range (0..4095).
      - Convert to 8-bit via bit shift: img8 = img16 >> 4 (0..255).
      - Normalize by /255 and replicate to 3 channels.
    Returns: dict {roi_id (str): mask float32 HxW (0/1)}
    """
    files = os.listdir(root_path)
    pro_dict = match_files(files, "791")  # uses get_number(...) :contentReference[oaicite:3]{index=3}
    if not pro_dict:
        return {}

    model = MultiScaleUNet().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # tolerate DataParallel saves
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    masks = {}
    for roi, fname in pro_dict.items():
        img16 = cv2.imread(os.path.join(root_path, fname), cv2.IMREAD_UNCHANGED)
        if img16 is None:
            continue

        img8 = (img16 >> 4).astype(np.uint8)  # 12-bit -> 8-bit

        x = (
            torch.from_numpy(img8)
            .float()
            .div(255.0)
            .unsqueeze(0)          # 1xHxW
            .repeat(3, 1, 1)       # 3xHxW
            .unsqueeze(0)          # 1x3xHxW
            .to(device)
        )

        with torch.inference_mode():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        masks[roi] = (prob >= float(threshold)).astype(np.float32)

    return masks
