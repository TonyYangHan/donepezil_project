import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def get_number(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def match_files(files, label):
    return {get_number(f): f for f in files if label in f and (f.endswith('.tiff') or f.endswith('.tif'))}


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


def _prepare_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert an image array to a 3-channel float tensor in [0, 1]."""
    if img.ndim == 2:
        tensor = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)
    elif img.ndim == 3:
        tensor = torch.from_numpy(img).float()
        tensor = tensor.permute(2, 0, 1)
        c = tensor.shape[0]
        if c == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif c == 2:
            tensor = torch.cat([tensor, tensor[:1]], dim=0)
        elif c > 3:
            tensor = tensor[:3]
    else:
        raise ValueError(f"Unsupported image dimensions: {img.shape}")

    max_val = tensor.max()
    if max_val > 0:
        tensor = tensor / max_val
    return tensor.unsqueeze(0)


def generate_model_masks(root_path: str, weights_path: str, device: str = "cpu", threshold: float = 0.5):
    """Generate ROI masks from 791-channel images using a trained MultiScaleUNet.

    Returns a dict mapping ROI number (as string) to a float mask array in [0, 1].
    """
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Mask weights not found: {weights_path}")

    files = os.listdir(root_path)
    pro_dict = match_files(files, "791")
    if not pro_dict:
        return {}

    model = MultiScaleUNet().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    masks = {}
    for roi, fname in pro_dict.items():
        img_path = os.path.join(root_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        tensor = _prepare_tensor(img).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(tensor))
        mask = (pred[0, 0].cpu().numpy() >= float(threshold)).astype(np.float32)
        masks[roi] = mask
    return masks