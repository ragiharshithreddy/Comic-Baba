"""
RIFEInterpolator — wrapper for the RIFE model.

Strategy
--------
For each consecutive pair of input frames (A, B), generate (factor-1)
intermediate frames using the RIFE model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comic_baba.models.interpolators.base import BaseInterpolator

# Simple RIFE-like architecture placeholder for demonstration
class TinyRIFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x1, x2, t):
        # x1, x2: [B, 3, H, W]
        # t: scalar [0, 1]
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.conv1(x))
        residual = self.conv2(x)
        # Simple blend + residual
        return (1-t) * x1 + t * x2 + residual

class RIFEInterpolator(BaseInterpolator):
    """
    RIFE-based frame interpolator.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = TinyRIFE().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def interpolate(self, frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
        """Return a RIFE-interpolated up-sampled frame sequence."""
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        if factor == 1:
            return list(frames)
        if len(frames) < 2:
            return list(frames)

        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                a = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                b = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                out.append(frames[i])

                for step in range(1, factor):
                    t = step / factor
                    res = self.model(a, b, t)
                    res = (res.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    out.append(res)

        out.append(frames[-1])
        return out
