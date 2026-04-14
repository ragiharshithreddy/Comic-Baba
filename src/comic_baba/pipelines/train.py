"""
Training pipeline for interpolation models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from comic_baba.data.dataset import ComicClipDataset
from comic_baba.models.interpolators.rife import TinyRIFE
from comic_baba.utils.config import load_config
from comic_baba.utils.seed import set_seed

logger = logging.getLogger(__name__)

class TrainingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x1, x2, t):
        return self.model(x1, x2, t)

def run_train(config_path: str | Path, checkpoint_path: str | Path | None = None):
    config = load_config(config_path)
    set_seed(config.get("run", {}).get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRIFE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get("train", {}).get("lr", 1e-4))
    criterion = nn.MSELoss()

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    dataset = ComicClipDataset(
        manifest_path=config["paths"]["manifest"],
        split="train",
        resize=(config["data"]["resize"]["width"], config["data"]["resize"]["height"])
    )

    # We need a custom collation or just handle it in the loop since ComicClipDataset yields clips
    num_epochs = config.get("train", {}).get("epochs", 1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        count = 0

        for clip in dataset:
            frames = clip["frames"]
            if len(frames) < 3:
                continue

            # Simple triplet training: frame 0 and 2 as inputs, 1 as target (t=0.5)
            for i in range(len(frames) - 2):
                x1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                x2 = torch.from_numpy(frames[i+2]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                target = torch.from_numpy(frames[i+1]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                optimizer.zero_grad()
                output = model(x1, x2, 0.5)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else 0
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Save checkpoint
        out_dir = Path(config["paths"]["outputs_root"]) / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), out_dir / "latest.pt")

    logger.info("Training complete.")
