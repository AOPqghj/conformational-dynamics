"""Portable torch checkpoints."""

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, **metadata: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state_dict, **metadata}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint
