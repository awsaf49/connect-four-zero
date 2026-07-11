from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .game import COLS, ROWS, ConnectFour


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, COLS),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(ROWS * COLS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.tower(self.stem(x))
        return self.policy_head(trunk), self.value_head(self.value_conv(trunk)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def resolve_device(allow_cpu: bool = False) -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    reason = "PyTorch was not built with MPS" if not torch.backends.mps.is_built() else "MPS is unavailable"
    raise RuntimeError(f"{reason}; production training requires the Apple GPU")


class PolicyValueEvaluator:
    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device | str | None = None,
        allow_cpu: bool = False,
    ) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        self.device = torch.device(device) if device is not None else resolve_device(allow_cpu)
        if self.device.type == "cpu" and not allow_cpu:
            raise RuntimeError("CPU evaluator requires allow_cpu=True")
        self.model = model.to(self.device)

    def predict(self, states: Sequence[ConnectFour]) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.empty((0, COLS), np.float32), np.empty((0,), np.float32)
        batch = torch.from_numpy(np.stack([state.encode() for state in states])).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            logits, values = self.model(batch)
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "version": 1,
                "model": self.model.state_dict(),
                "channels": self.model.channels,
                "blocks": self.model.blocks,
                "metadata": metadata or {},
            },
            temporary,
        )
        temporary.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device | str | None = None,
        allow_cpu: bool = False,
    ) -> tuple[PolicyValueEvaluator, dict[str, object]]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = PolicyValueNet(int(payload["channels"]), int(payload["blocks"]))
        model.load_state_dict(payload["model"])
        return cls(model, device=device, allow_cpu=allow_cpu), dict(payload.get("metadata", {}))

    def mps_metrics(self) -> dict[str, float | str | bool]:
        metrics: dict[str, float | str | bool] = {
            "device": str(self.device),
            "mps_available": torch.backends.mps.is_available(),
        }
        if self.device.type == "mps":
            metrics.update(
                {
                    "mps_allocated_bytes": float(torch.mps.current_allocated_memory()),
                    "mps_driver_bytes": float(torch.mps.driver_allocated_memory()),
                    "mps_recommended_bytes": float(torch.mps.recommended_max_memory()),
                }
            )
        return metrics

