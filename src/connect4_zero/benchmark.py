from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from .network import PolicyValueNet, resolve_device

Precision = Literal["float32", "float16"]


@dataclass(slots=True)
class BenchmarkResult:
    channels: int
    blocks: int
    parameters: int
    batch_size: int
    precision: Precision
    training_positions_per_second: float
    inference_positions_per_second: float
    peak_mps_bytes: int
    finite: bool

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


def benchmark_model(
    *,
    channels: int,
    blocks: int,
    batch_size: int,
    precision: Precision,
    warmup_steps: int = 2,
    measured_steps: int = 8,
    allow_cpu: bool = False,
) -> BenchmarkResult:
    device = resolve_device(allow_cpu)
    model = PolicyValueNet(channels, blocks).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(batch_size, 3, 6, 7, device=device)
    pi = torch.softmax(torch.randn(batch_size, 7, device=device), dim=1)
    z = torch.empty(batch_size, device=device).uniform_(-1, 1)
    enabled = precision == "float16"

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
            logits, values = model(x)
            loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss = loss + nn.functional.mse_loss(values.float(), z)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(warmup_steps):
        train_step()
    _synchronize(device)
    peak = int(torch.mps.current_allocated_memory()) if device.type == "mps" else 0
    started = time.perf_counter()
    loss = torch.tensor(float("nan"), device=device)
    for _ in range(measured_steps):
        loss = train_step()
        if device.type == "mps":
            peak = max(peak, int(torch.mps.current_allocated_memory()))
    _synchronize(device)
    training_elapsed = max(time.perf_counter() - started, 1e-9)
    finite = bool(torch.isfinite(loss).item()) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                model(x)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(measured_steps * 4):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                model(x)
            if device.type == "mps":
                peak = max(peak, int(torch.mps.current_allocated_memory()))
        _synchronize(device)
    inference_elapsed = max(time.perf_counter() - started, 1e-9)
    return BenchmarkResult(
        channels=channels,
        blocks=blocks,
        parameters=model.parameter_count,
        batch_size=batch_size,
        precision=precision,
        training_positions_per_second=measured_steps * batch_size / training_elapsed,
        inference_positions_per_second=measured_steps * 4 * batch_size / inference_elapsed,
        peak_mps_bytes=peak,
        finite=finite,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
