from pathlib import Path

import numpy as np
import pytest
import torch

from connect4_zero.game import ConnectFour
from connect4_zero.network import PolicyValueEvaluator, PolicyValueNet, resolve_device


def test_network_shapes_and_gradients() -> None:
    model = PolicyValueNet()
    inputs = torch.randn(8, 3, 6, 7)
    policy, value = model(inputs)
    assert policy.shape == (8, 7)
    assert value.shape == (8,)
    (policy.mean() + value.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert 440_000 < model.parameter_count < 460_000


def test_evaluator_round_trip(tmp_path: Path) -> None:
    evaluator = PolicyValueEvaluator(PolicyValueNet(), device="cpu", allow_cpu=True)
    logits, values = evaluator.predict([ConnectFour.new(), ConnectFour.from_moves([3])])
    assert logits.shape == (2, 7)
    assert values.shape == (2,)
    checkpoint = tmp_path / "model.pt"
    evaluator.save(checkpoint, {"generation": 2})
    loaded, metadata = PolicyValueEvaluator.load(checkpoint, device="cpu", allow_cpu=True)
    loaded_logits, loaded_values = loaded.predict([ConnectFour.new()])
    assert np.allclose(logits[:1], loaded_logits)
    assert np.allclose(values[:1], loaded_values)
    assert metadata["generation"] == 2


def test_cpu_requires_explicit_opt_in() -> None:
    with pytest.raises(RuntimeError):
        PolicyValueEvaluator(PolicyValueNet(), device="cpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_forward_backward() -> None:
    device = resolve_device()
    model = PolicyValueNet().to(device)
    policy, value = model(torch.randn(16, 3, 6, 7, device=device))
    (policy.mean() + value.mean()).backward()
    torch.mps.synchronize()
    assert device.type == "mps"

