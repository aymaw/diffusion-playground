import numpy as np
import pytest
import torch

from diffusionlab.core import (
    Denoiser,
    ddim_step,
    noisy,
    sample,
    schedule,
    shapes,
    sliced_wasserstein,
    train,
)


def test_forward_noise_has_exact_formula():
    x = torch.tensor([[2.0, -1.0]])
    eps = torch.tensor([[0.5, 0.2]])
    a = schedule()
    t = torch.tensor([45])
    actual = noisy(x, t, eps, a)
    assert torch.allclose(actual, a[45].sqrt() * x + (1 - a[45]).sqrt() * eps)
    assert torch.allclose(ddim_step(actual, eps, a[45], torch.tensor(1.0)), x, atol=1e-6)


def test_schedule_decreases_and_approaches_normal():
    a = schedule()
    assert torch.all((a > 0) & (a < 1))
    assert torch.all(a[1:] < a[:-1])
    assert a[-1] < 0.01


def test_training_updates_parameters_and_samples_reload(tmp_path):
    torch.manual_seed(3)
    before = Denoiser().net[0].weight.detach().clone()
    model, _ = train(3, 3)
    assert not torch.equal(before, model.net[0].weight)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    other = Denoiser()
    other.load_state_dict(torch.load(path, weights_only=True))
    y = torch.tensor([0, 1, 2])
    a, history = sample(model, y, 4, 4)
    b, _ = sample(other, y, 4, 4)
    assert torch.equal(a, b) and torch.isfinite(a).all()
    assert len(history) == 5


def test_guidance_zero_ignores_labels():
    model = Denoiser()
    a, _ = sample(model, torch.zeros(8, dtype=torch.long), 9, 4, 0)
    b, _ = sample(model, torch.ones(8, dtype=torch.long), 9, 4, 0)
    assert torch.equal(a, b)


def test_distance_translation_and_reordering():
    x = np.array([[0.0, 0.0], [1.0, 2.0]])
    assert sliced_wasserstein(x, x[::-1]) == 0
    assert sliced_wasserstein(x, x + 1) > 0.1


@pytest.mark.parametrize("steps,guidance", [(1, 1), (129, 1), (4, float("nan")), (4, -1)])
def test_invalid_sampling(steps, guidance):
    with pytest.raises(ValueError):
        sample(Denoiser(), torch.tensor([0]), 1, steps, guidance)


def test_synthetic_split_is_independent():
    a, ya = shapes(100, 3)
    b, yb = shapes(100, 4)
    assert not torch.equal(a, b)
    assert set(ya.tolist()) == set(yb.tolist()) == {0, 1, 2}
