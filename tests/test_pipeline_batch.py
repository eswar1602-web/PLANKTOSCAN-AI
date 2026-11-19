"""Pipeline.classify fixed-batch padding and stage timing (tiny CPU model, no trained weights)."""

import copy

import numpy as np
import torch
import torch.nn as nn

from src.common.config import load_config
from src.common.timing import stage
from src.pipeline.pipeline import Pipeline


class TinyNet(nn.Module):
    """Per-sample model (no batch coupling) that records every input batch size."""

    def __init__(self, n_classes: int = 8):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3, n_classes)
        self.batch_sizes: list[int] = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        return self.fc(x.mean(dim=(2, 3)))


def _pipe(batch: int) -> Pipeline:
    p = Pipeline.__new__(Pipeline)                  # skip loading real weights
    p.cfg = copy.deepcopy(load_config())
    p.cfg["pipeline"]["classifier_batch"] = batch
    p.device = "cpu"
    p.classifier = TinyNet().eval()
    p.meta = {"image_size": 32, "preprocess": None, "grayscale": False}
    p.classes = [str(i) for i in range(8)]
    return p


def _crops(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(n)
    return [rng.integers(0, 255, (int(rng.integers(20, 60)), int(rng.integers(20, 60)), 3), dtype=np.uint8)
            for _ in range(n)]


def test_fixed_batch_gives_same_probs_and_one_shape() -> None:
    for n in (1, 3, 8, 9, 20):
        crops = _crops(n)
        fixed, var = _pipe(8), _pipe(0)
        a, b = fixed.classify(crops), var.classify(crops)
        assert a.shape == (n, 8) and np.allclose(a, b, atol=1e-6)
        assert set(fixed.classifier.batch_sizes) == {8}               # always the same input shape
        assert np.allclose(a.sum(1), 1)


def test_empty_crop_list() -> None:
    assert _pipe(8).classify([]).shape == (0, 8)


def test_stage_timer_accumulates_and_none_is_noop() -> None:
    t: dict[str, float] = {}
    with stage(t, "a"):
        pass
    with stage(t, "a"):
        pass
    assert set(t) == {"a"} and t["a"] >= 0
    with stage(None, "b"):
        pass
