"""Pipeline.classify fixed-batch padding and stage timing (tiny CPU model, no trained weights)."""

import copy

import numpy as np
import torch
import torch.nn as nn

from src.common.config import load_config
from src.common.timing import stage
from src.pipeline.pipeline import Pipeline


class TinyNet(nn.Module):
    """Per-sample model (no batch coupling) with timm's feature/head split; records every input batch size."""

    num_features = 3

    def __init__(self, n_classes: int = 8):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3, n_classes)
        self.batch_sizes: list[int] = []

    def forward_features(self, x):
        self.batch_sizes.append(x.shape[0])
        return x

    def forward_head(self, x, pre_logits: bool = False):
        f = x.mean(dim=(2, 3))
        return f if pre_logits else self.fc(f)

    def get_classifier(self):
        return self.fc

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


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
    logits, feats = _pipe(8).classify_full([])
    assert logits.shape == (0, 8) and feats.shape == (0, 3)


def test_features_and_logits_come_from_one_pass() -> None:
    p = _pipe(8)
    logits, feats = p.classify_full(_crops(5))
    assert feats.shape == (5, 3)
    assert np.allclose(logits, p.classifier.fc(torch.from_numpy(feats)).detach().numpy(), atol=1e-5)
    assert np.allclose(torch.softmax(torch.from_numpy(logits), 1).numpy(), p.classify(_crops(5)), atol=1e-6)


def test_stage_timer_accumulates_and_none_is_noop() -> None:
    t: dict[str, float] = {}
    with stage(t, "a"):
        pass
    with stage(t, "a"):
        pass
    assert set(t) == {"a"} and t["a"] >= 0
    with stage(None, "b"):
        pass
