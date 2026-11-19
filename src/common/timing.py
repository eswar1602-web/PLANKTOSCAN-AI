"""Wall-clock stage timing that is correct on the GPU (CUDA work is asynchronous)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def _sync(device: str) -> None:
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


@contextmanager
def stage(timings: dict[str, float] | None, name: str, device: str = "cpu") -> Iterator[None]:
    """Add the elapsed milliseconds of the block to timings[name] (no-op if timings is None).

    On CUDA the device is synchronised before and after, so queued kernels are not attributed to
    the wrong stage (or missed entirely).
    """
    if timings is None:
        yield
        return
    _sync(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
