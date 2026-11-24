# INT8 quantization (PyTorch FX, onednn, CPU), run `20261122_180914_quantize_int8`

Calibration: 512 TRAIN crops. Evaluation: 4066 VAL crops (original + frame crops). No test data.

| classifier (EfficientNet-B0) | FP32 | INT8 |
|---|---:|---:|
| size (MB) | 16.35 | 5.6 |
| VAL accuracy | 0.9727 | 0.8389 |
| VAL macro F1 | 0.9447 | 0.8083 |
| VAL uncertain share at 0.89 | 0.0888 | 0.6299 |
| latency per crop, 16 threads (ms) | 23.57 | 105.67 |
| latency per crop, 1 thread (ms) | 24.71 | 82.72 |

Predicted class agreement INT8 vs FP32: 84.63%; confirmed/uncertain status agreement: 43.73%.

Detector: failed: TraceError: Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanatio

Rule (written after the first measurement; the result fails every part of it by a wide margin): deploy INT8 only if it is faster than FP32 at 1 thread, VAL macro F1 drops by at most 0.01, and the detector can be quantized too.

Decision: not deployed: INT8 is not faster at 1 thread (82.72 vs 24.71 ms); VAL macro F1 drops by 0.136 (> 0.01); the detector could not be quantized with the installed tools.

Likely cause of the loss and the slowdown (not tested): post-training per-tensor activation ranges fit EfficientNet's SiLU and squeeze-excitation layers poorly, and oneDNN converts between INT8 and float around every op it cannot run in INT8. Quantization-aware training would be needed, which is retraining.
