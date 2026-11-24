# Confidence calibration and unknown flag (fitted on TRAIN/VAL; open set measured once)

Temperature T = 0.691 (folds 0.699, 0.683). VAL NLL 0.1866 -> 0.1321 out of fold; ECE 0.0841 -> 0.0058.

Raw confidence ceiling: 99th percentile 0.9231, label-smoothing ceiling 0.9125. VAL crops within +-0.01 of the threshold 0.89: 13.4%.

| raw threshold | VAL coverage | accepted accuracy |
|---|---|---|
| 0.85 | 0.9567 | 0.9879 |
| 0.86 | 0.9518 | 0.9889 |
| 0.87 | 0.9447 | 0.9898 |
| 0.88 | 0.9338 | 0.99 |
| 0.89 | 0.9110 | 0.9908 |
| 0.90 | 0.8003 | 0.992 |
| 0.91 | 0.2678 | 0.9835 |
| 0.92 | 0.0229 | 0.9247 |

Unknown cut-off 10028.7 (95% of known VAL scores). Known VAL flagged: original crops 3.8%, frame crops 6.2%, confirmed organisms 0.5%.

| open-set class | n | uncertain | unknown flag | uncertain or unknown | counted as organism | counted without any flag |
|---|---|---|---|---|---|---|
| ALL | 3895 | 0.734 | 0.757 | 0.843 | 0.061 | 0.053 |
| ALL excluding maybe_cyano | 2531 | 0.835 | 0.823 | 0.899 | 0.053 | 0.042 |
| aphanizomenon | 225 | 0.858 | 0.871 | 0.924 | 0.027 | 0.009 |
| chaoborus | 10 | 0.900 | 0.800 | 1.000 | 0.100 | 0.000 |
| filament | 405 | 0.731 | 0.751 | 0.840 | 0.114 | 0.094 |
| fish | 222 | 0.775 | 0.811 | 0.869 | 0.086 | 0.041 |
| hydra | 18 | 0.389 | 0.111 | 0.389 | 0.611 | 0.611 |
| leptodora | 203 | 0.763 | 0.581 | 0.768 | 0.108 | 0.103 |
| maybe_cyano | 1364 | 0.548 | 0.635 | 0.738 | 0.075 | 0.073 |
| paradileptus | 424 | 0.799 | 0.934 | 0.962 | 0.031 | 0.028 |
| unknown_plankton | 71 | 0.578 | 0.549 | 0.648 | 0.155 | 0.155 |
| uroglena | 953 | 0.945 | 0.880 | 0.953 | 0.005 | 0.001 |
