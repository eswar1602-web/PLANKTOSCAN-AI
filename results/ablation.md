# Ablations (VAL only, single seed)

## Classifier (EfficientNet-B0)

| experiment   | init         | grayscale   | preprocess   | class_weighting   |   epochs_budget |   best_epoch |   val_accuracy |   val_macro_f1 |   val_macro_f1_organisms |
|:-------------|:-------------|:------------|:-------------|:------------------|----------------:|-------------:|---------------:|---------------:|-------------------------:|
| abl_pre0     | from scratch | False       | False        | none              |              25 |           23 |         0.9115 |         0.8608 |                   0.9161 |
| abl_pre1     | from scratch | False       | True         | none              |              25 |           23 |         0.9149 |         0.8621 |                   0.9223 |
| cls_w0       | from scratch | False       | True         | none              |              40 |           37 |         0.9518 |         0.9055 |                   0.9576 |
| cls_w1       | from scratch | False       | True         | inverse_sqrt_freq |              40 |           37 |         0.9469 |         0.9051 |                   0.9549 |
| cls_pt_rgb   | ImageNet     | False       | True         | none              |              40 |           29 |         0.9823 |         0.9593 |                   0.9895 |
| cls_pt_gray  | ImageNet     | True        | True         | none              |              40 |           28 |         0.9784 |         0.959  |                   0.9827 |

## Detector (YOLOv8n, SYNTHETIC val frames, Ultralytics default val: NMS IoU 0.7)

| init              |   time_cap_h |   epochs_trained | stopped_early_by_user   |   best_epoch |   best_val_mAP50 |   best_val_mAP50_95 |   common_epoch |   val_mAP50_at_common_epoch |   val_mAP50_95_at_common_epoch |
|:------------------|-------------:|-----------------:|:------------------------|-------------:|-----------------:|--------------------:|---------------:|----------------------------:|-------------------------------:|
| from scratch      |        1.25  |                7 | True                    |            7 |           0.937  |              0.6825 |              7 |                      0.937  |                         0.6825 |
| COCO (yolov8n.pt) |        0.667 |               63 | False                   |           63 |           0.9836 |              0.8402 |              7 |                      0.9666 |                         0.7669 |
