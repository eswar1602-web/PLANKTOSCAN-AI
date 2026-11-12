# Classifier ablations (VAL split only, single seed)

| experiment   | preprocess   | class_weighting   |   epochs_budget |   best_epoch |   val_accuracy |   val_macro_f1 |   val_macro_f1_organisms |
|:-------------|:-------------|:------------------|----------------:|-------------:|---------------:|---------------:|-------------------------:|
| abl_pre0     | False        | none              |              25 |           23 |         0.9115 |         0.8608 |                   0.9161 |
| abl_pre1     | True         | none              |              25 |           23 |         0.9149 |         0.8621 |                   0.9223 |
