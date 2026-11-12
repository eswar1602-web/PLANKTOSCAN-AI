# Split and class-imbalance report

Seed 42; blocks = whole days, class-stratified greedy assignment; target {'train': 0.7, 'val': 0.15, 'test': 0.15}. Run folder: `20261112_044746_split`.
This report counts images only. No model metric has been computed on any split; the test split stays untouched until final evaluation.

- Mapped: 14,048 images in 8 classes; removed as byte-identical files: 5 ({'label_conflict': 4, 'duplicate': 1}; see `results/removed_duplicates.csv`).
- Closed set used: **14,043 images, 8 classes**; open set (never trained on, never used for tuning): **3,895 images, 10 raw classes**.
- Stratified on: target, raw_class.
- Imbalance ratio (largest : smallest class): **7.3 : 1** (dinobryon 3,319 : reject 455).
- Hour-block fallback: **{'diatom': 'tried, reverted (min val/test fill 0.31 -> 0.31, no gain)'}**.
- (class, block) groups shared across splits: **0**.
- Classes with < 30 test images: **none**.

## Per-class counts and days per split

| target     |   train_images |   val_images |   test_images |   train_days |   val_days |   test_days |   total_images |   percent_of_dataset |   train_share |   val_share |   test_share |
|:-----------|---------------:|-------------:|--------------:|-------------:|-----------:|------------:|---------------:|---------------------:|--------------:|------------:|-------------:|
| dinobryon  |           2315 |          489 |           515 |           36 |         25 |          14 |           3319 |                23.63 |         0.697 |       0.147 |        0.155 |
| rotifer    |           2471 |          415 |           392 |          125 |         70 |          73 |           3278 |                23.34 |         0.754 |       0.127 |        0.12  |
| cladoceran |           1326 |          281 |           283 |          107 |         51 |          52 |           1890 |                13.46 |         0.702 |       0.149 |        0.15  |
| nauplius   |           1054 |          228 |           225 |          112 |         43 |          46 |           1507 |                10.73 |         0.699 |       0.151 |        0.149 |
| copepod    |            986 |          210 |           207 |          109 |         44 |          47 |           1403 |                 9.99 |         0.703 |       0.15  |        0.148 |
| diatom     |           1093 |          220 |            64 |           21 |          5 |           9 |           1377 |                 9.81 |         0.794 |       0.16  |        0.046 |
| ceratium   |            568 |          123 |           123 |           46 |         25 |          24 |            814 |                 5.8  |         0.698 |       0.151 |        0.151 |
| reject     |            333 |           67 |            55 |           15 |          7 |           8 |            455 |                 3.24 |         0.732 |       0.147 |        0.121 |

## Raw subtypes per split

|                                     |   train |   val |   test |
|:------------------------------------|--------:|------:|-------:|
| ('ceratium', 'ceratium')            |     568 |   123 |    123 |
| ('cladoceran', 'bosmina')           |      57 |    11 |     12 |
| ('cladoceran', 'daphnia')           |     508 |   106 |    107 |
| ('cladoceran', 'diaphanosoma')      |     761 |   164 |    164 |
| ('copepod', 'cyclops')              |     611 |   128 |    127 |
| ('copepod', 'eudiaptomus')          |     375 |    82 |     80 |
| ('diatom', 'asterionella')          |     874 |   172 |      8 |
| ('diatom', 'diatom_chain')          |      13 |     2 |      2 |
| ('diatom', 'fragilaria')            |     206 |    46 |     54 |
| ('dinobryon', 'dinobryon')          |    2315 |   489 |    515 |
| ('nauplius', 'nauplius')            |    1054 |   228 |    225 |
| ('reject', 'copepod_skins')         |      26 |     4 |      3 |
| ('reject', 'daphnia_skins')         |      36 |     6 |      4 |
| ('reject', 'dirt')                  |      89 |    18 |     24 |
| ('reject', 'unknown')               |     182 |    39 |     24 |
| ('rotifer', 'asplanchna')           |     424 |    91 |     91 |
| ('rotifer', 'brachionus')           |     111 |    13 |     13 |
| ('rotifer', 'conochilus')           |     184 |    40 |     40 |
| ('rotifer', 'kellicottia')          |     362 |    78 |     79 |
| ('rotifer', 'keratella_cochlearis') |      96 |     8 |      8 |
| ('rotifer', 'keratella_quadrata')   |     290 |    66 |     64 |
| ('rotifer', 'polyarthra')           |      50 |    15 |     15 |
| ('rotifer', 'rotifers')             |     665 |    51 |     27 |
| ('rotifer', 'synchaeta')            |     114 |    14 |     14 |
| ('rotifer', 'trichocerca')          |     175 |    39 |     41 |

Raw classes with any split share more than 10 points from target: asterionella, brachionus, keratella_cochlearis, rotifers, synchaeta.

## Busiest single day per class

| class      | busiest_day   |   images |   pct_of_class | split   |
|:-----------|:--------------|---------:|---------------:|:--------|
| diatom     | 2019-06-02    |      882 |           64.1 | train   |
| ceratium   | 2020-05-14    |      236 |           29   | train   |
| reject     | 2020-02-18    |      109 |           24   | train   |
| nauplius   | 2019-04-15    |      360 |           23.9 | train   |
| copepod    | 2020-05-15    |      165 |           11.8 | train   |
| rotifer    | 2020-05-14    |      381 |           11.6 | train   |
| dinobryon  | 2018-06-13    |      229 |            6.9 | train   |
| cladoceran | 2018-07-20    |      100 |            5.3 | train   |

**Diatom bloom-day warning.** Diatom images are concentrated in a few bloom days (see table). With whole-day blocks the bloom day must go to train, so diatom val/test come from few days and the asterionella/fragilaria mix differs between splits. Diatom test F1 therefore rests on few independent sampling days and mostly measures the minority genus; treat it as a less reliable estimate than the other classes.

## Leakage check

1. Time blocks: no (class, block) group appears in more than one split (above).
2. Stage 1, pHash candidates: 64-bit pHash, Hamming distance <= 4, every cross-split pair. Caveat: pHash downsamples to 32x32 and these are small dark-field crops (median ~72x76 px; min side < 32px: 234 of 14,043), so unrelated blobs collide often.
3. Stage 2, confirmation: a candidate is a near-duplicate only if the 64x64 grayscale pixel correlation is >= 0.98.

- pHash candidate pairs across splits: 4,563
- **Confirmed near-duplicates across splits: 0**
  - candidates by split pair: {('train', 'test'): 1988, ('train', 'val'): 2145, ('val', 'test'): 430}
  - candidates of different classes: 2207; both images small: 5; median time gap 3069 h; max pixel corr 0.977
  - details: `results/leakage_pairs.csv`
