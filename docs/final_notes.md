## What was built

An end-to-end pipeline: a YOLOv8n single-class detector finds every object in a frame; each box is
cropped from the original image, preprocessed (border-median illumination correction, NL-means
denoising, CLAHE) and classified by EfficientNet-B0 into 7 organism classes plus `reject`. Items
below the val-chosen confidence threshold are counted as **uncertain** instead of being dropped.
A Streamlit app shows the annotated image and counts and exports CSV, HTML and PDF reports.

## How it was evaluated

- All choices (preprocessing, class weighting, initialisation, checkpoint, uncertainty threshold,
  NMS IoU and detector confidence) were made on the VAL split or the synthetic VAL frames.
- The test split and the synthetic test frames were used once, by `scripts/evaluate.py`.
- Confidence intervals are 95% bootstrap intervals that resample whole sampling days, because images from
  one day are strongly correlated.

## How to read the results

- **Detector and end-to-end numbers are from synthetic frames.** These were built by pasting real
  test-split crops onto generated backgrounds. They measure whether the system works under
  controlled conditions, not on real microscope slides.
- **Diatom test F1 is a weak estimate.** It rests on few days and is mostly one subtype.
- **`reject` is the hardest class.** It is a mixed bag, so its F1 is expected to trail the organism classes.
- **Open-set behaviour is partial.** Some never-seen organisms are flagged uncertain, but others are
  confidently assigned a known class. Treat the counts of rare classes with caution.

## What did not work or was changed

- The from-scratch detector run was stopped by the user after 7 epochs, so it is reported only as an
  ablation row.
- The first uncertainty target (97%) became degenerate once the pretrained classifier was better than
  97% on val. It was raised to 99%, still chosen on val, before any test use.
- Class weighting (inverse sqrt frequency) and preprocessing gave differences within single-seed noise.
- Grayscale input matched RGB, so the colour information did not measurably help.

## Not done

- No validation on real multi-organism microscope images (none with labels are available).
- No marine data. No repeated seeds. No test-time augmentation.

## Main weakness found in the final evaluation

On synthetic frames about half of the correctly detected organisms are flagged **uncertain**, so
per-class counts are much lower than the true counts. The uncertain items are listed, never dropped.
A val-only diagnostic (`scripts/diagnose_frame_crops.py`) shows the cause. The same objects are
classified almost perfectly as their original ZooLake crops, but much worse when re-cut from a frame.
The frame crop has lost its native background and been resampled by the paste, and the classifier
was never trained on that kind of crop. Changing the crop padding does not help.

The likely fix is to also train the classifier on crops cut from the synthetic **train** frames, or
to paste with the crop's own background. That would be a new model, and it would need a new,
untouched test set (or fresh synthetic test frames) to be evaluated honestly. It was not done here,
because the test split had already been used.

The uncertainty flag did its job: accepted predictions stay mostly correct, and the doubtful ones are
shown to the user instead of being silently miscounted.
