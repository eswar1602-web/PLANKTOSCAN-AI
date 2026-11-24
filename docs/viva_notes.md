# Viva notes: design decisions

One line per decision. Added to as each phase is built.

## Phase 1: setup
- **Two-stage design (YOLOv8n detector + EfficientNet-B0 classifier):** ZooLake only has labelled single-organism crops, so the classifier trains on real images and the detector only has to learn "organism vs background" from synthetic frames.
- **Single-class detector:** species identity comes from the classifier, which sees real crops at higher resolution; a multi-class detector would learn species only from synthetic pastes.
- **Python 3.12:** the newest version with mature, tested PyTorch and Ultralytics builds, so setup is predictable.
- **Macro F1 as the headline metric:** the classes are imbalanced, so accuracy can look high while rare classes fail; macro F1 weights every class equally.
- **Best checkpoint chosen by val macro F1, not val loss:** the checkpoint is picked by the same metric we report.
- **Inverse-sqrt-frequency class weights:** they help rare classes without the instability that full inverse-frequency weights can cause on very rare classes.
- **All settings in one config.yaml, copied into every run folder with seed, git commit and library versions:** any reported number can be traced back to the exact code and settings that produced it.
- **Leakage-safe split:** near-identical images of the same organism in both train and test would inflate scores (detailed in Phase 2 below).
- **Synthetic frames built per split:** train frames use only train crops, val only val, test only test, so the detector is never tested on organisms it has seen.
- **Alpha-blended pasting:** hard paste edges are a shortcut the detector could learn instead of organism shape.
- **"Uncertain" label instead of dropping low-confidence detections:** counts stay complete, and a human can review doubtful organisms.
- **Mixed precision + batch 32/8:** fits the 4 GB RTX 3050 and roughly halves memory use with negligible accuracy cost.
- **reportlab for PDF:** pure Python, so there are no system dependencies to break on Windows or Colab.

## Phase 2: data
- **ZooLake only, chosen from the paper's data statement:** a verified source (DOI, published SHA-256) with taxonomist labels; the freshwater-vs-marine mismatch is stated as a limitation.
- **Merging raw classes into 8 project classes:** this cuts the imbalance ratio from 332:1 to 7.3:1 and leaves at least 55 test images per class, so per-class F1 is not just noise.
- **dinobryon included:** it is the most common object in the lake, and leaving it out would make real samples produce many forced or "uncertain" labels.
- **`reject` rather than "non-organism":** the dataset README says "unknown" may contain organism fragments, so the honest name is "not countable as a target organism".
- **Open set kept apart, never trained or tuned on:** it measures what happens when the system meets organisms it was never taught. That is the realistic failure mode in the field.
- **Whole-day blocks, not a random split:** images from the same day share water, lighting and often the same bloom, so a random split leaks; a pure chronological cut would put seasonal classes in only one split.
- **Stratified on class and raw subtype together:** otherwise one split can be all asterionella and another all fragilaria; this cut the number of off-balance subtypes from 15 to 5 of 25.
- **Hour fallback kept only if it helps:** finer blocks raise leakage risk, so they are used only when they actually improve the val/test fill (for diatom it didn't, so it was reverted).
- **Two-stage leakage check (pHash, then pixel correlation):** pHash alone flagged about 4,600 cross-split pairs on these tiny dark-field crops, mostly different classes months apart. Confirmation found 0 real near-duplicates.
- **Byte-identical files removed (5 rows):** two images were filed under two different classes. A contradictory label is worse than no label, so both copies were dropped.
- **Test split untouched until Phase 8:** any decision made after looking at test results turns the test set into a second validation set and inflates the reported numbers.

## Phase 3: preprocessing
- **Subtract the background, don't divide (flat-field):** in dark field the background is near-black, so dividing by it amplifies noise; subtracting the background level is the dark-field equivalent of flat-field correction.
- **Border-median background for crops:** a large blur on a 72 px crop is mostly the organism itself, so its border pixels are a safer, parameter-free estimate of the background.
- **CLAHE tile size in pixels, not a fixed grid:** an 8x8 grid on a 16 px crop would create 2 px tiles; tile_px=32 keeps tiles meaningful on any image size.
- **One `prepare_crop` function for training and inference:** a train/inference mismatch in preprocessing silently lowers accuracy in deployment; one shared function removes that risk.
- **Preprocessing ablation result:** val macro F1 0.8621 with vs 0.8608 without (`results/ablation.csv`). Kept on by the rule set beforehand, but the gap is within single-seed noise: the camera software had already enhanced these crops.
- **No frame-level preprocessing for the detector:** it was never evaluated within the time budget, and an untested step in the detection path is a risk without a measured benefit.

## Phase 4: classifier
- **EfficientNet-B0 at 128 px, not 224:** crops are tiny (median 72x76), so 128 px keeps nearly all detail at a third of the compute, which matters on a 4 GB GPU.
- **Letterbox with black padding, not stretching:** stretching distorts shape, a key cue for plankton; black padding matches the dark-field background.
- **Rotations up to 180 degrees and both flips:** organisms drift in any orientation, so orientation carries no class information.
- **GPU batch augmentation, no DataLoader workers:** Windows multiprocessing data loading is fragile; augmenting whole batches on the GPU is fast and deterministic.
- **LR 1e-3 + warmup + cosine + label smoothing 0.1, same for scratch and pretrained:** identical hyperparameters make the init comparison clean; label smoothing curbs over-confidence (it also caps max softmax near 0.93, which bounds the useful threshold range).
- **ImageNet fine-tuning chosen over from scratch:** on the same split and seed, val macro F1 rose from 0.906 to 0.959; small datasets benefit from generic edge/texture features.
- **RGB kept, not grayscale x3:** crops are coloured and colour carries class information (diatoms are distinctly less blue); both were trained and RGB (0.9593) vs gray (0.9590) is a tie, so the rule 'highest val macro F1' picked RGB.
- **Checkpoint chosen by val macro F1:** we select on the metric we report, not on val loss or accuracy.

## Phase 5: detector
- **Single-class detector:** the detector only answers "is there an object"; species identity comes from the classifier trained on real crops.
- **Reject crops are pasted as detectable objects:** debris also appears in real samples; the classifier, not the detector, decides what is rejected.
- **Alpha mask from the crop's own brightness, edges forced transparent:** no rectangular seams for the detector to exploit.
- **Native pixel scale (0.8-1.25x), large crops capped at 192 px:** keeps organism sizes realistic relative to the frame.
- **Tight boxes from the alpha mask, not the crop rectangle:** labels follow the organism, not the padding around it.
- **Per-split frames with an assertion:** a detector tested on organisms it was trained on would report inflated mAP.
- **YOLOv8n fine-tuned from COCO yolov8n.pt, 40 min cap:** 63 epochs reached val mAP@0.5:0.95 0.840 vs 0.683 for the from-scratch run (stopped at 7 epochs; 0.767 for pretrained at the same epoch).
- **Only two allow-listed weight files, SHA-256 checked against the publisher:** the offline guard still blocks every other download.
- **Ultralytics network side effects disabled and blocked:** the project rule forbids downloads, and the library would otherwise fetch a model for its AMP check and a font for plots.
- **NMS IoU chosen by val mAP@0.5:0.95; confidence by val detection F1:** both are tuned on synthetic val frames only.

## Phase 6-7: pipeline and app
- **"uncertain" when max softmax < threshold, chosen on VAL as the smallest threshold with >=99% accepted accuracy (raised from 97%, which gave t=0 for the pretrained model):** low-confidence items are shown and counted separately rather than silently counted wrong or dropped.
- **Low confidence takes priority over the reject label:** an unsure "reject" is still unsure.
- **Crops are cut from the original image, then preprocessed:** identical to how the classifier was trained.
- **reportlab PDF, jinja2 HTML, CSV:** pure-Python exports with no system dependencies.
- **Domain warning in the app, HTML and PDF:** results on real microscope images are not validated, and the user must see that wherever the numbers appear.

## Phase 8: evaluation
- **Test split used once, by one script:** every number in the README is regenerated by `scripts/evaluate.py`, so no metric is typed by hand.
- **Day-block bootstrap for CIs:** images from one day are correlated; resampling single images would give overconfident intervals.
- **Open-set reported with and without maybe_cyano:** its labels are themselves uncertain, so it could flatter or penalise the uncertainty flag.
- **End-to-end count error per class on synthetic test frames:** counting is the actual product output, so it is evaluated directly, not only through mAP and F1.
- **Frame-crop diagnostic on VAL with ground-truth boxes:** it separates detector errors from classifier domain shift; it showed the end-to-end uncertainty comes from the classifier, not the detector.
- **No fix after seeing test:** retraining on frame crops after the test evaluation would turn the test set into a tuning set; it is listed as future work.

## After the final evaluation: real pilot and embedded feasibility
- **Real-image pilot is a separate script, tagged on every output and never used for tuning:** it can only report how the frozen system transfers; using it to pick thresholds would leak the only real-data check.
- **Two predicted totals (with and without uncertain items):** an uncertain item is usually an organism the classifier is unsure about, so the truth is likely between the two; showing both keeps the uncertainty visible.
- **Works with total counts only:** manual per-class counts are expensive, and a total count by an expert is the realistic minimum.
- **Real pilot outputs are git-ignored:** real images may not be ours to publish.
- **ONNX + OpenCV DNN on CPU with the GPU hidden:** ONNX is the portable format for edge runtimes; OpenCV DNN was already installed, and no new downloads were allowed.
- **Exporter step skipped instead of installing `onnx`:** torch only needs `onnx` to insert custom onnx-script functions, which these models do not have; parity tests (logit difference, box matching) prove the exported models are equivalent.
- **YOLO decoded and NMS'd in NumPy/OpenCV:** the edge device should not need PyTorch or Ultralytics; parity is checked against Ultralytics boxes at IoU >= 0.9.
- **Median over 100 different VAL inputs after warm-up, at all threads and at 1 thread:** the median resists OS jitter; 1 thread approximates a small CPU without claiming to be one.
- **Classifier timed per crop at batch 1, with and without preprocessing:** that is how a small device would run it, and it shows the handcrafted preprocessing is not free: about as costly as the network forward with all threads, about a third of it on one thread.

## Backend v2, Phase A: fixing undercounting (classifier v2)
- **Train on crops re-cut from synthetic frames:** the diagnostic showed the end-to-end loss is a crop-vs-frame domain gap in the classifier, so the fix is to show the classifier the kind of crop the pipeline will actually give it.
- **Per-split frame crops only (train frames -> train, val frames -> val), asserted in code; test frames refused:** the leakage rule for the detector frames carries over to classifier data.
- **Ground-truth box + random jitter, not detector boxes:** GT boxes exist for every object and do not depend on detector errors; the jitter imitates the detector's box error instead.
- **Jitter std 0.06 of box side, measured on VAL frames (pooled edge std 0.0605):** the augmentation is calibrated to the real detector rather than guessed; the Gaussian covers the heavy tail (90th percentile only 0.041).
- **Same cropping function in training and inference (`pad_crop`, padding 0.1), tested against the old pipeline code:** a train/inference mismatch is exactly the bug being fixed, so it must not reappear in the crop step.
- **50/50 mix (one frame crop per original crop) in train and VAL:** keeps the original-crop accuracy while adding the frame domain; val is mixed the same way so selection rewards both.
- **Same recipe as cls_pt_rgb (ImageNet init, RGB, preprocessing on, unweighted, same LR/epochs/seed):** only the training data changes, so any difference is attributable to the frame crops.
- **v1 kept as classifier_v1.pt; v2 promoted only if it wins on mixed VAL macro F1:** model choice stays a val-only decision, and v1 stays reproducible.
- **Threshold for v2 re-chosen on mixed VAL with the same 99% rule; v1 keeps 0.84:** each model is compared at its own val-chosen operating point.
- **Fresh synthetic test frames with a new seed, evaluated once (script refuses a second run):** the old test frames informed the decision to build v2, so they can no longer give an unbiased v2 number.
- **`estimated_counts` alongside the strict counts:** strict counts undercount by design (uncertain items are not counted); the estimate puts every item under its argmax class and shows how much of each class came from uncertain items, so nothing is hidden or silently merged.
- **Success rule fixed before any v2 number was seen on VAL, and checked on a VAL dry run:** the bar could not be moved to fit the result, and the one-shot test run could not fail on a bug.
- **Dry run repeated after every change to the evaluation script:** a one-shot evaluation must run the exact code that was checked.
- **Hash lists of `data/synthetic/` and `results/` around the fresh-frame generation:** proves the new frames did not overwrite anything used before.
- **v1 outputs frozen in `results/v1/`:** the v1 numbers stay traceable after `results/` moves on to v2.
- **Threshold sensitivity reported, not used for selection:** picking 0.85 because it looks better on test would be tuning on test; the table only shows how fragile 0.89 is.
- **Reject regression accepted and documented instead of retrained:** reject is not counted, the organism counts improved a lot, and a new run would need yet another fresh test set; the weakness is stated with numbers.

## Application backend, app and packaging (2026-11-20)
- **One service (`src/service.py`) for the app and the CLI:** inference, validation and counting live in one place, so the two front ends cannot drift apart; the service adds no model logic of its own.
- **One canonical result (`AnalysisResult.to_dict()`) renders every report:** CSV, JSON, HTML and PDF are views of the same dict, so they cannot disagree on a count (tested).
- **Confirmed, estimated, uncertain and rejected kept as separate fields everywhere:** merging them would hide exactly the uncertainty the threshold was chosen to expose.
- **Typed errors with stable codes; zero detections is a success with a warning:** callers can react per error type, and an empty frame is a valid answer, not a failure.
- **Header-only size check before decoding (PIL), then OpenCV decode:** a decompression bomb is rejected before its pixels are allocated; a decoded size that disagrees with the header means a truncated file.
- **Content type checked, not only the extension:** a GIF renamed to .png is rejected as unsupported instead of failing somewhere inside the model.
- **Fixed classifier batch of 32 (padded):** on the RTX 3050 cuDNN spent 6-12 s on every new batch shape, and the batch size is the number of detections; on 300 VAL frames the fixed batch changed 0 labels and 0 statuses (max prob difference 2.9e-3), and was 3x faster overall.
- **Warm-up of detector AND classifier at load:** a blank frame has no detections, so warming only the detector left the first real image 6 s slow.
- **CUDA-synchronised stage timers:** GPU work is asynchronous, so unsynchronised timers attribute kernel time to the wrong stage.
- **Enhanced image is display-only and capped at 800 px:** the detector was trained on unenhanced frames; NL-means on a full large frame took up to 8.5 s for a preview.
- **SQLite stores metadata and file paths, never images; one connection per call, one transaction per write:** keeps the database small and safe with the API server's worker threads; a failed insert rolls back.
- **Failures are stored in the history too:** a batch log that silently omits failed images overstates success.
- **Each analysis writes to a new folder `<name>_<UTC time>-<random>` with `exist_ok=False`:** two analyses can never overwrite each other.
- **History deletes files only inside its own folder:** a CLI output folder chosen by the user is never removed by a history clean-up.
- **Ordering by time, then insertion order (rowid):** second-resolution timestamps plus random ids listed same-second analyses in random order (found by a test).
- **Model manifest with SHA-256 instead of copying weights to a dist/ folder:** identifies the exact models without duplicating 39 MB; `--check` verifies a copy.
- **History folder redirectable by an environment variable (`PLANKTOSCAN_HISTORY_DIR`):** tests and demos run the real server without writing into the real history.
- **No real-image numbers:** no real microscope images were available; the README says so instead of showing synthetic results as if they were real.

## Web application: FastAPI backend + React frontend (2026-11-20)
- **FastAPI over the existing service, no new inference code:** the API only validates the request, calls `AnalysisService`, saves outputs and writes the history, exactly as the CLI does; a test checks that neither imports the models.
- **One JSON error envelope `{"error": {code, message}}` with an HTTP status per error type:** the frontend shows the server's own message and code, so a user sees why an image was refused, not a generic failure.
- **Upload read only up to the size limit:** a huge upload is refused without being held in memory.
- **One analysis at a time on the GPU (a lock), inference in a worker thread:** the event loop stays free for history and downloads while a 4 GB GPU never runs two models at once.
- **Models load at server start, and a server without models still serves the history (503 only for analysis):** a missing weight file is reported clearly instead of crashing the whole app.
- **App factory (`create_app`) instead of a module-level app:** importing the API in tests builds nothing and never touches the real history.
- **The API serves the built frontend:** one command (`python -m api`) runs the whole application; Vite's proxy keeps the same URLs in development.
- **View state in the URL (`?view=history&analysis=<id>`):** a previous analysis can be linked, reloaded and reached with the back button.
- **The specimen frame draws the boxes as SVG in image pixel coordinates over the original:** the boxes stay aligned at any size and link to the detection table; the server-annotated image stays as the report version.
- **Browser validation of type and size before upload, the server validates again:** quick feedback without trusting the client.
- **Every data region has loading, empty, error and full states:** offline server, missing models, refused image and empty history each say what happened and what to do.
- **Terrazzo theme (Hallmark deal): ruled specification tables, mono codes, one rust signal:** a lab tool reads like a specimen catalogue; colour is kept for state and focus, never decoration.
- **Fonts bundled from npm (@fontsource), not loaded from a CDN:** the app works offline in a lab or at a viva.
- **The earlier prototype UI and its packages were removed:** one frontend to maintain; its reusable parts (analyse once per upload, failures in the history, the history folder override, example frames) moved into the API.

## Count intervals and concentration (2026-11-22)
- **Two uncertainties, kept apart:** which class each detection is (classification) and how far one image's
  volume represents the sample (sampling). They have different causes, so they get different intervals and are
  never added together.
- **Classification range = Poisson-binomial over the classifier probabilities:** each detection is class c with
  its probability, independently; the exact distribution of the sum gives the 2.5% and 97.5% quantiles. Exact by
  dynamic programming (O(n^2), trivial for a frame); no normal approximation, which fails at small counts.
- **Coverage measured on VAL, not assumed:** 99.7% of (frame, class) pairs inside the nominal 95% range on 300
  synthetic VAL frames. It is conservative, partly because the classifier keeps about 0.014 on every class.
  Test frames were not used.
- **Concentration uses the exact Garwood Poisson interval:** the standard counting-chamber statistic, valid at 0
  (an upper bound of 3.69 organisms per volume) where +-2 sqrt(n) breaks down.
- **Concentration from confirmed counts, not estimated:** confirmed counts are the conservative, primary output;
  the estimated counts and their range stay visible next to them.
- **Calibration as imaged volume, or pixel size times field depth, plus a concentration factor:** these match
  how a lab knows its setup (counting-chamber volume, or a calibrated micrometre scale and a chamber depth). Giving
  both forms is refused, so the volume is never ambiguous.
- **Warning below 10 confirmed organisms and pooling in the CLI:** one image rarely holds enough organisms for a
  usable interval. Summing counts and volumes over the images of one sample is the correct way to narrow it, and
  that is a folder run.
- **Old saved results still render:** the reports and the web app check whether intervals exist, so analyses
  stored before this change keep working.

## Calibrated confidence, unknown flag, threshold sensitivity (2026-11-22)
- **Diagnosed the cliff before fixing it:** label smoothing 0.1 over 8 classes makes the training target for the
  true class 0.9125, so confident softmax piles up just above 0.9 and the 0.89 threshold is on that pile (13.4% of
  VAL crops within +-0.01).
- **Temperature scaling (T = 0.69 on VAL) makes probabilities honest, not the threshold stable:** ECE 0.084 to
  0.006 out of fold. It barely changes the order of the scores, so the crowding remains; the same rule on
  calibrated scores agrees with the current one on 97.5% of VAL crops. Claiming it fixed the fragility would be
  false.
- **The deployed status rule is unchanged:** all reported test numbers describe it, and test data is not touched
  again. Calibrated probabilities are reported next to it and used for the count ranges.
- **Calibrated count ranges adopted by a rule set before the check:** keep them if VAL end-to-end coverage stays
  at or above 95%. It was 96.7%, with a width of 0.58 against 1.65 for raw probabilities.
- **Threshold sensitivity shown, not hidden:** each result reports confirmed counts at -0.01 and +0.01 and the
  number of borderline detections, with a warning above 20%. The user sees when a count depends on the exact
  threshold.
- **Unknown flag = Mahalanobis distance on the penultimate features:** class means from TRAIN, one shared
  covariance with Ledoit-Wolf shrinkage (1,280 features, so the plain estimate would be unstable). It needs no
  retraining and no unknown examples.
- **Cut-off from known VAL crops only (95th percentile):** the open-set crops were already an evaluation set, so
  they were scored once, after every choice was fixed.
- **The flag never changes a count:** it is information for the user. Removing flagged items from the counts
  would drop 0.5% of confirmed known organisms on VAL, and that trade-off was never evaluated on test.
- **Reported the weak result as it is:** silently counted unknowns go only from 6.1% to 5.3%; most unknowns that
  get through look like a trained class in feature space as well.
- **One forward pass for features and logits:** the head is split from the backbone, and the logits were checked
  equal to the old path (0 argmax changes on 4,066 VAL crops), so no result moved.

## Label corrections and INT8 (2026-11-22)
- **Corrections are stored beside the result, never in it:** the model's counts, reports and report.json stay what
  the model produced; reviewed counts are shown next to them. An evaluation can always tell model output from
  human judgement.
- **Four kinds of verdict:** a trained class, debris (reject), other organism (an open-set example for later) and
  not an object (a detector false positive). These are the cases a retraining set needs, and the data a person
  can actually give.
- **One correction per detection, replaced on change:** the latest judgement counts; created_at is kept.
- **Snapshot of the detection in the correction row:** predicted label, status, probability, box, image size and
  crop padding. The export is self-describing and does not depend on today's config.
- **Export checks the image is the same:** a CLI source file replaced after the analysis has a different size
  and is skipped, never cropped under old labels. A same-size replacement would not be caught; storing a hash at
  analysis time would close that gap.
- **No orphans:** the existence check and the insert are one SQL statement, so a concurrent delete cannot leave a
  correction without its analysis.
- **Overlapping saves in the browser:** each answer carries the whole view, and only the answer to the latest
  request is applied. A late answer cannot undo a newer correction on screen.
- **Deleting an analysis deletes its corrections, and the confirmation says so:** the user chose deletion; the
  export exists to keep corrections first.
- **Nothing goes into the training data automatically:** the export is a folder to review. Mixing unreviewed
  labels, or images from the test period, into training would leak or poison the evaluation.
- **INT8 tried with the tools allowed (PyTorch FX, oneDNN), measured, and rejected on numbers:**
  - the classifier is 3x smaller, but 3.3x slower on 1 thread and loses 0.136 VAL macro F1;
  - the detector cannot be traced by torch.fx.
  - The rule was written after the first measurement; that is stated, and the result fails it by a wide margin.
- **The real INT8 route is onnxruntime (static QDQ) on the target board:** it needs a package that is not
  installed and was not installed without permission. EfficientNet may need quantization-aware training, which
  is retraining.
