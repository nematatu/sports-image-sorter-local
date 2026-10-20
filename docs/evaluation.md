# Evaluation

This document records the reference run honestly, including the ways it is
biased. The private photos and their labels are **not** distributed with this
repository.

## Dataset

- 23 images from a personal sports-photo folder.
- 19 JPEG, 4 WebP; 0 decode errors.
- Ground truth after visual inspection by one annotator:
  - badminton: 8
  - volleyball: 11
  - table tennis: 2
  - unclassified: 2 (no identifiable sport)
- Labels were assigned from image content (equipment, court/venue, uniform,
  action, signage), never from filenames or EXIF.
- No image, thumbnail, filename, or absolute path is committed.
- Duplicate check: one exact duplicate pair (two byte-identical WebP files);
  zero near-duplicate pairs at dHash distance <= 6.

The exact-duplicate pair means two rows are not statistically independent
examples. This is a minor additional source of optimism.

## Protocol

The following was fixed before looking at evaluation results:

- Default taxonomy and prompts in `sports_sorter/config.py`.
- Preprocessing: max side 1024, RGB, EXIF orientation applied technically.
- Zero-shot engines averaged the full image and one centered 75% square crop
  (`--crops center`), batch size 4, device `auto` (MPS).
- Fusion: top-label vote weighted by `engine_weight * confidence`; abstain
  votes counted separately; thresholds `min_confidence=0.35` and
  `ambiguity_margin=0.08`.
- Accuracy is exact label match, including `未分類` as a real class.
- The report includes a 95% Wilson interval for accuracy.
- Human labels are written only to the `labels` table; engine output is never
  used as truth.

All six local engines and fusion were evaluated on all 23 images. OCR covered
only 3 images because it abstains when no configured keyword is visible.

## Pre-registered fusion vs revised weights

The first fusion weights were:

```python
{
    "sports100_vit": 1.0,
    "sportsnet7": 0.9,
    "clip": 1.0,
    "siglip2": 1.1,
    "apple_vision": 0.4,
    "ocr_text": 0.6,
}
```

They scored **18/23 = 78.3%** (95% Wilson CI 58.1–90.3%). The five errors were
unclassified -> table tennis, badminton -> tennis, badminton -> tennis,
volleyball -> unclassified, and volleyball -> unclassified.

After inspecting those errors, the weights were changed to reflect each
engine's measured scope and reliability:

```python
{
    "siglip2": 1.0,
    "clip": 0.8,
    "ocr_text": 0.5,
    "sports100_vit": 0.25,
    "sportsnet7": 0.25,
    "apple_vision": 0.2,
}
```

This scored **21/23 = 91.3%** (95% Wilson CI 73.2–97.6%), with macro F1 0.680.

**This second number is post-hoc and optimistic.** It is useful as a regression
reference for this library and this pipeline, not as an unbiased estimate of
performance on new sports libraries. A clean evaluation would freeze the
weights before collecting a new labelled sample.

## Per-engine results

| engine | correct / 23 | accuracy | 95% CI | macro F1 | coverage |
|---|---:|---:|---:|---:|---:|
| fusion (revised weights) | 21 | 91.3% | 73.2–97.6% | 0.680 | 100% |
| siglip2 | 20 | 87.0% | 67.9–95.5% | 0.520 | 100% |
| clip | 19 | 82.6% | 62.9–93.0% | 0.496 | 100% |
| apple_vision | 12 | 52.2% | 33.0–70.8% | 0.342 | 100% |
| sportsnet7 | 6 | 26.1% | 12.5–46.5% | 0.082 | 100% |
| sports100_vit | 5 | 21.7% | 9.7–41.9% | 0.133 | 100% |
| ocr_text | 3 / 3 covered | 100% of covered | 43.8–100%* | 1.000 | 13.0% (3/23) |

\* Conditional on the three images where OCR found a keyword. It says nothing
about the other 20, so 13% coverage is essential context.

### Fusion detail

| truth | predicted | count |
|---|---|---:|
| badminton | badminton | 7 |
| badminton | tennis | 1 |
| volleyball | volleyball | 11 |
| table_tennis | table_tennis | 2 |
| unclassified | unclassified | 1 |
| unclassified | table_tennis | 1 |

Per-label fusion metrics:

| label | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| badminton | 1.000 | 0.875 | 0.933 | 8 |
| volleyball | 1.000 | 1.000 | 1.000 | 11 |
| table_tennis | 0.667 | 1.000 | 0.800 | 2 |
| unclassified | 1.000 | 0.500 | 0.667 | 2 |

Macro F1 is 0.680. Tennis has zero support in the gold set, so it is excluded
from the macro average; predictions of tennis still count as errors.

### The two fusion errors

Image order is the database id; filenames and paths are intentionally omitted.

| id | truth | prediction | confidence | interpretation |
|---:|---|---|---:|---|
| 1 | unclassified | table_tennis | 0.521 | Weak visual cues were enough for a sports label; the abstain bucket did not win. |
| 3 | badminton | tennis | 0.477 | Racket + net + indoor court are genuinely similar at this scale. |

Reviewing these two is the intended workflow: the confidence is displayed, and
a human correction in `review` overrides the model at export time.

## Separate cloud comparison: Gemini 3.8 Flash

After the local-only reference run, a second run was explicitly approved to
send the same 23 private images to Google's Gemini API. It used
`gemini-3.8-flash` through the existing `gemini` engine and was evaluated
against the same gold labels. It was **not** added to fusion.

| metric | local fusion | Gemini 3.8 Flash |
|---|---:|---:|
| correct | 21/23 | 21/23 |
| accuracy | 91.3% | 91.3% |
| 95% Wilson CI | 73.2–97.6% | 73.2–97.6% |
| macro F1 | 0.680 | **0.724** |
| total inference time* | 44.6 s for all six engines | 137.5 s |
| per-image time | 1.94 s (six engines) | 5.98 s |
| API cost | $0 | provider billing; usage metadata not captured |

The errors were different:

- Gemini: unclassified -> volleyball (0.88) and unclassified -> badminton
  (0.98). It got every image with an actual sport correct, but had **zero
  recall on `未分類`**.
- Local fusion: unclassified -> table tennis (0.521) and badminton -> tennis
  (0.477).

\* Engine wall time from the run ledger; model download/load is excluded. The
local number is the sum of six separate engine runs, so process startup and
model loading would add to an end-to-end wall-clock measurement.

This is a real trade-off rather than a clean win. Gemini was better at the
sport-vs-sport boundary but confidently forced both ceremony/portrait images
into a sport folder, which violates the project's conservative `未分類` policy.
For a workflow where a human reviews every image, Gemini's higher macro F1 may
be preferable; for unattended safety, local fusion's abstain behavior is
preferable.

Gemini 2.5 Flash was retired for new API users and returned HTTP 404; the
engine default is now `gemini-3.8-flash`, overridable with `GEMINI_MODEL`.

## What was exported

All 23 images received human labels, so `export` used **human label > fusion >
engine** and produced:

- badminton: 8
- volleyball: 11
- table tennis: 2
- unclassified: 2

This is why the exported tree is not a raw fusion snapshot: it is the corrected
set after review. The export ledger verified 23/23 copies by SHA-256, with zero
modified or missing source files. The destination already contained 23
unrelated files; they were preserved, and new name collisions were written with
`__1` suffixes rather than overwriting.

## Reproducing the method

The repository ships tests and a synthetic-data generator, but not the private
photos or labels. On your own labelled sample:

```bash
sports-sorter --db data/sports.db --input "$INPUT" scan
sports-sorter --db data/sports.db --input "$INPUT" classify \
  --engine siglip2 --engine clip --engine sports100_vit \
  --engine sportsnet7 --engine apple_vision --engine ocr_text
sports-sorter --db data/sports.db finalize
# Add human labels via the review UI or a CSV with rel_path,label:
sports-sorter --db data/sports.db label --csv labels.csv
sports-sorter --db data/sports.db evaluate --all --json data/eval-all.json
sports-sorter --db data/sports.db report --out data/report.md
```

`evaluate` reports `n/a` rather than inventing accuracy when no human labels
exist. Fusion weights and thresholds are in the taxonomy config; changing them
only requires re-running `finalize`, not re-running the models.

## Runtime, memory, and cost

Environment: macOS 27.0, arm64 Apple M1, 16 GB RAM, Python 3.14.8, PyTorch
2.14.1, Transformers 5.20.2, Pillow 13.0.0.

Warm per-engine runtime for all 23 images, including the center-crop pass:

| engine | seconds | peak RSS |
|---|---:|---:|
| siglip2 | 6.4 | ~935 MB |
| clip | 2.6 | ~681 MB |
| sports100_vit | 2.3 | ~878 MB |
| sportsnet7 | 2.3 | not separately recorded |
| apple_vision | 3.5 | ~333 MB |
| ocr_text | 3.0 | not separately recorded |

No swap was observed. The first Apple Vision call compiled the Swift helper
(~35 s). First-time Hugging Face downloads and model loading are additional.

API cost: **$0**. No cloud engine was enabled; no image was uploaded.

## Limitations

1. **Small sample.** 23 images, two dominant classes, one annotator. The 95% CI
   for 91.3% is roughly 73–98%; the true value could be much lower.
2. **Post-hoc weights.** The reported fusion score is exploratory, not a
   pre-registered confirmatory result.
3. **Distribution shift.** The set is indoor racket sports. Basketball,
   athletics, and swimming have configured prompts but zero gold examples, so
   their accuracy is unknown.
4. **Single-label truth.** A photo showing a venue with multiple sports is
   forced to one label; the protocol has no "multi-label" target.
5. **MPS nondeterminism.** Minor floating-point differences can change a
   borderline fusion decision; use `--device cpu` if bit-stable runs matter.
6. **Cloud untested.** Gemini/OpenAI/Jev implementations are code-complete but
   unverified against this dataset.
7. **No confidence calibration.** Engine confidences are model outputs, not
   calibrated probabilities; fusion weights are hand-set, not learned.
8. **Duplicate leakage.** One exact pair is counted twice, slightly inflating
   the apparent result.
