# Engines

An engine receives an image path and the taxonomy config, and returns a
`Prediction` with a label id, a confidence, per-label scores, and raw evidence.
Persistence, fusion, review, export, and verification are engine-agnostic.

The registry lives in `sports_sorter/engines/base.py`; implementations are
imported lazily so an optional dependency never breaks the core CLI.

## Availability

```bash
sports-sorter engines
sports-sorter engines --json
```

A `--` prefix means unavailable. The reason is printed below each engine, for
example "needs the 'local' extra", "macOS only", or "cloud engines disabled".
`[SENDS PHOTOS OFF-DEVICE]` is always shown for a cloud engine.

## Local engines

All local engines run offline after their weights/helper are cached. None of
them sends image pixels or embeddings anywhere.

### `siglip2` — default primary signal

- Model: `google/siglip2-base-patch16-224`
- Type: zero-shot image/text classification
- Requires: `pip install -e ".[local]"` (PyTorch + Transformers)
- Device: Apple MPS when available, otherwise CPU; override with
  `--device cpu|mps|cuda|auto`
- Default weight: `1.0`
- Reference accuracy: 20/23 (87.0%)

The taxonomy prompts are embedded with the text tower and compared with each
image. Prompt scores are averaged per label. A separate abstain-prompt group
(portraits, medal ceremonies, group photos, no equipment) competes with the
sports; if it wins, the result is `unclassified`.

**Important implementation detail:** SigLIP's text tower has no attention mask,
so it must receive fixed-length text sequences. The engine uses
`padding="max_length"` plus truncation for SigLIP. With dynamic padding every
image collapsed to the same label in early testing. CLIP uses normal dynamic
padding.

### `clip` — second zero-shot signal

- Model: `laion/CLIP-ViT-B-32-laion2b-s34b-b79k`
- Type: zero-shot image/text classification
- Requires: the `local` extra
- Default weight: `0.8`
- Reference accuracy: 19/23 (82.6%)

CLIP is fast and complements SigLIP on this set, but it over-predicted
volleyball and table tennis in the reference run. It is not a substitute for
the abstain logic.

### `ocr_text` — content overlay signal

- API: Apple Vision `VNRecognizeTextRequest` (Japanese + English)
- Platform: macOS with `swiftc`; the helper is compiled once into
  `~/.cache/sports-image-sorter/vision_helper`
- Default weight: `0.5` (but it only votes when a keyword is found)
- Reference coverage: 3/23 (13%); 3/3 correct on covered images

This engine reads scoreboards, banners, jerseys, and venue signage and matches
configured `ocr_keywords`/aliases. It never reads the filename or EXIF. When no
keyword is present it produces no prediction at all (`label_id=None`), so it
does not create false abstain votes.

OCR is a **content** signal: a scoreboard that says "バレーボール" is visual
evidence in the image, not metadata. It is still fallible; Japanese venue text
can be small, stylized, or absent.

### `apple_vision` — free on-device sanity check

- API: Apple Vision `VNClassifyImageRequest`
- Platform: macOS with `swiftc`
- Default weight: `0.2`
- Reference accuracy: 12/23 (52.2%)

Apple's bundled classifier is fast, free, and generic (~1300 everyday classes).
It knows "volleyball" and "tennis" but has no "badminton" concept, and it labels
portraits "person". When none of its classes map onto the taxonomy it returns
`unclassified` with **confidence 0.0**. That is deliberate: "no opinion" must
not act as a strong abstain vote in fusion.

### `sports100_vit` — advisory supervised classifier

- Model: `vumin001/vit-base-sports-classification`
- Type: ViT fine-tuned on 100 sports classes
- Requires: the `local` extra
- Default weight: `0.25`
- Reference accuracy: 5/23 (21.7%)

Classes that map to the taxonomy are summed; every unmatched class becomes
`other` (`その他の競技`). The model's fixed label set and domain shift made it
weak on this small, badminton/volleyball-heavy set, hence the low weight.

### `sportsnet7` — advisory supervised classifier

- Model: `prithivMLmods/SportsNet-7`
- Type: SigLIP 2 fine-tune on seven sports
- Requires: the `local` extra
- Default weight: `0.25`
- Reference accuracy: 6/23 (26.1%)

It recognized badminton reasonably but mapped most volleyball to badminton or
`other`. Useful as one weak vote, not as a decision-maker.

### `fake` — tests only

A deterministic stub that hashes the path and returns a label. It never opens
pixels and must not be used for a real sorting run. It exists so the full CLI,
database, review, export, and verification flow can be tested without models.

## Default fusion weights

```python
{
    "siglip2": 1.00,
    "clip": 0.80,
    "ocr_text": 0.50,
    "sports100_vit": 0.25,
    "sportsnet7": 0.25,
    "apple_vision": 0.20,
}
```

These are the **post-hoc** weights selected after the reference errors were
inspected. The pre-registered weights scored 78.3%; the revised weights scored
91.3%. See [evaluation.md](evaluation.md) for the exact sets and the caveat.

A recommended lower-cost setup is:

```bash
sports-sorter classify --engine siglip2 --engine clip --engine ocr_text --engine apple_vision
```

It avoids the two supervised models that contributed little on the reference
set. Fusion simply ignores engines that are absent or weight 0.

## Cloud engines (opt-in, unverified)

`gemini`, `openai`, and `jev` are implemented to show that the pipeline is
provider-agnostic. They were **not executed** for the original reference run.
There was no approval to upload private photos and no API key was used, so that
evaluation did not measure their accuracy or cost. A later, separately approved
local-vs-Gemini comparison used `gemini-3.8-flash`; its results are recorded in
the project's private run data, not in this repository's distributed fixtures.

The Gemini default is `gemini-3.8-flash`; override it with `GEMINI_MODEL`.
Gemini 2.5 Flash was retired for new users and returns HTTP 404.

Every cloud engine requires:

1. `SPORTS_SORTER_ALLOW_CLOUD=1` in the environment, **and**
2. the provider key: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or
   `TYPESAFE_API_KEY`.

Jev additionally needs `TYPESAFE_DANGER_ZONE_IMAGES=1`; its image input is an
experimental "danger zone" API and its response shape may change.

When enabled, the image is decoded, oriented, resized (max side 1024, or 768 for
Jev), JPEG-encoded in memory, and sent to the provider. No API key or photo is
written to the database; only the model's textual answer is stored in `raw`.
Sending private photos to a third party is a human decision. Review the
provider's retention/training terms before opting in.

Relevant files: `sports_sorter/engines/external.py`, `.env.example`.

## Adding an engine

1. Subclass `Engine` (or a shared helper) in a module.
2. Set `name`, `version`, `description`, `sends_images_offsite`, and
   `requires_credentials` accurately.
3. Implement `availability()` and `classify(path)`. Return
   `Prediction(label_id=..., confidence=..., scores=...)`; use `self._error(exc)`
   for per-image failures and `label_id=None` for "no opinion", not a fake
   sport.
4. Register it in `_REGISTRY` in `base.py`.
5. Add a synthetic unit test using `FakeEngine` or a pure function. Never add a
   real photo fixture.

Keep `version` and prompt/config options stable enough for the prediction cache
key. `prediction_config_hash()` intentionally includes prompts, taxonomy,
preprocess settings, engine name, version, and options, but **excludes fusion
weights and thresholds**.

## Costs and resource use

Local engines have no per-image API cost. The only external cost is the one-time
model download from Hugging Face and local CPU/GPU time. Reference run: **$0**
API spend, no image uploads.

Measured warm runtimes for 23 images on an Apple M1 / 16 GB machine:

| engine | seconds | observed peak RSS |
|---|---:|---:|
| SigLIP 2 | 6.4 | ~935 MB |
| CLIP | 2.6 | ~681 MB |
| Sports-100 ViT | 2.3 | ~878 MB |
| SportsNet-7 | 2.3 | not separately recorded |
| Apple Vision | 3.5 | ~333 MB |
| OCR | 3.0 | not separately recorded |

The first Apple Vision run also compiled the Swift helper (~35 s). Model loading
and first-run downloads are outside these numbers. No swap was observed.
