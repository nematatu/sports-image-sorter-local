# sports-image-sorter

[![CI](https://github.com/nematatu/sports-image-sorter/actions/workflows/ci.yml/badge.svg)](https://github.com/nematatu/sports-image-sorter/actions/workflows/ci.yml)

Classify sports photos into per-sport folders with local-first vision engines,
a review UI for corrections, and a copy-only export that never touches the
originals.

The project was built for a real 23-image library and keeps the boring safety
properties explicit: **read-only input, no overwrites, no filename/EXIF
classification, no image uploads by default, and a valid `未分類` outcome**.

日本語の最小手順は [クイックスタート（日本語）](#クイックスタート日本語) にあります。

## What it does

```text
scan -> classify (one or more engines) -> finalize (weighted fusion)
     -> review/correct -> export (copy only) -> verify (SHA-256)
```

- **Local by default.** Apple Vision / OCR, CLIP, SigLIP 2, and two supervised
  sports classifiers run on the machine. Cloud engines are implemented but
  disabled behind two explicit gates.
- **Content-only decisions.** Filenames and EXIF are never features. EXIF
  orientation is applied only so the model and the review UI see the same
  pixels a human sees.
- **Conservative fusion.** Ambiguous, low-confidence, portrait, ceremony, or
  equipment-free images go to `未分類` instead of being forced into a sport.
- **Never destructive.** Originals are opened read-only. Exports are copies
  created with `O_CREAT | O_EXCL`; an existing destination is never overwritten
  (a collision becomes `name__1.ext`).
- **Resumable.** SQLite stores hashes, engine predictions, human labels, and
  the export ledger. Re-running a command skips completed work.
- **Auditable.** `verify` re-hashes both exported copies and originals.

## Quick start

Requirements: Python 3.10+, Pillow. The neural engines also need the optional
`local` extra (PyTorch + Transformers, several hundred MB of model weights on
first use). Apple Vision requires macOS with the Xcode Command Line Tools.

```bash
git clone https://github.com/nematatu/sports-image-sorter.git
cd sports-image-sorter

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local,dev]"     # use `pip install -e .` for the core CLI only
sports-sorter engines             # show availability and off-device warnings
```

Run it on a directory you own:

```bash
export INPUT="$HOME/Pictures/sports"
export OUTPUT="$HOME/Pictures/sports-sorted"

sports-sorter --input "$INPUT" scan
sports-sorter --input "$INPUT" classify \
  --engine siglip2 --engine clip --engine ocr_text --engine apple_vision
sports-sorter finalize
sports-sorter review --open       # optional: correct labels in a local browser UI
sports-sorter export --out "$OUTPUT"
sports-sorter verify
sports-sorter report --out data/report.md
```

`classify` accepts `--engine` repeatedly or comma-separated. The default engine
list is in `sports_sorter/config.py`; pass an explicit subset when a model is not
installed or not useful for your library.

### クイックスタート（日本語）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"

INPUT="$HOME/Pictures/sports"
OUTPUT="$HOME/Pictures/sports-sorted"

sports-sorter --input "$INPUT" scan
sports-sorter --input "$INPUT" classify --engine siglip2 --engine clip \
  --engine ocr_text --engine apple_vision
sports-sorter finalize
sports-sorter review --open
sports-sorter export --out "$OUTPUT"
sports-sorter verify
```

- 元画像は読み取り専用で、移動・削除・上書きはしません。出力はコピーです。
- 判定に自信がない画像は `未分類` に入ります。`未分類` は失敗ではなく、人が確認するための正常な結果です。
- 人名・ファイル名・EXIF から競技を推測しません。看板やスコアボードの OCR だけを補助信号として使います。
- `review` で修正した人間ラベルは、モデルの判定より優先して `export` に使われます。

## Reference result

A private, manually labelled set of 23 photos (8 badminton, 11 volleyball,
2 table tennis, 2 unclassified; 19 JPEG + 4 WebP) was used. No images,
thumbnails, filenames, or absolute paths are included in this repository.

| system | accuracy (95% Wilson CI) | macro F1 | coverage |
|---|---:|---:|---:|
| **fusion (default weights)** | **91.3%** (21/23; 73.2–97.6%) | **0.680** | 100% |
| SigLIP 2 base zero-shot | 87.0% (20/23) | 0.520 | 100% |
| CLIP ViT-B/32 LAION zero-shot | 82.6% (19/23) | 0.496 | 100% |
| Apple Vision classify | 52.2% (12/23) | 0.342 | 100% |
| SportsNet-7 | 26.1% (6/23) | 0.082 | 100% |
| Sports-100 ViT | 21.7% (5/23) | 0.133 | 100% |
| Apple Vision OCR keywords | 100% of covered (3/3) | 1.000 | 13% (3/23) |

The two fusion errors were `未分類 -> 卓球` (confidence 0.521) and
`バドミントン -> テニス` (confidence 0.477). See
[docs/evaluation.md](docs/evaluation.md) for the protocol, the pre-registered
weights (78.3%), the post-hoc revised weights (91.3%), timings, memory, and
limitations.

> The 91.3% number is **not** an unbiased benchmark. The weights were revised
> after inspecting the 23 errors, and a single annotator labelled the set. Use it
> as a regression reference, not as a claim about arbitrary sports libraries.

## How it works

1. **Scan** hashes every supported image, records dimensions/format, and stores
   a perceptual dHash for duplicate reporting. Decode failures are retained as
   rows with an error, not silently dropped.
2. **Classify** runs each requested engine. Each prediction is cached by
   `(image, engine, engine-version, prompt/config, options)`, so changing fusion
   weights does not force an expensive model re-run.
3. **Finalize** gives every engine a configurable weight and combines its
   top-label confidence. It abstains when the abstain bucket wins, when the
   winner is below `min_confidence`, or when the top two labels are inside
   `ambiguity_margin`.
4. **Review** is a stdlib HTTP server bound to `127.0.0.1` by default. It
   renders thumbnails, shows engine disagreements, and writes only the human
   `labels` table.
5. **Export** resolves each image as human label > fusion > fallback engine,
   copies it into the matching configured directory, and records a hash in the
   export ledger. All configured sport directories are created, even empty ones.
6. **Verify** re-hashes exported files and originals. A changed original is a
   problem, not a silent success.

## Engines

| name | type | platform / requirements | off-device | default weight |
|---|---|---|---:|---:|
| `siglip2` | zero-shot | `local` extra; `google/siglip2-base-patch16-224` | no | 1.00 |
| `clip` | zero-shot | `local` extra; LAION CLIP ViT-B/32 | no | 0.80 |
| `ocr_text` | OCR keyword matcher | macOS + `swiftc` | no | 0.50 |
| `sports100_vit` | supervised | `local` extra; Sports-100 ViT | no | 0.25 |
| `sportsnet7` | supervised | `local` extra; SportsNet-7 | no | 0.25 |
| `apple_vision` | generic classifier | macOS + `swiftc` | no | 0.20 |
| `gemini` | cloud vision | `SPORTS_SORTER_ALLOW_CLOUD=1` + key | **yes** | 0 (not run) |
| `openai` | cloud vision | `SPORTS_SORTER_ALLOW_CLOUD=1` + key | **yes** | 0 (not run) |
| `jev` | cloud choice model | cloud gates + `TYPESAFE_DANGER_ZONE_IMAGES=1` | **yes** | 0 (not run) |
| `fake` | deterministic stub | none | no | 0 (tests only) |

Cloud engines were **not executed** for the reference run: there was no approval
to upload private photos and no API key was used. Total API cost: **$0**.

Details, model IDs, and how to add an engine: [docs/engines.md](docs/engines.md).

## Custom taxonomy

The taxonomy, prompts, OCR keywords, threshold, and weights are data, not code.

```bash
sports-sorter init-config --out config.json
$EDITOR config.json
sports-sorter --config config.json --input "$INPUT" classify --engine siglip2
# or:
SPORTS_SORTER_CONFIG=config.json sports-sorter --input "$INPUT" scan
```

Each label has an `id`, Japanese directory name (`name_ja`), English name,
aliases, zero-shot `prompts`, visual `evidence`, and OCR keywords. Keep `id`
stable once a database exists. Add a sport by adding one object under `labels`;
add abstain behavior through `abstain_prompts`. Re-run `classify` when prompts
change (the cache key changes automatically).

## Safety model

- Input directories such as `~/Downloads`, `~/Library`, `~/Movies`, `~/Music`,
  `/System`, `/Library`, and `/usr` are blocked by default. The explicit
  `--i-know-what-i-am-doing` flag is the only override.
- The output directory may not be the input, contain the input, or live inside
  it. Export paths are created with exclusive semantics; existing files survive.
- Nothing is moved or deleted. There is no automatic cleanup of output.
- Cloud engines require both `SPORTS_SORTER_ALLOW_CLOUD=1` and a provider key.
  Jev image input additionally requires `TYPESAFE_DANGER_ZONE_IMAGES=1`.
- Reports omit absolute paths and filenames unless `--include-paths` is passed.
  Treat the SQLite database (`data/sports.db`) and thumbnails (`data/cache`) as
  private local data.
- The review server has no authentication and is intended for loopback use.
  Do not bind it to a public interface.

Full rationale and recovery notes: [docs/safety.md](docs/safety.md).

## CLI reference

```text
sports-sorter [--db PATH] [--config PATH] [--input DIR] <command>

scan        Index and hash images (read-only)
classify    Run engines; caches results and resumes after interruption
finalize    Fuse cached predictions into the final label
export      Copy images into per-sport folders (no overwrites)
verify      Re-hash exports and originals
evaluate    Accuracy against human labels for one engine
compare     Accuracy table for all engines
engines     List availability, version, and off-device status
dedupe      Exact and perceptual near-duplicate report
label       Set human labels by image id/path or CSV
review      Start the local review UI
report      Aggregate JSON/Markdown report (path-free by default)
init-config Write an editable taxonomy JSON
```

Useful flags: `--dry-run` (plan without writing), `--force` (ignore prediction
cache), `--limit N`, `--no-scan` (classify from the current index), `--json`,
`--exclude-unclassified`, and `--recursive`.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

All tests use synthetic Pillow drawings and a fake engine; no real photos are
required. CI runs the core test suite on Python 3.10, 3.12, and 3.13, rejects
committed image/media files, and checks that the package compiles. The optional
neural and macOS engines are not exercised in CI.

Generate a small synthetic set for a pipeline smoke test:

```bash
python scripts/gen_synthetic_images.py --out data/synthetic
sports-sorter --db data/smoke.db --input data/synthetic scan
sports-sorter --db data/smoke.db classify --engine fake --no-scan
sports-sorter --db data/smoke.db finalize
sports-sorter --db data/smoke.db export --out data/smoke-sorted
```

The synthetic images are **not** an accuracy benchmark.

## Repository layout

```text
sports_sorter/          package (CLI, DB, engines, exporter, review server)
sports_sorter/web/      no-build local review UI
sports_sorter/tools/    Swift helper for Apple Vision / Vision OCR
tests/                  synthetic-data tests, no network or model weights
scripts/                synthetic pipeline smoke-test generator
docs/                   engine, evaluation and safety documentation
```

## Known limitations

- The reference set is small (23 images), single-annotator, and dominated by
  badminton/volleyball. Confidence intervals are wide.
- Fusion weights were tuned after seeing this set; the 91.3% figure is
  exploratory. Re-evaluate on your own labelled sample.
- `未分類` deliberately trades coverage for precision; portraits, ceremonies,
  and heavily cropped action shots may need review.
- Sports-100 and SportsNet-7 have fixed label sets that only partially overlap
  the default taxonomy; their votes are intentionally low-weight.
- Cloud engines are implemented but unverified in this release.
- The tool copies files; it does not rename, deduplicate, or delete them.

## License

MIT. See [LICENSE](LICENSE).
