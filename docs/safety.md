# Safety and privacy

The tool operates on a photo library the user cares about. Safety is therefore
part of the design, not a wrapper around it. This document describes the
guarantees, where they are enforced, and where they stop.

## Threat model

The primary risks are accidental, not adversarial:

- reading or writing the wrong directory;
- modifying, moving, or deleting originals;
- overwriting an existing file in the destination;
- silently uploading private images to a cloud provider;
- committing photos, thumbnails, labels, or personal paths to a repository;
- trusting a low-confidence model prediction as fact;
- leaving a half-written export that looks complete.

The tool is **not** a sandbox and does not protect against a malicious local
user, a compromised Python environment, or an attacker who already has your
account. The review server has no authentication; keep it on loopback.

## Input is read-only and path-restricted

`assert_input_allowed()` in `sports_sorter/safety.py` resolves the input path
and refuses protected locations unless the explicit
`--i-know-what-i-am-doing` flag is supplied:

```text
~/Downloads
~/Library
~/Movies
~/Music
~/Pictures/Photos Library.photoslibrary
/System
/Library
/private/var/db
/usr
/bin
/sbin
/etc
```

Resolution follows symlinks, so a symlink whose target is protected is blocked.
The override exists for unusual but legitimate layouts; it is not a normal
workflow and it does not make uploads or overwrites safe.

Every image is opened read-only (`open(..., "rb")`). The scanner hashes and
decodes a file, then closes it. `load_prepared()` copies the decoded pixels into
memory and returns an independent image; no writable handle to the original is
kept. EXIF orientation is applied because it is a technical display transform.
EXIF, filenames, and directory names are never classification features.

`verify_originals()` re-hashes every source file and reports `changed` or
`missing`. The reference run ended with 0 changed and 0 missing originals.

## Output cannot wrap or nest inside input

`assert_safe_output()` refuses a destination that:

- is the input directory,
- is a parent/ancestor of the input directory,
- is inside the input directory,
- is inside a protected location.

This prevents an export from polluting the source tree, racing the scanner, or
turning a typo into recursive copies. The check is performed before the
destination is created.

## No overwrites

The exporter copies, it never moves or deletes originals. A destination is
created with:

```python
os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
```

If the name already exists, `_unique_destination()` selects `name__1.ext`,
`name__2.ext`, and so on. An existing file is never truncated or replaced.
This also applies when the destination already contains unrelated files from a
previous workflow: those files are left untouched.

After copying, the destination is re-hashed and compared with the source. On a
mismatch the newly created destination is removed and the row is reported as an
error. A partial file created by this process is also removed if the copy is
interrupted, including Ctrl-C; it is never mistaken for a successful export.

The `exports` table records `(image_id, label, destination, bytes, sha256)` so
`verify` can later prove that the copy still matches and that the original has
not changed. The source hash is also stored on the image row from the initial
scan.

## No automatic deletion or cleanup

There is no command that deletes source photos and no automatic cleanup of the
output. If you want to remove a mistaken export, inspect the destination and do
it yourself. The tool never treats an empty folder as permission to prune.

Thumbnails are generated into `data/cache/thumbs` (or
`$SPORTS_SORTER_CACHE/thumbs`) and are local derived data. They can be deleted
manually; they are not exports and are never written next to the originals.

## Cloud uploads are opt-in, twice

The default pipeline is fully local. Cloud engines are OFF unless both are true:

1. `SPORTS_SORTER_ALLOW_CLOUD=1`, and
2. the provider key is present (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or
   `TYPESAFE_API_KEY`).

Jev image input additionally requires `TYPESAFE_DANGER_ZONE_IMAGES=1`.

`availability()` returns a visible reason when a gate is closed. There is no
silent fallback from a local engine to a cloud engine. Cloud engines are marked
`[SENDS PHOTOS OFF-DEVICE]` in `sports-sorter engines`.

If enabled, the image is resized, JPEG-encoded in memory, and sent to the
provider. Only the response text is stored in `raw`; API keys and image bytes
are not written to the database. The provider's retention, training, and
cross-border transfer terms apply. Never enable this for images of other people
without the right to share them.

The reference run did not enable any cloud engine and spent **$0** on APIs.

## What is stored locally

Assume the SQLite database is private:

- `images`: relative and absolute paths, size, SHA-256, perceptual hash,
  dimensions, format, mtime, decode errors.
- `runs`: engine name/version, options, timestamps.
- `predictions`: label, confidence, score maps, raw model evidence.
- `labels`: human ground truth and notes.
- `exports`: destination paths and hashes.

`data/` is gitignored, including `data/sports.db`, reports, thumbnails, and
labels. Do not publish it. The generated report is path-free by default:
absolute paths, filenames in misclassification lists, and duplicate file names
are included only with `--include-paths`, which also adds a warning field.

The `.gitignore` blocks image/media extensions, `data/`, `.env`, SQLite files,
and common secret/key files. CI also fails if an image/media file is tracked.

## Review UI

- Binds to `127.0.0.1` by default; no CDN, no external font, no build step.
- Serves only images already indexed in SQLite, looked up by primary key; a
  request cannot name an arbitrary path.
- Thumbnails are bounded (96–640 px) and generated from the indexed file.
- POST endpoints check the `Origin` header against the local host/port, reject
  unknown labels, and cap request bodies at 2 MB.
- There is **no login**. If you bind it to a non-loopback address (`--host
  0.0.0.0`), anyone who can reach the port can view photos and change labels.
  Do not expose it to a LAN or the public internet.

## Export verification

```bash
sports-sorter verify
sports-sorter verify --json data/verify-report.json
```

`verify` checks:

- destination exists;
- destination SHA-256 matches the recorded source hash;
- source still exists;
- source SHA-256 still matches the scan record.

A non-zero exit code means at least one check failed. Do not delete the
destination or "fix" a hash mismatch by editing the database; investigate the
filesystem and restore from backup if needed.

## Limitations

- Safety checks prevent the tool's own writes. They cannot stop another process
  from changing files while a run is in progress.
- The output directory could later be moved into the input directory manually;
  the guard only checks the path at export time.
- SQLite and thumbnails are private data; filesystem permissions are your
  responsibility.
- No encryption at rest and no secure deletion.
- Reports are only as anonymous as their aggregates; a tiny dataset can still
  be identifying.
- A human label is authoritative for export. Verify the review decisions before
  treating an export as ground truth.
