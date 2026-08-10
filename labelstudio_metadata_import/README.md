# labelstudio_metadata_import

Runs Glaswacht (`risk_assessment`) headlessly — no TUI — on a batch of glass
plate images, then imports only the images that need human review into Label
Studio as tasks. Everything goes over the Label Studio HTTP API, so nothing
needs to be mounted or shared with the Label Studio host.

Two different upload strategies are used, depending on the command — see
"How images actually get to Label Studio" below for why: `assess`/`import`/
`pipeline`/`test-run` embed each image directly in its task's JSON; `sample`
uploads to Label Studio's file storage first and references that.

## Setup

The token must never end up in a script or commit — set it as an environment
variable (or in a local `.env`, which is gitignored and picked up
automatically by `glashelder`'s `load_dotenv()`):

```bash
export LS_URL=https://labelstudio.jouwdomein.be   # default: http://localhost:8080
export LS_TOKEN=...                                 # Label Studio API token — never pass this on the CLI in a script
export LS_PROJECT_ID=1                              # used by assess/import/pipeline/test-run/health
export LS_SAMPLE_PROJECT_ID=2                       # used by sample — a separate project, so random spot-checks don't land in the review queue
```

`sample` reads its own project ID (`LS_SAMPLE_PROJECT_ID`/`--project-id`), independent from the rest — random calibration batches and Glaswacht-flagged review batches are meant to go to different Label Studio projects. Everything else shares `LS_PROJECT_ID`.

## Commands

```bash
# 0. Health check — verifies LS_URL is reachable, LS_TOKEN is valid, and
#    LS_PROJECT_ID is accessible, before you run anything against real data
uv run glashelder labelstudio-metadata-import health

# 1. Quick smoke test on a single image — run Glaswacht + preview the LS
#    import without touching Label Studio (dry-run by default)
uv run glashelder labelstudio-metadata-import test-run --image plate.jpg

# same, but also actually create the task in Label Studio
uv run glashelder labelstudio-metadata-import test-run --image plate.jpg --commit

# faster iteration: only run one model instead of all three VLMs
uv run glashelder labelstudio-metadata-import test-run --image plate.jpg --model qwen3

# 2. Run Glaswacht on a full folder, writing the Label Studio-ready CSV
uv run glashelder labelstudio-metadata-import assess --path ./glasplaten \
    --output-csv results.csv

# 3. Import images + a results CSV into Label Studio (dry-run: only writes
#    taken.json for inspection)
uv run glashelder labelstudio-metadata-import import \
    --image-path ./glasplaten --csv-path results.csv

# once taken.json looks right:
uv run glashelder labelstudio-metadata-import import \
    --image-path ./glasplaten --csv-path results.csv --commit

# 4. Or do assess + import in one go
uv run glashelder labelstudio-metadata-import pipeline --path ./glasplaten --commit

# 4b. Same, but resumable: process in checkpointed chunks, uploading each
#     chunk's flagged images as soon as that chunk finishes. Re-running the
#     exact same command later picks up where it left off.
uv run glashelder labelstudio-metadata-import pipeline --path ./glasplaten \
    --checkpoint-dir ./ckpt/batch1 --chunk-size 200 --commit

# 5. Random spot-check batch: push a random subset of a folder to Label
#    Studio, no Glaswacht run, no metadata — just the images
uv run glashelder labelstudio-metadata-import sample --path ./glasplaten --percent 5 --commit
uv run glashelder labelstudio-metadata-import sample --path ./glasplaten --count 25 --commit

# 5b. Same, but stratified: N images per collection instead of one flat draw
#     over the whole folder (so a big collection doesn't drown out a small one)
uv run glashelder labelstudio-metadata-import sample --path ./glasplaten \
    --group-csv GLAS_metadata.csv --count 10 --commit
```

Every command supports `--help` for the full option list.

## What gets sent to Label Studio

**Filtering:** an image is only included if at least one of the models that
ran rated it something other than `Safe`. If every model that ran rates an
image `Safe`, it's dropped — not written to the CSV, and never uploaded to
Label Studio. `assess`/`pipeline`/`test-run` all report how many images were
kept vs. skipped.

**Fields:** only `image`, and per model `model_N_rating` / `model_N_rationale`
— `category` and `majority_vote` are never sent. `N` is a fixed number per
physical model (not affected by the order `--model` flags are given on the
CLI), currently:

```
image, model_1_rating, model_1_rationale, model_2_rating, model_2_rationale, model_3_rating, model_3_rationale
model_1 = lavaguard
model_2 = qwen3
model_3 = pixtral
```

If you restrict a run to fewer models with `--model`, the numbering
compresses (e.g. `--model qwen3 --model pixtral` → `model_1` = qwen3,
`model_2` = pixtral); it always follows the model registry's order, not the
order the flags were typed in. Point your Label Studio labeling config at
these fields with `$model_1_rating`, `$model_2_rationale`, etc.

## Random sampling (`sample`)

Independent of Glaswacht: pick a truly random subset of a folder and push
just the images to Label Studio (tasks are `{"image": ...}`, no rating/
rationale fields — there's nothing to attach since no model ran). Useful for
spot checks or calibration batches, e.g. "grab 5% of this drop and get a
human baseline."

- Exactly one of `--percent` (0–100) or `--count` (absolute) is required.
- Sampling uses `secrets.SystemRandom` (the OS's CSPRNG, `os.urandom`-backed),
  not the default seeded PRNG — there's no `--seed` option and none is
  planned: the selection is intentionally not reproducible.
- Only the selected images are uploaded, same as `import`/`pipeline`.
- Uses its own `LS_SAMPLE_PROJECT_ID` (see Setup), not `LS_PROJECT_ID`.

**Stratified mode (`--group-csv`):** without it, `sample` draws `--count`/
`--percent` from the whole folder in one flat pool — a collection with 30,000
images ends up massively over-represented next to one with 80. Pass
`--group-csv` (the GLAS metadata export) to instead spread that same total
`--count`/`--percent` as evenly as possible *across every collection*, so
each one ends up roughly equally represented regardless of how large it is.
`--count`/`--percent` is always a total, not a per-collection amount — e.g.
`--count 200` selects 200 images altogether, not 200 per collection. If a
collection has fewer images available than its equal share, the shortfall is
handed to the other collections rather than left unused, so the requested
total is still met whenever enough images exist somewhere. Defaults match
that export: `--id-column pid` (matched against filenames in `--path` as
either the whole stem or the part before the first `_`, e.g. `pid=abc123`
matches `abc123_tiff.jpg`) and `--group-column "Het geheel waartoe de
glasplaat behoort"` (column U — the collection name). Both are overridable
for a differently-shaped CSV. Rows whose image isn't present in `--path` are
skipped and reported, not treated as an error — you'll usually only have a
subset of the full export downloaded locally.

## Resumable, checkpointed runs (`pipeline --checkpoint-dir`)

`assess`/`pipeline` normally process the *entire* folder through every model
before writing anything — for a big batch that can be hours of GPU work with
nothing on disk and nothing in Label Studio until the very end. If it crashes
partway (or you need to stop it), you lose that work and start over.

`pipeline --checkpoint-dir DIR [--chunk-size N]` (default `N=200`) instead:

1. Splits the folder into chunks of `--chunk-size` images.
2. For each chunk: runs every model over just that chunk, filters (same
   Safe-only-dropped rule as `assess`), and uploads the flagged images to
   Label Studio right away — you see results in LS as the run progresses,
   not only at the end.
3. Records the chunk's images as done in `DIR/manifest.json` only *after*
   that upload (or dry-run write) succeeds, and appends every processed
   image's row (flagged or not) to `DIR/results.csv`, a running full record.
4. Moves to the next chunk.

Re-running the exact same command (same `--checkpoint-dir`, same `--path`,
same `--model` selection) skips every image already recorded as done and
resumes with whatever's left — so an interrupted run redoes at most one
chunk's worth of work, not the whole batch. New images dropped into `--path`
between runs are picked up automatically. Resuming with a *different*
`--model` selection than the checkpoint was created with is rejected with a
clear error — mixing model sets under the same `model_N` slots would produce
nonsense — use the same models, or a fresh `--checkpoint-dir`.

Two things worth knowing about the failure-recovery granularity: a chunk is
the unit of both crash-safety and "done" — if the process dies mid-chunk,
that whole chunk (not just the remaining images in it) gets redone on
restart, and if one image within an otherwise-successful chunk fails
assessment (corrupt file, decode error), it's still marked done along with
the rest of that chunk rather than retried forever.

`DIR/results.csv` is always a valid `import --csv-path` input, so if a chunk
was only ever dry-run, you can commit it later with the plain `import`
command without re-running Glaswacht.

## How images actually get to Label Studio

Label Studio has a real, confirmed-against-a-live-instance restriction: its
"direct file upload" endpoint (`commit_to_project=false`, meant to pre-upload
a file so a task can later reference its storage path) flatly **refuses any
project whose labeling config has more than one data key** — the response is
`400 Bad Request`, `"Your label config has more than one data key and direct
file upload supports only one data key."` Any project that shows a human
reviewer the model's rating and rationale alongside the image (i.e. anything
built with this tool in mind) necessarily has more than one key — `$image`
plus `$model_1_rating`, `$model_1_rationale`, etc. — so that upload path is a
dead end here, not a bug to work around in the usual sense.

- **`assess`/`import`/`pipeline`/`test-run`** (anything carrying model
  ratings) sidestep it entirely: each image is read locally and embedded
  directly in its task's `data.image` field as a base64 `data:` URI, sent in
  the same JSON request as the rating fields. No separate upload step, no
  dependency on the project's label config shape. Tradeoff: ~33% larger
  request payloads (a 500KB image becomes ~650KB in the request), and every
  task's data permanently carries its own image rather than a lightweight
  reference to shared storage.
- **`sample`** has no metadata to attach — its target project only needs
  `$image` — so it still uses the more efficient upload-then-reference path:
  one file per HTTP request to Label Studio's storage (Label Studio's import
  endpoint only registers the *first* file when several are bundled under
  one multipart request — confirmed live: 3 files in one POST produced only
  1 `file_upload_id` — so batching there would silently drop most of every
  batch), then the resulting `/data/upload/{project}/...` path is referenced
  in a plain `{"image": ...}` task. After uploading, it briefly retries the
  file-uploads listing if a just-uploaded name isn't visible yet (eventual
  consistency; also follows pagination on larger projects).

## Notes

- `import`/`pipeline` always write `--tasks-json` (default `taken.json`)
  first; inspect it before re-running with `--commit`.
- `import` only imports the images actually listed in `--csv-path` — if
  `--image-path` is a folder with more images than the CSV (e.g. because
  `assess` filtered out the Safe ones), those extra files are ignored.
- Glaswacht's own `risk_assessment` package always writes its own full,
  unfiltered record — `combined_risk_assessment_results.csv` in the current
  directory, with `category` and `majority_vote` included — as a side effect
  of running the models. That's unrelated to (and unaffected by) this
  package's `--output-csv` and exists independently for archival purposes.
- Glaswacht's models are heavy (three VLMs, needs a strong GPU). For a fast
  test, restrict `--model` to one.
