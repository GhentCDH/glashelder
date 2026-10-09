# Embed

Generates image embedding vectors, keeps them in stores, and provides a web interface
for searching through them.

Supports CLIP (default, with text search) and DINOv2 (no text support).

## Commands

Paths and defaults come from `glashelder.toml` (run `uv run glashelder setup` to
write one), so the arguments are only needed to override it.

```bash
# embed the scraped artworks (paths.artworks) into paths.embeddings.reference
uv run glashelder embed artworks

# embed a directory of JPGs (paths.images.glashelder) into paths.embeddings.glashelder
uv run glashelder embed images

# save the top-k references of every glashelder image in the reference store,
# searching only what is new since the last run
uv run glashelder embed neighbours

# what a store (or old pickle) holds, and whether it fits in RAM or VRAM here
uv run glashelder embed info embeddings/artworks-clip.store

# turn an old pickle into a store, or merge several into one
uv run glashelder embed convert artworks-clip.pkl artworks-clip.store

# copy the 10 images that match a text prompt best into search_results/
uv run glashelder embed search "a dog in the park" --top-k 10

# start the search web interface
uv run glashelder embed serve
```

The web interface supports text-to-image search, image-to-image search, and similarity browsing.

## Stores

A store is a directory:

```
artworks-clip.store/
  meta.json      model, model_id, dim, dtype (float16 by default) and the image root
  vectors.bin    one fixed-size row per image
  hashes.bin     a hash of the image bytes each row was made from
  refs.txt       one reference per line: a URL, or a path relative to the root
```

- **Incremental.** `embed artworks` and `embed images` only read images whose
  reference is not in the store yet. `--verify` reads every image and embeds again
  the ones whose bytes changed. `--prune` drops rows that are no longer in the input.
- **Crash safe.** Every batch is written as soon as it is embedded. After an
  interrupted run, the next run continues where it stopped.
- **Stable indices.** A changed image is written again in its own row. Only
  `--prune` moves rows.
- **Portable.** Local images are stored relative to the directory they were embedded
  from. A reader joins them onto its own image directory, so a store made on the GPU
  machine works on a laptop with the images mounted elsewhere.
- **Larger than RAM.** The matrix is memory-mapped. Commands that search it say
  whether it went to VRAM and whether it fits in RAM. If it does not fit, every full
  scan reads it from disk again, so `artmatcher` searches all glashelder images in
  one pass instead of one pass per image.
- **Saved top-k.** `embed neighbours` saves the exact top-k references of every
  glashelder image in `<reference store>/neighbours/`. It stays exact as the stores
  grow: new references are searched and merged, new glashelder images get their own
  search. A store counts rows changed in place (`--verify`) or removed (`--prune`)
  as `edits`, and a saved result made before an edit is searched again from scratch.
  `artmatcher serve` and `batch-classify` use it, so they start at once.

## Fetching politely

Scraped images are fetched from the backends' servers. Every host gets its own
workers and rate limit, so a slow host does not hold up the others:

```toml
[images]
concurrency = 16 # connections over all hosts
per_host = 4     # in flight to one host
rate = 5.0       # requests per second to one host

[images.hosts."uploads.wikiart.org"]
rate = 20.0      # a host that can take more
```

`--concurrency`, `--per-host` and `--rate` override the `[images]` values for one
run. At start, `embed artworks` shows the limits for each host, and how long they
make the run take at the least.
