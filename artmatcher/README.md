# Artmatcher

Given a dataset of known artworks, find glass plates that contain any of them.

Uses a calibrated Random Forest classifier that combines three types of features:
- **Embedding similarity** distributions (gaps, z-scores, decay rates)
- **ORB geometric** verification (keypoint matches, inliers, homography)
- **Perceptual hashes** (pHash, dHash, wHash, aHash distances)

See [`notebooks/artmatcher/artmatcher.ipynb`](../notebooks/artmatcher/artmatcher.ipynb) for a walkthrough.

A reference artwork is identified by a string that is either a local path or an
image URL from an `artscraper` scrape, so no reference images need to be kept on
disk. Passing `--artworks` (or setting `paths.artworks`) joins each match back to
its title, artist, year and source link.

## Commands

Artifact locations and tuning defaults live in `glashelder.toml` at the repo root,
so the options below are only needed when overriding it.

```bash
# embed what is new in the scrape, fetching the images while the model works
uv run glashelder embed artworks

# label a starting set, then train the classifier on it
uv run glashelder artmatcher serve
uv run glashelder artmatcher train-classifier labeling_dataset.json model.pkl --workers 48

# classify the whole set
uv run glashelder artmatcher batch-classify -o classes.json

# review the results
uv run glashelder artmatcher serve --results-json classes.json
uv run glashelder artmatcher plot-results -i classes.json -o plots/
```

All backends share one reference store. `--source` (repeatable) matches against the
artworks of some backends only, for example `--source wikiart --source kikirpa`.

At start, `serve` and `batch-classify` search every glashelder image against the
references in one pass. So a reference store larger than RAM is read from disk once
per start, not once per image. They also report whether the references went to VRAM
and whether they fit in RAM.
