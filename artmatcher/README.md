# Artmatcher

Given a dataset of known artworks, find glass plates that contain any of them.

Uses a calibrated Random Forest classifier that combines three types of features:
- **Embedding similarity** distributions (gaps, z-scores, decay rates)
- **ORB geometric** verification (keypoint matches, inliers, homography)
- **Perceptual hashes** (pHash, dHash, wHash, aHash distances)

See [`notebooks/artmatcher/artmatcher.ipynb`](../notebooks/artmatcher/artmatcher.ipynb) for a walkthrough.

## Commands

```bash
# train the classifier on a labeled dataset
uv run glashelder artmatcher train-classifier

# classify image pairs in batch
uv run glashelder artmatcher batch-classify

# start the web interface
uv run glashelder artmatcher serve
```
