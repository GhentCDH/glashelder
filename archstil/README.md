# archstil

Segments buildings in images with [SAM3](https://huggingface.co/facebook/sam3)
(text-prompted "promptable concept segmentation").

`facebook/sam3` is a gated model: request access on Hugging Face and run
`hf auth login` (or set `HF_TOKEN`) before the first run.

```sh
uv run glashelder archstil segment IMAGES_DIR -o OUTPUT_DIR
# options: -p building -p church --threshold 0.5 --device cuda
```

For each image it writes to `OUTPUT_DIR`:

- `<name>_overlay.png`: the image with coloured masks, boxes and scores
- `<name>_masks/<i>.png`: one binary mask per detected building
- `<name>.json`: box, score, prompt and mask area per building

## Reference photos with their style (Onroerend Erfgoed)

`scrape` collects photos of Flemish heritage objects from the
[Inventaris Onroerend Erfgoed](https://inventaris.onroerenderfgoed.be) together with
their architectural style, dating and type. It reuses artscraper's HTTP client
(throttling, retries, backoff) but keeps its own records, since artscraper's
`Artwork` has no place for style.

```sh
uv run glashelder archstil styles                       # every style, with its id
uv run glashelder archstil scrape -o archstil_references \
    -s gotiek -s "art deco" -s modernisme --limit 200   # 200 objects per style
```

- `-s/--style` takes a label, id or URI and can be repeated; `--limit` is per
  style, so you get a balanced set. Without `-s`, every object is listed.
- `--typologie 513` keeps only one heritage type (513 = chapels).
- `--size medium|large|full`: 500 px, 1024 px (default) or the original.
- Objects whose record has no style are skipped (`--any-style` keeps them).

Output:

- `images/<image id>.jpg`: the object's primary photo
- `references.jsonl`: one line per object with `styles`, `datings`, `types`,
  `location`, the photo's `attribution` and `licenses`, and the `query` it was
  found with
- `cache/`: every object and image record as the API returned it

Rerun the same command to resume or to raise `--limit`: saved objects are skipped
and cached records are never fetched again.

The photos are CC BY 4.0 / Modellicentie gratis hergebruik; keep the `attribution`
line with any image you publish. The API's
[fair use policy](https://developers.onroerenderfgoed.be/docs/fair-use) asks for no
high parallel load, caching on our side, and waiting before retrying, hence the
defaults of 4 requests in flight and 2 requests per second per host.

## Label while downloading

Run these three side by side, in separate terminals, and start labelling right away:

```sh
uv run glashelder archstil verify archstil_references/segments            # http://localhost:5004
uv run glashelder archstil segment archstil_references/images \
    -o archstil_references/segments --watch                               # SAM3, as photos arrive
uv run glashelder archstil scrape -o archstil_references -s 4 -s 22 --limit 100
```

`segment` skips photos it already did, and `--watch` keeps looking for new ones every
10 seconds until Ctrl+C. The viewer picks up new photos and references as they come,
in the order they were segmented.

### The viewer

A reference photo often shows more than the heritage object: neighbouring houses, a
church behind it. Per photo you pick the SAM3 segment(s) that are the object, and its
class: one of the object's styles, by default the one it was downloaded for.

The largest segment is preselected. Only selected segments are coloured on the photo;
hovering a building shows it lightly.

| Key | Action |
| --- | --- |
| click, or `1`–`9` | select that one segment (replaces the selection) |
| `Shift` + click, or `Shift` + `1`–`9` | add or remove a segment, for an object SAM3 split in parts |
| right-click, `Ctrl` + click, or the `−` on a segment's row | cut that segment out of the selection (shown black), e.g. a neighbour inside a larger mask; again to undo |
| `A` / `C` | select all / clear (also the cut-outs) |
| `S` | switch class among the object's styles |
| `Enter` | verify and go to the next unchecked photo |
| `X` | no segment fits: reject |
| `H` (hold) | hide the masks |
| `U`, `←` / `→` | next unchecked photo, browse | The panel shows the object's
metadata: styles, dating, type, location, photo, attribution and licences.

### The master file

Every save writes `archstil_references/master.json` (next to `references.jsonl`; `-m`
to put it elsewhere), with all metadata and the label of every checked photo:

```json
{
  "classes": ["art deco", "neogotiek"],
  "counts": {"art deco": 1, "neogotiek": 1},
  "verified": 2, "rejected": 1,
  "images": [{
    "stem": "108018", "image": "images/108018.jpg", "width": 1024, "height": 738,
    "status": "verified", "class": "neogotiek", "class_id": 1,
    "annotation": {"bbox_xyxy": [53, 101, 969, 612], "polygons": [[x, y, x, y, ...]],
                   "area_px": 301266},
    "segments": [0, 4],
    "name": "Heilige Geestkapel", "location": "...", "styles": ["gotiek", "neogotiek"],
    "datings": ["..."], "types": ["..."], "object_id": 84, "object_uri": "...",
    "image_id": 108018, "image_uri": "...", "attribution": "...", "licenses": ["..."]
  }]
}
```

The annotation is the union of the chosen segments minus the cut-out ones
(`removed_segments`), so one instance per photo: its box and outline in pixels. The
cut is made on the mask pixels before the outline is traced, so box and polygons
leave the removed parts out. A cut-out inside the object leaves a hole; polygon
formats like YOLO's have no holes, so it is joined to the outline by a zero-width
cut and every polygon stays one simple ring. Specks (parts under 1% of the largest)
are dropped, and outlines are simplified to within 1 pixel. An outline can be several polygons when something stands
in front of the building, like trees. `class_id` follows the sorted `classes`, so it
can shift while new classes come in. Choices themselves live in
`segments/selections.json`; `archstil master archstil_references/segments` writes the
master file again from it.
