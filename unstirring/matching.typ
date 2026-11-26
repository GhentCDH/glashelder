#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set page(width: 32cm, height: auto, margin: 1.5em)
#set par(leading: 1mm)
#set text(font: "IBM Plex Sans")
#show heading.where(level: 1): set block(below: 1.2em)

#let db = (shape: fletcher.shapes.cylinder)
#let storage = (stroke: (thickness: 1pt, dash: "dashed"), corner-radius: 4pt)
#let person = (stroke: 1pt + blue, fill: blue.lighten(90%), shape: fletcher.shapes.pill)
#let gpu = (stroke: 1pt + green.darken(30%), fill: green.lighten(90%), corner-radius: 4pt)
#let new = (stroke: 1pt + teal.darken(20%), fill: teal.lighten(90%), corner-radius: 4pt)
#let ask = (shape: fletcher.shapes.diamond, fill: yellow.lighten(85%))
#let data-edge = (stroke: 1pt + luma(40%))
#let manual-edge = (stroke: (paint: blue, thickness: 1pt, dash: "dashed"))
#let new-edge = (stroke: 1pt + teal.darken(20%))
#let small(body) = text(size: 9pt, body)
#let group(clr, label, ..nodes, name: none) = node(
  enclose: nodes.pos(),
  fill: clr.lighten(92%),
  stroke: 1pt + clr.lighten(40%),
  label: align(top, text(fill: clr.darken(50%), label)),
  name: name,
)
#let badge(clr, body) = box(fill: clr.lighten(80%), inset: (x: 4pt, y: 2pt), radius: 3pt, text(size: 8.5pt, fill: clr.darken(40%), body))
#let t(..args) = table(stroke: 0.5pt + luma(70%), inset: 6pt, ..args)

#let legend = [
  #grid(
    columns: 10,
    column-gutter: (0.6em, 2em),
    row-gutter: 0.8em,
    align: horizon,
    diagram(edge((0, 0), (1, 0), "->", ..data-edge)), [data flow],
    diagram(edge((0, 0), (1, 0), "->", ..manual-edge)), [by a person],
    diagram(node-stroke: 1pt, node((0, 0), [person], ..person)), [person],
    diagram(node-stroke: 1pt, node((0, 0), [step], ..gpu)), [GPU step (DGX Spark)],
    diagram(node-stroke: 1pt, node((0, 0), `files`, ..storage)), [storage],
  )
]

#let legend-new = [
  #grid(
    columns: 8,
    column-gutter: (0.6em, 2em),
    row-gutter: 0.8em,
    align: horizon,
    diagram(node-stroke: 1pt, node((0, 0), [step], corner-radius: 4pt)), [exists now],
    diagram(node-stroke: 1pt, node((0, 0), [step], ..new)), [proposed],
    diagram(edge((0, 0), (1, 0), "->", ..new-edge)), [proposed flow],
    diagram(node-stroke: 1pt, node((0, 0), [?], ..ask)), [decision],
  )
]

/// What runs now, in `notebooks/featmatch/`.
#let current = diagram(
  spacing: (3.2em, 3.2em),
  node-stroke: 1pt,
  edge-stroke: 1pt,
  label-size: 9pt,
  label-sep: 0.15em,

  // INPUTS
  group(gray, [inputs], (0, -0.95), <inv>, <churches>, <plates>),
  node((0, 0), align(center)[`unstirring`\ #small[scraper: Onroerend Erfgoed]\ #small[inventory, type "kerken"]], name: <inv>),
  node((0, 1.2), align(center)[`onroerend/kerken`\ #small[1337 modern church photos]], name: <churches>, ..storage),
  node((0, 2.4), align(center)[`assets-crop`\ #small[61,283 glass plates]], name: <plates>, ..storage),

  // GLOBAL
  group(purple, [1. global search], (1.6, 1.1), <stores>, <short>),
  node((1.6, 1.8), align(center)[DINOv2 stores\ #small[`onroerend-dino`, `glashelder-dino`]\ #small[`dinov2-large`, 1024 dim]], name: <stores>, ..storage),
  node((1.6, 3.0), align(center)[shortlist\ #small[top K plates per church]\ #small[cosine, `embed.search.topk`]], name: <short>, ..gpu),

  // LOCAL
  group(orange, [2. local features], (3.4, 1.1), <prep>, <det>, <match>),
  node((3.4, 0), align(center)[prepare\ #small[grayscale, long side 1024]], name: <prep>),
  node((3.4, 1.2), align(center)[detect + describe\ #small[ORB, AKAZE, BRISK, RootSIFT,]\ #small[SuperPoint, DISK, ALIKED, DoGHardNet]], name: <det>, ..gpu),
  node((3.4, 2.4), align(center)[match\ #small[nearest neighbour: ratio 0.8, mutual]\ #small[or LightGlue]], name: <match>, ..gpu),

  // VERIFY
  group(teal, [3. check and rank], (5.2, 1.1), <verify>, <score>, <rank>),
  node((5.2, 0), align(center)[geometric check\ #small[homography, MAGSAC, 8 px]], name: <verify>),
  node((5.2, 1.2), align(center)[score\ #small[`inliers`]\ #small[`cells`: 8 x 8 grid cells hit]], name: <score>),
  node((5.2, 2.4), align(center)[rank pairs\ #small[best first]], name: <rank>),

  node((6.8, 2.4), [you], name: <you>, ..person),

  edge(<inv>, <churches>, "->", label: [download], label-side: left, ..data-edge),
  edge(<churches>, <stores>, "->", label: [`embed images`], ..data-edge),
  edge(<plates>, <stores.south-west>, "->", label: [`embed images`], label-side: right, ..data-edge),
  edge(<stores>, <short>, "->", ..data-edge),
  edge(<short.east>, (2.5, 3.0), (2.5, 0), <prep>, "->", label: [pairs], label-pos: 0.75, ..data-edge),
  edge(<prep>, <det>, "->", ..data-edge),
  edge(<det>, <match>, "->", label: [once per image], label-side: left, ..data-edge),
  edge(<match.east>, (4.3, 2.4), (4.3, 0), <verify>, "->", label: [per pair], label-pos: 0.75, ..data-edge),
  edge(<verify>, <score>, "->", ..data-edge),
  edge(<score>, <rank>, "->", ..data-edge),
  edge(<rank>, <you>, "<-", label: [look at the best], label-side: left, ..manual-edge),
)

/// The proposed flow. The new parts are teal.
#let proposed = diagram(
  spacing: (2.2em, 3.2em),
  node-stroke: 1pt,
  edge-stroke: 1pt,
  label-size: 9pt,
  label-sep: 0.15em,

  node((0, 0), align(center)[plate\ #small[61,283]], name: <plate>, ..storage),
  node((0, 1.4), [place name?], name: <place>, ..ask),
  node((1.6, 0.6), align(center)[candidates: the churches\ of that town #small[(1 to 5)]], name: <town>, ..new),
  node((1.6, 2.2), align(center)[candidates: DINOv2\ shortlist #small[(top K)]], name: <dino>),

  node((3.2, 1.4), align(center)[LightGlue match\ #small[ALIKED, DISK, SuperPoint,]\ #small[DoGHardNet]], name: <lg>, ..gpu),
  node((4.6, 1.4), align(center)[homography\ #small[inliers, cells]], name: <h>),

  group(teal, [new signals], (6.2, 1.4), <agree>, <comp>, <support>),
  node((6.2, 0.2), align(center)[*agreement*\ #small[how many methods rank it high]], name: <agree>, ..new),
  node((6.2, 1.4), align(center)[*competition*\ #small[score vs. the church's other plates]\ #small[and the plate's other churches]], name: <comp>, ..new),
  node((6.2, 2.6), align(center)[*descriptor support*\ #small[inliers that are also]\ #small[mutual nearest neighbours]], name: <support>, ..new),

  node((7.9, 1.4), align(center)[decision\ #small[thresholds set from labels]], name: <decide>, ..new),
  node((7.9, 3.0), align(center)[`labels.json`\ #small[y / n per pair]], name: <labels>, ..storage),
  node((6.2, 4.0), [you], name: <you>, ..person),
  node((7.9, 0.1), align(center)[matches\ #small[for review]], name: <out>, ..storage),

  edge(<plate>, <place>, "->", ..data-edge),
  edge(<place>, <town>, "->", label: [yes], ..new-edge),
  edge(<place>, <dino>, "->", label: [no], label-side: right, ..data-edge),
  edge(<town>, <lg>, "->", ..new-edge),
  edge(<dino>, <lg>, "->", ..data-edge),
  edge(<lg>, <h>, "->", ..data-edge),
  edge(<h>, <agree>, "->", ..new-edge),
  edge(<h>, <comp>, "->", ..new-edge),
  edge(<h>, <support>, "->", ..new-edge),
  edge(<agree>, <decide>, "->", ..new-edge),
  edge(<comp>, <decide>, "->", ..new-edge),
  edge(<support>, <decide>, "->", ..new-edge),
  edge(<decide>, <out>, "->", ..new-edge),
  edge(<you>, (7.9, 4.0), <labels>, "->", label: [label the best pairs], label-pos: 0.3, ..manual-edge),
  edge(<labels>, <decide>, "->", label: [tune], label-side: right, ..new-edge),
)

/// A horizontal range on an inlier scale from 0 to `max`.
#let scale-width = 14cm
#let scale-max = 320
#let range(lo, hi, clr) = box(width: scale-width, height: 1.1em, {
  place(dx: lo / scale-max * scale-width, box(width: calc.max((hi - lo) / scale-max * scale-width, 3pt), height: 1.1em, fill: clr.lighten(40%), stroke: 0.5pt + clr.darken(20%), radius: 2pt))
})
#let axis = box(width: scale-width, height: 1.4em, {
  place(dy: 0pt, line(length: scale-width, stroke: 0.5pt + luma(50%)))
  for v in (0, 50, 100, 150, 200, 250, 300) {
    place(dx: v / scale-max * scale-width, line(length: 3pt, angle: 90deg, stroke: 0.5pt + luma(50%)))
    place(dx: v / scale-max * scale-width - 0.6em, dy: 4pt, text(size: 8pt, fill: luma(40%), str(v)))
  }
})

= Church matching: what runs now

#current

#legend

#v(1em)

#t(
  columns: 4,
  table.header([*step*], [*what*], [*why*], [*where*]),
  [inputs], [1337 modern photos of churches from the Onroerend Erfgoed inventory; 61,283 glass plates], [the two sets to match], [`unstirring/`, `assets-crop`],
  [1. global search], [one DINOv2 vector per image; the K plates closest to each church], [all pairs is 82M pairs, about 350 h of matching; a shortlist is about 1 h], [`embed`, both notebooks],
  [2. local features], [keypoints and descriptors once per image; then matches per pair], [find the same points of the same building in two photos], [`churches.ipynb` (ALIKED + LightGlue), `compare.ipynb` (12 methods)],
  [3. check and rank], [fit a homography to the matches; count the matches that agree (inliers) and the grid cells they cover], [random matches do not agree on one geometry], [both notebooks],
)

#pagebreak()

= What the runs showed

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  [
    == Speed on the DGX Spark (GB10)

    #t(
      columns: 2,
      [read one image], [25 ms],
      [ALIKED, one image], [32 ms],
      [LightGlue, one pair], [11 ms],
      [homography, one pair], [1 ms],
      [all plates and churches, features], [about 1 h],
      [all pairs (82M)], [about 350 h],
      [shortlist (about 300k pairs)], [about 1.3 h],
    )

    Speed is not the problem. The shortlist makes the cost small.
  ],
  [
    == Pairs you checked

    #t(
      columns: 3,
      table.header([*church*], [*plate*], [*result*]),
      [`114900`], [`fx73v2hq54`], badge(green)[same church],
      [`114900`], [`6t0gt8jn7v`], badge(green)[same church],
      [`132754`], [`707wm46537`], badge(green)[same church],
      [`238234`], [`gt5fb80q3j`], badge(red)[different church],
    )

    Inliers per method, where the pair was in the method's top 4 (·: not in the top 4):

    #t(
      columns: 5,
      table.header([*method*], [*114900 fx73*], [*114900 6t0g*], [*132754 707w*], [*238234 gt5f*]),
      [`aliked+lightglue`], [221], [171], [185], [·],
      [`disk+lightglue`], [·], [228], [298], text(fill: red.darken(30%))[175],
      [`doghardnet+lightglue`], [120], [101], [244], [·],
      [`superpoint+lightglue`], [·], [·], [164], text(fill: red.darken(30%))[141],
      [`sift+lightglue`], [·], [·], [130], [·],
      [all `+nn` and OpenCV], [·], [·], [·], [·],
    )
  ],
)

#v(1em)

== Inlier counts on one scale

#grid(
  columns: (auto, auto, auto),
  column-gutter: 1em,
  row-gutter: 0.7em,
  align: (right + horizon, left + horizon, left + horizon),
  [best pairs of the nearest neighbour methods], range(5, 9, gray), [5 to 9: the noise floor of a homography],
  [same church, LightGlue methods], range(101, 298, green), [101 to 298],
  [different church, LightGlue methods], range(141, 175, red), [141 to 175: inside the true range],
  [], axis, [inliers],
)

#v(1em)

#t(
  columns: (auto, 1fr),
  [*nearest neighbour*], [Finds nothing. Plain descriptors do not survive 100 years of light, film and weather. The ratio test removes almost all matches, and what is left is noise.],
  [*LightGlue*], [Finds the true matches, but also gives look-alike churches the same score. Its matches come from context, not only from descriptor similarity: it is trained to align two images, and it can align two towers that only look alike.],
  [*one threshold*], [Cannot work. The false pair is inside the range of the true pairs.],
  [*checks that failed*], [The fundamental matrix accepted any two towers (it only checks a line, not a point). The homography plausibility and spread checks then rejected a true pair (`114900` / `fx73v2hq54`).],
  [*hubs*], [Some churches score high against many plates: `181724` had 12 of the best 30 pairs in the first run.],
  [*bug*], [`disk+nn` gives 0 matches for every pair. Not a result.],
)

#pagebreak()

= Proposed

#proposed

#legend-new

#v(1em)

Keep LightGlue, because only LightGlue finds the true matches. Stop deciding on the inlier count alone. Decide on signals that a look-alike cannot fake as easily, and tune them on pairs that you have labelled.

#v(0.5em)

#t(
  columns: (auto, 1fr, 1fr, auto),
  table.header([*signal*], [*what it measures*], [*why a look-alike should fail it*], [*evidence so far*]),
  [*place name*], [limit the candidates of a plate to the churches of its town], [look-alikes from other towns never compete], badge(gray)[needs plate metadata],
  [*agreement*], [the number of methods (ALIKED, DISK, SuperPoint, DoGHardNet) that rank the pair high], [each detector picks other points; a false alignment does not repeat as often], [#badge(green)[true pairs: 2 to 5 methods] #badge(orange)[false pair: 2] #small[(top 4 only, so weak)]],
  [*competition*], [the score minus the church's typical high score over its plates, and the same per plate], [a hub church scores high everywhere, so its excess is small], badge(gray)[not measured],
  [*descriptor support*], [the share of LightGlue inliers that are also mutual nearest neighbours], [matches that only context made have no descriptor similarity], badge(gray)[not measured],
)

#v(1em)

== Order of work

#t(
  columns: (auto, 1fr, auto),
  table.header([*\#*], [*what*], [*who*]),
  [1], [Find out if the plates have a place name (caption, inventory note, OCR of the label). If yes, this filter comes first.], [you],
  [2], [Add a label cell to `compare.ipynb`: it shows the best candidate pairs one by one, you answer y or n, the answers go to `labels.json`.], [notebook, then you],
  [3], [Label 30 to 50 pairs.], [you],
  [4], [Compute the three signals for every labelled pair and compare true with false.], [notebook],
  [5], [Fix `disk+nn`.], [notebook],
  [6], [Pick the signals and thresholds that separate the labels; run on the full shortlist.], [notebook, then you],
)

#pagebreak()

= Open decisions

#t(
  columns: (auto, 1fr, 1fr),
  table.header([*question*], [*why it matters*], [*what to do*]),
  [Do the plates have a place name?], [The strongest filter. It changes the task from "which of 1337" to "which of a few".], [match the town names of plates and churches, then use LightGlue only within a town],
  [Is DINOv2 a good enough shortlist?], [A true pair that is not in the shortlist is never matched. Both true churches were in it, but that is 2 out of 2.], [keep it; else try MegaLoc, which is trained to find the same place],
  [Are some plates mirrored?], [ALIKED and LightGlue do not match a mirrored image.], [match each plate also flipped left to right],
  [Is a dense matcher worth it?], [RoMa or MASt3R match every pixel and are stronger across large changes, but take about 0.3 s per pair.], [only on the pairs that the signals leave unclear],
)
