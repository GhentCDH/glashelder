#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2"

#set page(width: 30cm, height: auto, margin: 1.5em)
#set par(leading: 1mm)
#set text(font: "IBM Plex Sans")
#show heading.where(level: 1): set block(below: 1.2em)

// STYLE
#let step = (corner-radius: 4pt)
#let gpu = (stroke: 1pt + green.darken(30%), fill: green.lighten(90%), corner-radius: 4pt)
#let ask = (shape: fletcher.shapes.diamond, fill: yellow.lighten(85%))
#let data-edge = (stroke: 1pt + luma(40%))
#let knob = teal.darken(20%)
#let good = green.darken(30%)
#let bad = red.darken(20%)
#let small(body) = text(size: 9pt, body)
#let badge(clr, body) = box(fill: clr.lighten(80%), inset: (x: 4pt, y: 2pt), radius: 3pt, text(size: 8.5pt, fill: clr.darken(40%), body))
#let t(..args) = table(stroke: 0.5pt + luma(70%), inset: 6pt, ..args)
#let f-px = text(fill: knob, raw("f_px"))
#let h-px = text(fill: knob, raw("h_px"))

// image boxes are 6 x 4.5 cm, for 1024 x 768 px
#let W = 6
#let H = 4.5
#let photo(origin, label) = {
  import cetz.draw: *
  rect(origin, (origin.at(0) + W, origin.at(1) + H), stroke: 1pt, fill: luma(97%))
  content((origin.at(0) + W / 2, origin.at(1) + H + 0.35), label)
}

/// The match pipeline of one pair, in `unstir/src/unstir/match.py`.
#let pipeline = diagram(
  spacing: (3.6em, 2em),
  node-stroke: 1pt,
  edge-stroke: 1pt,
  label-size: 9pt,
  label-sep: 0.15em,

  node((0, 0), align(center)[prepare\ #small[grayscale, long side 1024 px]], name: <prep>, ..step),
  node((1, 0), align(center)[ALIKED + LightGlue\ #small[about 45 matches]\ #small[on an unrelated pair]], name: <lg>, ..gpu),
  node((2, 0), align(center)[RANSAC (MAGSAC)\ #small[tries up to 10,000 F]\ #small[keeps the F with most inliers]], name: <ransac>, ..step),
  node((3, 0), align(center)[distance to\ epipolar line\ #small[$<=$] #f-px ?], name: <test>, ..ask),
  node((4, -0.6), align(center)[inlier], name: <in>, stroke: 1pt + good),
  node((4, 0.6), align(center)[outlier], name: <out>, stroke: 1pt + bad),
  node((5, -0.6), align(center)[`f_inliers`, `cells`\ #small[cells: 8 x 8 grid,]\ #small[lower of the two images]], name: <score>, ..step),

  edge(<prep>, <lg>, "->", ..data-edge),
  edge(<lg>, <ransac>, "->", ..data-edge, label: "matches"),
  edge(<ransac>, <test>, "->", ..data-edge, label: "every match"),
  edge(<test>, <in>, "->", ..data-edge, label: "yes", label-side: left),
  edge(<test>, <out>, "->", ..data-edge, label: "no", label-side: right),
  edge(<in>, <score>, "->", ..data-edge),
)

/// A point in photo A, its epipolar line in photo B, the band of 2 x f_px, and two matches.
#let epipolar = cetz.canvas(length: 1cm, {
  import cetz.draw: *

  photo((0, 0), [photo A (plate)])
  photo((8, 0), [photo B (reference)])

  // the point in A
  circle((2.2, 2.9), radius: 0.09, fill: black)
  content((2.2, 2.9), anchor: "south-west", padding: 0.15, small[$x$: top of a window])

  // F maps it to a line in B
  line((6.3, 2.9), (7.7, 2.9), mark: (end: ">"), stroke: 1pt + luma(40%))
  content((7, 3.2), small[F])

  let a = (8, 0.9)
  let b = (14, 3.7)
  line(a, b, stroke: (paint: knob.transparentize(70%), thickness: 0.5cm, cap: "butt"))
  line(a, b, stroke: (paint: knob, thickness: 0.8pt, dash: "dashed"))
  content((14.1, 3.7), anchor: "west", small[epipolar line of $x$])

  // possible positions of x along the line: depth is unknown
  for (p, l) in (((9.0, 1.37), [near]), ((10.6, 2.11), [far])) {
    circle(p, radius: 0.08, stroke: 0.8pt + luma(40%), fill: white)
    content(p, anchor: "north-west", padding: 0.12, text(size: 8pt, fill: luma(40%), l))
  }

  // an inlier and an outlier match
  circle((12.0, 2.85), radius: 0.1, fill: good)
  content((12.0, 2.85), anchor: "north-west", padding: (left: 0.15, top: 0.3), text(size: 9pt, fill: good)[match 1: 1 px off, inlier])
  circle((11.0, 3.9), radius: 0.1, fill: bad)
  line((11.0, 3.9), (11.47, 2.7), stroke: (paint: bad, thickness: 0.8pt), mark: (start: "|", end: "|"))
  content((11.0, 3.9), anchor: "south", padding: 0.15, text(size: 9pt, fill: bad)[match 2: 30 px off, outlier])

  // band width
  content((8.6, 0.35), anchor: "west", text(size: 9pt, fill: knob)[band: #raw("f_px") on each side of the line (drawn wider)])
})

/// The H disc next to the F band, at the same scale (both 10x bigger than real).
#let areas = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let r = 0.47 // 8 px at 6 cm per 1024 px, times 10

  photo((0, 0), [homography: a point])
  circle((3.4, 2.4), radius: r, fill: knob.transparentize(70%), stroke: 0.8pt + knob)
  circle((3.4, 2.4), radius: 0.06, fill: black)
  content((3.4, 2.4 - r), anchor: "north", padding: 0.12, small[radius #raw("h_px") = 8])
  content((W / 2, -0.45), small[about 200 px², 0.03% of the image])

  photo((8, 0), [fundamental matrix: a line])
  line((8, 1.2), (14, 3.5), stroke: (paint: knob.transparentize(70%), thickness: 0.47cm, cap: "butt"))
  line((8, 1.2), (14, 3.5), stroke: (paint: knob, thickness: 0.8pt, dash: "dashed"))
  content((11, 2.35), anchor: "north-west", padding: (top: 0.35), small[width 2 x #raw("f_px") = 8])
  content((8 + W / 2, -0.45), small[about 8,000 px², 1% of the image])
})

/// Ranges on one cells axis: unrelated pairs and the best pairs of the 50 plate test run.
#let cells-axis = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let s = 0.5 // cm per cell
  let x(v) = v * s

  line((0, 0), (x(32), 0), mark: (end: ">"))
  for v in range(0, 33, step: 4) {
    line((x(v), -0.1), (x(v), 0.1))
    content((x(v), -0.15), anchor: "north", text(size: 8pt)[#v])
  }
  content((x(32) + 0.2, 0), anchor: "west", small[`cells`])

  // unrelated pairs: p10 5, p50 7, p90 11
  rect((x(5), 0.4), (x(11), 0.9), fill: bad.lighten(80%), stroke: 0.8pt + bad)
  line((x(7), 0.4), (x(7), 0.9), stroke: 1.5pt + bad)
  content((x(11), 0.65), anchor: "west", padding: 0.15, text(size: 9pt, fill: bad)[most pairs (10% to 90%), median 7: noise floor at #raw("f_px") 4])

  // the 10 pairs with the most h_inliers (73 to 309), one dot each
  for v in (17, 31, 10, 10, 8, 9, 5, 9, 21, 16).enumerate() {
    let (i, c) = v
    circle((x(c), 1.3 + calc.rem(i, 2) * 0.3), radius: 0.1, fill: good, stroke: none)
  }
  content((x(32) + 0.2, 1.45), anchor: "west", text(size: 9pt, fill: good)[10 pairs with most `h_inliers` (73 to 309)])
})

/// Expected shift of the noise floor when f_px goes down. Not measured yet.
#let shift-axis = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let s = 0.5
  let x(v) = v * s

  line((0, 0), (x(32), 0), mark: (end: ">"))
  for v in range(0, 33, step: 4) {
    line((x(v), -0.1), (x(v), 0.1))
    content((x(v), -0.15), anchor: "north", text(size: 8pt)[#v])
  }
  content((x(32) + 0.2, 0), anchor: "west", small[`cells`])

  rect((x(5), 1.3), (x(11), 1.8), fill: bad.lighten(80%), stroke: 0.8pt + bad)
  content((x(11), 1.55), anchor: "west", padding: 0.15, text(size: 9pt, fill: bad)[unrelated, #raw("f_px") 4 (measured)])
  rect((x(2), 0.4), (x(7), 0.9), fill: bad.lighten(90%), stroke: (paint: bad, thickness: 0.8pt, dash: "dashed"))
  content((x(7), 0.65), anchor: "west", padding: 0.15, text(size: 9pt, fill: bad)[unrelated, #raw("f_px") 2 (expected: fewer chance inliers)])
  line((x(8), 1.25), (x(5), 0.95), mark: (end: ">"), stroke: 0.8pt + bad)

  content((x(16), 2.45), anchor: "center", text(size: 9pt, fill: good)[real matches: range not known yet, expected to stay about the same (LightGlue is accurate to 1 to 2 px)])
})

// PAGES

= What #raw("--f-px") is

#raw("--f-px") is the distance in pixels that a match may be off from what the fundamental matrix F predicts, and still count as an inlier. Both images are 1024 px on the long side, so 4 px is 0.4% of the width. Default: 4.

#pipeline

#v(1em)

== The epipolar line

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  epipolar,
  [
    A point $x$ in photo A can be anywhere on one line in photo B. Where it is on that line depends on its depth, which is not known. That line is the *epipolar line*. F is the rule that gives this line for every point in A.

    #v(0.5em)
    A match is an inlier when its point in B is at most #f-px from the line. So the test is a *band* of 2 x #f-px around the line, over the full image.

    #v(0.5em)
    F is the right model for the same building from another viewpoint. A homography is only right for one plane (one facade) or almost the same view.
  ],
)

#pagebreak()

= Why F lets more wrong matches through than a homography

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  areas,
  [
    A homography predicts one *point*. The test is a disc of radius #h-px.

    #v(0.5em)
    F predicts a *line*. The test only measures the distance across the line, not along it. So the band covers about 40 times more of the image than the disc.

    #v(0.5em)
    That alone gives about 1% per match: 45 random matches times 1% is 0.5 inliers. But the measured value is about 20. Two more reasons:
  ],
)

#v(1em)

#t(
  columns: (auto, 1fr),
  [*RANSAC chooses F*], [F has 7 degrees of freedom. Any 7 matches fit an F exactly. RANSAC tries up to 10,000 F and keeps the one with the most inliers. With only 45 matches, the best of 10,000 tries catches many matches by chance.],
  [*Wrong matches are not random*], [Two churches are both upright: tower at the top, ground at the bottom. LightGlue matches window to window and tower to tower. These wrong matches already have a rough common layout, and a general model like F fits that layout.],
  [*They spread out*], [The chance inliers are spread over the image, so they also fill grid cells. That is why `cells` has a noise floor too.],
)

#pagebreak()

= What the test run measured

50 plates of clusters 53, 60, 61, 74, 84, 85. 1005 pairs, the top 20 references of each plate. #f-px 4, #h-px 8. Almost all of these pairs are different buildings.

#v(0.5em)
#cells-axis
#v(1em)

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  t(
    columns: 5,
    [], [*10%*], [*median*], [*90%*], [*max*],
    [`matches`], [23], [45], [77], [584],
    [`f_inliers`], [12], [20], [44], [583],
    [`h_inliers`], [5], [8], [24], [309],
    [`cells`], [5], [7], [11], [31],
    [`f_inliers / matches`], [0.36], [0.50], [0.70], [1.00],
  ),
  [
    #badge(orange)[noise floor] A typical unrelated pair keeps 20 of its 45 matches as F inliers, over 7 cells.

    #v(0.5em)
    #badge(orange)[weak alone] The 10 pairs with the most `h_inliers` (73 to 309) have 5 to 31 cells. Half of them are inside the noise band. Their inliers sit on one part of the image, so `cells` alone does not separate them. `h_inliers` and `f_inliers` do.

    #v(0.5em)
    #badge(gray)[unknown] Which of these pairs are real matches, and where real matches from another viewpoint fall. Those can have few `h_inliers`, so they are the important case. Only labels can show this.
  ],
)

#pagebreak()

= What a smaller #raw("--f-px") changes

#shift-axis
#v(1em)

#t(
  columns: (auto, 1fr, 1fr),
  [*#raw("f_px")*], [*unrelated pairs*], [*real matches*],
  [8], [more chance inliers, the noise floor goes up], [all good matches pass, also with lens distortion],
  [4 (now)], [median 20 inliers, 7 cells (measured)], [most good matches pass],
  [2], [fewer chance inliers, the noise floor goes down (expected)], [most still pass: LightGlue keypoints are accurate to 1 to 2 px at 1024 px],
  [1], [very few chance inliers], [real matches start to fail too: old lenses, a scanned plate, and a 1024 px resize all add error],
)

#v(1em)

#t(
  columns: (auto, 1fr),
  [*What changes*], [Only which matches count as inliers, so `f_inliers` and `cells`. `matches` and the DINOv2 rank stay the same.],
  [*What it costs*], [Nothing. The run time is the same.],
  [*What to do*], [Keep 4 for round 1. After labelling, run `unstir pairs` again on the same plates with `--f-px 2` and compare how well `cells` separates match from no match. The labels stay valid, because a label belongs to a (plate, reference) pair.],
)
