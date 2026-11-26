#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2"

#set page(width: 30cm, height: auto, margin: 1.5em)
#set par(leading: 1mm)
#set text(font: "IBM Plex Sans")
#show heading.where(level: 1): set block(below: 1.2em)

// STYLE
#let step = (corner-radius: 4pt)
#let new = (stroke: 1pt + teal.darken(20%), fill: teal.lighten(90%), corner-radius: 4pt)
#let ask = (shape: fletcher.shapes.diamond, fill: yellow.lighten(85%))
#let data-edge = (stroke: 1pt + luma(40%))
#let new-edge = (stroke: 1pt + teal.darken(20%))
#let good = green.darken(30%)
#let bad = red.darken(20%)
#let small(body) = text(size: 9pt, body)
#let badge(clr, body) = box(fill: clr.lighten(80%), inset: (x: 4pt, y: 2pt), radius: 3pt, text(size: 8.5pt, fill: clr.darken(40%), body))
#let t(..args) = table(stroke: 0.5pt + luma(70%), inset: 6pt, ..args)

// one colour per building, the same on every page
#let spaengien = blue.darken(10%)
#let spieghel = orange.darken(10%)
#let mouwe = green.darken(25%)
#let arend = purple
#let church = red.darken(10%)

/// A photo frame with a title above it.
#let frame(origin, w, h, title) = {
  import cetz.draw: *
  rect(origin, (origin.at(0) + w, origin.at(1) + h), stroke: 1pt, fill: luma(97%))
  content((origin.at(0) + w / 2, origin.at(1) + h + 0.3), small(title))
}

/// A guild house: a narrow facade with a stepped gable and windows.
#let house(x, y, w, h, clr, name: none) = {
  import cetz.draw: *
  let top = y + h
  rect((x, y), (x + w, top), stroke: 0.8pt + clr, fill: clr.lighten(85%))
  // stepped gable
  let s = w / 6
  line(
    (x, top),
    (x, top + 0.25), (x + s, top + 0.25), (x + s, top + 0.5), (x + 2 * s, top + 0.5),
    (x + 2 * s, top + 0.8), (x + 4 * s, top + 0.8), (x + 4 * s, top + 0.5),
    (x + 5 * s, top + 0.5), (x + 5 * s, top + 0.25), (x + w, top + 0.25), (x + w, top),
    close: true, stroke: 0.8pt + clr, fill: clr.lighten(85%),
  )
  for i in range(3) {
    for j in range(3) {
      let wx = x + w * (0.15 + i * 0.27)
      let wy = y + h * (0.2 + j * 0.27)
      rect((wx, wy), (wx + w * 0.17, wy + h * 0.17), stroke: 0.5pt + clr, fill: white)
    }
  }
  if name != none {
    content((x + w / 2, y - 0.25), text(size: 8pt, fill: clr.darken(20%), name))
  }
}

/// A church: a nave with a tower and a spire, `k` times the base size.
#let church-drawing(x, y, clr, spire: 1.4, k: 1) = {
  import cetz.draw: *
  let p(dx, dy) = (x + dx * k, y + dy * k)
  rect(p(0, 0), p(2.2, 1.3), stroke: 0.8pt + clr, fill: clr.lighten(85%))
  line(p(0, 1.3), p(1.1, 1.9), p(2.2, 1.3), close: true, stroke: 0.8pt + clr, fill: clr.lighten(85%))
  rect(p(0.75, 0), p(1.45, 2.6), stroke: 0.8pt + clr, fill: clr.lighten(80%))
  line(p(0.75, 2.6), p(1.1, 2.6 + spire), p(1.45, 2.6), close: true, stroke: 0.8pt + clr, fill: clr.lighten(70%))
  rect(p(0.95, 1.9), p(1.25, 2.3), stroke: 0.5pt + clr, fill: white)
}

/// Matched points: a small grid of dots in a box.
#let points(x0, y0, x1, y1, clr, n: 4) = {
  import cetz.draw: *
  for i in range(n) {
    for j in range(n) {
      let px = x0 + (x1 - x0) * (i + 0.5) / n
      let py = y0 + (y1 - y0) * (j + 0.5) / n
      circle((px, py), radius: 0.05, fill: clr, stroke: none)
    }
  }
}

/// The area of the photo that a candidate's matched points cover.
#let area(x0, y0, x1, y1, clr) = {
  import cetz.draw: *
  rect((x0, y0), (x1, y1), stroke: (paint: clr, thickness: 1.2pt, dash: "dashed"), fill: clr.transparentize(85%))
}

// FIGURES

/// Page 1: your photo of the Grote Markt and what the page got.
#let grote-markt = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  frame((0, 0), 9, 5.2, [your photo: a row of guild houses])
  house(0.3, 0.6, 1.9, 3.0, arend, name: [Den Arend])
  house(2.3, 0.6, 2.1, 3.3, mouwe, name: [De Mouwe])
  house(4.5, 0.6, 2.1, 3.4, spieghel, name: [De Spieghel])
  house(6.7, 0.6, 2.0, 3.2, spaengien, name: [Spaengien])
  points(6.8, 0.8, 8.6, 3.6, spaengien)
  points(4.6, 0.8, 6.5, 3.8, spieghel)
  points(2.4, 0.8, 4.3, 3.7, mouwe)
  points(0.4, 0.8, 2.1, 3.4, arend, n: 3)
})

/// Page 2, left: two look-alike churches claim the same part of your photo.
#let lookalikes = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  frame((0, 0), 4.5, 5, [your photo])
  church-drawing(1.1, 0.5, luma(30%))
  area(1.75, 0.5, 2.65, 4.4, church)
  area(1.6, 1.4, 2.8, 4.6, spieghel)

  frame((6.2, 2.7), 3, 2.6, [reference A])
  frame((6.2, -0.5), 3, 2.6, [reference B])
  church-drawing(6.75, 2.85, church, spire: 1.0, k: 0.6)
  church-drawing(6.75, -0.35, spieghel, spire: 0.8, k: 0.6)
  line((5.9, 4), (2.9, 3.4), mark: (end: ">"), stroke: 1pt + church)
  line((5.9, 0.8), (2.9, 2.2), mark: (end: ">"), stroke: 1pt + spieghel)
  content((4.5, -0.9), small[both cover the *same* part: they compete, the margin decides])
})

/// Page 2, right: neighbours each claim their own part.
#let neighbours = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  frame((0, 0), 6.6, 5, [your photo])
  house(0.3, 0.6, 1.9, 3.0, mouwe)
  house(2.3, 0.6, 2.0, 3.3, spieghel)
  house(4.4, 0.6, 1.9, 3.1, spaengien)
  area(0.25, 0.5, 2.25, 4.0, mouwe)
  area(2.25, 0.5, 4.35, 4.25, spieghel)
  area(4.35, 0.5, 6.35, 4.0, spaengien)

  frame((7.8, 3.4), 2, 2.3, [De Mouwe])
  frame((7.8, 0.6), 2, 2.3, [De Spieghel])
  frame((7.8, -2.2), 2, 2.3, [Spaengien])
  house(8.15, 3.55, 1.3, 1.2, mouwe)
  house(8.15, 0.75, 1.3, 1.2, spieghel)
  house(8.15, -2.05, 1.3, 1.2, spaengien)
  line((7.6, 4.4), (1.3, 3.2), mark: (end: ">"), stroke: 1pt + mouwe)
  line((7.6, 1.6), (3.3, 2.4), mark: (end: ">"), stroke: 1pt + spieghel)
  line((7.6, -1.2), (5.3, 1.6), mark: (end: ">"), stroke: 1pt + spaengien)
  content((3.3, -0.4), small[each covers its *own* part: all three are in the photo])
})

/// Page 3: the reference side. Your photo shows De Spieghel; two reference
/// photos match it, but only one has its matched points in its centre.
#let reference-side = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  frame((0, 0), 4.2, 5, [your photo: De Spieghel])
  house(0.9, 0.6, 2.4, 3.4, spieghel)
  area(0.85, 0.5, 3.35, 4.9, spieghel)

  // reference photo of De Spieghel: the facade in the middle
  frame((6.5, 2.9), 4.4, 3.2, [reference photo of *De Spieghel*])
  house(7.1, 3.15, 1.0, 1.9, mouwe)
  house(8.15, 3.15, 1.5, 2.15, spieghel)
  house(9.7, 3.15, 1.0, 1.9, spaengien)
  points(8.25, 3.3, 9.55, 5.1, spieghel)
  rect((7.6, 3.0), (10.2, 6.0), stroke: (paint: good, thickness: 1pt, dash: "dotted"))
  content((10.95, 4.5), anchor: "west", text(size: 9pt, fill: good)[points in the centre:\ this photo is *about* it])

  // reference photo of De Mouwe: half of De Spieghel at its edge
  frame((6.5, -1.3), 4.4, 3.2, [reference photo of *De Mouwe*])
  house(6.6, -1.05, 1.3, 2.1, spieghel)
  house(8.0, -1.05, 1.6, 2.3, mouwe)
  house(9.7, -1.05, 1.1, 2.0, arend)
  points(6.65, -0.9, 7.85, 0.9, spieghel, n: 3)
  rect((7.6, -1.2), (10.2, 1.8), stroke: (paint: good, thickness: 1pt, dash: "dotted"))
  content((10.95, 0.3), anchor: "west", text(size: 9pt, fill: bad)[points at the edge:\ only a neighbour in view])

  line((6.3, 4.4), (3.4, 3.4), mark: (end: ">"), stroke: 1pt + spieghel)
  line((6.3, 0.2), (3.4, 1.8), mark: (end: ">"), stroke: (paint: spieghel, thickness: 1pt, dash: "dashed"))
  content((2.1, -0.4), small[both claim the same part of your photo])
  content((8.7, -1.8), small[dotted: the centre of each reference photo])
})

/// Page 4: the proposed decision, for collections with one object per photo.
#let flow = diagram(
  spacing: (3.4em, 2.2em),
  node-stroke: 1pt,
  edge-stroke: 1pt,
  label-size: 9pt,
  label-sep: 0.15em,

  node((0, 0), align(center)[top 10 candidates\ #small[DINOv2 retrieval]], name: <cand>, ..step),
  node((1, 0), align(center)[point check\ #small[ALIKED + LightGlue, F]], name: <check>, ..step),
  node((2, 0), align(center)[passes?\ #small[`min_f`, `min_ratio`]], name: <pass>, ..ask),
  node((2, 1.2), align(center)[not a match], name: <drop>, stroke: 1pt + bad),
  node((3, 0), align(center)[area in your photo\ #small[cells of the 8 x 8 grid]\ #small[that hold its points]], name: <area>, ..new),
  node((4, 0), align(center)[group by overlap\ #small[two areas that share]\ #small[most of their cells]], name: <group>, ..new),
  node((5, 0), align(center)[one per group?], name: <one>, ..ask),
  node((5, -1.3), align(center)[match], name: <yes>, stroke: 1pt + good),
  node((5, 1.3), align(center)[most centred in its\ own photo, by a margin?], name: <centre>, ..new),
  node((3.6, 1.3), align(center)[match], name: <yes2>, stroke: 1pt + good),
  node((5, 2.7), align(center)[old margin rule\ #small[`margin` f_inliers]], name: <margin>, ..step),

  edge(<cand>, <check>, "->", ..data-edge),
  edge(<check>, <pass>, "->", ..data-edge),
  edge(<pass>, <drop>, "->", ..data-edge, label: "no", label-side: left),
  edge(<pass>, <area>, "->", ..data-edge, label: "yes"),
  edge(<area>, <group>, "->", ..new-edge),
  edge(<group>, <one>, "->", ..new-edge, label: "per group"),
  edge(<one>, <yes>, "->", ..new-edge, label: "yes", label-side: left),
  edge(<one>, <centre>, "->", ..new-edge, label: "no", label-side: left),
  edge(<centre>, <yes2>, "->", ..new-edge, label: "yes", label-side: right),
  edge(<centre>, <margin>, "->", ..new-edge, label: "no", label-side: left),
)

#let legend = grid(
  columns: 8,
  column-gutter: (0.6em, 2em),
  row-gutter: 0.8em,
  align: horizon,
  diagram(node-stroke: 1pt, node((0, 0), [step], ..step)), [exists now],
  diagram(node-stroke: 1pt, node((0, 0), [step], ..new)), [proposed],
  diagram(node-stroke: 1pt, node((0, 0), [?], ..ask)), [decision],
  diagram(edge((0, 0), (1, 0), "->", ..new-edge)), [proposed flow],
)

// PAGES

= Several heritage objects in one photo

The lookup page and `unstir decide` assume that a photo shows *one* building. The best candidate is a match when it beats the next one by a margin of 50 `f_inliers`. If two candidates score about the same, they are taken for look-alikes, and neither is trusted.

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  grote-markt,
  [
    A photo of the Grote Markt in Antwerpen shows a row of guild houses. Each of them is a separate object in Onroerend Erfgoed, with its own photo. All of them really are in the photo, so all of them match:

    #v(0.5em)
    #t(
      columns: 4,
      [*candidate*], [*points*], [*fit*], [*page said*],
      text(fill: spaengien)[Spaengien], [579], [100%], badge(green)[match],
      text(fill: spieghel)[De Spieghel], [528], [100%], [],
      text(fill: mouwe)[De Mouwe], [421], [100%], [],
      text(fill: arend)[Den Arend], [391], [100%], [],
      [Den Engel], [279], [99%], [],
    )

    #v(0.5em)
    #badge(orange)[luck] Spaengien was only 51 points ahead of De Spieghel, just above the margin of 50. With one more house in the frame, or a slightly different angle, the page says *no match*, although four correct objects are in the photo.
  ],
)

#pagebreak()

= Look-alikes and neighbours look different in your photo

Both cases give several candidates with a high score. The difference is *where in your photo* their matched points are.

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  [
    == Look-alikes
    #lookalikes
    #v(0.5em)
    Two different churches with the same tower type. Both match *the same tower* in your photo. At most one of them can be right. Here the margin rule is correct.
  ],
  [
    == Neighbours
    #neighbours
    #v(0.5em)
    Three guild houses next to each other. Each reference matches *its own facade* in your photo. All three can be right at the same time. Here the margin rule is wrong.
  ],
)

#v(1em)

So the margin should only compare candidates that claim the *same part* of your photo. Candidates on separate parts do not compete.

The part a candidate claims is cheap to compute: the matched points of each candidate are already there. The `cells` score already counts the cells of an 8 x 8 grid that hold an inlier. The same cells, as a set instead of a count, are the area. Two candidates overlap when they share most of their cells.

#pagebreak()

= But a reference photo can also show a neighbour

Onroerend Erfgoed photos are taken in the street, so the photo of De Mouwe can also show half of De Spieghel. Then your photo of De Spieghel matches *both* reference photos, on the same part. That looks like two look-alikes, and the result would be *no match*, or the wrong one.

#reference-side

#v(1em)

The *reference side* separates them. An Onroerend Erfgoed photo is about one object, and that object is mostly in its centre:

- In the photo of De Spieghel, the matched points are in the *centre*.
- In the photo of De Mouwe, they are at the *edge*, on the neighbour.

So when candidates claim the same part of your photo, the one whose points lie in the centre of *its own* photo is the object you photographed. Only when none of them is clearly more centred, the old margin decides.

#t(
  columns: (auto, 1fr),
  [*centred*], [The share of a candidate's inlier points that fall in the middle part of its own photo, for example the middle 60% of the width. Computed on the reference side of the same matches, so it costs nothing extra.],
  [*when it fails*], [A reference photo that is not centred on its object (a street view, a wide shot of a square). Then the margin decides, as now. The error is mild: the page names a neighbour that really is in your photo, not a building elsewhere.],
)

#pagebreak()

= Proposed decision

For a collection with one object per photo (Onroerend Erfgoed). The glass plates keep their own rule: every plate that passes is a match.

#flow
#v(0.5em)
#legend
#v(1em)

#t(
  columns: (auto, 1fr, 1fr),
  [*step*], [*what*], [*why*],
  [area], [The cells of an 8 x 8 grid on your photo that hold an inlier of the candidate.], [The same cells that `cells` counts now. Keep them as a set.],
  [group], [Two candidates are in one group when they share most of their cells (for example half of the smaller area).], [Look-alikes share the same part of your photo; neighbours do not.],
  [one per group], [A group of one candidate is a match.], [Nothing competes with it.],
  [centred], [In a group of several, the candidate with clearly more of its points in the centre of its own photo wins.], [A reference photo is about the object in its centre.],
  [margin], [If none is clearly more centred, the old rule: best `f_inliers` with a margin.], [The case of real look-alikes.],
)

#v(1em)

The result can now be *several matches*, one per group: Spaengien, De Spieghel, De Mouwe and Den Arend for the Grote Markt photo. The page shows all of them.

#pagebreak()

= Open points

#t(
  columns: (auto, 1fr, 1fr),
  [*question*], [*why it matters*], [*what to do*],
  [How much overlap makes one group?], [Too little: look-alikes get through as separate matches. Too much: neighbours with a shared wall compete.], [Start with half of the smaller area. Check on photos of squares and rows of houses.],
  [How to measure "centred"?], [A wide reference photo has its object smaller in the middle.], [Start with the middle 60% of the width. Compare with the feedback labels.],
  [What if the match is only at the edge of your photo?], [A neighbour that is half in view still matches strongly.], [It is in the photo, so it is not wrong. Maybe show it as "also in view" instead of "match".],
  [Does it also help `unstir match` for the plates?], [Plates of streets and squares show several objects too.], [The same grouping, with the plate as "your photo".],
)

#v(1em)

#badge(green)[data] The feedback buttons on the lookup page give exactly the labels to set these thresholds: photos of squares and streets, with "same building" or "different" per candidate.
