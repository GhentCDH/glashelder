#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2"
#import "@preview/lilaq:0.6.0" as lq

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

// GEOMETRY
// A synthetic scene: a church with a facade 14 m in front of camera A and a tower 4 m
// behind the facade. Camera B is 4 m to the right of A and turned to the facade.
// All plots of the theory pages are computed from it. World axes: X right, Y down, Z forward.

#let vdot(a, b) = a.zip(b).map(((x, y)) => x * y).sum()
#let mv(m, v) = m.map(r => vdot(r, v))
#let tr(m) = range(m.first().len()).map(j => m.map(r => r.at(j)))
#let mm(a, b) = a.map(r => tr(b).map(c => vdot(r, c)))
#let madd(a, b) = a.zip(b).map(((r, s)) => r.zip(s).map(((x, y)) => x + y))
#let add(a, b) = a.zip(b).map(((x, y)) => x + y)
#let sub(a, b) = a.zip(b).map(((x, y)) => x - y)
#let mul(a, s) = a.map(x => x * s)
#let skew(t) = ((0, -t.at(2), t.at(1)), (t.at(2), 0, -t.at(0)), (-t.at(1), t.at(0), 0))
#let hom(p) = (..p, 1)
#let dehom(p) = (p.at(0) / p.at(2), p.at(1) / p.at(2))
#let dist(a, b) = calc.sqrt(vdot(sub(a, b), sub(a, b)))

#let IW = 1024
#let IH = 768
#let focal = 700
#let K = ((focal, 0, IW / 2), (0, focal, IH / 2), (0, 0, 1))
#let K-inv = ((1 / focal, 0, -IW / 2 / focal), (0, 1 / focal, -IH / 2 / focal), (0, 0, 1))

#let yaw = calc.asin(4 / calc.sqrt(4 * 4 + 14 * 14))
#let cam-a = (C: (0, 0, 0), R: ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
#let cam-b = (
  C: (4, 0, 0),
  R: ((calc.cos(yaw), 0, calc.sin(yaw)), (0, 1, 0), (-calc.sin(yaw), 0, calc.cos(yaw))),
)
#let project(cam, X) = dehom(mv(K, mv(cam.R, sub(X, cam.C))))
#let proj-a = X => project(cam-a, X)
#let proj-b = X => project(cam-b, X)
/// The 3D point at depth `s` on the ray of pixel `x`.
#let ray-point(cam, x, s) = add(cam.C, mul(mv(tr(cam.R), mv(K-inv, hom(x))), s))

// B relative to A: x_B ~ K (R X + t)
#let t-ab = mul(mv(cam-b.R, cam-b.C), -1)
#let F = mm(mm(tr(K-inv), skew(t-ab)), mm(cam-b.R, K-inv))
#let plane-d = 14
#let H = mm(mm(K, madd(cam-b.R, t-ab.map(ti => (0, 0, ti / plane-d)))), K-inv)
#let apply-h(x) = dehom(mv(H, hom(x)))
#let epi-b(x) = mv(F, hom(x))
#let epi-a(x) = mv(tr(F), hom(x))
#let epipole-b = dehom(mv(K, t-ab))
#let line-dist(l, x) = calc.abs(vdot(l, hom(x))) / calc.sqrt(l.at(0) * l.at(0) + l.at(1) * l.at(1))
/// The part of line l = (a, b, c) between u0 and u1.
#let line-seg(l, u0, u1) = ((u0, -(l.at(0) * u0 + l.at(2)) / l.at(1)), (u1, -(l.at(0) * u1 + l.at(2)) / l.at(1)))

#let circle-pts(c, r, n: 20) = range(n).map(i => {
  let a = i / n * 360deg
  (c.at(0) + r * calc.cos(a), c.at(1) + r * calc.sin(a), c.at(2))
})
#let rect-pts(x0, x1, y0, y1, z) = ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))
#let church = (
  tower: ((-1.2, -3.5, 18), (1.2, -3.5, 18), (1.2, -6.5, 18), (0, -8.5, 18), (-1.2, -6.5, 18)),
  louvre: rect-pts(-0.4, 0.4, -6, -5, 18),
  facade: ((-3.5, 4, 14), (3.5, 4, 14), (3.5, -1, 14), (0, -4, 14), (-3.5, -1, 14)),
  door: rect-pts(-0.7, 0.7, 2, 4, 14),
  win-l: rect-pts(-2.6, -1.6, 0, 2.8, 14),
  win-r: rect-pts(1.6, 2.6, 0, 2.8, 14),
  rose: circle-pts((0, -2, 14), 0.6),
)
// points used to show matches: on the facade plane, and on the tower (off the plane)
#let facade-pts = ((0, -4, 14), (-3.5, 4, 14), (-2.6, 0, 14), (3.5, -1, 14))
#let tower-pts = ((0, -8.5, 18), (-1.2, -6.5, 18), (1.2, -3.5, 18))
#let parallax = tower-pts.map(X => dist(apply-h(proj-a(X)), proj-b(X)))

#for X in facade-pts + tower-pts {
  assert(line-dist(epi-b(proj-a(X)), proj-b(X)) < 1e-6, message: "F is wrong")
}
#for X in facade-pts { assert(dist(apply-h(proj-a(X)), proj-b(X)) < 1e-6, message: "H is wrong") }

#let h-col = purple.darken(10%)
#let pt-col = black

// PLOTS
#let image-plot(title, ..children) = lq.diagram(
  width: 8.4cm,
  height: 8.4cm * IH / IW,
  xlim: (0, IW),
  ylim: (IH, 0),
  title: small(title),
  grid: none,
  xaxis: (tick-distance: 256, exponent: 0),
  yaxis: (tick-distance: 256, exponent: 0),
  ..children,
)

#let church-paths(proj, stroke: 0.8pt + luma(30%), fill: white) = (
  lq.path(..church.tower.map(proj), closed: true, stroke: stroke, fill: luma(92%)),
  lq.path(..church.louvre.map(proj), closed: true, stroke: stroke),
  lq.path(..church.facade.map(proj), closed: true, stroke: stroke, fill: fill),
  ..("door", "win-l", "win-r", "rose").map(k => lq.path(..church.at(k).map(proj), closed: true, stroke: stroke)),
)

#let dots(pts, color: pt-col, size: 5pt, mark: "o") = lq.scatter(
  pts.map(p => p.at(0)),
  pts.map(p => p.at(1)),
  color: color,
  size: size,
  mark: mark,
)

/// Photo A and photo B, with the facade homography. The tower is 4 m off the facade plane.
#let h-plots = {
  let tower-h = X => apply-h(proj-a(X))
  grid(
    columns: 2,
    column-gutter: 1.5em,
    image-plot(
      [photo A],
      ..church-paths(proj-a),
      dots(facade-pts.map(proj-a)),
      dots(tower-pts.map(proj-a)),
    ),
    image-plot(
      [photo B: true position (black), $H x$ (purple)],
      ..church-paths(proj-b),
      lq.path(..church.tower.map(tower-h), closed: true, stroke: (paint: h-col, thickness: 0.8pt, dash: "dashed")),
      ..tower-pts.map(X => lq.line(tower-h(X), proj-b(X), stroke: 1.5pt + bad)),
      dots(facade-pts.map(X => apply-h(proj-a(X))), color: h-col, size: 9pt, mark: "o"),
      dots(tower-pts.map(tower-h), color: h-col, size: 5pt, mark: "s"),
      dots(facade-pts.map(proj-b)),
      dots(tower-pts.map(proj-b)),
    ),
  )
}

/// Epipolar lines in both photos for the same matches, and one wrong match that slides along a line.
#let f-plots = {
  let pts = facade-pts + tower-pts
  let wrong-x = pts.at(2)
  let l = epi-b(proj-a(wrong-x))
  let true-b = proj-b(wrong-x)
  let wrong = (true-b.at(0) + 230, -(l.at(0) * (true-b.at(0) + 230) + l.at(2)) / l.at(1))
  grid(
    columns: 2,
    column-gutter: 1.5em,
    image-plot(
      [photo A: lines $F^T x'$ are parallel],
      ..church-paths(proj-a),
      ..pts.map(X => lq.line(..line-seg(epi-a(proj-b(X)), 0, IW), stroke: 0.7pt + knob)),
      dots(pts.map(proj-a)),
    ),
    image-plot(
      [photo B: lines $F x$, all through $e'$ (left, outside)],
      ..church-paths(proj-b),
      ..pts.map(X => lq.line(..line-seg(epi-b(proj-a(X)), 0, IW), stroke: 0.7pt + knob)),
      dots(pts.map(proj-b)),
      dots((wrong,), color: bad, size: 6pt),
      lq.place(wrong.at(0), wrong.at(1), align: bottom, pad(bottom: 4pt, text(size: 7.5pt, fill: bad)[wrong, but inlier])),
    ),
  )
}

/// All epipolar lines of photo B meet in the epipole e', the image of camera A's center.
#let pencil = {
  let pts = facade-pts + tower-pts
  let (eu, ev) = epipole-b
  lq.diagram(
    width: 15cm,
    height: 15cm * 1300 / 3500,
    xlim: (-2400, 1100),
    ylim: (1050, -250),
    grid: none,
    xaxis: (tick-distance: 500, exponent: 0),
    yaxis: (tick-distance: 500, exponent: 0),
    lq.rect(0, 0, width: IW, height: IH, fill: luma(97%), stroke: 1pt),
    lq.place(IW / 2, -60, small[photo B]),
    ..pts.map(X => lq.line(..line-seg(epi-b(proj-a(X)), -2400, 1100), stroke: 0.7pt + knob)),
    dots(pts.map(proj-b), size: 4pt),
    dots(((eu, ev),), color: knob, size: 7pt),
    lq.place(eu, ev, align: top + left, pad(top: 6pt, left: 2pt, text(size: 9pt, fill: knob)[$e'$ = image of camera A's center])),
  )
}

/// The chance that a random point passes the test, by threshold.
#let chance = {
  let area = IW * IH
  let xs = lq.linspace(0.5, 16, num: 64)
  let p-h = r => calc.pi * r * r / area * 100
  let p-f = f => 2 * f * 1000 / area * 100
  lq.diagram(
    width: 11cm,
    height: 6cm,
    yscale: "log",
    xlim: (0, 16.5),
    xlabel: small[threshold (px)],
    ylabel: small[chance (%)],
    legend: (position: bottom + right),
    xaxis: (tick-distance: 2),
    lq.plot(xs, xs.map(p-f), mark: none, stroke: 1.2pt + knob, label: small[F: band $2 f_"px" dot 1000 slash (1024 dot 768)$]),
    lq.plot(xs, xs.map(p-h), mark: none, stroke: 1.2pt + h-col, label: small[H: disc $pi h_"px"^2 slash (1024 dot 768)$]),
    lq.scatter((4, 2), (p-f(4), p-f(2)), color: knob, size: 6pt),
    lq.place(4, p-f(4), align: right + bottom, pad(right: 4pt, bottom: 2pt, text(size: 8pt, fill: knob)[4: #calc.round(p-f(4), digits: 2)%])),
    lq.place(2, p-f(2), align: left + top, pad(left: 4pt, top: 2pt, text(size: 8pt, fill: knob)[2: #calc.round(p-f(2), digits: 2)%])),
    lq.scatter((8,), (p-h(8),), color: h-col, size: 6pt),
    lq.place(8, p-h(8), align: left + top, pad(left: 4pt, top: 2pt, text(size: 8pt, fill: h-col)[8: #calc.round(p-h(8), digits: 3)%])),
  )
}

// 3D
#let to3(p) = (p.at(2), -p.at(1), p.at(0))

#let frustum(cam, s, clr, label) = {
  import cetz.draw: *
  let c = to3(cam.C)
  let corners = ((0, 0), (IW, 0), (IW, IH), (0, IH)).map(x => to3(ray-point(cam, x, s)))
  line(..corners, close: true, fill: clr.lighten(85%).transparentize(20%), stroke: 0.8pt + clr)
  for p in corners { line(c, p, stroke: 0.5pt + clr) }
  circle(c, radius: 0.12, fill: clr, stroke: none)
  content(c, anchor: "east", padding: 0.25, box(fill: white, inset: 1pt, text(size: 9pt, fill: clr, label)))
}

#let church3d(face: white) = {
  import cetz.draw: *
  line(..church.tower.map(to3), close: true, fill: luma(92%), stroke: 0.6pt + luma(30%))
  line(..church.facade.map(to3), close: true, fill: face, stroke: 0.6pt + luma(30%))
  for k in ("door", "win-l", "win-r", "rose") {
    line(..church.at(k).map(to3), close: true, stroke: 0.5pt + luma(30%))
  }
}

#let view3d(body) = cetz.canvas(length: 0.6cm, {
  import cetz.draw: *
  ortho(x: 30deg, y: -40deg, sorted: false, body)
})

/// One camera, one ray: every point on the ray gives the same pixel.
#let pinhole3d = view3d({
  import cetz.draw: *
  let s = 5
  frustum(cam-a, s, knob, [camera center $C$])
  church3d()
  let X = (0, -4, 14)
  let x = proj-a(X)
  line(to3(cam-a.C), to3(ray-point(cam-a, x, 20)), stroke: (paint: bad, thickness: 0.8pt, dash: "dashed"))
  for d in (8, 20) {
    circle(to3(ray-point(cam-a, x, d)), radius: 0.12, fill: white, stroke: 0.8pt + bad)
  }
  content(to3(ray-point(cam-a, x, 8)), anchor: "north-east", padding: 0.2, text(size: 8pt, fill: bad)[same pixel])
  circle(to3(X), radius: 0.14, fill: bad, stroke: none)
  content(to3(X), anchor: "south-west", padding: 0.2, text(size: 9pt, fill: bad)[$X$])
  circle(to3(ray-point(cam-a, x, s)), radius: 0.12, fill: black, stroke: none)
  content(to3(ray-point(cam-a, x, s)), anchor: "south-east", padding: 0.2, small[$x = (u, v, 1)$])
  content(to3(ray-point(cam-a, (IW, IH), s)), anchor: "north-west", padding: 0.2, text(size: 9pt, fill: knob)[image plane at depth $f$])
})

/// Two cameras and the facade plane: a homography between the two images.
#let homography3d = view3d({
  import cetz.draw: *
  let s = 2.2
  church3d(face: h-col.lighten(85%))
  frustum(cam-a, s, knob, [$C$ (A)])
  frustum(cam-b, s, knob, [$C'$ (B)])
  for (X, clr) in ((facade-pts.at(2), h-col), (tower-pts.at(0), bad)) {
    for cam in (cam-a, cam-b) {
      line(to3(cam.C), to3(X), stroke: 0.7pt + clr)
      circle(to3(ray-point(cam, project(cam, X), s)), radius: 0.1, fill: clr, stroke: none)
    }
    circle(to3(X), radius: 0.14, fill: clr, stroke: none)
  }
  content(to3(facade-pts.at(2)), anchor: "east", padding: 0.25, text(size: 9pt, fill: h-col)[on the plane])
  content(to3(tower-pts.at(0)), anchor: "south", padding: 0.2, text(size: 9pt, fill: bad)[off the plane])
})

/// Epipolar geometry: the plane through C, C' and X cuts both image planes in the epipolar lines.
#let epipolar3d = view3d({
  import cetz.draw: *
  let s = 2.2
  let x = proj-a((0, -4, 14))
  let near = ray-point(cam-a, x, 9)
  let far = ray-point(cam-a, x, 22)
  line(to3(cam-a.C), to3(cam-b.C), to3(far), close: true, fill: knob.lighten(80%).transparentize(40%), stroke: none)
  church3d()
  frustum(cam-a, s, knob, [$C$ (A)])
  frustum(cam-b, s, knob, [$C'$ (B)])
  line(to3(cam-a.C), to3(cam-b.C), stroke: 1pt + black)
  content(to3(mul(cam-b.C, 0.5)), anchor: "north-east", padding: 0.1, box(fill: white, inset: 1pt, small[baseline $t$]))
  line(to3(cam-a.C), to3(far), stroke: (paint: bad, thickness: 0.8pt, dash: "dashed"))
  // the epipolar line in B: images of the points on the ray, drawn in B's image plane
  line(..(near, far).map(X => to3(ray-point(cam-b, proj-b(X), s))), stroke: 1.2pt + knob)
  for (X, l) in ((near, [near]), ((0, -4, 14), [$X$]), (far, [far])) {
    line(to3(cam-b.C), to3(X), stroke: 0.4pt + luma(50%))
    circle(to3(X), radius: 0.13, fill: if l == [$X$] { bad } else { white }, stroke: 0.8pt + bad)
    content(to3(X), anchor: "south", padding: 0.2, text(size: 9pt, fill: bad, l))
    circle(to3(ray-point(cam-b, proj-b(X), s)), radius: 0.09, fill: bad, stroke: none)
  }
  circle(to3(ray-point(cam-a, x, s)), radius: 0.1, fill: black, stroke: none)
  content(to3(far), anchor: "west", padding: 0.3, text(size: 9pt, fill: knob)[epipolar plane])
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

= Background: cameras and homogeneous coordinates

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  pinhole3d,
  [
    The next pages use one synthetic scene. A church facade is 14 m in front of camera A. The tower is 4 m behind the facade. Camera B is 4 m to the right and turned #calc.round(yaw / 1deg, digits: 1)° to the facade. Both images are #IW x #IH px, focal length #focal px. Typst computes all plots from this scene, so the numbers are exact.

    #v(0.5em)
    *A pixel is a ray.* A camera is a pinhole at its center $C$. All 3D points on one ray through $C$ give the same pixel. The camera cannot measure where on the ray the point is (its depth).

    #v(0.5em)
    *Homogeneous coordinates* write this down. Pixel $(u, v)$ becomes $x = (u, v, 1)^T$, and $lambda x$ is the same pixel for every $lambda != 0$. To go back, divide by the last value:
    $ pi(u, v, w) = (u / w, v / w) $
    $(u, v, 0)$ is a point at infinity: a direction, not a place.
  ],
)

#v(1em)

#t(
  columns: (auto, 1fr),
  [*projection*], [$x tilde K [R | t] X$, with $X = (X, Y, Z, 1)^T$ the 3D point. $tilde$ means "equal up to a scale". $R$ (rotation) and $t$ (translation) tell where the camera is and where it looks.],
  [*intrinsics*], [$K = mat(f, 0, c_u; 0, f, c_v; 0, 0, 1)$: focal length $f$ in px and the image center $(c_u, c_v)$. $K^(-1) x$ is the direction of the ray of pixel $x$.],
  [*lines*], [A line is also 3 numbers: $l = (a, b, c)^T$ is the line $a u + b v + c = 0$. A point is on the line when $l^T x = 0$. The distance from a point to the line is $|l^T x| slash sqrt(a^2 + b^2)$, with $x = (u, v, 1)^T$. The F test uses this.],
  [*cross product*], [$[t]_times = mat(0, -t_3, t_2; t_3, 0, -t_1; -t_2, t_1, 0)$, so that $[t]_times y = t times y$. The line through two points $x_1, x_2$ is $x_1 times x_2$.],
)

#pagebreak()

= Homography: point to point

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  homography3d,
  [
    A homography $H$ is a 3 x 3 matrix that sends a point to a point:
    $ x' tilde H x, quad H = mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) $
    $H$ is only known up to a scale, so it has *8 degrees of freedom*. Each match gives 2 equations, so *4 matches* fix $H$ (no 3 of them on one line).

    #v(0.5em)
    *When is it exact?*
    - All 3D points are on one plane $n^T X = d$ (one facade):
      $ H = K' (R + (t n^T) / d) K^(-1) $
    - The camera only turns and does not move ($t = 0$): $H = K' R K^(-1)$.
    - Almost: the scene is far away compared to the distance between the cameras.
  ],
)

#v(1em)

#h-plots

#v(0.5em)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  [
    *The test.* $H$ predicts one point. The error is the distance to it:
    $ d_H = ||x' - pi(H x)|| <= h_"px" $
    So the inlier area is a disc of radius #h-px around the predicted point.
  ],
  [
    *Parallax.* Here $H$ belongs to the facade plane ($d = #plane-d$ m). It sends the facade points (circles) exactly onto their match. The tower is 4 m behind that plane, so $H$ puts it #calc.round(calc.min(..parallax)) to #calc.round(calc.max(..parallax)) px off (purple squares, red lines). A church has many planes (facade, tower, side walls, roof), so one $H$ only explains one part of it.
  ],
)

#pagebreak()

= Fundamental matrix: point to line

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  epipolar3d,
  [
    The plane through $C$, $C'$ and $X$ is the *epipolar plane*. It cuts each image plane in a line. If $X$ moves along its ray (near, far), its image in B moves along that line. That is the *epipolar line*.

    #v(0.5em)
    The *fundamental matrix* $F$ (3 x 3) gives this line for every point:
    $ l' = F x, quad x'^T F x = 0 $
    and from the cameras:
    $ F = K'^(-T) [t]_times R K^(-1) $

    #v(0.5em)
    $[t]_times$ has rank 2, so $det F = 0$. Scale and $det F = 0$ remove 2 of the 9 values: *7 degrees of freedom*, *7 matches* minimum (8 with the linear algorithm).
  ],
)

#v(1em)

#f-plots

#v(0.5em)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  [
    *The test.* $F$ predicts a line $l' = F x = (a, b, c)^T$. The error is the distance to that line:
    $ d_F = (|x'^T F x|) / sqrt(a^2 + b^2) <= f_"px" $
    OpenCV (USAC) uses the Sampson distance. It is an approximation of this distance in both images at the same time, and almost the same value.

    #v(0.5em)
    #badge(bad)[weak point] The test does not see an error *along* the line. The red match is 230 px from the right place, but it is on the line, so it is an inlier.
  ],
  [
    *Epipoles.* $F e = 0$ and $F^T e' = 0$. The epipole $e'$ is the image of camera A's center in photo B. All lines in B go through it.

    #v(0.5em)
    Here B is to the right of A, so $e'$ is far to the left of photo B (next figure). Camera B is not in front of A, so in photo A the epipole is a point at infinity, $e = (1, 0, 0)^T$, and all lines in A are parallel.

    #v(0.5em)
    *F is valid for any static scene*, also the tower: all 7 points are on their lines. It only needs a camera that moves ($t != 0$).
  ],
)

#v(0.5em)
#pencil

#pagebreak()

= Homography and fundamental matrix side by side

#t(
  columns: (auto, 1fr, 1fr),
  [], [*homography $H$*], [*fundamental matrix $F$*],
  [*maps*], [point to point], [point to line],
  [*equation*], [$x' tilde H x$], [$x'^T F x = 0$],
  [*from the cameras*], [$K' (R + t n^T slash d) K^(-1)$], [$K'^(-T) [t]_times R K^(-1)$],
  [*degrees of freedom*], [8], [7 (rank 2)],
  [*minimum matches*], [4], [7 (8 linear)],
  [*valid for*], [one plane, or a camera that only turns], [any static scene, if the camera moves],
  [*error*], [distance to the predicted point], [distance to the epipolar line],
  [*inlier area*], [disc, radius #h-px], [band, 2 x #f-px wide, across the image],
  [*option*], [#raw("--h-px") (8)], [#raw("--f-px") (4)],
  [*in `unstir`*], [`h_inliers`], [`f_inliers`, `cells`],
)

#v(1em)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  [
    *H inliers are also F inliers.* If $H$ is true for a set of matches, then $F = [e']_times H$ is true for them too, for any $e'$. So a set of matches on one facade fits many different $F$, and RANSAC can always find one. This is why `f_inliers` is almost always larger than `h_inliers` (median 20 against 8 in the test run).
  ],
  [
    *F is more general, so it is less strict.* A real match from another viewpoint of the same church fits $F$, but not $H$. A wrong match only has to be close to a line, not to a point. So $F$ finds more real matches, and more wrong matches too. The next page shows how many.
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

#grid(
  columns: (auto, 1fr),
  column-gutter: 2em,
  chance,
  [
    *Chance that one random match passes*, before RANSAC chooses the model. The band grows linearly with #f-px, the disc with the square of #h-px. A line across the image is about 1000 px long.

    #v(0.5em)
    #f-px 2 halves the chance. RANSAC choice and the common layout of churches (previous page) make the real numbers higher, but they scale in the same direction.
  ],
)

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
