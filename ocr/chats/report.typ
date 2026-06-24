#import "@preview/lilaq:0.6.0" as lq
#import "lib.typ": bg, fg

#let config-table(report) = context table(
  columns: (1fr, 1fr),
  stroke: bg.darken(10%) + 1pt,
  [model], [#report.config.model],
  [$tau$], [#report.config.tau],
)

#let array-min(arr) = {
  arr.reduce((acc, x) => calc.min(acc, x))
}

#let cycle = (
  bg.darken(30%).saturate(30%),
  fg.rotate(180deg).lighten(30%).saturate(40%),
  fg.rotate(-120deg).lighten(70%).saturate(40%),
  fg.rotate(175deg).lighten(70%).saturate(50%),
)

#let extract-metrics(metrics) = {
  let metrics = metrics
    .pairs()
    .filter(i => not i.at(0).starts-with("n_"))
    .map(i => {
      let key = i.at(0)
      if key == "recall_at_tau" or key == "recall_any_section" or key == "mean_best_sim" {
        ("1 - " + i.at(0), 1 - i.at(1))
      } else if key == "cer" {
        ("cer * 10", i.at(1) * 10)
      } else {
        i
      }
    })
    .to-dict()

  let metrics-keys = metrics.keys().enumerate()
  let metrics-values = metrics.values()

  (metrics-keys, metrics-values)
}

// Compare several reports: grouped bars of the `combined` metrics, one colour
// per report (model). `reports` is an array of (label, report) pairs.
#let plot-combined-comparison(reports) = {
  set text(size: 8pt, font: "Maple Mono")
  let n = reports.len()
  let width = 0.8 / n

  let keys = extract-metrics(reports.at(0).at(1).overall.combined).at(0)

  let bars = reports
    .enumerate()
    .map(((i, entry)) => {
      let (label, report) = entry
      let values = extract-metrics(report.overall.combined).at(1)
      let offset = (i - (n - 1) / 2) * width
      lq.bar(range(values.len()), values, offset: offset, width: width * 0.9, label: label)
    })

  lq.diagram(
    cycle: cycle,
    margin: (top: 30%),
    xaxis: (ticks: keys, subticks: none),
    ..bars,
  )
}

#let plot-overall-metrics(report) = {
  set text(size: 8pt, font: "Maple Mono")
  let meta = report.overall.meta
  let content = report.overall.content
  // let combined = report.overall.combined

  let (meta-keys, meta-values) = extract-metrics(meta)
  let (content-keys, content-values) = extract-metrics(content)
  // let (combined-keys, combined-values) = extract-metrics(combined)

  lq.diagram(
    margin: (top: 30%),
    xaxis: (ticks: meta-keys, subticks: none),
    cycle: cycle,
    lq.bar(range(meta-values.len()), meta-values, offset: -0.2, width: 0.35, label: [meta]),
    lq.bar(range(content-values.len()), content-values, offset: 0.2, width: 0.35, label: [content]),
    // lq.bar(range(combined-values.len()), combined-values, offset: 0.25, width: 0.21, label: [combined]),
  )
}

#show: lq.show_(
  lq.tick-label.with(kind: "x"),
  it => box(
    width: 0pt,
    align(right, rotate(-45deg, reflow: true, it)),
  ),
)

#show lq.selector(lq.tick-label): it => {
  show math.equation: set text(font: "Maple Mono", weight: 300)
  it
}

= Validation Report

Hier een aantal grafieken die de performance van de verschillende methoden toont en vergelijkt.
De metrieken die worden geplot zijn steeds to gekozen dat _een lagere waarde beter is_.
Zo wordt bijvoorbeeld `1 - recall` geplot aangezien een hogere `recall` beter is.
`cer` (character error rate) wordt geplot als `10 * cer` aangezien het anders te klein zou zijn om duidelijk
op dezelfde assen te tonen.

ss Conditional Prompt
#let report = json("./assets/reports/report.json")

=== Config

#config-table(report)

=== Overall

#align(center, figure(plot-overall-metrics(report), caption: [Conditional Prompt metrics]))

== Single Prompt
#let report = json("./assets/reports/report-no-bounding-boxes.json")

=== Config

#config-table(report)

=== Overall

#align(center, figure(plot-overall-metrics(report), caption: [Single Prompt metrics]))

== Comparison (combined)

#align(center)[
  #figure(
    plot-combined-comparison((
      ([single], json("./assets/reports/report-no-bounding-boxes.json")),
      ([split], json("./assets/reports/report-split.json")),
      ([conditional], json("./assets/reports/report.json")),
      ([cond. typed], json("./assets/reports/report-typed.json")),
    )),
    caption: [Comparison of metrics],
  )<comparison>
]
