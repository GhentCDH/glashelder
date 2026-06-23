#import "@preview/lilaq:0.6.0" as lq
#import "lib.typ": _theme

#let config-table(report) = context table(
  columns: (1fr, 1fr),
  stroke: _theme.get().bg.darken(10%) + 1pt,
  [model], [#report.config.model],
  [$tau$], [#report.config.tau],
)

#let array-min(arr) = {
  arr.reduce((acc, x) => calc.min(acc, x))
}

#let bg = oklch(rgb("#e6eff1"))
#let fg = bg.darken(70%)

#let cycle = (bg.darken(30%).saturate(30%), fg.rotate(180deg).lighten(30%).saturate(40%))

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
    xaxis: (ticks: meta-keys),
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

= Validation Report

== Conditional Prompt
#let report = json("./assets/reports/report.json")

=== Config

#config-table(report)

=== Overall

#plot-overall-metrics(report)

== Single Prompt
#let report = json("./assets/reports/report-no-bounding-boxes.json")

=== Config

#config-table(report)

=== Overall

#plot-overall-metrics(report)
