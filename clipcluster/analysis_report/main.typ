#set text(font: "IBM Plex Sans")
#set page(margin: 1cm)

#let data = json("./data/cluster_stats.json")
#let images = json("cluster_images.json")

#let most-dense = data.clusters.sorted(key: it => -it.density).slice(0, 10)
#let recursive = data.clusters.filter(it => it.cluster_id.contains(".")).sorted(key: it => it.cluster_id)

#let show-cluster(i, cluster) = block(
  fill: if calc.rem-euclid(i, 2) == 0 { luma(235) } else { white },
  inset: 0.5cm,
  breakable: false,
)[
  #grid(
    columns: (3cm, 1fr),
    [=== Cluster #cluster.cluster_id],
    [#cluster.size images (#{ calc.round(float(cluster.percentage), digits: 1) } %).],
  )

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    column-gutter: 5pt,
    ..images
      .at(str(cluster.cluster_id))
      .map(img => grid.cell(align: center, image(img, height: 4cm, width: 100%, fit: "cover")))
  )
]


= Cluster Density Report

== Most Dense Clusters

#for (i, cluster) in most-dense.enumerate() {
  show-cluster(i, cluster)
}

#pagebreak(weak: true)

== Recursively Clustered

#for (i, cluster) in recursive.enumerate() {
  show-cluster(i, cluster)
}

