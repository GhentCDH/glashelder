#set text(font: "IBM Plex Sans")
#set page(margin: 1cm)

#let data = json("./data/cluster_stats.json")
#let images = json("cluster_images.json")

#let most-dense = data.clusters.sorted(key: it => -it.density).slice(0, 10)
#let recursive = data.clusters.filter(it => it.cluster_id.contains(".")).sorted(key: it => it.cluster_id)

#let cluster-stats(cluster) = table(
  columns: (5cm, auto),
  stroke: 1pt + luma(200),
  [Density], [#calc.round(cluster.density, digits: 2)],
  [Compactness], [#calc.round(cluster.compactness, digits: 2)],
  [Std distance], [#calc.round(cluster.std_distance, digits: 2)],
  [Max distance], [#calc.round(cluster.max_distance, digits: 2)],
  [Avg Pairwise Distance], [#calc.round(cluster.avg_pairwise_distance, digits: 2)],
  [Persistence], [#calc.round(cluster.persistence, digits: 2)],
)

#let show-cluster(i, cluster, stats: false) = block(
  fill: if calc.rem-euclid(i, 2) == 0 { luma(235) } else { white },
  inset: 0.5cm,
  breakable: false,
)[
  #grid(
    columns: (3cm, 1fr),
    [=== Cluster #cluster.cluster_id],
    [#cluster.size images (#{ calc.round(float(cluster.percentage), digits: 1) } %).],
  )

  #if stats {
    cluster-stats(cluster)
  }

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    column-gutter: 5pt,
    ..images
      .at(str(cluster.cluster_id))
      .map(img => grid.cell(align: center, image(img, height: 4cm, width: 100%, fit: "cover")))
  )
]


= Cluster Density Report

== Metadata

#table(
  columns: (5cm, 1fr),
  [Embeddings model], data.metadata.embedding_model,
  [Clustering dimensions], [#data.metadata.reduction_dims],
  [Timestamp], [#data.metadata.timestamp],
  [Minumum Cluster Size], [#data.metadata.min_cluster_size],
  table.hline(stroke: 2pt),
  [Hierarchical], [#data.metadata_hierarchical.enabled],
  ..if data.metadata_hierarchical.enabled {
    (
      [Density threshold],
      [#data.metadata_hierarchical.density_threshold],
      [Max depth],
      [#data.metadata_hierarchical.max_depth],
    )
  },
)

#grid(
  columns: (1fr, 1fr),
  image("./data/clusters_2d.png"), image("./data/clusters_3d.png"),
)

== Overall Characteristics

#table(
  columns: (5cm, 1fr),
  [\# Clusters], [#data.overall.n_clusters],
  [\# Images], [#data.overall.n_samples],
  [\# Noise], [#data.overall.n_noise],
  [Noise Ratio], [#data.overall.noise_ratio],
  [Silhouette Score], [#data.overall.silhouette_score],
)

#pagebreak(weak: true)

== Most Dense Clusters

#for (i, cluster) in most-dense.enumerate() {
  show-cluster(i, cluster)
}

#pagebreak(weak: true)

== Recursively Clustered

#for (i, cluster) in recursive.enumerate() {
  show-cluster(i, cluster)
}

#pagebreak(weak: true)

== All Clusters
#for (i, cluster) in data.clusters.sorted(key: it => it.cluster_id).enumerate() {
  show-cluster(i, cluster, stats: true)
}
