// Cluster report, filled in by `glashelder cluster report` from data.json.
#set text(font: ("IBM Plex Sans", "Libertinus Serif"))
#set page(margin: 1cm)

#let data = json("data.json")
#let meta = data.meta

#let num(x, digits: 2) = if x == none { [-] } else { [#calc.round(x, digits: digits)] }

#let title(cluster) = if cluster.name == "" {
  [Cluster #cluster.id]
} else {
  [Cluster #cluster.id: #cluster.name]
}

#let cluster-stats(cluster) = table(
  columns: (5cm, auto),
  stroke: 1pt + luma(200),
  [Density], num(cluster.density),
  [Compactness], num(cluster.compactness),
  [Std distance], num(cluster.std_distance),
  [Max distance], num(cluster.max_distance),
  [Avg pairwise distance], num(cluster.avg_pairwise_distance),
  [Persistence], num(cluster.persistence),
)

#let show-cluster(i, cluster, stats: false) = block(
  fill: if calc.rem-euclid(i, 2) == 0 { luma(235) } else { white },
  inset: 0.5cm,
  breakable: false,
  width: 100%,
)[
  #grid(
    columns: (1fr, auto),
    [=== #title(cluster)],
    [#cluster.size images (#num(cluster.percentage, digits: 1) %)],
  )

  #if stats {
    cluster-stats(cluster)
  }

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    column-gutter: 5pt,
    ..cluster.samples.map(img => grid.cell(
      align: center,
      image(img, height: 4cm, width: 100%, fit: "cover"),
    ))
  )
]

= Cluster Report

== Metadata

#table(
  columns: (5cm, 1fr),
  [Embeddings], raw(meta.embeddings),
  [Embeddings model], [#meta.model (#raw(meta.model_id))],
  [Created], meta.created,
  [UMAP], [#meta.umap.dimensions dimensions, n_neighbors #meta.umap.n_neighbors, min_dist #meta.umap.min_dist],
  [Minimum cluster size], [#meta.hdbscan.min_cluster_size],
)

#grid(
  columns: (1fr, 1fr),
  image("clusters_2d.png"), image("clusters_3d.png"),
)

== Overall Characteristics

#table(
  columns: (5cm, 1fr),
  [\# Clusters], [#data.overall.n_clusters],
  [\# Images], [#data.overall.n_samples],
  [\# Noise], [#data.overall.n_noise],
  [Noise ratio], num(data.overall.noise_ratio, digits: 3),
  [Silhouette score], num(data.overall.silhouette_score, digits: 3),
)

#pagebreak(weak: true)

== Most Dense Clusters

#for (i, cluster) in data.clusters.sorted(key: it => -it.density).slice(0, calc.min(10, data.clusters.len())).enumerate() {
  show-cluster(i, cluster)
}

#pagebreak(weak: true)

== All Clusters

#for (i, cluster) in data.clusters.sorted(key: it => it.id).enumerate() {
  show-cluster(i, cluster, stats: true)
}
