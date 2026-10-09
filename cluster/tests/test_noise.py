import matplotlib
import numpy as np

matplotlib.use("Agg")

from cluster.plots import create_2d_visualization
from cluster.stats import calculate_cluster_statistics, is_noise, noise_mask

HIERARCHICAL = np.array(["-1", "0", "0", "1.0", "1.0", "1.noise", "2.1.noise"], object)


def test_is_noise():
    assert is_noise(-1)
    assert is_noise("-1")
    assert is_noise("1.noise")
    assert not is_noise(0)
    assert not is_noise("1.0")


def test_noise_mask():
    assert noise_mask(np.array([-1, 0, 1, -1])).tolist() == [True, False, False, True]
    assert noise_mask(HIERARCHICAL).tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_statistics_with_hierarchical_labels():
    points = np.random.default_rng(0).normal(size=(len(HIERARCHICAL), 3))
    stats = calculate_cluster_statistics(points, HIERARCHICAL, None)
    assert stats["overall"]["n_noise"] == 3
    assert stats["overall"]["n_clusters"] == 2
    assert {c["cluster_id"] for c in stats["clusters"]} == {"0", "1.0"}


def test_plot_with_hierarchical_labels(tmp_path):
    points = np.random.default_rng(0).normal(size=(len(HIERARCHICAL), 2))
    create_2d_visualization(points, HIERARCHICAL, str(tmp_path / "plot.png"))
    assert (tmp_path / "plot.png").is_file()
