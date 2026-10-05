"""
Visualization and statistical analysis of image matching results.
Produces high-res PNGs with a modern monospaced aesthetic.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
BG = "#0f1117"
BG_PANEL = "#181b25"
ACCENT = "#6c8aff"
ACCENT2 = "#ff6c8a"
ACCENT3 = "#6cffa0"
ACCENT4 = "#ffc46c"
GRID_COLOR = "#2a2e3a"
TEXT_COLOR = "#c8cdd8"
TEXT_DIM = "#6b7280"

DPI = 300


def setup_style():
    """Apply a dark modern style globally."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": BG,
            "axes.facecolor": BG_PANEL,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_DIM,
            "ytick.color": TEXT_DIM,
            "legend.facecolor": BG_PANEL,
            "legend.edgecolor": GRID_COLOR,
            "legend.fontsize": 8,
            "savefig.facecolor": BG,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.3,
        }
    )


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_arrays(results: list[dict]):
    """Pull out arrays we care about from the results list."""
    confidence = []
    is_match = []
    sim_0 = []
    gap_01 = []
    mean_1to9 = []
    std_1to9 = []
    zscore_0 = []
    decay_rate = []
    sim_range = []
    phash = []
    orb_conf = []

    for r in results:
        confidence.append(r["confidence"])
        is_match.append(r["is_match"])
        sim_0.append(r["similarities"][0])
        f = r.get("features", {})
        gap_01.append(f.get("gap_01", np.nan))
        mean_1to9.append(f.get("mean_1to9", np.nan))
        std_1to9.append(f.get("std_1to9", np.nan))
        zscore_0.append(f.get("sim_0_zscore", np.nan))
        decay_rate.append(f.get("decay_rate", np.nan))
        sim_range.append(f.get("sim_range", np.nan))
        phash.append(f.get("phash_distance", np.nan))
        orb_conf.append(f.get("orb_confidence", np.nan))

    return {
        k: np.array(v)
        for k, v in {
            "confidence": confidence,
            "is_match": is_match,
            "sim_0": sim_0,
            "gap_01": gap_01,
            "mean_1to9": mean_1to9,
            "std_1to9": std_1to9,
            "zscore_0": zscore_0,
            "decay_rate": decay_rate,
            "sim_range": sim_range,
            "phash": phash,
            "orb_conf": orb_conf,
        }.items()
    }


def add_watermark(fig, text="glashelder match analysis"):
    fig.text(
        0.99,
        0.005,
        text,
        fontsize=7,
        color=TEXT_DIM,
        ha="right",
        va="bottom",
        alpha=0.4,
    )


# ---------------------------------------------------------------------------
# Plot 1: Confidence distribution
# ---------------------------------------------------------------------------
def plot_confidence_distribution(d: dict, meta: dict, summary: dict, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Confidence Score Distribution", fontsize=16, fontweight="bold", y=0.97
    )

    conf = d["confidence"]
    matches = d["is_match"]

    # (a) Histogram with KDE
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 80)
    ax.hist(conf, bins=bins, color=ACCENT, alpha=0.7, edgecolor="none", label="all")
    ax.hist(
        conf[matches],
        bins=bins,
        color=ACCENT2,
        alpha=0.8,
        edgecolor="none",
        label="matches",
    )

    # KDE overlay
    if len(conf) > 10:
        kde = stats.gaussian_kde(conf, bw_method=0.02)
        xs = np.linspace(0, 1, 300)
        kde_y = kde(xs) * len(conf) * (bins[1] - bins[0])
        ax.plot(xs, kde_y, color=ACCENT3, lw=1.5, alpha=0.8, label="KDE")

    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    ax.set_title("histogram + KDE", fontsize=11, color=TEXT_DIM)
    ax.legend()

    # (b) Log-scale histogram of matches vs non-matches
    ax = axes[0, 1]
    ax.hist(
        conf[~matches],
        bins=bins,
        color=ACCENT,
        alpha=0.6,
        edgecolor="none",
        label="non-match",
    )
    ax.hist(
        conf[matches],
        bins=bins,
        color=ACCENT2,
        alpha=0.8,
        edgecolor="none",
        label="match",
    )
    ax.set_yscale("log")
    ax.set_xlabel("confidence")
    ax.set_ylabel("count (log)")
    ax.set_title("match vs non-match (log scale)", fontsize=11, color=TEXT_DIM)
    ax.legend()

    # (c) ECDF
    ax = axes[1, 0]
    sorted_conf = np.sort(conf)
    ecdf_y = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
    ax.plot(sorted_conf, ecdf_y, color=ACCENT, lw=1.5)
    # mark threshold
    threshold = meta.get("threshold", 0.82)
    # mark median & percentiles
    for pct, ls in [(50, "--"), (90, ":"), (99, "-.")]:
        val = np.percentile(conf, pct)
        ax.axvline(val, color=ACCENT4, ls=ls, lw=1, alpha=0.7)
        ax.text(
            val + 0.01,
            0.05 + pct / 200,
            f"p{pct}={val:.3f}",
            fontsize=7,
            color=ACCENT4,
            rotation=90,
        )
    ax.set_xlabel("confidence")
    ax.set_ylabel("cumulative proportion")
    ax.set_title("ECDF with percentiles", fontsize=11, color=TEXT_DIM)

    # (d) Stats summary text panel
    ax = axes[1, 1]
    ax.axis("off")
    match_rate = summary.get("match_rate", np.nan)
    total = summary.get("total_processed", 0)
    n_matches = summary.get("total_matches", 0)

    stats_lines = [
        f"  total images    {total:>10,}",
        f"  matches         {n_matches:>10,}",
        f"  match rate      {match_rate:>10.4%}",
        "",
        "  confidence stats",
        f"    mean          {np.mean(conf):>10.4f}",
        f"    median        {np.median(conf):>10.4f}",
        f"    std           {np.std(conf):>10.4f}",
        f"    min           {np.min(conf):>10.4f}",
        f"    max           {np.max(conf):>10.4f}",
        f"    skewness      {stats.skew(conf):>10.4f}",
        f"    kurtosis      {stats.kurtosis(conf):>10.4f}",
        "",
        "  match confidence",
        f"    mean          {np.mean(conf[matches]):>10.4f}"
        if matches.any()
        else "    (no matches)",
        f"    median        {np.median(conf[matches]):>10.4f}" if matches.any() else "",
        "",
        f"  threshold       {threshold}",
        f"  top_k           {meta.get('top_k', '?')}",
        f"  duration        {meta.get('duration_seconds', '?'):.1f}s",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        color=TEXT_COLOR,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BG, "edgecolor": GRID_COLOR},
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    add_watermark(fig)
    out = outdir / "01_confidence_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Feature correlations with confidence
# ---------------------------------------------------------------------------
def plot_feature_correlations(d: dict, outdir: Path):
    features = [
        ("sim_0", "top-1 similarity"),
        ("gap_01", "gap (sim₀ - sim₁)"),
        ("zscore_0", "sim₀ z-score"),
        ("mean_1to9", "mean sim 1-9"),
        ("std_1to9", "std sim 1-9"),
        ("sim_range", "sim range"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Feature Correlations with Confidence", fontsize=16, fontweight="bold", y=0.97
    )

    conf = d["confidence"]
    matches = d["is_match"]

    for ax, (key, label) in zip(axes.flat, features, strict=False):
        x = d[key]
        valid = ~np.isnan(x) & ~np.isnan(conf)
        xv, cv, mv = x[valid], conf[valid], matches[valid]

        # subsample for scatter if huge
        n = len(xv)
        if n > 8000:
            idx = np.random.default_rng(42).choice(n, 8000, replace=False)
            xs, cs, ms = xv[idx], cv[idx], mv[idx]
        else:
            xs, cs, ms = xv, cv, mv

        ax.scatter(xs[~ms], cs[~ms], s=1.5, alpha=0.15, color=ACCENT, rasterized=True)
        ax.scatter(
            xs[ms], cs[ms], s=6, alpha=0.6, color=ACCENT2, rasterized=True, zorder=5
        )

        # Pearson r
        if len(xv) > 2:
            r, _p = stats.pearsonr(xv, cv)
            sr, _sp = stats.spearmanr(xv, cv)
            ax.text(
                0.03,
                0.97,
                f"r={r:.3f}  ρ={sr:.3f}",  # noqa: RUF001
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                color=ACCENT3,
                bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.8, "pad": 2},
            )

        # trend line
        if len(xv) > 10:
            z = np.polyfit(xv, cv, 1)
            xline = np.linspace(np.nanpercentile(xv, 1), np.nanpercentile(xv, 99), 100)
            ax.plot(xline, np.polyval(z, xline), color=ACCENT4, lw=1.5, alpha=0.8)

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("confidence", fontsize=9)

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    add_watermark(fig)
    out = outdir / "02_feature_correlations.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: Hash & ORB features
# ---------------------------------------------------------------------------
def plot_hash_orb(d: dict, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Perceptual Hash & ORB Features", fontsize=16, fontweight="bold", y=0.97
    )

    conf = d["confidence"]
    matches = d["is_match"]

    # (a) phash vs confidence
    ax = axes[0, 0]
    ph = d["phash"]
    valid = ~np.isnan(ph)
    if valid.any():
        n = valid.sum()
        if n > 6000:
            idx = np.random.default_rng(42).choice(n, 6000, replace=False)
            phv, cv, mv = ph[valid][idx], conf[valid][idx], matches[valid][idx]
        else:
            phv, cv, mv = ph[valid], conf[valid], matches[valid]
        ax.scatter(phv[~mv], cv[~mv], s=2, alpha=0.15, color=ACCENT, rasterized=True)
        ax.scatter(
            phv[mv], cv[mv], s=8, alpha=0.6, color=ACCENT2, rasterized=True, zorder=5
        )
        r, _ = stats.pearsonr(ph[valid], conf[valid])
        ax.text(
            0.03,
            0.97,
            f"r={r:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            color=ACCENT3,
            bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
    ax.set_xlabel("phash distance")
    ax.set_ylabel("confidence")
    ax.set_title("phash distance vs confidence", fontsize=11, color=TEXT_DIM)

    # (b) ORB confidence vs match confidence
    ax = axes[0, 1]
    orb = d["orb_conf"]
    valid = ~np.isnan(orb)
    if valid.any():
        n = valid.sum()
        if n > 6000:
            idx = np.random.default_rng(42).choice(n, 6000, replace=False)
            ov, cv, mv = orb[valid][idx], conf[valid][idx], matches[valid][idx]
        else:
            ov, cv, mv = orb[valid], conf[valid], matches[valid]
        ax.scatter(ov[~mv], cv[~mv], s=2, alpha=0.15, color=ACCENT, rasterized=True)
        ax.scatter(
            ov[mv], cv[mv], s=8, alpha=0.6, color=ACCENT2, rasterized=True, zorder=5
        )
    ax.set_xlabel("ORB confidence")
    ax.set_ylabel("confidence")
    ax.set_title("ORB confidence vs match confidence", fontsize=11, color=TEXT_DIM)

    # (c) phash distance distribution: match vs non-match
    ax = axes[1, 0]
    valid = ~np.isnan(ph)
    if valid.any():
        bins_h = np.arange(0, 65, 1)
        ax.hist(
            ph[valid & ~matches],
            bins=bins_h,
            color=ACCENT,
            alpha=0.6,
            edgecolor="none",
            label="non-match",
            density=True,
        )
        if matches.any():
            ax.hist(
                ph[valid & matches],
                bins=bins_h,
                color=ACCENT2,
                alpha=0.8,
                edgecolor="none",
                label="match",
                density=True,
            )
        ax.set_xlabel("phash distance")
        ax.set_ylabel("density")
        ax.set_title("phash distribution by class", fontsize=11, color=TEXT_DIM)
        ax.legend()

    # (d) sim_0 vs gap_01 colored by confidence
    ax = axes[1, 1]
    s0 = d["sim_0"]
    g01 = d["gap_01"]
    valid = ~np.isnan(s0) & ~np.isnan(g01)
    if valid.any():
        n = valid.sum()
        if n > 6000:
            idx = np.random.default_rng(42).choice(n, 6000, replace=False)
            s0v, g01v, cv = s0[valid][idx], g01[valid][idx], conf[valid][idx]
        else:
            s0v, g01v, cv = s0[valid], conf[valid], conf[valid]
        sc = ax.scatter(
            s0v,
            g01v,
            c=cv,
            s=2,
            alpha=0.4,
            cmap="magma",
            rasterized=True,
            vmin=0,
            vmax=1,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("confidence", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("sim₀ (top-1 similarity)")
    ax.set_ylabel("gap₀₁")
    ax.set_title("sim₀ vs gap₀₁ (colored by confidence)", fontsize=11, color=TEXT_DIM)

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    add_watermark(fig)
    out = outdir / "03_hash_orb_features.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 4: Statistical deep-dive
# ---------------------------------------------------------------------------
def plot_statistical_analysis(d: dict, outdir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Statistical Analysis", fontsize=16, fontweight="bold", y=0.97)

    conf = d["confidence"]
    matches = d["is_match"]

    # (a) QQ-plot of confidence
    ax = axes[0, 0]
    sorted_conf = np.sort(conf)
    n = len(sorted_conf)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n))
    ax.scatter(
        theoretical, sorted_conf, s=0.5, alpha=0.3, color=ACCENT, rasterized=True
    )
    ax.plot(
        [theoretical[0], theoretical[-1]],
        [sorted_conf[0], sorted_conf[-1]],
        color=ACCENT2,
        lw=1,
        ls="--",
    )
    ax.set_xlabel("theoretical quantiles (normal)")
    ax.set_ylabel("sample quantiles")
    ax.set_title("Q-Q plot (confidence)", fontsize=11, color=TEXT_DIM)

    # (b) Box plots: confidence by match status
    ax = axes[0, 1]
    data_box = [conf[~matches], conf[matches]] if matches.any() else [conf[~matches]]
    labels_box = ["non-match", "match"] if matches.any() else ["non-match"]
    bp = ax.boxplot(
        data_box,
        labels=labels_box,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": ACCENT4, "lw": 2},
        flierprops={"marker": ".", "markersize": 1, "alpha": 0.3, "color": TEXT_DIM},
    )
    colors = [ACCENT, ACCENT2]
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("confidence")
    ax.set_title("confidence by match status", fontsize=11, color=TEXT_DIM)

    # Mann-Whitney U test
    if matches.any() and (~matches).any():
        _u_stat, u_p = stats.mannwhitneyu(
            conf[matches], conf[~matches], alternative="greater"
        )
        ax.text(
            0.5,
            0.02,
            f"Mann-Whitney U p={u_p:.2e}",
            transform=ax.transAxes,
            fontsize=8,
            ha="center",
            color=ACCENT3,
        )

    # (c) Correlation heatmap
    ax = axes[0, 2]
    feature_keys = [
        "confidence",
        "sim_0",
        "gap_01",
        "zscore_0",
        "mean_1to9",
        "std_1to9",
        "phash",
        "orb_conf",
    ]
    feature_labels = [
        "conf",
        "sim₀",
        "gap₀₁",
        "z-score",
        "μ₁₋₉",
        "σ₁₋₉",
        "phash",
        "orb",
    ]
    mat = np.column_stack([d[k] for k in feature_keys])
    # drop rows with any nan
    valid_rows = ~np.any(np.isnan(mat), axis=1)
    mat_clean = mat[valid_rows]
    if len(mat_clean) > 10:
        corr = np.corrcoef(mat_clean.T)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(feature_labels)))
        ax.set_yticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(feature_labels, fontsize=7)
        # annotate cells
        for i in range(len(feature_labels)):
            for j in range(len(feature_labels)):
                val = corr[i, j]
                color = "white" if abs(val) > 0.5 else TEXT_COLOR
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("feature correlation matrix", fontsize=11, color=TEXT_DIM)

    # (d) Similarity rank curves: matches vs non-matches
    ax = axes[1, 0]

    # Instead: decay_rate distribution
    dr = d["decay_rate"]
    valid = ~np.isnan(dr)
    if valid.any():
        bins_dr = np.linspace(np.nanpercentile(dr, 1), np.nanpercentile(dr, 99), 60)
        ax.hist(
            dr[valid & ~matches],
            bins=bins_dr,
            color=ACCENT,
            alpha=0.6,
            edgecolor="none",
            label="non-match",
            density=True,
        )
        if matches.any():
            ax.hist(
                dr[valid & matches],
                bins=bins_dr,
                color=ACCENT2,
                alpha=0.8,
                edgecolor="none",
                label="match",
                density=True,
            )
        ax.set_xlabel("decay rate")
        ax.set_ylabel("density")
        ax.set_title("decay rate by match status", fontsize=11, color=TEXT_DIM)
        ax.legend()

    # (e) z-score distribution
    ax = axes[1, 1]
    zs = d["zscore_0"]
    valid = ~np.isnan(zs)
    if valid.any():
        bins_z = np.linspace(np.nanpercentile(zs, 0.5), np.nanpercentile(zs, 99.5), 60)
        ax.hist(
            zs[valid & ~matches],
            bins=bins_z,
            color=ACCENT,
            alpha=0.6,
            edgecolor="none",
            label="non-match",
            density=True,
        )
        if matches.any():
            ax.hist(
                zs[valid & matches],
                bins=bins_z,
                color=ACCENT2,
                alpha=0.8,
                edgecolor="none",
                label="match",
                density=True,
            )
        ax.set_xlabel("sim₀ z-score")
        ax.set_ylabel("density")
        ax.set_title("z-score distribution by class", fontsize=11, color=TEXT_DIM)
        ax.legend()

    # precision and recall need ground truth, which this file does not have:
    # `artmatcher report --labels` measures them against labeled sessions
    fig.delaxes(axes[1, 2])

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    add_watermark(fig)
    out = outdir / "04_statistical_analysis.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")
