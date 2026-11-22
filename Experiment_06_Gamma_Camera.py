"""Experiment 06. Paste this complete file into one Google Colab cell.

Dependencies: NumPy, SciPy and Matplotlib (preinstalled in Colab).
Run locally with: python Experiment_06_Gamma_Camera.py
The script saves color figures (450 dpi PNG and PDF) and ROI statistics (CSV).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter


def main():
    # Synthetic anterior projection. The patient's right is on the image left.
    n, seed, sigma = 160, 6, 2.0
    row, col = np.indices((n, n))
    extent = (-0.5, n - 0.5, n - 0.5, -0.5)

    def ellipse(x, y, rx, ry):
        return ((col - x) / rx) ** 2 + ((row - y) / ry) ** 2 <= 1

    torso = ellipse(80, 80, 59, 73)
    regions = [
        ("Liver", ellipse(57, 59, 24, 16), 8, "#b85c00"),
        ("Right kidney", ellipse(52, 93, 10, 15), 17, "#007b74"),
        ("Left kidney", ellipse(108, 91, 10, 15), 15, "#2461b2"),
        ("Bladder", ellipse(80, 125, 13, 11), 30, "#a53985"),
        ("Background", ellipse(109, 53, 9, 9), 0, "#656a71"),
    ]
    source = 2.0 * torso
    for _, mask, increment, _ in regions:
        assert np.all(torso[mask])
        source[mask] += increment
    assert np.max(np.sum([mask for _, mask, _, _ in regions], axis=0)) == 1

    # Blur projected uptake, then draw independent Poisson counts per pixel.
    expected = gaussian_filter(source, sigma=sigma, mode="constant")
    assert np.isclose(expected.sum(), source.sum(), rtol=1e-5)
    counts = np.random.default_rng(seed).poisson(expected)
    bg_mean = counts[regions[-1][1]].mean()
    metrics = []
    for name, mask, _, _ in regions:
        roi = counts[mask]
        metrics.append((name, roi.size, roi.sum(), roi.mean(), expected[mask].mean(),
                        roi.sum() ** 0.5 / roi.size, roi.mean() / bg_mean))

    output = Path("experiment_06_gamma_camera_outputs")
    output.mkdir(exist_ok=True)
    with (output / "roi_statistics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ROI", "Pixels", "Total counts", "Measured mean counts/pixel",
                         "Expected mean counts/pixel", "Estimated Poisson SE", "ROI/background"])
        writer.writerows(metrics)

    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": "black",
                         "text.color": "black", "figure.facecolor": "white",
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.34), layout="constrained")
    panels = [(source, "A  Input phantom"), (expected, "B  Blurred expectation"),
              (counts, "C  Simulated acquisition")]
    vmax = 45
    for ax, (data, title) in zip(axes, panels):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax, extent=extent,
                       interpolation="nearest")
        ax.set_axis_off()
    # A shared scale makes intensities comparable. Values above the cap saturate.
    fig.colorbar(im, ax=axes, shrink=0.77, pad=0.025, extend="max",
                 label="Counts or count-equivalent per pixel")
    fig.savefig(output / "01_planar_gamma_images.png", dpi=450)
    fig.savefig(output / "01_planar_gamma_images.pdf")

    fig2, (ax, chart) = plt.subplots(1, 2, figsize=(9.3, 4.15),
                                     gridspec_kw={"width_ratios": [1, 1.35]},
                                     layout="constrained")
    ax.set_title("A  ROI localization", loc="left", fontweight="bold")
    ax.imshow(counts, cmap="magma", vmin=0, vmax=vmax, extent=extent,
              interpolation="nearest")
    ax.set_axis_off()
    for index, (_, mask, _, color) in enumerate(regions, 1):
        ax.contour(col, row, mask, levels=[0.5], colors="white", linewidths=1.8)
        ax.contour(col, row, mask, levels=[0.5], colors=[color], linewidths=0.8)
        ax.text(col[mask].mean(), row[mask].mean(), str(index), ha="center",
                va="center", fontweight="bold",
                bbox={"boxstyle": "circle,pad=0.18", "fc": "white", "ec": color})

    y = np.arange(len(regions))
    chart.set_title("B  Regional count density", loc="left", fontweight="bold")
    chart.set_axisbelow(True)
    chart.grid(axis="x", color="#dfe2e7")
    for spine in ("top", "right", "left"):
        chart.spines[spine].set_visible(False)
    for i, (_, _, _, color) in enumerate(regions):
        mean, model, se = metrics[i][3:6]
        chart.barh(i, mean, height=0.53, color=color, alpha=0.86)
        chart.errorbar(mean, i, xerr=1.96 * se, fmt="none", color="black",
                       capsize=3, linewidth=1.15, zorder=3)
        chart.scatter(model, i, marker="D", facecolor="white", edgecolor="black",
                      s=21, zorder=4)
        chart.text(max(mean + 1.96 * se, model) + 0.7, i, f"{mean:.2f}",
                   va="center", fontsize=9)
    chart.set_yticks(y, [f"{i}  {r[0]}" for i, r in enumerate(regions, 1)])
    chart.tick_params(axis="y", length=0)
    chart.set_ylim(4.65, -1.70)
    chart.set_xlim(0, 36)
    chart.set_xlabel("Mean detected counts per pixel")
    chart.legend(handles=[
        Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor="black",
               linestyle="none", markersize=4, label="Model expectation"),
        Line2D([], [], color="black", marker="|", label="Approx. 95% Poisson interval"),
    ], loc="upper right", frameon=False, fontsize=8, ncol=1, borderaxespad=0.15)
    fig2.savefig(output / "02_localization_and_roi_graph.png", dpi=450)
    fig2.savefig(output / "02_localization_and_roi_graph.pdf")

    print(f"Expected total {expected.sum():,.0f}. Detected total {counts.sum():,}.")
    print(f"{'ROI':<15} {'Pixels':>6} {'Total':>7} {'Mean':>7} {'Model':>7} {'ROI/BG':>7}")
    for name, pixels, total, mean, model, _, ratio in metrics:
        print(f"{name:<15} {pixels:>6d} {total:>7d} {mean:>7.2f} {model:>7.2f} {ratio:>7.2f}")
    print(f"Color figure and CSV output: {output.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
