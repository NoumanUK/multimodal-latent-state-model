from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("experiments/results/static_vs_temporal_results.csv")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication-friendly global settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {RESULTS_CSV}. "
            "Run scripts.evaluate_static_vs_temporal first."
        )

    df = pd.read_csv(RESULTS_CSV)

    required = {
        "family",
        "condition",
        "noise_level",
        "missing_rate",
        "available_modalities",
        "static_mse",
        "temporal_mse",
        "temporal_improvement_percent",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Results CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    return df


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        print(f"Saved: {path}")


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.03)


def annotate_line_points(ax, x, y, dy, fmt="{:.3f}") -> None:
    for xi, yi in zip(x, y):
        ax.annotate(
            fmt.format(yi),
            xy=(xi, yi),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=8,
        )


# Figure 02: Noise robustness

def figure_02_noise_robustness(df: pd.DataFrame) -> None:
    noise = df[df["family"] == "noise"].sort_values("noise_level").copy()

    x = noise["noise_level"].to_numpy() * 100.0
    static = noise["static_mse"].to_numpy()
    temporal = noise["temporal_mse"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)

    ax.plot(x, static, marker="o", linewidth=2.0, markersize=6,
            label="Robust Static MVAE")
    ax.plot(x, temporal, marker="s", linewidth=2.0, markersize=6,
            label="Robust Temporal MVAE")

    annotate_line_points(ax, x, static, dy=9)
    annotate_line_points(ax, x, temporal, dy=-16)

    ax.set_xlabel("Observation noise level (%)")
    ax.set_ylabel("Current-state MSE")
    ax.set_xticks(x)
    ax.set_ylim(0.23, max(static.max(), temporal.max()) + 0.035)
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False)

    save_figure(fig, "figure_02_noise_robustness")
    plt.close(fig)


# Figure 03: Missingness robustness

def figure_03_missingness_robustness(df: pd.DataFrame) -> None:
    missing = df[df["family"] == "missingness"].sort_values("missing_rate").copy()

    x = missing["missing_rate"].to_numpy() * 100.0
    static = missing["static_mse"].to_numpy()
    temporal = missing["temporal_mse"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)

    ax.plot(x, static, marker="o", linewidth=2.0, markersize=6,
            label="Robust Static MVAE")
    ax.plot(x, temporal, marker="s", linewidth=2.0, markersize=6,
            label="Robust Temporal MVAE")

    annotate_line_points(ax, x, static, dy=9)
    annotate_line_points(ax, x, temporal, dy=-16)

    ax.set_xlabel("Elementwise missingness (%)")
    ax.set_ylabel("Current-state MSE")
    ax.set_xticks(x)
    ax.set_ylim(0.20, max(static.max(), temporal.max()) + 0.07)
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False)

    save_figure(fig, "figure_03_missingness_robustness")
    plt.close(fig)


# Figure 04: Modality dropout

def figure_04_modality_dropout(df: pd.DataFrame) -> None:
    order = ["ABC", "AB", "AC", "BC", "A", "B", "C"]

    modality = df[df["family"] == "modalities"].copy()
    modality["available_modalities"] = pd.Categorical(
        modality["available_modalities"], categories=order, ordered=True
    )
    modality = modality.sort_values("available_modalities")

    labels = modality["available_modalities"].astype(str).to_list()
    static = modality["static_mse"].to_numpy()
    temporal = modality["temporal_mse"].to_numpy()

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.3, 4.6), constrained_layout=True)

    bars_static = ax.bar(x - width / 2, static, width=width,
                         label="Robust Static MVAE")
    bars_temporal = ax.bar(x + width / 2, temporal, width=width,
                           label="Robust Temporal MVAE")

    ax.bar_label(bars_static, labels=[f"{v:.3f}" for v in static],
                 padding=3, fontsize=7.5, rotation=90)
    ax.bar_label(bars_temporal, labels=[f"{v:.3f}" for v in temporal],
                 padding=3, fontsize=7.5, rotation=90)

    ax.set_xlabel("Available modalities")
    ax.set_ylabel("Current-state MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(static.max(), temporal.max()) * 1.22)
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    save_figure(fig, "figure_04_modality_dropout")
    plt.close(fig)


# Figure 05: Compact horizontal improvement plot

def figure_05_temporal_improvement(df: pd.DataFrame) -> None:
    selected = pd.concat(
        [
            df[(df["family"] == "noise") & (df["noise_level"] > 0)],
            df[(df["family"] == "missingness") & (df["missing_rate"] > 0)],
            df[(df["family"] == "modalities") &
               (df["available_modalities"] != "ABC")],
        ],
        ignore_index=True,
    )

    labels = []
    for _, row in selected.iterrows():
        if row["family"] == "noise":
            labels.append(f"Noise {int(row['noise_level'] * 100)}%")
        elif row["family"] == "missingness":
            labels.append(f"Missing {int(row['missing_rate'] * 100)}%")
        else:
            labels.append(f"Modalities {row['available_modalities']}")

    values = selected["temporal_improvement_percent"].to_numpy()
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
    bars = ax.barh(y, values, height=0.62)

    ax.axvline(0, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative MSE reduction of Temporal vs Static (%)")
    ax.set_xlim(min(-4, values.min() - 3), values.max() + 9)
    clean_axis(ax, grid_axis="x")

    for bar, value in zip(bars, values):
        if value >= 0:
            x_text, ha = value + 0.9, "left"
        else:
            x_text, ha = value - 0.9, "right"

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            va="center",
            ha=ha,
            fontsize=8.5,
        )

    save_figure(fig, "figure_05_temporal_improvement")
    plt.close(fig)


# Figure C: Grouped improvement chart without overlap

def figure_c_relative_improvement(df: pd.DataFrame) -> None:
    noise = df[df["family"] == "noise"].sort_values("noise_level").copy()
    missing = df[df["family"] == "missingness"].sort_values("missing_rate").copy()

    order = ["ABC", "AB", "AC", "BC", "A", "B", "C"]
    modalities = df[df["family"] == "modalities"].copy()
    modalities["available_modalities"] = pd.Categorical(
        modalities["available_modalities"], categories=order, ordered=True
    )
    modalities = modalities.sort_values("available_modalities")

    labels = (
        [f"Noise {int(v * 100)}%" for v in noise["noise_level"]]
        + [f"Missing {int(v * 100)}%" for v in missing["missing_rate"]]
        + modalities["available_modalities"].astype(str).to_list()
    )

    values = np.array(
        noise["temporal_improvement_percent"].tolist()
        + missing["temporal_improvement_percent"].tolist()
        + modalities["temporal_improvement_percent"].tolist()
    )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.6, 5.15), constrained_layout=True)

    n_noise = len(noise)
    n_missing = len(missing)
    n_modalities = len(modalities)

    ax.axvspan(-0.5, n_noise - 0.5, alpha=0.04)
    ax.axvspan(n_noise - 0.5, n_noise + n_missing - 0.5, alpha=0.07)
    ax.axvspan(n_noise + n_missing - 0.5, len(labels) - 0.5, alpha=0.04)

    bars = ax.bar(x, values, width=0.68)

    ax.axhline(0, linewidth=0.9)
    ax.set_ylabel("Relative MSE reduction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylim(values.min() - 5, values.max() + 12)
    clean_axis(ax)

    ymax = ax.get_ylim()[1]
    group_y = ymax - 2.0

    ax.text((n_noise - 1) / 2, group_y, "Observation noise",
            ha="center", va="top", fontweight="bold", fontsize=9)
    ax.text(n_noise + (n_missing - 1) / 2, group_y,
            "Elementwise missingness",
            ha="center", va="top", fontweight="bold", fontsize=9)
    ax.text(n_noise + n_missing + (n_modalities - 1) / 2,
            group_y, "Available modalities",
            ha="center", va="top", fontweight="bold", fontsize=9)

    for bar, value in zip(bars, values):
        if value >= 0:
            y_text, va = value + 0.8, "bottom"
        else:
            y_text, va = value - 0.8, "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{value:+.2f}%",
            ha="center",
            va=va,
            fontsize=7.7,
        )

    # Intentionally no long title or formula inside the figure.
    # Put the formula in the manuscript caption/text instead.
    save_figure(fig, "figure_c_relative_improvement")
    plt.close(fig)


# Figure F: Clean summary table preview

def figure_f_results_table(df: pd.DataFrame) -> None:
    noise = df[df["family"] == "noise"].sort_values("noise_level").copy()
    missing = df[df["family"] == "missingness"].sort_values("missing_rate").copy()

    rows = []

    for _, row in noise.iterrows():
        rows.append([
            "Noise",
            f"{int(row['noise_level'] * 100)}%",
            f"{row['static_mse']:.4f}",
            f"{row['temporal_mse']:.4f}",
            f"{row['temporal_improvement_percent']:+.2f}%",
        ])

    for _, row in missing.iterrows():
        rows.append([
            "Missingness",
            f"{int(row['missing_rate'] * 100)}%",
            f"{row['static_mse']:.4f}",
            f"{row['temporal_mse']:.4f}",
            f"{row['temporal_improvement_percent']:+.2f}%",
        ])

    columns = [
        "Scenario",
        "Level",
        "Static MSE",
        "Temporal MSE",
        "Relative change",
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.3), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.19, 0.14, 0.19, 0.19, 0.21],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    for col in range(len(columns)):
        cell = table[(0, col)]
        cell.set_text_props(weight="bold")
        cell.set_linewidth(0.8)

    for row_idx in range(1, len(rows) + 1):
        for col_idx in range(len(columns)):
            table[(row_idx, col_idx)].set_linewidth(0.45)
        table[(row_idx, 0)].set_text_props(weight="bold")

    save_figure(fig, "figure_f_noise_missingness_table")
    plt.close(fig)


def main() -> None:
    df = load_results()

    print("Loaded:", RESULTS_CSV)
    print("Rows:", len(df))
    print()

    figure_02_noise_robustness(df)
    figure_03_missingness_robustness(df)
    figure_04_modality_dropout(df)
    figure_05_temporal_improvement(df)
    figure_c_relative_improvement(df)
    figure_f_results_table(df)

    print()
    print("All six publication-quality outputs generated.")


if __name__ == "__main__":
    main()