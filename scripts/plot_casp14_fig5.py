"""Plot the CASP14 Stage-2 retrospective comparison (Figure 5)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd


FOCUS = {"SaProt + DeepAccNet", "DeepAccNet", "ESM3"}


def plot_combined_metrics(
    ax,
    rows: pd.DataFrame,
    title: str,
    coverage_note: str | None = None,
) -> None:
    plot_rows = rows.sort_values(
        ["mean_rank_two_correlations", "pearson", "method"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    x = np.arange(len(plot_rows))
    width = 0.34
    bar_metrics = [
        ("pearson", "Pearson", "#6F86A6"),
        ("spearman", "Spearman", "#73B7B2"),
    ]
    for offset, (column, label, color) in zip((-width / 2, width / 2), bar_metrics):
        ax.bar(x + offset, plot_rows[column], width=width, color=color,
               edgecolor="white", linewidth=0.45, label=label, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_rows["plot_name"],
        rotation=62,
        ha="right",
        rotation_mode="anchor",
        fontsize=12.5,
    )
    for tick in ax.get_xticklabels():
        if tick.get_text() in FOCUS:
            tick.set_color("#D43F2A")
            tick.set_fontweight("bold")
    ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
    set_score_scale(ax, (0.15, 0.70), [0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
    ax.set_ylabel("Pearson / Spearman (higher is better)")
    if coverage_note:
        ax.text(
            0.99,
            0.90,
            coverage_note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11.5,
            color="#303030",
        )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#202020")


def plot_roc_axis(
    ax,
    rows: pd.DataFrame,
    curves: pd.DataFrame,
    subset: str,
    title: str,
    coverage_note: str | None = None,
) -> None:
    display_names = rows.set_index("method")["plot_name"].to_dict()
    sources = rows.set_index("method")["source"].to_dict()
    plot_curves = curves[curves["subset"].eq(subset)].copy()
    summary = (
        plot_curves.groupby("method", as_index=False)
        .agg(macro_auc=("macro_auc", "first"))
    )
    summary["is_ours"] = summary["method"].map(sources).eq("ours")
    summary = summary.sort_values(
        ["is_ours", "macro_auc", "method"], ascending=[False, False, True]
    )
    focus_colors = {
        "SaProt + DeepAccNet": "#D43F2A",
        "ESM3": "#E07A2D",
        "DeepAccNet": "#315F8C",
    }
    official_colors = plt.get_cmap("Greys")(np.linspace(0.42, 0.72, (~summary["is_ours"]).sum()))
    official_index = 0
    for row in summary.itertuples(index=False):
        method_curve = plot_curves[plot_curves["method"].eq(row.method)].sort_values("fpr")
        display_name = display_names[row.method]
        is_ours = bool(row.is_ours)
        if is_ours:
            color = focus_colors[display_name]
        else:
            color = official_colors[official_index]
            official_index += 1
        ax.plot(
            method_curve["fpr"],
            method_curve["tpr"],
            color=color,
            linewidth=3.0 if is_ours else 1.0,
            alpha=1.0 if is_ours else 0.62,
            label=f"{display_name} (AUC={row.macro_auc:.3f})",
            zorder=4 if is_ours else 2,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if coverage_note:
        ax.text(
            0.99,
            0.90,
            coverage_note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11.0,
            color="#303030",
        )
    ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    left_handles, right_handles = handles[:5], handles[5:]
    left_labels, right_labels = labels[:5], labels[5:]
    left_legend = ax.legend(
        left_handles,
        left_labels,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.15, 0.035),
        bbox_transform=ax.transAxes,
        ncol=1,
        fontsize=7.6,
        handlelength=1.45,
        handletextpad=0.30,
        labelspacing=0.16,
        borderaxespad=0.0,
    )
    right_legend = ax.legend(
        right_handles,
        right_labels,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.57, 0.035),
        bbox_transform=ax.transAxes,
        ncol=1,
        fontsize=7.6,
        handlelength=1.45,
        handletextpad=0.30,
        labelspacing=0.16,
        borderaxespad=0.0,
    )
    ax.add_artist(left_legend)
    for legend in (left_legend, right_legend):
        for text in legend.get_texts():
            if any(text.get_text().startswith(name) for name in FOCUS):
                text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#202020")


def plot_topone_selection(
    ax,
    rows: pd.DataFrame,
    title: str,
    loss_limit: float,
):
    plot_rows = rows.sort_values(
        ["topone_loss", "selected_model_lddt", "method"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    x = np.arange(len(plot_rows))
    colors = ["#D43F2A" if name in FOCUS else "#6F86A6" for name in plot_rows["plot_name"]]
    ours = plot_rows[plot_rows["source"].eq("ours")]
    best_reference = float(ours["best_available_lddt"].median())

    ax.bar(
        x,
        plot_rows["selected_model_lddt"],
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        label="Selected-model lDDT",
        zorder=2,
    )
    ax.axhline(
        best_reference,
        color="#3F6B52",
        linestyle="--",
        linewidth=1.5,
        label="_nolegend_",
        zorder=3,
    )
    ax.text(
        0.02,
        best_reference + 0.008,
        f"Best available model lDDT = {best_reference:.3f}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=11.5,
        color="#3F6B52",
    )
    ax_loss = ax.twinx()
    ax_loss.plot(
        x,
        plot_rows["topone_loss"],
        color="#292929",
        marker="D",
        markersize=4.8,
        linewidth=1.3,
        label="Top-one loss",
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_rows["plot_name"],
        rotation=62,
        ha="right",
        rotation_mode="anchor",
        fontsize=12.5,
    )
    for tick in ax.get_xticklabels():
        if tick.get_text() in FOCUS:
            tick.set_color("#D43F2A")
            tick.set_fontweight("bold")
    ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
    if ax.get_subplotspec().rowspan.start == 0:
        set_score_scale(ax, (0.40, 0.68), [0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    else:
        set_score_scale(ax, (0.42, 0.65), [0.42, 0.46, 0.50, 0.54, 0.58, 0.62])
    set_score_scale(ax_loss, (0.01, loss_limit), [0.01, 0.04, 0.07, 0.10, 0.13])
    ax.set_ylabel("Selected-model lDDT to native (higher is better)")
    ax_loss.set_ylabel("Top-one loss (lower is better)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_loss.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="upper right",
        ncol=2,
        fontsize=11.0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#202020")
    ax_loss.spines["right"].set_linewidth(1.0)
    ax_loss.spines["right"].set_color("#202020")
    return ax_loss


def set_score_scale(ax, limits, ticks):
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def summarize(rows):
    result = (
        rows.rename(columns={"display_name": "plot_name"})
        .groupby(["method", "source", "plot_name"], as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            selected_model_lddt=("chosen_true_lddt", "mean"),
            best_available_lddt=("best_true_lddt", "mean"),
            topone_loss=("topone_loss", "mean"),
        )
    )
    ranks = result[["pearson", "spearman"]].rank(ascending=False, method="min")
    result["mean_rank_two_correlations"] = ranks.mean(axis=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-target", type=Path, required=True)
    parser.add_argument("--roc-curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-file", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = pd.read_csv(args.per_target)
    required = {
        "display_name", "method", "source", "target", "multidomain",
        "pearson", "spearman", "chosen_true_lddt", "best_true_lddt", "topone_loss",
    }
    if missing := required.difference(rows.columns):
        raise ValueError(f"Missing per-target columns: {sorted(missing)}")
    rows = rows[rows.target.ne("T1045s1")]
    multi = rows[rows.multidomain.astype(str).str.lower().eq("true")]
    if (
        rows.duplicated(["method", "source", "target"]).any()
        or rows.target.nunique() != 70
        or multi.target.nunique() != 22
        or rows.method.nunique() != 17
    ):
        raise ValueError("Expected the manuscript's 17-method, 70-target CASP14 comparison and its 22-target subset")

    metric_rows = summarize(rows)
    multidomain_rows = summarize(multi)
    roc_curves = pd.read_csv(args.roc_curves)
    for subset, summary in (("all", metric_rows), ("multidomain", multidomain_rows)):
        subset_roc = roc_curves.loc[roc_curves.subset.eq(subset)]
        counts = subset_roc.groupby("method")["n_targets"].first().sort_index()
        expected = summary.set_index("method")["n_targets"].sort_index()
        if not counts.equals(expected):
            raise ValueError(f"ROC curves and per-target data have different coverage for {subset}")
    if args.font_file:
        fontManager.addfont(args.font_file)
        plt.rcParams["font.family"] = FontProperties(fname=args.font_file).get_name()
    plt.rcParams.update({"font.size": 13.0, "axes.labelsize": 14.0,
                         "xtick.labelsize": 12.0, "ytick.labelsize": 12.0})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_limit = 0.15
    all_target_label = "all evaluated targets"

    fig, axes = plt.subplots(
        2, 3, figsize=(24.5, 14.0), constrained_layout=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.0],
            "width_ratios": [1.18, 0.90, 1.18],
        },
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.10, wspace=0.015, hspace=0.08)
    plot_combined_metrics(
        axes[0, 0],
        metric_rows,
        f"Pearson and Spearman\n({all_target_label})",
    )
    plot_roc_axis(
        axes[0, 1],
        metric_rows,
        roc_curves,
        "all",
        f"Macro-average residue-level ROC\n({all_target_label})",
    )
    top_loss_ax = plot_topone_selection(
        axes[0, 2],
        metric_rows,
        f"Top-one model selection\n({all_target_label})",
        loss_limit,
    )
    plot_combined_metrics(
        axes[1, 0],
        multidomain_rows,
        "Pearson and Spearman\n(22 multidomain targets)",
    )
    plot_roc_axis(
        axes[1, 1],
        multidomain_rows,
        roc_curves,
        "multidomain",
        "Macro-average residue-level ROC\n(22 multidomain targets)",
    )
    bottom_loss_ax = plot_topone_selection(
        axes[1, 2],
        multidomain_rows,
        "Top-one model selection\n(22 multidomain targets)",
        loss_limit,
    )

    for label, ax in zip(("a", "b", "c", "d", "e", "f"), axes.flat):
        two_mm_up = ScaledTranslation(0, 2 / 25.4, fig.dpi_scale_trans)
        label_x = -0.16 if label in {"b", "e"} else -0.09
        ax.text(label_x, 1.015, label, transform=ax.transAxes + two_mm_up,
                fontsize=27, fontweight="bold", va="center", ha="left")

    # Preserve the automatic layout, then narrow only the b-c and e-f gaps.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    for primary_ax, loss_ax in (
        (axes[0, 2], top_loss_ax),
        (axes[1, 2], bottom_loss_ax),
    ):
        position = primary_ax.get_position()
        shifted = [position.x0 - 0.025, position.y0, position.width, position.height]
        primary_ax.set_position(shifted)
        loss_ax.set_position(shifted)

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
