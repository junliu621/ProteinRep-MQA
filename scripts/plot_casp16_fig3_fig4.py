"""Plot CASP16 direct and augmented MQA performance from saved evaluations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd


COLORS = {
    "DeepAccNet": "#7C8491",
    "ESM3": "#436289",
    "SaProt": "#8ACCC9",
    "ProstT5": "#B0A0BB",
    "ProteinMPNN": "#D9B1B7",
    "ESM-IF1": "#EEC994",
    "ESM2": "#B9D2C8",
}
DIRECT = ("ESM3", "SaProt", "ProstT5", "ProteinMPNN", "ESM-IF1")
AUGMENTED = ("ESM3", "SaProt", "ProstT5", "ESM2", "ProteinMPNN", "ESM-IF1")
HIDDEN = {
    (3, "lDDT", "lDDT of top-1 model"): {"SaProt", "ESM-IF1"},
    (3, "TM-score", "TM-score of top-1 model"): {"ProstT5", "ESM-IF1"},
    (4, "lDDT", "Pearson"): {"+ProteinMPNN"},
    (4, "TM-score", "Pearson"): {"+ProstT5"},
    (4, "TM-score", "Spearman"): {"+ProstT5"},
    (4, "TM-score", "TM-score of top-1 model"): {"+ProstT5"},
}
LABEL_Y_SHIFT = {
    ("lDDT", "Spearman", "+ESM2"): 0.006,
    ("TM-score", "Pearson", "ESM3"): 0.012,
}


def read_metric_tables(summary_path, top1_path, expected_decoys):
    summary = pd.read_csv(summary_path)
    if "scope" in summary:
        summary = summary[summary.scope.eq("common_12_methods_52targets")]
    summary = summary.set_index("method", verify_integrity=True)
    selected = pd.read_csv(top1_path)
    methods = {"DeepAccNet", *DIRECT, *(f"{m}+DeepAccNet" for m in AUGMENTED)}
    by_method = selected.groupby("method")
    if (
        set(summary.index) != methods
        or set(by_method.groups) != methods
        or selected.duplicated(["target", "method"]).any()
        or selected.target.nunique() != 52
        or not by_method.target.nunique().eq(52).all()
        or not by_method.n_decoys.sum().eq(expected_decoys).all()
        or not selected.selected_true_quality.between(0, 1).all()
        or not np.isfinite(summary[["pearson", "spearman"]].to_numpy()).all()
    ):
        raise ValueError("Expected complete summary and Top-1 records for the 52-target CASP16 cohort")
    return summary, by_method.selected_true_quality.mean()


def rounded_bar(ax, x, height, color, y_range):
    width = 0.62
    left, right = x - width / 2, x + width / 2
    rx, ry = width * 0.16, min(y_range * 0.01, height * 0.20)
    points = [
        (left, 0), (left, height - ry), (left, height), (left + rx, height),
        (right - rx, height), (right, height), (right, height - ry),
        (right, 0), (left, 0),
    ]
    codes = [
        PlotPath.MOVETO, PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
        PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3, PlotPath.LINETO,
        PlotPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(PlotPath(points, codes), facecolor=color,
                           edgecolor="none", alpha=0.86, zorder=3))


def draw_panel(ax, entries, figure_number, task, metric, shared_limits=None):
    baseline = entries[0][1]
    entries = [entries[0], *sorted(entries[1:], key=lambda entry: -entry[1])]
    values = [value for _, value in entries]
    ymin = max(0, min(values) - 0.025)
    ymax = min(1, max(values) + 0.035)
    value_font, delta_font = (13, 12) if figure_number == 3 else (12, 11)
    hidden = HIDDEN.get((figure_number, task, metric), ())

    for x, (model, value) in enumerate(entries):
        rounded_bar(ax, x, value, COLORS[model.lstrip("+")], ymax - ymin)
        above_baseline = value >= baseline
        ax.text(x, value + (0.001 if above_baseline else -0.002), f"{value:.3f}",
                ha="center", va="bottom" if above_baseline else "top",
                fontsize=value_font, color="#222222", zorder=6)
        delta = value - baseline
        if x == 0 or model in hidden or abs(delta) < 1e-6:
            continue
        small = abs(delta) <= 0.003
        arrow_x = x - 0.62 * 0.23
        ax.add_patch(FancyArrowPatch(
            (arrow_x, baseline), (arrow_x, value), arrowstyle="<->",
            mutation_scale=5 if small else 8, linewidth=0.75 if small else 1.15,
            color="#2F3640", alpha=0.96, shrinkA=0, shrinkB=0, zorder=4,
        ))
        note_y = (baseline + value) / 2 + LABEL_Y_SHIFT.get((task, metric, model), 0.0)
        ax.text(arrow_x + (0.018 if small else 0.028), note_y, f"{delta:+.3f}",
                fontsize=max(8, delta_font - 3) if small else delta_font,
                color="#1E2329", va="center", ha="left", zorder=7)

    ax.axhline(baseline, color="#4B4F54", linestyle="--", linewidth=0.95,
               alpha=0.68, zorder=2)
    ax.set_xticks(range(len(entries)), [model for model, _ in entries],
                  rotation=30, ha="right", fontsize=14)
    ax.set_ylabel(metric, fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-0.65, len(entries) - 0.35)
    ax.set_ylim(shared_limits or (ymin, ymax))
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.16, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#444444")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lddt-summary", type=Path, required=True)
    parser.add_argument("--tm-summary", type=Path, required=True)
    parser.add_argument("--lddt-top1", type=Path, required=True)
    parser.add_argument("--tm-top1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, action="append", default=[],
                        help="Register a font file. Repeat for other styles in the same family.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    for font in args.font_file:
        fontManager.addfont(font)
    if args.font_file:
        plt.rcParams["font.family"] = FontProperties(fname=args.font_file[0]).get_name()
    plt.rcParams["font.size"] = 12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "lDDT": read_metric_tables(args.lddt_summary, args.lddt_top1, 17_742),
        "TM-score": read_metric_tables(args.tm_summary, args.tm_top1, 17_539),
    }
    for number, names in ((3, DIRECT), (4, AUGMENTED)):
        methods = [("DeepAccNet", "DeepAccNet")]
        methods.extend(
            (n, n) if number == 3 else (f"+{n}", f"{n}+DeepAccNet") for n in names
        )
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for row, (task, (summary, top1)) in enumerate(inputs.items()):
            data = (("Pearson", summary.pearson), ("Spearman", summary.spearman),
                    (f"{task} of top-1 model", top1))
            for col, (ax, (metric, scores)) in enumerate(zip(axes[row], data)):
                entries = [(name, scores.loc[key]) for name, key in methods]
                limits = (top1.min() - 0.025, top1.max() + 0.035) if col == 2 else None
                draw_panel(ax, entries, number, task, metric, limits)
            title = ("Direct quality prediction from protein representations" if number == 3
                     else "DeepAccNet augmentation with protein representations")
            axes[row, 1].set_title(f"{title} ({task})", fontsize=23,
                                   fontweight="bold", pad=18, color="#111111")
            fig.text(0.048, 0.933 if row == 0 else 0.454, "ab"[row], fontsize=23,
                     fontweight="bold", ha="left", va="top", color="#111111")
        fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.0)
        fig.savefig(args.output_dir / f"Figure{number}_CASP16_Top1.png",
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
