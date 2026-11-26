"""Plot the CASP14 T1079 case study (Figure 6)."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


TARGET = "T1079"
CASE_DECOY = "T1079TS453_1"

D_SHIFT = 0.024
SCORE_TEXT_INSET = 85
SCORE_TEXT_DROP = 24


COLORS = {
    "true": "#4F6D7A",
    "esm": "#7E6A9E",
    "deep_esm": "#C97C5D",
    "deep": "#8A9A5B",
    "ref": "#666666",
    "native": "#2D96E6",
    "model": "#D75FC8",
}


def load_global_points(path):
    with path.open(newline="") as handle:
        rows = [
            {"model": row["model"], **{key: float(row[key]) for key in ("true", "esm", "deep_esm", "deep")}}
            for row in csv.DictReader(handle)
        ]
    if len(rows) != 477 or len({row["model"] for row in rows}) != 477:
        raise ValueError("Expected 477 unique decoys for the T1079 case study")
    return rows


def parse_true_residue_lddt(path):
    scored = {}
    max_resnum = 0
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                resnum = int(parts[1])
            except ValueError:
                continue
            max_resnum = max(max_resnum, resnum)
            if len(parts) >= 5:
                try:
                    score = float(parts[4])
                except ValueError:
                    continue
                if 0.0 <= score <= 1.0:
                    scored[resnum] = score
    if not scored:
        raise ValueError(f"No residue lDDT values parsed from {path}")
    values = np.full(max_resnum, np.nan, dtype=np.float32)
    for resnum, score in scored.items():
        values[resnum - 1] = score
    return values


def load_np_vector(path, key=None):
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            key = "lddt" if "lddt" in data.files else data.files[0]
        arr = np.asarray(data[key], dtype=np.float32)
    else:
        arr = np.asarray(np.load(path), dtype=np.float32)
    arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"No finite values in {path}")
    return np.clip(arr, 0.0, 1.0)


def content_bounds(img, tolerance=8):
    if img.mode != "RGB":
        img = img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg).convert("L")
    diff = diff.point(lambda p: 255 if p > tolerance else 0)
    return diff.getbbox() or (0, 0, *img.size)


def make_structure_panel_image(path, font_path, case, align_annotations=False):
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font = ImageFont.truetype(str(font_path), 32)

    pink = (215, 95, 200)
    blue = (45, 150, 230)
    black = (25, 25, 25)

    score_x = int(w * 0.67)
    score_y = int(h * 0.70)
    legend_x = int(w * 0.66)
    if align_annotations:
        score_x -= SCORE_TEXT_INSET
        score_y += SCORE_TEXT_DROP
        legend_x = score_x
    legend_y = int(h * 0.075)
    dot_r = 8
    line_gap = 38
    for i, (color, text) in enumerate([
        (pink, f"Model {CASE_DECOY}"),
        (blue, f"Native {TARGET}"),
    ]):
        y = legend_y + i * line_gap
        draw.ellipse((legend_x, y, legend_x + dot_r * 2, y + dot_r * 2), fill=color)
        draw.text((legend_x + 30, y - 7), text, fill=black, font=font)

    for i, text in enumerate([
        f"True lDDT: {case['true']:.3f}",
        f"DeepAccNet+ESM3: {case['deep_esm']:.3f}",
        f"ESM3: {case['esm']:.3f}",
        f"DeepAccNet: {case['deep']:.3f}",
    ]):
        draw.text((score_x, score_y + i * 38), text, fill=black, font=font)

    return img


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))


def add_panel_label(fig, label, x, y):
    fig.text(
        x,
        y,
        label,
        fontsize=28,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_scatter(ax, rows, pred_key, title, color):
    true = np.asarray([r["true"] for r in rows], dtype=np.float32)
    pred = np.asarray([r[pred_key] for r in rows], dtype=np.float32)

    ax.scatter(true, pred, s=22, alpha=0.72, color=color, edgecolors="none")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=COLORS["ref"], alpha=0.75)
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.1, 0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.6, alpha=0.25)
    ax.set_title(title, fontsize=17, pad=9)
    ax.set_xlabel("True score", fontsize=14)
    ax.set_ylabel("Predicted global score", fontsize=14)
    ax.tick_params(labelsize=12)

    text = (
        f"Samples = {true.size}\n"
        f"Pearson = {pearson(true, pred):.3f}\n"
        f"Spearman = {spearman(true, pred):.3f}\n"
        f"MAE = {mae(true, pred):.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.85, boxstyle="round,pad=0.35"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-points", type=Path, required=True)
    parser.add_argument("--true-lddt", type=Path, required=True)
    parser.add_argument("--deepaccnet", type=Path, required=True)
    parser.add_argument("--esm3", type=Path, required=True)
    parser.add_argument("--esm3-fusion", type=Path, required=True)
    parser.add_argument("--structure-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-file", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = load_global_points(args.global_points)
    case = next((r for r in rows if r["model"] == CASE_DECOY), None)
    if case is None:
        raise ValueError(f"{CASE_DECOY} not present in merged global points.")

    true_res = parse_true_residue_lddt(args.true_lddt)
    esm_res = load_np_vector(args.esm3)
    deep_res = load_np_vector(args.deepaccnet, key="lddt")
    deep_esm_res = load_np_vector(args.esm3_fusion, key="lddt")

    lengths = [len(true_res), len(esm_res), len(deep_res), len(deep_esm_res)]
    if len(set(lengths)) != 1:
        raise ValueError(
            "Residue-level lengths do not match: "
            f"true={len(true_res)}, esm={len(esm_res)}, "
            f"deep_esm={len(deep_esm_res)}, deep={len(deep_res)}"
        )

    font_path = args.font_file or Path(font_manager.findfont("Liberation Sans"))
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()

    plt.rcParams.update({
        "font.family": font_name,
        "font.sans-serif": [font_name, "Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(17.0, 11.5), dpi=args.dpi)

    # Manual panel placement. Each list is [left, bottom, width, height]
    # in figure-relative coordinates. Change these values to tune panels.
    PANEL_POS = {
        "a": [0.060, 0.60, 0.31, 0.370],
        "b": [0.360, 0.60, 0.31, 0.370],
        "c": [0.660, 0.60, 0.31, 0.370],
        "d": [0.015, 0.105, 0.372, 0.390],
        "e": [0.400, 0.105, 0.545, 0.390],
    }

    # Panel letter positions in figure-relative coordinates.
    # Change these values to tune only the a/b/c/d/e letters.
    LABEL_POS = {
        "a": [0.065, 0.998],
        "b": [0.365, 0.998],
        "c": [0.665, 0.998],
        "d": [0.065, 0.53],
        "e": [0.365, 0.53],
    }

    ax_a = fig.add_axes(PANEL_POS["a"])
    ax_b = fig.add_axes(PANEL_POS["b"])
    ax_c = fig.add_axes(PANEL_POS["c"])
    ax_d = fig.add_axes(PANEL_POS["d"])

    e_left, e_bottom, e_w, e_h = PANEL_POS["e"]
    track_h = 0.090
    track_gap = 0.055
    e_top = e_bottom + e_h
    ax_e_tracks = [
        fig.add_axes([e_left, e_top - (track_h + track_gap) * i - track_h, e_w, track_h])
        for i in range(3)
    ]

    plot_scatter(ax_a, rows, "deep", "DeepAccNet", COLORS["deep"])
    plot_scatter(ax_b, rows, "esm", "ESM3", COLORS["esm"])
    plot_scatter(ax_c, rows, "deep_esm", "DeepAccNet + ESM3", COLORS["deep_esm"])
    ax_b.set_ylabel("")
    ax_c.set_ylabel("")

    for label, (x_pos, y_pos) in LABEL_POS.items():
        add_panel_label(fig, label, x_pos, y_pos)

    structure_img = make_structure_panel_image(args.structure_image, font_path, case)
    structure_crop = content_bounds(structure_img)
    structure_artist = ax_d.imshow(structure_img.crop(structure_crop))
    ax_d.axis("off")

    x = np.arange(1, len(true_res) + 1)
    tracks = [
        ("DeepAccNet", deep_res, COLORS["deep"]),
        ("ESM3", esm_res, COLORS["esm"]),
        ("DeepAccNet + ESM3", deep_esm_res, COLORS["deep_esm"]),
    ]

    for i, (ax, (label, values, color)) in enumerate(zip(ax_e_tracks, tracks)):
        ax.plot(x, true_res, color="#6F8791", linewidth=1.75, alpha=0.55, zorder=1)
        ax.plot(x, values, color=color, linewidth=1.35, zorder=2)
        ax.set_xlim(1, len(true_res))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linewidth=0.5, alpha=0.24)
        ax.tick_params(labelsize=11, length=2, pad=1)
        ax.set_yticks([0.0, 0.5, 1.0])
        bbox = ax.get_position()
        fig.text(
            bbox.x0,
            bbox.y1 + 0.010,
            label,
            va="bottom",
            ha="left",
            fontsize=15,
            color=color,
            fontweight="bold",
        )
        if i == 0:
            ax.set_title(f"Residue-level quality profile for {CASE_DECOY}", fontsize=18, pad=8)
        if i < len(ax_e_tracks) - 1:
            ax.tick_params(labelbottom=False)

    ax_e_tracks[-1].set_xlabel("Residue index", fontsize=17)
    ax_e_tracks[1].set_ylabel("Per-residue lDDT / predicted score", fontsize=17)

    # Freeze the original crop so moving d does not shift the other panels.
    fig.canvas.draw()
    crop = fig.get_tightbbox(fig.canvas.get_renderer()).padded(plt.rcParams["savefig.pad_inches"])
    left, bottom, width, height = PANEL_POS["d"]
    structure_img = make_structure_panel_image(
        args.structure_image, font_path, case, align_annotations=True
    )
    structure_artist.set_data(structure_img.crop(structure_crop))
    ax_d.set_position([left + D_SHIFT, bottom, width, height])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches=crop, dpi=args.dpi)
    plt.close(fig)

    print(f"Global common decoys: {len(rows)}")
    print(f"Residue length: {len(true_res)}")
    print(f"Case global scores: true={case['true']:.6f}, esm={case['esm']:.6f}, "
          f"deep_esm={case['deep_esm']:.6f}, deep={case['deep']:.6f}")
    print(f"Case residue means: true={np.nanmean(true_res):.6f}, esm={np.mean(esm_res):.6f}, "
          f"deep_esm={np.mean(deep_esm_res):.6f}, deep={np.mean(deep_res):.6f}")
    print(f"Saved PNG: {args.output}")


if __name__ == "__main__":
    main()
