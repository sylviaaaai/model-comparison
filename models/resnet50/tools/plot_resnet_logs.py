#!/usr/bin/env python3
"""Create an HTML/SVG dashboard for ResNet 5-fold training logs."""

from __future__ import annotations

import csv
import html
import math
import os
import argparse
from collections import defaultdict
from pathlib import Path


ROOT = Path("results/resnet50_unfreeze_5fold")
OUT = ROOT / "plots"
PNG_OUT = OUT / "png"
METRICS = [
    "accuracy",
    "auc",
    "loss",
    "val_accuracy",
    "val_auc",
    "val_loss",
    "learning_rate",
]
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HTML/SVG and PNG dashboards for ResNet 5-fold training logs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Experiment result directory that contains setting/fold training_log.csv files.",
    )
    return parser.parse_args()


def configure_paths(root: Path) -> None:
    global ROOT
    global OUT
    global PNG_OUT

    ROOT = root
    OUT = ROOT / "plots"
    PNG_OUT = OUT / "png"


def setup_matplotlib():
    """Import matplotlib only when available, keeping this script stdlib-friendly."""
    mpl_cache = OUT / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_logs() -> dict[str, dict[str, list[dict[str, float]]]]:
    data: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(dict)
    for path in sorted(ROOT.glob("*/fold_*/training_log.csv")):
        experiment = path.parts[-3]
        fold = path.parts[-2]
        with path.open(newline="", encoding="utf-8") as f:
            rows = []
            for row in csv.DictReader(f):
                parsed = {}
                for key, value in row.items():
                    parsed[key] = float(value)
                rows.append(parsed)
        if rows:
            data[experiment][fold] = rows
    return dict(data)


def nice_range(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo, hi = min(finite), max(finite)
    if lo == hi:
        pad = abs(lo) * 0.05 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def fmt_num(value: float) -> str:
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def svg_line_chart(
    series: list[tuple[str, list[tuple[float, float]], str]],
    title: str,
    y_label: str,
    width: int = 860,
    height: int = 430,
) -> str:
    margin = {"left": 64, "right": 24, "top": 48, "bottom": 58}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_points = [point for _, points, _ in series for point in points]
    if not all_points:
        return ""
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = nice_range(ys)

    def sx(x: float) -> float:
        if x_min == x_max:
            return margin["left"] + plot_w / 2
        return margin["left"] + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin["left"]}" y="28" class="chart-title">{html.escape(title)}</text>',
    ]

    for i in range(6):
        t = i / 5
        y = margin["top"] + t * plot_h
        value = y_max - t * (y_max - y_min)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt_num(value)}</text>')

    x_ticks = sorted(set([x_min, x_max] + [round(x_min + i * (x_max - x_min) / 4) for i in range(1, 4)]))
    for x in x_ticks:
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{margin["top"]}" x2="{px:.1f}" y2="{margin["top"] + plot_h}" class="grid"/>')
        parts.append(f'<text x="{px:.1f}" y="{height - 30}" class="tick" text-anchor="middle">{fmt_num(x)}</text>')

    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<text x="{margin["left"] + plot_w / 2}" y="{height - 8}" class="axis-label" text-anchor="middle">epoch</text>')
    parts.append(f'<text x="18" y="{margin["top"] + plot_h / 2}" class="axis-label" transform="rotate(-90 18 {margin["top"] + plot_h / 2})" text-anchor="middle">{html.escape(y_label)}</text>')

    legend_x = margin["left"] + 8
    legend_y = height - 50
    for idx, (name, points, color) in enumerate(series):
        path = " ".join(f"{'M' if j == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}" for j, (x, y) in enumerate(points))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{color}"><title>{html.escape(name)} epoch {fmt_num(x)}: {fmt_num(y)}</title></circle>')
        lx = legend_x + (idx % 4) * 188
        ly = legend_y + (idx // 4) * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<text x="{lx + 24}" y="{ly + 4}" class="legend">{html.escape(name)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def mean_series(rows_by_fold: dict[str, list[dict[str, float]]], metric: str) -> list[tuple[float, float]]:
    by_epoch: dict[float, list[float]] = defaultdict(list)
    for rows in rows_by_fold.values():
        for row in rows:
            by_epoch[row["epoch"]].append(row[metric])
    return [(epoch, sum(values) / len(values)) for epoch, values in sorted(by_epoch.items())]


def save_png_chart(
    plt,
    series: list[tuple[str, list[tuple[float, float]], str]],
    title: str,
    y_label: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=150)
    for name, points, color in series:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker="o", markersize=3.2, linewidth=1.8, label=name, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, color="#d9deea", linewidth=0.8, alpha=0.9)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_png_charts(data: dict[str, dict[str, list[dict[str, float]]]]) -> int:
    plt = setup_matplotlib()
    count = 0

    for metric in METRICS:
        aggregate = []
        for idx, (experiment, folds) in enumerate(sorted(data.items())):
            aggregate.append((experiment, mean_series(folds, metric), COLORS[idx % len(COLORS)]))
        save_png_chart(
            plt,
            aggregate,
            f"5-fold mean {metric}",
            metric,
            PNG_OUT / "cross_strategy_mean" / f"{metric}.png",
        )
        count += 1

    for experiment, folds in sorted(data.items()):
        for metric in METRICS:
            fold_series = []
            for idx, (fold, rows) in enumerate(sorted(folds.items())):
                points = [(row["epoch"], row[metric]) for row in rows]
                fold_series.append((fold, points, COLORS[idx % len(COLORS)]))
            save_png_chart(
                plt,
                fold_series,
                f"{experiment} {metric}",
                metric,
                PNG_OUT / "folds" / experiment / f"{metric}.png",
            )
            count += 1

    return count


def summarize(data: dict[str, dict[str, list[dict[str, float]]]]) -> list[dict[str, str]]:
    summary = []
    for experiment, folds in sorted(data.items()):
        best_points = []
        for fold, rows in sorted(folds.items()):
            best = max(rows, key=lambda r: r["val_auc"])
            best_points.append(best["val_auc"])
            summary.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "best_epoch": str(int(best["epoch"])),
                    "best_val_auc": fmt_num(best["val_auc"]),
                    "best_val_accuracy": fmt_num(best["val_accuracy"]),
                    "best_val_loss": fmt_num(best["val_loss"]),
                    "final_val_auc": fmt_num(rows[-1]["val_auc"]),
                    "final_val_loss": fmt_num(rows[-1]["val_loss"]),
                }
            )
        if best_points:
            summary.append(
                {
                    "experiment": experiment,
                    "fold": "MEAN",
                    "best_epoch": "",
                    "best_val_auc": fmt_num(sum(best_points) / len(best_points)),
                    "best_val_accuracy": "",
                    "best_val_loss": "",
                    "final_val_auc": "",
                    "final_val_loss": "",
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "experiment",
        "fold",
        "best_epoch",
        "best_val_auc",
        "best_val_accuracy",
        "best_val_loss",
        "final_val_auc",
        "final_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_html(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys()) if rows else []
    cells = ["<table>", "<thead><tr>"]
    for header in headers:
        cells.append(f"<th>{html.escape(header)}</th>")
    cells.append("</tr></thead><tbody>")
    for row in rows:
        cls = ' class="mean-row"' if row.get("fold") == "MEAN" else ""
        cells.append(f"<tr{cls}>")
        for header in headers:
            cells.append(f"<td>{html.escape(row.get(header, ''))}</td>")
        cells.append("</tr>")
    cells.append("</tbody></table>")
    return "\n".join(cells)


def main() -> None:
    args = parse_args()
    configure_paths(args.root)
    data = read_logs()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summarize(data)
    write_csv(OUT / "summary.csv", summary)
    png_count = 0
    try:
        png_count = write_png_charts(data)
    except ImportError:
        print("matplotlib is not available; skipped PNG export")

    sections = []
    for metric in METRICS:
        aggregate = []
        for idx, (experiment, folds) in enumerate(sorted(data.items())):
            aggregate.append((experiment, mean_series(folds, metric), COLORS[idx % len(COLORS)]))
        sections.append(
            f'<section><h2>Cross-strategy mean: {html.escape(metric)}</h2>'
            + svg_line_chart(aggregate, f"5-fold mean {metric}", metric)
            + "</section>"
        )

    for experiment, folds in sorted(data.items()):
        charts = []
        for metric in METRICS:
            fold_series = []
            for idx, (fold, rows) in enumerate(sorted(folds.items())):
                points = [(row["epoch"], row[metric]) for row in rows]
                fold_series.append((fold, points, COLORS[idx % len(COLORS)]))
            charts.append(
                f'<div class="chart-card"><h3>{html.escape(metric)}</h3>'
                + svg_line_chart(fold_series, f"{experiment} {metric}", metric)
                + "</div>"
            )
        sections.append(f'<section><h2>{html.escape(experiment)} folds</h2><div class="chart-grid">{"".join(charts)}</div></section>')

    best_mean = [r for r in summary if r["fold"] == "MEAN"]
    best_mean.sort(key=lambda r: float(r["best_val_auc"]), reverse=True)
    winner = best_mean[0] if best_mean else None
    headline = ""
    if winner:
        headline = (
            f"<p><strong>Best mean best-val-AUC:</strong> "
            f"{html.escape(winner['experiment'])} = {winner['best_val_auc']}</p>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ResNet50 5-fold training curves</title>
<style>
  :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  body {{ margin: 0; color: #172033; background: #f7f8fb; }}
  header {{ padding: 28px 36px 18px; background: #ffffff; border-bottom: 1px solid #d9deea; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
  h2 {{ margin: 28px 0 14px; font-size: 21px; letter-spacing: 0; }}
  h3 {{ margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }}
  p {{ margin: 6px 0; }}
  main {{ padding: 0 28px 36px; }}
  section {{ margin: 22px 0; }}
  svg {{ width: 100%; height: auto; border: 1px solid #d9deea; background: white; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
  .chart-card {{ min-width: 0; }}
  .chart-title {{ font-size: 18px; font-weight: 700; fill: #172033; }}
  .axis {{ stroke: #4b5563; stroke-width: 1.2; }}
  .grid {{ stroke: #e5e8f0; stroke-width: 1; }}
  .tick, .axis-label, .legend {{ fill: #4b5563; font-size: 12px; }}
  .axis-label {{ font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d9deea; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; text-align: left; font-size: 13px; }}
  th {{ background: #eef2f7; font-weight: 700; }}
  .mean-row td {{ font-weight: 700; background: #fbfcff; }}
  @media (max-width: 640px) {{
    header {{ padding: 22px 18px 14px; }}
    main {{ padding: 0 14px 28px; }}
    .chart-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>ResNet50 5-fold training curves</h1>
  <p>Source: {html.escape(str(ROOT))}</p>
  {headline}
</header>
<main>
  <section>
    <h2>Best validation AUC summary</h2>
    {table_html(summary)}
  </section>
  {''.join(sections)}
</main>
</body>
</html>
"""
    (OUT / "index.html").write_text(document, encoding="utf-8")
    print(f"Wrote {OUT / 'index.html'}")
    print(f"Wrote {OUT / 'summary.csv'}")
    if png_count:
        print(f"Wrote {png_count} PNG charts under {PNG_OUT}")


if __name__ == "__main__":
    main()
