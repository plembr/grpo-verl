from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAG_PATTERNS = [
    "acc",
    "reward",
    "kl",
    "entropy",
    "loss",
    "grad_norm",
    "lr",
    "clipfrac",
    "advantages",
]


def _safe_filename(value: str) -> str:
    value = value.replace("@", "_at_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "__", value)
    return value.strip("._") or "scalar"


def _matches(tag: str, patterns: list[str]) -> bool:
    lower = tag.lower()
    return any(pattern.lower() in lower for pattern in patterns)


def _smooth(values: list[float], weight: float) -> list[float]:
    if not values or weight <= 0:
        return values

    smoothed: list[float] = []
    last = values[0]
    for value in values:
        last = last * weight + value * (1 - weight)
        smoothed.append(last)
    return smoothed


def _write_csv(path: Path, points) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["wall_time", "step", "value"])
        for point in points:
            writer.writerow([point.wall_time, point.step, point.value])


def _plot_scalar(path: Path, tag: str, points, smooth_weight: float) -> None:
    steps = [point.step for point in points]
    values = [point.value for point in points]
    smoothed = _smooth(values, smooth_weight)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8), dpi=160)
    plt.plot(steps, values, color="#94a3b8", linewidth=1.1, alpha=0.55, label="raw")
    if smooth_weight > 0:
        plt.plot(steps, smoothed, color="#0f172a", linewidth=2.0, label=f"smoothed {smooth_weight:g}")
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_index(path: Path, title: str, image_entries: list[tuple[str, str]], summary_rows: list[dict[str, str]]) -> None:
    rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['tag'])}</td>"
        f"<td>{html.escape(row['points'])}</td>"
        f"<td>{html.escape(row['first_step'])}</td>"
        f"<td>{html.escape(row['first_value'])}</td>"
        f"<td>{html.escape(row['last_step'])}</td>"
        f"<td>{html.escape(row['last_value'])}</td>"
        f"<td>{html.escape(row['best_step'])}</td>"
        f"<td>{html.escape(row['best_value'])}</td>"
        "</tr>"
        for row in summary_rows
    )
    cards_html = "\n".join(
        f'<section><h2>{html.escape(tag)}</h2><img src="{html.escape(src)}" alt="{html.escape(tag)}"></section>'
        for tag, src in image_entries
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #111827; background: #f8fafc; }}
    h1 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0 32px; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #e5e7eb; }}
    section {{ margin: 24px 0; padding: 16px; background: white; border: 1px solid #e5e7eb; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Exported TensorBoard scalar curves. Raw values are light; smoothed values are dark.</p>
  <table>
    <thead>
      <tr>
        <th>tag</th><th>points</th><th>first step</th><th>first value</th>
        <th>last step</th><th>last value</th><th>best step</th><th>best value</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  {cards_html}
</body>
</html>
""",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export selected TensorBoard scalars to CSV, PNG, and HTML.")
    parser.add_argument("--logdir", required=True, help="Directory containing TensorBoard event files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="TensorBoard Scalars")
    parser.add_argument("--pattern", action="append", default=[], help="Case-insensitive tag substring to include.")
    parser.add_argument("--smooth", type=float, default=0.6, help="EMA smoothing weight, 0 disables smoothing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logdir = Path(args.logdir)
    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    png_dir = output_dir / "png"

    patterns = args.pattern or DEFAULT_TAG_PATTERNS
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()

    tags = [tag for tag in accumulator.Tags().get("scalars", []) if _matches(tag, patterns)]
    if not tags:
        raise RuntimeError(f"No scalar tags matched {patterns!r} in {logdir}")

    image_entries: list[tuple[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for tag in sorted(tags):
        points = accumulator.Scalars(tag)
        if not points:
            continue

        safe = _safe_filename(tag)
        _write_csv(csv_dir / f"{safe}.csv", points)
        image_path = png_dir / f"{safe}.png"
        _plot_scalar(image_path, tag, points, args.smooth)
        image_entries.append((tag, f"png/{image_path.name}"))

        values = [point.value for point in points]
        steps = [point.step for point in points]
        best_index = max(range(len(values)), key=lambda index: values[index])
        summary_rows.append(
            {
                "tag": tag,
                "points": str(len(points)),
                "first_step": str(steps[0]),
                "first_value": f"{values[0]:.8g}",
                "last_step": str(steps[-1]),
                "last_value": f"{values[-1]:.8g}",
                "best_step": str(steps[best_index]),
                "best_value": f"{values[best_index]:.8g}",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_index(output_dir / "index.html", args.title, image_entries, summary_rows)

    print(f"exported {len(image_entries)} scalar charts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
