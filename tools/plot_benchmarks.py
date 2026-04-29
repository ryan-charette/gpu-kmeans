#!/usr/bin/env python3
"""Plot benchmark CSV output from kmeans_benchmark."""

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks.png"))
    parser.add_argument("--x", choices=["rows", "dims", "k"], default="rows")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting") from exc

    series: dict[str, list[tuple[float, float]]] = {}
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status"] != "ok":
                continue
            label = f"{row['backend']}:{row['kernel']}"
            series.setdefault(label, []).append((float(row[args.x]), float(row["total_ms"])))

    if not series:
        raise SystemExit("no successful benchmark rows found")

    plt.figure(figsize=(8, 5))
    for label, points in sorted(series.items()):
        points.sort()
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(args.x)
    plt.ylabel("total time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
