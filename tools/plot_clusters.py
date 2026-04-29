#!/usr/bin/env python3
"""Plot 2D input data with cluster assignments and optional centroids."""

import argparse
import csv
from pathlib import Path


def read_matrix(path: Path, drop_first_column: bool) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        delimiter = "," if "," in sample else None
        reader = csv.reader(handle, delimiter=delimiter or " ")
        for raw in reader:
            parts = [part for part in raw if part != ""]
            if not parts or parts[0].startswith("#"):
                continue
            try:
                row = [float(part) for part in parts]
            except ValueError:
                continue
            if len(row) == 1 and not rows:
                continue
            if drop_first_column:
                row = row[1:]
            rows.append(row)
    return rows


def read_assignments(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [int(row["label"]) for row in reader]


def read_centroids(path: Path | None) -> list[list[float]]:
    if path is None:
        return []
    centroids: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            centroids.append([float(value) for value in row[1:]])
    return centroids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--centroids", type=Path)
    parser.add_argument("--output", type=Path, default=Path("clusters.png"))
    parser.add_argument("--drop-first-column", action="store_true")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting") from exc

    data = read_matrix(args.data, args.drop_first_column)
    labels = read_assignments(args.assignments)
    centroids = read_centroids(args.centroids)
    if not data or len(data[0]) < 2:
        raise SystemExit("cluster plotting requires at least two dimensions")
    if len(labels) != len(data):
        raise SystemExit("assignment count does not match data row count")

    plt.figure(figsize=(8, 6))
    plt.scatter([row[0] for row in data], [row[1] for row in data], c=labels, s=16, cmap="tab20")
    if centroids:
        plt.scatter([row[0] for row in centroids], [row[1] for row in centroids],
                    marker="x", s=120, c="black", linewidths=2)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
