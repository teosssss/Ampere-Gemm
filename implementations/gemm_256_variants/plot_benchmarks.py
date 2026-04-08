from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "plots"

KMEANS_FILE = RESULTS / "l4-tritonbench-20260408-111218.json"
LARGE_FILE = RESULTS / "l4-tritonbench-20260408-110716.json"

BACKENDS = [
    "torch_mm",
    "reg_pingpong_256",
    "reg_pingpong_256_mma",
    "reg_pingpong_256_colb",
    "reg_pingpong_256_colb_mma",
]

COLORS = {
    "torch_mm": "#4C78A8",
    "reg_pingpong_256": "#F58518",
    "reg_pingpong_256_mma": "#E45756",
    "reg_pingpong_256_colb": "#72B7B2",
    "reg_pingpong_256_colb_mma": "#54A24B",
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shape_label(shape: dict) -> str:
    return f"{shape['m']}x{shape['n']}x{shape['k']}"


def to_map(case: dict) -> dict:
    return {r["backend"]: r["tflops"] for r in case["results"] if r.get("tflops") is not None}


def plot_file(payload: dict, title: str, out_path: Path) -> None:
    cases = payload["cases"]
    labels = [shape_label(c["shape"]) for c in cases]
    width = 0.15
    x = list(range(len(labels)))

    plt.figure(figsize=(12, 4.8))
    for bi, backend in enumerate(BACKENDS):
        vals = []
        for case in cases:
            vals.append(to_map(case).get(backend, 0.0))
        x_off = [xi + (bi - (len(BACKENDS) - 1) / 2) * width for xi in x]
        plt.bar(x_off, vals, width=width, label=backend, color=COLORS.get(backend))

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("TFLOPS")
    plt.title(title)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_vs_baseline(kmeans: dict, large: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharey=True)

    for ax, payload, title in [
        (axes[0], kmeans, "K-means Shapes vs torch_mm"),
        (axes[1], large, "Larger Shapes vs torch_mm"),
    ]:
        cases = payload["cases"]
        labels = [shape_label(c["shape"]) for c in cases]
        x = list(range(len(labels)))

        for backend in BACKENDS[1:]:
            vals = []
            for case in cases:
                case_map = to_map(case)
                baseline = case_map["torch_mm"]
                vals.append(case_map.get(backend, 0.0) / baseline if baseline else 0.0)
            ax.plot(x, vals, marker="o", linewidth=2, label=backend, color=COLORS.get(backend))

        ax.axhline(1.0, color="#4C78A8", linestyle="--", linewidth=1.5, label="torch_mm baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("TFLOPS / torch_mm")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    kmeans = load(KMEANS_FILE)
    large = load(LARGE_FILE)
    plot_file(kmeans, "L4 TritonBench: K-means Shapes", OUT_DIR / "kmeans_tflops.png")
    plot_file(large, "L4 TritonBench: Larger Shapes", OUT_DIR / "large_tflops.png")
    plot_vs_baseline(kmeans, large, OUT_DIR / "baseline_relative_tflops.png")


if __name__ == "__main__":
    main()
