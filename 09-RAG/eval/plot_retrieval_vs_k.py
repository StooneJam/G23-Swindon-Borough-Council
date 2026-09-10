"""
Plot mean retrieval metrics vs k for dissertation Figure 4.x.

Usage:
  pip install matplotlib
  python eval/plot_retrieval_vs_k.py
  python eval/plot_retrieval_vs_k.py eval/results/eval_20260826T142148Z_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = ROOT / "eval" / "results" / "eval_20260826T142148Z_summary.json"
OUT_DIR = ROOT / "eval" / "results" / "figures"


def load_metrics(summary_path: Path) -> tuple[list[int], dict[str, list[float]]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    by_k = data["retrieval_metrics"]["by_k"]
    ks = sorted(int(k) for k in by_k)
    series = {
        "Precision": [by_k[str(k)]["precision_at_k_mean"] for k in ks],
        "Recall": [by_k[str(k)]["recall_at_k_mean"] for k in ks],
        "NDCG": [by_k[str(k)]["ndcg_at_k_mean"] for k in ks],
    }
    return ks, series


def plot(ks: list[int], series: dict[str, list[float]], out_base: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {
        "Precision": ("o-", "#4472C4"),
        "Recall": ("s-", "#ED7D31"),
        "NDCG": ("^-", "#70AD47"),
    }
    for name, values in series.items():
        marker, color = styles[name]
        ax.plot(ks, values, marker, linestyle="-", color=color, label=name, linewidth=2, markersize=7)

    ax.set_xlabel("k (top retrieved chunks scored)")
    ax.set_ylabel("Mean score")
    ax.set_title("Mean retrieval metrics vs k (n = 3 gold questions)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUMMARY
    if not summary_path.is_file():
        raise SystemExit(f"Summary not found: {summary_path}")

    ks, series = load_metrics(summary_path)
    out_base = OUT_DIR / "retrieval_metrics_vs_k"
    plot(ks, series, out_base)


if __name__ == "__main__":
    main()
