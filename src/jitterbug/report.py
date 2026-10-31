"""Turn sweep results into a table and a chart."""

from __future__ import annotations

import json
from pathlib import Path

from jitterbug.runner import CallResult, sweep_table


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'profile':16} {'loss%':>6} {'jit ms':>6} {'calls':>5} {'task ok':>8} "
        f"{'WER':>6} {'MOS':>5} {'p95 ms':>7} {'repeats':>7}  verdicts"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        mos = f"{r['mean_mos']:.2f}" if r["mean_mos"] is not None else "-"
        verdicts = ", ".join(f"{k} {v}" for k, v in r["verdicts"].items())
        print(
            f"{r['profile']:16} {r['loss_pct']:>6g} {r['jitter_ms']:>6g} {r['calls']:>5} "
            f"{r['task_success_pct']:>7.0f}% {r['mean_wer']:>6.2f} {mos:>5} "
            f"{r['p95_latency_ms']:>7.0f} {r['mean_repeats']:>7.2f}  {verdicts}"
        )


def save_results(results: list[CallResult], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    payload = {"summary": sweep_table(results), "calls": [r.to_dict() for r in results]}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def loss_chart(rows: list[dict], path: str | Path, title: str) -> Path:
    """Task success, WER and MOS against packet loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["loss_pct"])
    loss = [r["loss_pct"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    panels = [
        ("Task success (%)", [r["task_success_pct"] for r in rows], (0, 105)),
        ("Word error rate", [r["mean_wer"] for r in rows], (0, None)),
        ("Speech quality (DNSMOS)", [r["mean_mos"] or float("nan") for r in rows], (1, 5)),
    ]
    for ax, (label, values, ylim) in zip(axes, panels, strict=True):
        ax.plot(loss, values, marker="o", color="#2a6fdb", linewidth=2)
        ax.set_xlabel("Packet loss (%)")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
