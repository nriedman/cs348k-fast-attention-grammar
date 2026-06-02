"""Plot the Design-for-Descent ablation from run-directory logs.

Reads one or more <run_dir>/log.jsonl files (written by autotune) and produces:
  1. ablation.png  -- best feasible runtime vs iteration, one line per run. The
     core deliverable: how the reachable optimum / convergence depends on which
     grammar property is present.
  2. flops_vs_runtime.png (optional, --headline RUN_DIR) -- for one run, FLOPs
     (flat: same computation) against runtime (dropping), showing the speedup is
     scheduling, not less work.

Usage:
  python plot_ablation.py \
      full=runs/attn_full no_reversibility=runs/attn_norev \
      no_repairability=runs/attn_norepair no_jump_continuity=runs/attn_nojump \
      no_local_control=runs/attn_nolocal \
      --headline runs/attn_full --out-dir figs

Each positional arg is LABEL=RUN_DIR. Labels become the legend entries.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")            # headless / no display needed
import matplotlib.pyplot as plt


PENALTY_BASE = 1e9               # matches optimization.MEM_PENALTY_BASE


def load_log(run_dir: str):
    """Read log.jsonl -> list of per-iteration records (dicts)."""
    path = os.path.join(run_dir, "log.jsonl")
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _to_float(x):
    """JSONL stores inf as the string 'inf'."""
    if x == "inf":
        return math.inf
    return float(x)


def best_feasible_curve(recs):
    """(iterations, best_feasible_ms) -- the running best among FEASIBLE programs.
    Penalty/infeasible iterations carry the last feasible best forward (NaN until
    the first feasible program, so the line starts where feasibility begins)."""
    its, ys = [], []
    best = math.inf
    for r in recs:
        bl = _to_float(r["best_loss"])
        # best_loss is feasible only if below the penalty floor
        if bl < PENALTY_BASE:
            best = min(best, bl)
        its.append(r["iter"])
        ys.append(best if best < PENALTY_BASE else float("nan"))
    return its, ys


def current_loss_regime(recs):
    """Per-iteration current loss, split into feasible ms vs a flag for penalty/
    infeasible -- useful to annotate when a run never reaches feasibility."""
    its, ms, infeasible = [], [], []
    for r in recs:
        l = _to_float(r["loss"])
        its.append(r["iter"])
        if l < PENALTY_BASE:
            ms.append(l); infeasible.append(False)
        else:
            ms.append(float("nan")); infeasible.append(True)
    return its, ms, infeasible


def plot_ablation(runs: dict, out_path: str):
    """runs: {label: run_dir}. Best feasible ms vs iteration, one line each."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_feasible = {}
    for label, rd in runs.items():
        recs = load_log(rd)
        its, ys = best_feasible_curve(recs)
        feasible = any(not math.isnan(y) for y in ys)
        any_feasible[label] = feasible
        if feasible:
            ax.plot(its, ys, marker="o", ms=3, lw=1.8, label=label)
        else:
            # never reached a compilable kernel: draw a flat line at the top so it
            # reads as "pinned at infeasible" rather than vanishing.
            ax.plot(its, [float("nan")] * len(its), label=f"{label} (never feasible)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best feasible runtime (ms)")
    ax.set_yscale("log")
    ax.set_title("Design for Descent: reachable optimum by grammar property")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    for label, feas in any_feasible.items():
        note = "reached feasible" if feas else "NEVER feasible (pinned at penalty)"
        print(f"  {label}: {note}")


def plot_flops_vs_runtime(run_dir: str, out_path: str):
    """For one run: FLOPs (flat) and runtime (dropping) over iterations."""
    recs = load_log(run_dir)
    its, ms, _ = current_loss_regime(recs)
    flops = [r.get("flops") for r in recs]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(its, ms, color="tab:blue", marker="o", ms=3, lw=1.8, label="runtime (ms)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("runtime (ms)", color="tab:blue")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(its, [f / 1e6 if f else float("nan") for f in flops],
             color="tab:red", lw=1.8, ls="--", label="FLOPs (M)")
    ax2.set_ylabel("FLOPs (millions)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Speedup is scheduling, not less work\n(FLOPs constant; runtime drops)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_DIR pairs")
    ap.add_argument("--headline", default=None,
                    help="run_dir for the FLOPs-vs-runtime figure")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()

    runs = {}
    for item in args.runs:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=RUN_DIR, got {item!r}")
        label, rd = item.split("=", 1)
        runs[label] = rd

    os.makedirs(args.out_dir, exist_ok=True)
    plot_ablation(runs, os.path.join(args.out_dir, "ablation.png"))
    if args.headline:
        plot_flops_vs_runtime(args.headline, os.path.join(args.out_dir, "flops_vs_runtime.png"))


if __name__ == "__main__":
    main()