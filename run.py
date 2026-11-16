"""Evolve one strategy on pre-split data and report a live-consistent holdout.

Usage:  py -m gp_eth.run  [--pop 100] [--gens 40] [--seed 42] [--folds]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import apply_risk_overlay, summarize
from .evolve import evolve, year_folds
from .export_pine import to_pine
from .individual import to_expr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "cache_ETHUSDT_1d_3000.csv"
OUT = HERE / "out"


def fmt(m: dict) -> str:
    return (f"total={m['total']:+9.1%}  ann={m['ann']:+8.1%}  sharpe={m['sharpe']:6.2f}  "
            f"calmar={m['calmar']:6.2f}  mdd={m['mdd']:7.2%}  flips={m['flips']:4d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--robust", action="store_true",
                    help="fitness = worst of two half-windows + small mean bonus")
    ap.add_argument("--folds", action="store_true",
                    help="fitness = mean calendar-year fold score - 0.5*std (recommended)")
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    train = df[df["open_time"] < args.split].reset_index(drop=True)
    test = df[df["open_time"] >= args.split].reset_index(drop=True)
    print(f"train {train.open_time.iloc[0].date()} -> {train.open_time.iloc[-1].date()}  ({len(train)} bars)")
    print(f"test  {test.open_time.iloc[0].date()}  -> {test.open_time.iloc[-1].date()}   ({len(test)} bars)\n")

    t0 = time.time()
    folds = year_folds(train["open_time"]) if args.folds else None
    if folds:
        print("folds: " + ", ".join(f"{train.open_time.iloc[a].date()}..{train.open_time.iloc[b-1].date()}"
                                    for a, b in folds) + "\n")
    best, log = evolve(train, pop_size=args.pop, generations=args.gens,
                       seed=args.seed, robust=args.robust, folds=folds)
    dt = time.time() - t0
    print(f"\nevolved in {dt:.1f}s")

    print("\n=== holdout ===")
    from .primitives import build_context, eval_tree
    from .individual import to_position

    # Evaluate on the full history and slice.  Every primitive is causal, so
    # train positions are identical to building on train alone; test positions
    # now carry the 200-bar z-score warmup from before the split, exactly as
    # live trading would.  Resetting the z-score at the split produced huge
    # fake positions on the first test bars.
    ctx_all = build_context(df)
    raw_all = eval_tree(best.tree, ctx_all)
    raw_all = np.nan_to_num(raw_all, nan=0.0, posinf=0.0, neginf=0.0)
    pos_all = to_position(raw_all)
    train_mask = (df["open_time"] < args.split).to_numpy()
    test_mask = ~train_mask

    results = {}
    for label, mask in (("train", train_mask), ("test ", test_mask)):
        close = ctx_all["close"][mask]
        pos = pos_all[mask]
        m_plain = summarize(close, pos)
        pos_risk = apply_risk_overlay(close, pos)
        m_risk = summarize(close, pos_risk)
        results[label.strip()] = {"plain": {k: v for k, v in m_plain.items() if k not in ("equity", "pos")},
                                  "risk": {k: v for k, v in m_risk.items() if k not in ("equity", "pos")}}
        bh = summarize(close, np.ones(len(close)))
        print(f"{label}  buy&hold: total={bh['total']:+9.1%} mdd={bh['mdd']:7.2%}")
        print(f"{label}  plain   : {fmt(m_plain)}  size={best.size}")
        print(f"{label}  +risk   : {fmt(m_risk)}")

    OUT.mkdir(exist_ok=True)
    suffix = "_folds" if args.folds else ("_robust" if args.robust else "")
    expr = best.expr
    pine = to_pine(best.tree, name=f"GP-Eth{'-Folds' if args.folds else ('-Robust' if args.robust else '')}")
    (OUT / f"best_strategy{suffix}.pine").write_text(pine, encoding="utf-8")
    (OUT / f"run_log{suffix}.json").write_text(json.dumps({
        "expr": expr,
        "tree": best.tree,
        "seed": args.seed,
        "pop": args.pop,
        "gens": args.gens,
        "robust": args.robust,
        "folds": args.folds,
        "train_test_split": args.split,
        "best_score_train": best.score,
        "results": results,
        "gen_best_scores": log.best,
        "gen_mean_scores": log.mean,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nexpr: {expr}")
    print(f"written: {OUT / f'best_strategy{suffix}.pine'}")
    print(f"written: {OUT / f'run_log{suffix}.json'}")


if __name__ == "__main__":
    main()
