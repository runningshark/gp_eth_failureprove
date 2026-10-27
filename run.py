"""Run GP evolution on ETHUSDT daily data with walk-forward evaluation.

Usage:  py -m gp_eth.run  [--pop 100] [--gens 40] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .backtest import apply_risk_overlay, summarize
from .evolve import evolve
from .export_pine import to_pine
from .individual import to_expr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "cache_ETHUSDT_1d_3000.csv"
OUT = ROOT / "gp_eth" / "out"


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
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    train = df[df["open_time"] < args.split].reset_index(drop=True)
    test = df[df["open_time"] >= args.split].reset_index(drop=True)
    print(f"train {train.open_time.iloc[0].date()} -> {train.open_time.iloc[-1].date()}  ({len(train)} bars)")
    print(f"test  {test.open_time.iloc[0].date()}  -> {test.open_time.iloc[-1].date()}   ({len(test)} bars)\n")

    t0 = time.time()
    best, log = evolve(train, pop_size=args.pop, generations=args.gens,
                       seed=args.seed, robust=args.robust)
    dt = time.time() - t0
    print(f"\nevolved in {dt:.1f}s")

    print("\n=== walk-forward ===")
    from .primitives import build_context, eval_tree
    from .individual import to_position
    import numpy as np

    results = {}
    for label, part in (("train", train), ("test ", test)):
        ctx = build_context(part)
        raw = eval_tree(best.tree, ctx)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        pos = to_position(raw)
        m_plain = summarize(ctx["close"], pos)
        pos_risk = apply_risk_overlay(ctx["close"], pos)
        m_risk = summarize(ctx["close"], pos_risk)
        results[label.strip()] = {"plain": {k: v for k, v in m_plain.items() if k not in ("equity", "pos")},
                                  "risk": {k: v for k, v in m_risk.items() if k not in ("equity", "pos")}}
        bh = summarize(ctx["close"], np.ones(len(ctx["close"])))
        print(f"{label}  buy&hold: total={bh['total']:+9.1%} mdd={bh['mdd']:7.2%}")
        print(f"{label}  plain   : {fmt(m_plain)}  size={best.size}")
        print(f"{label}  +risk   : {fmt(m_risk)}")

    OUT.mkdir(exist_ok=True)
    suffix = "_robust" if args.robust else ""
    expr = best.expr
    pine = to_pine(best.tree, name=f"GP-Eth{'-Robust' if args.robust else ''}")
    (OUT / f"best_strategy{suffix}.pine").write_text(pine, encoding="utf-8")
    (OUT / f"run_log{suffix}.json").write_text(json.dumps({
        "expr": expr,
        "tree": best.tree,
        "seed": args.seed,
        "pop": args.pop,
        "gens": args.gens,
        "robust": args.robust,
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
