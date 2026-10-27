"""Null calibration: run the same GP search on block-bootstrapped ETH data.

The point is to answer "how good does this search look when there is no
predictable structure?"  We preserve intra-block OHLC shape and volatility
clustering, but destroy long-range trend/predictability by resampling blocks.

Usage (from the Desktop):
    py -m gp_eth.null_calibration --reps 10 --pop 100 --gens 40 --block 20

Writes out/null_calibration.json.
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
from .experiments import evaluate_holdout, load_data, metric_row, positions_for
from .individual import tree_from_json

OUT = Path(__file__).resolve().parent / "out"


def block_bootstrap(df: pd.DataFrame, block: int, rng: np.random.Generator) -> pd.DataFrame:
    """Resample contiguous row blocks, rescaling each block to the prior close.

    This keeps the OHLC geometry inside a block and makes the splice point
    continuous, while removing any genuine multi-year trend.
    """
    n = len(df)
    starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
    cols = ["open", "high", "low", "close"]
    parts = []
    prev_close = None
    for s in starts:
        b = df.iloc[s:s + block].copy()
        if prev_close is not None and len(b):
            f = prev_close / float(b["open"].iloc[0])
            b[cols] = b[cols] * f
        parts.append(b)
        prev_close = float(b["close"].iloc[-1]) if len(b) else prev_close
    out = pd.concat(parts, ignore_index=True).iloc[:n].copy()
    out["open_time"] = df["open_time"].to_numpy()
    return out


def run_one(df: pd.DataFrame, split: str, pop: int, gens: int, seed: int,
            mode: str) -> dict:
    train = df[df["open_time"] < split].reset_index(drop=True)
    test_mask = (df["open_time"] >= split).to_numpy()
    from .primitives import build_context
    ctx = build_context(df)
    folds = year_folds(train["open_time"]) if mode == "folds" else None
    best, _ = evolve(train, pop_size=pop, generations=gens, seed=seed,
                     verbose=False, folds=folds, robust=(mode == "robust"))
    ev = evaluate_holdout(best.tree, ctx, test_mask)
    train_mask = ~test_mask
    pos = ev["pos"]
    tm = metric_row(ctx["close"][train_mask], pos[train_mask])
    return {"score": best.score, "size": best.size,
            "train": {"sharpe": tm["sharpe"], "total": tm["total"]},
            "test": ev["plain"], "test_risk": ev["risk"],
            "expr": best.expr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--mode", default="full", choices=("full", "folds", "robust"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_data(Path(args.data)) if args.data else load_data()
    print(f"real data {len(df)} bars; null reps={args.reps} block={args.block} mode={args.mode}", flush=True)

    real = run_one(df, args.split, args.pop, args.gens, args.seed, args.mode)
    print(f"REAL   score={real['score']:7.2f} train_sh={real['train']['sharpe']:5.2f} "
          f"test={real['test']['total']:+8.1%} test_sh={real['test']['sharpe']:5.2f}", flush=True)

    nulls = []
    for rep in range(args.reps):
        rng = np.random.default_rng(10_000 + rep)
        synth = block_bootstrap(df, args.block, rng)
        t0 = time.time()
        r = run_one(synth, args.split, args.pop, args.gens, args.seed + rep, args.mode)
        r["rep"] = rep
        r["seconds"] = round(time.time() - t0, 1)
        nulls.append(r)
        print(f"NULL {rep:02d} score={r['score']:7.2f} train_sh={r['train']['sharpe']:5.2f} "
              f"test={r['test']['total']:+8.1%} test_sh={r['test']['sharpe']:5.2f} "
              f"({r['seconds']}s)", flush=True)

    def pct(values, x):
        values = np.asarray(values, float)
        return float(np.mean(values >= x))

    out = {
        "config": vars(args),
        "real": real,
        "null": nulls,
        "p_value": {
            "train_sharpe": pct([r["train"]["sharpe"] for r in nulls], real["train"]["sharpe"]),
            "score": pct([r["score"] for r in nulls], real["score"]),
            "test_sharpe": pct([r["test"]["sharpe"] for r in nulls], real["test"]["sharpe"]),
        },
        "null_train_sharpe": [r["train"]["sharpe"] for r in nulls],
        "null_score": [r["score"] for r in nulls],
        "null_test_sharpe": [r["test"]["sharpe"] for r in nulls],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "null_calibration.json").write_text(json.dumps(out, indent=1, default=float),
                                               encoding="utf-8")
    print("\np(null >= real): " + ", ".join(f"{k}={v:.2f}" for k, v in out["p_value"].items()))
    print(f"written: {OUT / 'null_calibration.json'}")


if __name__ == "__main__":
    main()
