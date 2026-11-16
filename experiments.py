"""Multi-seed / multi-objective comparison for gp_eth.

The single stored run is one draw from a noisy search.  This script runs the
same GP with different fitness definitions and seeds, then evaluates every
result on the same holdout using the *live-consistent* warmup (the 200-bar
z-score is computed on the full history and sliced, not reset at the split).

Usage (from the repo root):
    py main.py experiments --seeds 0-9 --modes full,folds --pop 100 --gens 40

Writes out/experiment_results.json.
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
from .individual import tree_from_json, to_position
from .primitives import build_context, eval_tree

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "cache_ETHUSDT_1d_3000.csv"
OUT = Path(__file__).resolve().parent / "out"


def load_data(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time"]).sort_values("open_time")
    return df.reset_index(drop=True)


def positions_for(tree: tuple, ctx_all: dict) -> np.ndarray:
    raw = np.nan_to_num(eval_tree(tree, ctx_all), nan=0.0, posinf=0.0, neginf=0.0)
    return to_position(raw)


def metric_row(close: np.ndarray, pos: np.ndarray) -> dict:
    m = summarize(close, pos)
    if m is None:
        return {"total": float("nan"), "ann": float("nan"), "sharpe": float("nan"),
                "mdd": float("nan"), "calmar": float("nan"), "turnover": float("nan"),
                "flips": 0}
    return {k: m[k] for k in ("total", "ann", "sharpe", "mdd", "calmar", "turnover", "flips")}


def evaluate_holdout(tree: tuple, ctx_all: dict, test_mask: np.ndarray) -> dict:
    pos = positions_for(tree, ctx_all)
    close = ctx_all["close"]
    plain = metric_row(close[test_mask], pos[test_mask])
    risk = metric_row(close[test_mask], apply_risk_overlay(close[test_mask], pos[test_mask]))
    return {"plain": plain, "risk": risk, "pos": pos}


def baseline_rows(df: pd.DataFrame, ctx_all: dict, test_mask: np.ndarray) -> dict:
    close = ctx_all["close"]
    out = {}
    out["buy_hold"] = {"plain": metric_row(close[test_mask], np.ones(test_mask.sum()))}
    c = pd.Series(close)
    for fast, slow in ((10, 50), (50, 200)):
        sig = np.where(c.rolling(fast).mean() > c.rolling(slow).mean(), 1.0, -1.0)
        sig[:slow] = 0.0
        out[f"ma{fast}_{slow}"] = {"plain": metric_row(close[test_mask], sig[test_mask])}
    mom = np.zeros(len(c))
    mom[252:] = np.where(c.to_numpy()[252:] > c.to_numpy()[:-252], 1.0, 0.0)
    out["mom_12m"] = {"plain": metric_row(close[test_mask], mom[test_mask])}
    return out


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x != ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--modes", default="full,folds")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=40)
    args = ap.parse_args()

    df = load_data(Path(args.data))
    train = df[df["open_time"] < args.split].reset_index(drop=True)
    test_mask = (df["open_time"] >= args.split).to_numpy()
    ctx_all = build_context(df)
    folds = year_folds(train["open_time"])
    print(f"data {len(df)} bars {df.open_time.iloc[0].date()}..{df.open_time.iloc[-1].date()}  "
          f"train={len(train)} test={int(test_mask.sum())}")
    print("folds " + ", ".join(f"{train.open_time.iloc[a].date()}..{train.open_time.iloc[b-1].date()}"
                              for a, b in folds), flush=True)

    results = {"config": vars(args), "baselines": baseline_rows(df, ctx_all, test_mask),
               "runs": [], "ensembles": {}}
    for mode in args.modes.split(","):
        mode = mode.strip()
        for seed in parse_seeds(args.seeds):
            t0 = time.time()
            best, _ = evolve(train, pop_size=args.pop, generations=args.gens, seed=seed,
                             verbose=False, folds=folds if mode == "folds" else None,
                             robust=(mode == "robust"))
            ev = evaluate_holdout(best.tree, ctx_all, test_mask)
            row = {"mode": mode, "seed": seed, "score": best.score, "size": best.size,
                   "expr": best.expr, "tree": best.tree,
                   "train": metric_row(ctx_all["close"][~test_mask], ev["pos"][~test_mask]),
                   "test": ev["plain"], "test_risk": ev["risk"],
                   "seconds": round(time.time() - t0, 1)}
            results["runs"].append(row)
            print(f"{mode:5s} seed={seed:2d} score={best.score:7.2f} size={best.size:2d} "
                  f"train_sh={row['train']['sharpe']:5.2f} test={row['test']['total']:+8.1%} "
                  f"test_sh={row['test']['sharpe']:5.2f} risk={row['test_risk']['total']:+8.1%} "
                  f"risk_sh={row['test_risk']['sharpe']:5.2f} ({row['seconds']}s)", flush=True)

    # Equal-weight ensembles over seeds, per mode (bagging reduces seed variance).
    for mode in args.modes.split(","):
        mode = mode.strip()
        rows = [r for r in results["runs"] if r["mode"] == mode]
        if not rows:
            continue
        mats = np.vstack([positions_for(tree_from_json(r["tree"]), ctx_all) for r in rows])
        for name, pos in (("mean", mats.mean(axis=0)),
                          ("median", np.median(mats, axis=0)),
                          ("majority", np.sign(np.sign(mats).sum(axis=0)))):
            results["ensembles"].setdefault(mode, {})[name] = {
                "train": metric_row(ctx_all["close"][~test_mask], pos[~test_mask]),
                "test": metric_row(ctx_all["close"][test_mask], pos[test_mask]),
                "test_risk": metric_row(ctx_all["close"][test_mask],
                                        apply_risk_overlay(ctx_all["close"][test_mask],
                                                           pos[test_mask])),
            }
            print(f"ENSEMBLE {mode:5s} {name:8s} "
                  f"test={results['ensembles'][mode][name]['test']['total']:+8.1%} "
                  f"test_sh={results['ensembles'][mode][name]['test']['sharpe']:5.2f}")

    OUT.mkdir(exist_ok=True)
    path = OUT / "experiment_results.json"
    path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
