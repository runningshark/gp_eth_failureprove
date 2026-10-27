"""Rolling-origin walk-forward evaluation (the thing the old "walk-forward" label
did not actually do).

For each test year Y: train only on data before Y, use calendar-year fold
fitness, evolve several seeds, ensemble their positions, then trade year Y.
All positions at the start of Y carry the 200-bar warmup from earlier data.

Usage (from the Desktop):
    py -m gp_eth.walkforward --seeds 0-4 --pop 100 --gens 40 --years 2022-2026

Writes out/walkforward_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import apply_risk_overlay, run_backtest, summarize, BARS_PER_YEAR
from .evolve import evolve, year_folds
from .experiments import load_data
from .individual import to_position
from .primitives import build_context, eval_tree

OUT = Path(__file__).resolve().parent / "out"


def pos_on_history(tree: tuple, df: pd.DataFrame) -> np.ndarray:
    ctx = build_context(df)
    raw = np.nan_to_num(eval_tree(tree, ctx), nan=0.0, posinf=0.0, neginf=0.0)
    return to_position(raw)


def daily_strat(close: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return run_backtest(close, pos)["strat"]


def stats(s: np.ndarray) -> dict:
    s = np.asarray(s, dtype=float)
    eq = np.cumprod(1.0 + s)
    total = float(eq[-1] - 1.0)
    ann = float(eq[-1] ** (BARS_PER_YEAR / len(s)) - 1.0)
    sd = float(s.std(ddof=1))
    sharpe = float(s.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))
    return {"total": total, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "calmar": ann / mdd if mdd > 1e-9 else 0.0,
            "t_stat": sharpe * float(np.sqrt(len(s) / BARS_PER_YEAR)),
            "days": int(len(s))}


def aggregate(strats: list[np.ndarray]) -> dict:
    return stats(np.concatenate(strats))


def parse_years(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--years", default="2022-2026")
    args = ap.parse_args()
    seeds = (list(range(int(args.seeds.split("-")[0]), int(args.seeds.split("-")[1]) + 1))
             if "-" in args.seeds else [int(x) for x in args.seeds.split(",")])

    df = load_data(Path(args.data)) if args.data else load_data()
    years = parse_years(args.years)
    out = {"config": vars(args), "years": {}, "ensemble": {}}
    agg_ens, agg_bh, agg_ma, agg_risk = [], [], [], []
    dates_all: list[str] = []
    c = pd.Series(df["close"])

    for y in years:
        train = df[df["open_time"] < f"{y}-01-01"].reset_index(drop=True)
        test_start = int(np.flatnonzero((df["open_time"] >= f"{y}-01-01").to_numpy())[0])
        test_end = int((df["open_time"] < f"{y + 1}-01-01").sum())
        if y == years[-1]:
            test_end = len(df)
        folds = year_folds(train["open_time"])
        if len(folds) < 2:
            print(f"{y}: not enough full folds ({len(folds)}), skipped")
            continue
        # History up to the end of the test year; all primitives are causal.
        hist = df.iloc[:test_end].reset_index(drop=True)
        close = hist["close"].to_numpy()
        lo = max(0, test_start - 1)  # include one carry-in bar
        print(f"\n{y}: train<{y} n={len(train)} folds="
              f"{[(train.open_time.iloc[a].year, train.open_time.iloc[b-1].year) for a, b in folds]} "
              f"test={test_end - test_start} bars", flush=True)

        mats = []
        rows = []
        for seed in seeds:
            t0 = time.time()
            best, _ = evolve(train, pop_size=args.pop, generations=args.gens,
                             seed=seed, verbose=False, folds=folds)
            pos = pos_on_history(best.tree, hist)
            seg_close, seg_pos = close[lo:test_end], pos[lo:test_end]
            sm = summarize(seg_close, seg_pos)
            rows.append({"seed": seed, "score": best.score, "size": best.size,
                         "train_sharpe": best.metrics["sharpe"] if best.metrics else None,
                         "test": {k: sm[k] for k in ("total", "sharpe", "mdd", "flips")},
                         "expr": best.expr, "seconds": round(time.time() - t0, 1)})
            mats.append(pos)
            print(f"  seed={seed} size={best.size:2d} fit_sh={(best.metrics or {}).get('sharpe', float('nan')):5.2f} "
                  f"test={sm['total']:+8.1%} sh={sm['sharpe']:5.2f} ({rows[-1]['seconds']}s)", flush=True)

        M = np.vstack([m[lo:test_end] for m in mats])
        ens = np.median(M, axis=0)
        ens_mean = M.mean(axis=0)
        risk = apply_risk_overlay(seg_close, ens)
        out["years"][str(y)] = {
            "runs": rows,
            "ensemble_median": {k: summarize(seg_close, ens)[k] for k in ("total", "sharpe", "mdd", "flips")},
            "ensemble_mean": {k: summarize(seg_close, ens_mean)[k] for k in ("total", "sharpe", "mdd", "flips")},
            "ensemble_median_risk": {k: summarize(seg_close, risk)[k] for k in ("total", "sharpe", "mdd", "flips")},
            "buy_hold": {k: summarize(seg_close, np.ones(len(seg_close)))[k]
                         for k in ("total", "sharpe", "mdd")},
        }
        agg_ens.append(daily_strat(seg_close, ens))
        agg_risk.append(daily_strat(seg_close, apply_risk_overlay(seg_close, ens)))
        agg_bh.append(daily_strat(seg_close, np.ones(len(seg_close))))
        dates_all.extend(hist["open_time"].iloc[lo:test_end].dt.strftime("%Y-%m-%d").tolist())
        maf = c.rolling(10).mean().to_numpy()[lo:test_end]
        mas = c.rolling(50).mean().to_numpy()[lo:test_end]
        ma = np.where(maf > mas, 1.0, -1.0)
        ma[:50] = 0.0
        agg_ma.append(daily_strat(seg_close, ma))
        yrec = out["years"][str(y)]
        print(f"  ENS median test={yrec['ensemble_median']['total']:+8.1%} "
              f"sh={yrec['ensemble_median']['sharpe']:5.2f} | mean={yrec['ensemble_mean']['total']:+8.1%} "
              f"| +risk={yrec['ensemble_median_risk']['total']:+8.1%} | B&H={yrec['buy_hold']['total']:+8.1%}",
              flush=True)

    all_strats = {"median": agg_ens, "median_risk": agg_risk,
                  "buy_hold": agg_bh, "ma10_50": agg_ma}
    for name, strats in all_strats.items():
        out["ensemble"][name] = aggregate(strats)

    # Exact pre-2026 slice (all of the positive aggregate comes from 2026).
    mask_pre26 = np.array([d < "2026-01-01" for d in dates_all])
    out["ex_2026"] = {name: stats(np.concatenate(strats)[mask_pre26])
                      for name, strats in all_strats.items()}
    out["series"] = {
        "dates": dates_all,
        **{name: np.cumprod(1.0 + np.concatenate(strats)).round(6).tolist()
           for name, strats in all_strats.items()},
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "walkforward_results.json").write_text(json.dumps(out, indent=1, default=float),
                                                 encoding="utf-8")
    print("\n=== aggregate OOS ===")
    for k, v in out["ensemble"].items():
        print(f"{k:12s} total={v['total']:+9.1%} ann={v['ann']:+7.1%} sharpe={v['sharpe']:5.2f} "
              f"mdd={v['mdd']:6.1%} calmar={v['calmar']:5.2f} t={v['t_stat']:4.2f} days={v['days']}")
    print("\n=== ex-2026 (2022..2025) ===")
    for k, v in out["ex_2026"].items():
        print(f"{k:12s} total={v['total']:+9.1%} ann={v['ann']:+7.1%} sharpe={v['sharpe']:5.2f} "
              f"mdd={v['mdd']:6.1%} days={v['days']}")
    print(f"written: {OUT / 'walkforward_results.json'}")


if __name__ == "__main__":
    main()
