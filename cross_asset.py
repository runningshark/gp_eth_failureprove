"""Cross-asset check: apply the ETH-evolved fold trees to BTC daily bars.

This is not a training run.  The trees are used exactly as selected on ETH;
if the same feature logic still has an edge on a correlated but different
asset, that is mild evidence against pure ETH curve-fitting.

Usage (from the Desktop):
    py -m gp_eth.cross_asset --asset btc --start 2024-01-01

Writes out/cross_asset.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import summarize
from .experiments import load_data, positions_for
from .individual import tree_from_json
from .primitives import build_context

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BTC_4H = HERE.parent / "test" / "btc_4h.csv"


def load_btc_daily(path: Path = BTC_4H) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"]).sort_values("time").set_index("time")
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
    d = d.reset_index().rename(columns={"time": "open_time"})
    return d


def load_eth_trees(path: Path = OUT / "experiment_results.json") -> list[tuple]:
    data = json.loads(path.read_text(encoding="utf-8"))
    trees = []
    for r in data["runs"]:
        if r["mode"] == "folds":
            trees.append(tree_from_json(r["tree"]))
    return trees


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="btc", choices=("btc",))
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--data", default=str(BTC_4H))
    args = ap.parse_args()

    df = load_btc_daily(Path(args.data))
    ctx = build_context(df)
    close = ctx["close"]
    mask = (df["open_time"] >= args.start).to_numpy()
    trees = load_eth_trees()
    print(f"{args.asset}: {len(df)} daily bars {df.open_time.iloc[0].date()}..{df.open_time.iloc[-1].date()}  "
          f"trees={len(trees)}")

    mats = []
    rows = []
    for i, tree in enumerate(trees):
        pos = positions_for(tree, ctx)
        mats.append(pos)
        m = summarize(close[mask], pos[mask])
        rows.append({"i": i, "total": m["total"], "sharpe": m["sharpe"], "mdd": m["mdd"]})
        print(f"  tree {i:02d} test={m['total']:+9.1%} sh={m['sharpe']:5.2f} mdd={m['mdd']:6.1%}")

    M = np.vstack(mats)
    out = {"asset": args.asset, "start": args.start, "trees": rows, "ensemble": {}}
    for name, pos in (("median", np.median(M, axis=0)), ("mean", M.mean(axis=0))):
        m = summarize(close[mask], pos[mask])
        out["ensemble"][name] = {k: m[k] for k in ("total", "ann", "sharpe", "mdd", "calmar", "flips")}
        print(f"ENSEMBLE {name:6s} test={m['total']:+9.1%} sh={m['sharpe']:5.2f} mdd={m['mdd']:6.1%}")
    bh = summarize(close[mask], np.ones(mask.sum()))
    out["buy_hold"] = {k: bh[k] for k in ("total", "ann", "sharpe", "mdd")}
    print(f"BUY&HOLD      test={bh['total']:+9.1%} sh={bh['sharpe']:5.2f} mdd={bh['mdd']:6.1%}")

    OUT.mkdir(exist_ok=True)
    (OUT / "cross_asset.json").write_text(json.dumps(out, indent=1, default=float),
                                          encoding="utf-8")
    print(f"written: {OUT / 'cross_asset.json'}")


if __name__ == "__main__":
    main()
