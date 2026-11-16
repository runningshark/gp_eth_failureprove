"""Textbook benchmark strategies, evaluated walk-forward with fixed parameters.

The point of this module is honesty: instead of mining thousands of trees, we
test a small set of *pre-registered* strategy families that are known to work in
trending markets, with realistic fees and (optionally) volatility targeting.

No parameter is chosen on the evaluation window.  Every parameter set below is
a textbook default decided before looking at ETH results.  The optional
``--select`` mode picks parameters on *past* years only, then applies them to
the next year (a true out-of-sample parameter walk-forward).

Usage (from the package parent):
    py -m gp_eth.benchmarks                    # fixed params, 2018-2026
    py -m gp_eth.benchmarks --vol 0.4          # 40% annualised vol target
    py -m gp_eth.benchmarks --select           # yearly parameter re-selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BARS_PER_YEAR, FEE, run_backtest
from .trend import vol_scale

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DEFAULT_DATA = HERE / "cache_ETHUSDT_1d_3000.csv"


# ---------------------------------------------------------------- data helpers
def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    return df.sort_values("open_time").reset_index(drop=True)


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


# ------------------------------------------------------------------- signals
def sig_buy_hold(close, high, low, ret, **kw):
    return np.ones_like(close)


def sig_ma(close, high, low, ret, fast=10, slow=50, **kw):
    f, s = _sma(close, fast), _sma(close, slow)
    out = np.zeros_like(close)
    valid = ~(np.isnan(f) | np.isnan(s))
    out[valid] = np.sign(f[valid] - s[valid])
    return out


def sig_tsmom(close, high, low, ret, lookback=90, **kw):
    out = np.zeros_like(close)
    out[lookback:] = np.sign(close[lookback:] / close[:-lookback] - 1.0)
    return out


def sig_donchian(close, high, low, ret, lookback=55, **kw):
    """Turtle-style channel breakout; channel excludes the current bar."""
    upper = pd.Series(high).rolling(lookback, min_periods=lookback).max().shift(1).to_numpy()
    lower = pd.Series(low).rolling(lookback, min_periods=lookback).min().shift(1).to_numpy()
    out = np.zeros_like(close)
    state = 0.0
    for t in range(len(close)):
        if np.isnan(upper[t]):
            out[t] = 0.0
            continue
        if close[t] > upper[t]:
            state = 1.0
        elif close[t] < lower[t]:
            state = -1.0
        out[t] = state
    return out


def sig_rsi_revert(close, high, low, ret, n=14, lo=30.0, hi=70.0, **kw):
    from .primitives import _rsi
    r = _rsi(close, n)
    out = np.zeros_like(close)
    out[r < lo] = 1.0
    out[r > hi] = -1.0
    return out


def sig_combo(close, high, low, ret, **kw):
    """Average of independent trend signals; robust to any single lookback."""
    parts = [
        sig_tsmom(close, high, low, ret, lookback=30),
        sig_tsmom(close, high, low, ret, lookback=90),
        sig_tsmom(close, high, low, ret, lookback=180),
        sig_ma(close, high, low, ret, fast=10, slow=50),
        sig_ma(close, high, low, ret, fast=50, slow=200),
        sig_donchian(close, high, low, ret, lookback=55),
    ]
    return np.mean(parts, axis=0)


STRATEGIES = {
    "buy_hold": sig_buy_hold,
    "ma_10_50": sig_ma,
    "ma_20_100": lambda c, h, l, r, **k: sig_ma(c, h, l, r, fast=20, slow=100),
    "ma_50_200": lambda c, h, l, r, **k: sig_ma(c, h, l, r, fast=50, slow=200),
    "tsmom_30": lambda c, h, l, r, **k: sig_tsmom(c, h, l, r, lookback=30),
    "tsmom_90": lambda c, h, l, r, **k: sig_tsmom(c, h, l, r, lookback=90),
    "tsmom_180": lambda c, h, l, r, **k: sig_tsmom(c, h, l, r, lookback=180),
    "donchian_20": lambda c, h, l, r, **k: sig_donchian(c, h, l, r, lookback=20),
    "donchian_55": lambda c, h, l, r, **k: sig_donchian(c, h, l, r, lookback=55),
    "rsi_revert": sig_rsi_revert,
    "trend_combo": sig_combo,
}


# --------------------------------------------------------------- vol targeting
def apply_vol(raw: np.ndarray, ret: np.ndarray, target: float,
              lookback: int = 20, max_lev: float = 2.0,
              funding: float = 0.0) -> np.ndarray:
    pos = raw * vol_scale(ret, target, max_lev, lookback)
    return pos


# ------------------------------------------------------------------ evaluation
def stats(close: np.ndarray, pos: np.ndarray, funding: float = 0.0) -> dict | None:
    r = run_backtest(close, pos)
    s = r["strat"]
    if funding:
        # rough perp funding: longs pay `funding` per year, shorts receive it
        s = s - funding / BARS_PER_YEAR * np.abs(pos)
    eq = np.cumprod(1.0 + s)
    if not np.all(np.isfinite(eq)) or eq[-1] <= 0:
        return None
    n = len(s)
    total = float(eq[-1] - 1.0)
    ann = float(eq[-1] ** (BARS_PER_YEAR / n) - 1.0)
    sd = float(s.std(ddof=1))
    sharpe = float(s.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))
    return {"total": total, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "calmar": ann / mdd if mdd > 1e-9 else 0.0,
            "turnover": float(r["turnover"].mean()),
            "exposure": float(np.mean(np.abs(pos))), "days": n}


def fmt(name: str, m: dict | None) -> str:
    if m is None:
        return f"{name:14s}  n/a"
    return (f"{name:14s} total={m['total']:+8.1%} ann={m['ann']:+7.1%} "
            f"sharpe={m['sharpe']:5.2f} mdd={m['mdd']:6.1%} calmar={m['calmar']:5.2f} "
            f"turn={m['turnover']:.3f} exp={m['exposure']:.2f}")


def run_fixed(df: pd.DataFrame, vol: float, max_lev: float, funding: float,
              start: str | None) -> dict:
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    lo = 0
    if start:
        lo = int(np.searchsorted(df["open_time"].to_numpy(), np.datetime64(start)))
        lo = max(lo, 250)  # keep enough warmup for the longest lookback
    res = {}
    for name, fn in STRATEGIES.items():
        raw = fn(close, high, low, ret)
        pos = apply_vol(raw, ret, vol, max_lev=max_lev)
        m = stats(close[lo:], pos[lo:], funding)
        res[name] = m
        print(fmt(name, m), flush=True)
    return res


# ------------------------------------------------- out-of-sample param selection
GRID = {
    "ma": [{"fast": f, "slow": s} for f in (5, 10, 20, 50) for s in (50, 100, 200) if f < s],
    "tsmom": [{"lookback": n} for n in (20, 30, 60, 90, 120, 180)],
    "donchian": [{"lookback": n} for n in (20, 30, 55, 89)],
}


def select_year(df: pd.DataFrame, year: int, vol: float, max_lev: float,
                train_years: int = 3, funding: float = 0.0) -> dict:
    """Pick the best family+params on the `train_years` before `year`."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    years = df["open_time"].dt.year.to_numpy()
    tr = (years >= year - train_years) & (years < year)
    te = years == year
    if tr.sum() < 200 or te.sum() < 100:
        return {}
    best = None
    for fam, params in GRID.items():
        for p in params:
            raw = STRATEGIES[fam](close, high, low, ret, **p)
            pos = apply_vol(raw, ret, vol, max_lev=max_lev)
            m = stats(close[tr], pos[tr], funding)
            if m is None:
                continue
            score = m["sharpe"] + 0.5 * np.clip(m["calmar"], -2, 4) - 2.0 * max(0, m["mdd"] - 0.5)
            if best is None or score > best[0]:
                best = (score, fam, p)
    if best is None:
        return {}
    _, fam, p = best
    raw = STRATEGIES[fam](close, high, low, ret, **p)
    pos = apply_vol(raw, ret, vol, max_lev=max_lev)
    m = stats(close[te], pos[te], funding)
    return {"year": int(year), "family": fam, "params": p,
            "train": stats(close[tr], pos[tr], funding), "test": m}


def run_select(df: pd.DataFrame, vol: float, max_lev: float, funding: float,
               years: list[int]) -> dict:
    rows = []
    for y in years:
        r = select_year(df, y, vol, max_lev, funding=funding)
        if not r:
            continue
        rows.append(r)
        t = r["test"]
        print(f"{y} <- {r['family']} {r['params']}  "
              f"train={r['train']['total']:+7.1%} test={t['total']:+7.1%} "
              f"sharpe={t['sharpe']:5.2f} mdd={t['mdd']:6.1%}", flush=True)
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--start", default=None)
    ap.add_argument("--vol", type=float, default=0.0, help="annualised vol target, 0=off")
    ap.add_argument("--max-lev", type=float, default=2.0)
    ap.add_argument("--funding", type=float, default=0.0, help="annual funding cost on |pos|")
    ap.add_argument("--select", action="store_true", help="walk-forward param selection")
    ap.add_argument("--years", default="2022-2026")
    args = ap.parse_args()

    df = load(args.data)
    OUT.mkdir(exist_ok=True)
    tag = "select" if args.select else "fixed"
    suffix = f"vol{int(args.vol*100)}_lev{args.max_lev:g}_fund{args.funding:g}"

    if args.select:
        a, b = args.years.split("-")
        res = run_select(df, args.vol, args.max_lev, args.funding,
                         list(range(int(a), int(b) + 1)))
        path = OUT / f"benchmarks_{tag}_{suffix}.json"
    else:
        print(f"data: {args.data}  bars={len(df)}  "
              f"{df['open_time'].iloc[0].date()} -> {df['open_time'].iloc[-1].date()}")
        print(f"vol_target={args.vol:.0%} max_lev={args.max_lev:g} "
              f"funding={args.funding:.0%}/yr fee={FEE:.2%}")
        res = {"fixed": run_fixed(df, args.vol, args.max_lev, args.funding, args.start)}
        path = OUT / f"benchmarks_{tag}_{suffix}.json"
    path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
