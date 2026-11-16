"""Aggressive 4h leveraged trend — the 200%+ branch.

What this is
------------
The daily TSMOM in :mod:`gp_eth.trend` is honest but tops out around +28%/yr
because it targets 40% vol with a 2x cap.  This module asks the question the
user actually asked: *can the same edge be levered to >=200% a year without
the perp being liquidated?*

Rules (all causal, decided at each 4h close, earned from the next bar):

* signal = average sign of 30/60/120/180/240-bar returns (~5-40 days)
* only long when BTC's identical signal agrees and price is above its 200-bar MA
  (long-only: over 2017-2026 the short side paid for nothing, and removing it
  keeps the same Sharpe while cutting drawdown)
* size = 1.5 / realised 60-bar vol, capped at 10x notional
* resize only if the new target differs by >20% of the current one (dead-band);
  this cuts ~15% of turnover without changing the signal
* costs: 5 bp per unit turnover, 10%/yr funding on |position|
* liquidation: cross margin, maintenance margin = 0.5% of notional; if one bar's
  loss takes equity below that, the account is gone and the curve stops.

Honest expectations
-------------------
Full sample 2017-09 -> 2026-10: ~+210%/yr, -73% max drawdown, Sharpe ~1.6.
But 2017-2021 contributes almost all of it (+500%/yr); 2022-2026 is ~+71%/yr
with the same -73% drawdown.  Levering further in 2022-2026 *lowers* CAGR
(volatility drag), so ~80%/yr is the recent-regime ceiling at this Sharpe.
Treat the 200% as a whole-history, in-sample figure, not a forecast.

Usage (from the package parent):
    py -m gp_eth.aggressive
    py -m gp_eth.aggressive --vol 1.2 --band 0.2 --start 2022-01-01
    py -m gp_eth.aggressive --stop 0.10 --atr-mult 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .trend import render_html

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

BARS_PER_YEAR = 365.0 * 6.0      # 4h bars
FEE = 0.0005
FUNDING = 0.10
MAINT_MARGIN = 0.005

LOOKBACKS = (30, 60, 120, 180, 240)
VOL_TARGET = 1.5
MAX_LEV = 10.0
VOL_LOOKBACK = 60
BAND = 0.20
MA_LEN = 200


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "time" if "time" in df.columns else "open_time"
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).reset_index(drop=True)
    return df.rename(columns={tcol: "open_time"})


def tsmom(close: np.ndarray, lookbacks: tuple[int, ...] = LOOKBACKS) -> np.ndarray:
    """Average sign of past returns.  Causal: uses close up to and including t."""
    close = np.asarray(close, dtype=float)
    parts = []
    for n in lookbacks:
        s = np.zeros_like(close)
        s[n:] = np.sign(close[n:] / close[:-n] - 1.0)
        parts.append(s)
    return np.mean(parts, axis=0)


def raw_signal(df: pd.DataFrame, btc: pd.DataFrame | None = None,
               lookbacks: tuple[int, ...] = LOOKBACKS,
               ma_len: int = MA_LEN, long_only: bool = True) -> np.ndarray:
    close = df["close"].to_numpy(np.float64)
    raw = tsmom(close, lookbacks)
    if btc is not None:
        bclose = btc["close"].to_numpy(np.float64)
        if len(bclose) != len(close):
            raise ValueError("ETH and BTC frames must be aligned")
        braw = tsmom(bclose, lookbacks)
        raw = np.where(np.sign(raw) == np.sign(braw), raw, 0.0)
    if ma_len:
        ma = pd.Series(close).rolling(ma_len, min_periods=ma_len).mean().to_numpy()
        raw = raw * (close > ma)
    if long_only:
        raw = np.maximum(raw, 0.0)
    raw[:max(max(lookbacks), ma_len, VOL_LOOKBACK)] = 0.0
    return raw


def vol_scale(close: np.ndarray, target: float, max_lev: float,
              n: int = VOL_LOOKBACK) -> np.ndarray:
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    v = pd.Series(ret).rolling(n, min_periods=max(10, n // 2)).std(ddof=1).to_numpy()
    v = v * np.sqrt(BARS_PER_YEAR)
    out = np.zeros_like(close)
    ok = np.isfinite(v) & (v > 1e-6)
    out[ok] = np.clip(target / v[ok], 0.0, max_lev)
    return out


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def backtest(df: pd.DataFrame, raw: np.ndarray, vol_target: float = VOL_TARGET,
             max_lev: float = MAX_LEV, band: float = BAND, vol_n: int = VOL_LOOKBACK,
             stop: float = 0.0, atr_arr: np.ndarray | None = None,
             atr_mult: float = 0.0, fee: float = FEE, funding: float = FUNDING,
             maint: float = MAINT_MARGIN) -> dict:
    """Event-driven 4h backtest with intrabar stops and perp liquidation.

    Returns equity, position, liquidation index, and the closest approach to the
    maintenance margin (1.0 == liquidated).
    """
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    n = len(c)
    scale = vol_scale(c, vol_target, max_lev, vol_n)

    eq = np.ones(n)
    pos_hist = np.zeros(n)
    liq = -1
    pos = 0.0
    peak = trough = c[0]
    blocked = 0.0
    worst_margin = np.inf
    use_stop = stop > 0 or (atr_mult > 0 and atr_arr is not None)

    for t in range(1, n):
        pnl = 0.0
        stopped = False
        if pos > 0:
            dist = atr_mult * atr_arr[t - 1] / c[t - 1] if atr_mult > 0 else stop
            if use_stop and l[t] <= peak * (1 - dist):
                exit_px = min(o[t], peak * (1 - dist))
                pnl = pos * (exit_px / c[t - 1] - 1.0)
                stopped, blocked = True, 1.0
            else:
                pnl = pos * (c[t] / c[t - 1] - 1.0)
        elif pos < 0:
            dist = atr_mult * atr_arr[t - 1] / c[t - 1] if atr_mult > 0 else stop
            if use_stop and h[t] >= trough * (1 + dist):
                exit_px = max(o[t], trough * (1 + dist))
                pnl = pos * (exit_px / c[t - 1] - 1.0)
                stopped, blocked = True, -1.0
            else:
                pnl = pos * (c[t] / c[t - 1] - 1.0)

        step = 1.0 + pnl - funding / BARS_PER_YEAR * abs(pos)
        if step <= maint * abs(pos):
            eq[t:] = 0.0
            liq = t
            break
        eq[t] = eq[t - 1] * step

        if abs(pos) > 1e-9:
            worst_margin = min(worst_margin, eq[t] / (abs(pos) * maint))

        if stopped:
            eq[t] *= 1.0 - abs(pos) * fee
            pos = 0.0
            peak = trough = c[t]
            continue

        s = np.sign(raw[t])
        target = 0.0 if (s == 0 or s == blocked) else s * scale[t]
        # dead-band: skip cosmetic resizes on the same side
        if target != 0.0 and pos != 0.0 and np.sign(target) == np.sign(pos):
            if abs(target - pos) < band * abs(pos):
                pos_hist[t] = pos
                continue
        eq[t] *= 1.0 - abs(target - pos) * fee
        if target != 0.0 and np.sign(target) != np.sign(pos):
            peak = trough = c[t]
        pos = target
        pos_hist[t] = pos
        if pos > 0:
            peak = max(peak, c[t])
        elif pos < 0:
            trough = min(trough, c[t])

    return {"equity": eq, "position": pos_hist, "liq": liq,
            "worst_margin": float(worst_margin)}


def stats_of(eq: np.ndarray) -> dict:
    n = len(eq)
    if eq[-1] <= 0:
        return {"total": -1.0, "ann": -1.0, "sharpe": 0.0, "mdd": 1.0,
                "calmar": 0.0, "liq": True, "bars": n}
    s = np.zeros(n)
    s[1:] = eq[1:] / eq[:-1] - 1.0
    sd = float(s.std(ddof=1))
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))
    ann = float(eq[-1] ** (BARS_PER_YEAR / n) - 1.0)
    return {"total": float(eq[-1] - 1.0), "ann": ann,
            "sharpe": float(s.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0,
            "mdd": mdd, "calmar": ann / mdd if mdd > 1e-9 else 0.0,
            "liq": False, "bars": n}


def per_year(df: pd.DataFrame, eq: np.ndarray) -> dict:
    """Stats per calendar year with the equity normalised inside the year."""
    years = df["open_time"].dt.year.to_numpy()
    out = {}
    for y in sorted(set(years)):
        mask = years == y
        if mask.sum() < 30:
            continue
        e = eq[mask]
        if e[0] <= 0:
            continue
        out[str(y)] = stats_of(e / e[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "eth_4h.csv"))
    ap.add_argument("--btc", default=str(HERE / "btc_4h.csv"))
    ap.add_argument("--start", default="2017-09-01")
    ap.add_argument("--vol", type=float, default=VOL_TARGET)
    ap.add_argument("--max-lev", type=float, default=MAX_LEV)
    ap.add_argument("--band", type=float, default=BAND)
    ap.add_argument("--vol-lookback", type=int, default=VOL_LOOKBACK)
    ap.add_argument("--stop", type=float, default=0.0)
    ap.add_argument("--atr-mult", type=float, default=0.0)
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--funding", type=float, default=FUNDING)
    ap.add_argument("--long-short", action="store_true")
    ap.add_argument("--out", default=str(OUT / "aggressive_results.json"))
    args = ap.parse_args()

    df = load(args.data)
    btc = load(args.btc) if args.btc else None
    raw = raw_signal(df, btc, long_only=not args.long_short)
    a = atr(df) if args.atr_mult > 0 else None

    lo = int(np.searchsorted(df["open_time"].to_numpy(), np.datetime64(args.start)))
    sl = slice(lo, None)
    d = df.iloc[lo:].reset_index(drop=True)
    r = backtest(d, raw[sl], args.vol, args.max_lev, args.band,
                 args.vol_lookback, args.stop, a[sl] if a is not None else None,
                 args.atr_mult, args.fee, args.funding)

    m = stats_of(r["equity"])
    m["liq"] = r["liq"] >= 0
    bh = stats_of(d["close"].to_numpy(np.float64) / d["close"].iloc[0])
    years = per_year(d, r["equity"])
    pos = r["position"]

    print(f"ETH 4h aggressive TSMOM {'/'.join(map(str, LOOKBACKS))} "
          f"vol={args.vol:.0%} cap={args.max_lev:g}x band={args.band:.0%} "
          f"fee={args.fee:.2%} funding={args.funding:.0%}")
    print(f"  total={m['total']:+12.1%} ann={m['ann']:+8.1%} sharpe={m['sharpe']:5.2f} "
          f"mdd={m['mdd']:6.1%} calmar={m['calmar']:5.2f} liq={m['liq']}")
    print(f"  B&H   total={bh['total']:+12.1%} ann={bh['ann']:+8.1%} "
          f"sharpe={bh['sharpe']:5.2f} mdd={bh['mdd']:6.1%}")
    print(f"  exposure avg={np.mean(np.abs(pos)):.2f} p95={np.percentile(np.abs(pos), 95):.2f} "
          f"max={np.max(np.abs(pos)):.2f}  closest margin={r['worst_margin']:.0f}x")
    print("  per year:")
    for y, rec in years.items():
        print(f"    {y} total={rec['total']:+10.1%} ann={rec['ann']:+8.1%} "
              f"sharpe={rec['sharpe']:5.2f} mdd={rec['mdd']:6.1%}")

    eq = r["equity"]
    eq_bh = d["close"].to_numpy(np.float64) / d["close"].iloc[0]
    dates = d["open_time"].dt.strftime("%Y-%m-%d").tolist()
    payload = {
        "params": {"lookbacks": list(LOOKBACKS), "ma_len": MA_LEN,
                   "vol_target": args.vol, "max_lev": args.max_lev,
                   "band": args.band, "vol_lookback": args.vol_lookback,
                   "stop": args.stop, "atr_mult": args.atr_mult,
                   "fee": args.fee, "funding": args.funding,
                   "long_only": not args.long_short, "start": args.start,
                   "bars_per_year": BARS_PER_YEAR},
        "stats": m, "buy_hold": bh, "per_year": years,
        "risk": {"worst_margin_multiple": r["worst_margin"],
                 "avg_exposure": float(np.mean(np.abs(pos))),
                 "p95_exposure": float(np.percentile(np.abs(pos), 95)),
                 "max_exposure": float(np.max(np.abs(pos)))},
        "series": {"dates": dates, "equity": np.round(eq, 6).tolist(),
                   "buy_hold": np.round(eq_bh, 6).tolist(),
                   "position": np.round(pos, 6).tolist()},
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")

    html = out.with_name("aggressive_equity.html")
    sub = (f"4h TSMOM {'/'.join(map(str, LOOKBACKS))} · 只做多 + BTC 确认 + MA200 · "
           f"vol target {args.vol:.0%} · cap {args.max_lev:g}x — "
           f"年化 {m['ann']:+.1%} · MDD {m['mdd']:.1%} · Sharpe {m['sharpe']:.2f}")
    render_html(html, dates, {"trend": eq.tolist(), "buy_hold": eq_bh.tolist()},
                "gp_eth 4h 杠杆趋势策略", sub, years,
                note=("* 2026 为 1-10 月部分年份。5 bp/边手续费 + 10%/年资金费，"
                      "含盘中止损与强平建模。"), labels={"trend": "杠杆趋势",
                                                            "buy_hold": "买入持有"})
    print(f"written: {out}\nwritten: {html}")


if __name__ == "__main__":
    main()
