"""Fast 4h momentum — the 2-day branch.

The long-horizon 4h trend in :mod:`gp_eth.aggressive` holds for weeks and earns
~210%/yr over the whole history but only ~71%/yr in 2022-2026.  This module is
the short-horizon answer to the same question: trade a 2-day (12-bar) move, with
hysteresis so it holds a position for 1-3 days instead of re-balancing every bar.

Rules (all causal, decided at each 4h close, earned from the next bar):

* momentum ``r = close[t] / close[t-12] - 1``
* ``z`` = rolling std of the same 12-bar return over 72 bars
* enter long when ``r > 1.5 z``; exit when ``r < -0.25 z``
  (the asymmetry is the point: a 2-day move has to be strong to enter, but a
  small give-back is enough to leave — this is what cuts turnover from ~440x to
  ~160x a year at 2x notional)
* only long while ETH is above its 96-bar MA (16 days) **and** BTC is above its
  own 96-bar MA.  At the default point the BTC filter is within noise (+94.7% vs
  +95.9% in 2022-2026), but it makes the parameter surface much smoother: across
  the 60-point threshold grid the 2022-2026 median is +70% with it versus +50%
  without, and no cell is negative (three are without it).  ``--btc-ma 0`` turns
  it off.
* size = 1.5 / realised 240-bar vol (40 days), capped at 4x notional
* resize only if the new target differs by >40% of the current one
* costs: 5 bp per unit turnover, 10%/yr funding on |position|
* liquidation: cross margin, maintenance margin = 0.5% of notional

Honest expectations
-------------------
Full sample 2017-09 -> 2026-10: ~+127%/yr, -77% max drawdown, Sharpe ~1.46.
2022-2026: ~+95%/yr at 5 bp, ~+82%/yr at 10 bp, same -77% drawdown.
No liquidation in any of the 60-point parameter neighbourhood, closest approach
to maintenance margin is 71x (a ~35% one-bar move at the worst moment).

The headline is not a smooth curve.  2020, 2024 and 2025 carry almost all of it;
2019 and 2026 are losing years (2026 is -30% at the time of writing and the
strategy is ~60% below its 2026-08 peak).  The 12-bar edge is real but decays
quickly: adding one bar of execution delay costs ~20 pp of CAGR, two bars ~40 pp.
At 20 bp all-in costs the 2022-2026 edge is ~51%/yr, not 95%.

Usage (from the package parent):
    py -m gp_eth.fast4h
    py -m gp_eth.fast4h --vol 2.0 --cap 4 --fee 0.001
    py -m gp_eth.fast4h --btc-ma 0           # disable the BTC filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggressive import BARS_PER_YEAR, FUNDING, backtest, load, per_year, stats_of
from .trend import render_html

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# signal
N_BARS = 12          # 2 days
Z_WINDOW = 6         # z is 6 * N bars = 72
ENTRY_Z = 1.5
EXIT_Z = -0.25
MA_LEN = 96          # 16 days
BTC_MA_LEN = 96

# sizing
VOL_TARGET = 1.5
MAX_LEV = 4.0
VOL_LOOKBACK = 240   # 40 days
BAND = 0.40
FEE = 0.0005


def momentum_state(close: np.ndarray, n: int = N_BARS, entry_z: float = ENTRY_Z,
                   exit_z: float = EXIT_Z, z_window: int = Z_WINDOW) -> np.ndarray:
    """Hysteresis state machine on the n-bar return.  0 = flat, 1 = long.

    Causal: the state at t uses only closes up to and including t.
    """
    close = np.asarray(close, dtype=float)
    r = np.zeros_like(close)
    r[n:] = close[n:] / close[:-n] - 1.0
    z = pd.Series(r).rolling(n * z_window, min_periods=n * 3).std(ddof=1).to_numpy()
    out = np.zeros_like(close)
    state = 0.0
    for t in range(len(close)):
        if not np.isfinite(z[t]) or z[t] <= 0.0:
            out[t] = 0.0
            continue
        if state == 0.0 and r[t] > entry_z * z[t]:
            state = 1.0
        elif state == 1.0 and r[t] < exit_z * z[t]:
            state = 0.0
        out[t] = state
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def raw_signal(eth: pd.DataFrame, btc: pd.DataFrame | None = None,
               n: int = N_BARS, entry_z: float = ENTRY_Z, exit_z: float = EXIT_Z,
               ma_len: int = MA_LEN, btc_ma_len: int = BTC_MA_LEN) -> np.ndarray:
    """0/1 long-only target before vol sizing."""
    close = eth["close"].to_numpy(np.float64)
    raw = momentum_state(close, n, entry_z, exit_z)
    if ma_len:
        raw = raw * (close > sma(close, ma_len))
    if btc is not None and btc_ma_len:
        bclose = btc["close"].to_numpy(np.float64)
        if len(bclose) != len(close):
            raise ValueError("ETH and BTC frames must be aligned")
        raw = raw * (bclose > sma(bclose, btc_ma_len))
    raw[:max(n * Z_WINDOW, ma_len, btc_ma_len)] = 0.0
    return raw


def _sub(eq: np.ndarray, mask: np.ndarray) -> dict:
    e = eq[mask]
    if len(e) < 2 or e[0] <= 0 or e[-1] <= 0:
        return {"ann": -1.0, "mdd": 1.0, "sharpe": 0.0}
    return stats_of(e / e[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "eth_4h.csv"))
    ap.add_argument("--btc", default=str(HERE / "btc_4h.csv"))
    ap.add_argument("--start", default="2017-09-01")
    ap.add_argument("--vol", type=float, default=VOL_TARGET)
    ap.add_argument("--cap", type=float, default=MAX_LEV)
    ap.add_argument("--band", type=float, default=BAND)
    ap.add_argument("--vol-lookback", type=int, default=VOL_LOOKBACK)
    ap.add_argument("--btc-ma", type=int, default=BTC_MA_LEN,
                    help="BTC MA filter length in bars; 0 disables it")
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--funding", type=float, default=FUNDING)
    ap.add_argument("--out", default=str(OUT / "fast4h_results.json"))
    args = ap.parse_args()

    df = load(args.data)
    btc = load(args.btc) if (args.btc and args.btc_ma) else None
    raw = raw_signal(df, btc, btc_ma_len=args.btc_ma)

    lo = int(np.searchsorted(df["open_time"].to_numpy(), np.datetime64(args.start)))
    d = df.iloc[lo:].reset_index(drop=True)
    r = backtest(d, raw[lo:], args.vol, args.cap, args.band, args.vol_lookback,
                 fee=args.fee, funding=args.funding)

    m = stats_of(r["equity"])
    m["liq"] = r["liq"] >= 0
    years = per_year(d, r["equity"])
    pos = r["position"]
    eq = r["equity"]
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / np.maximum(peak, 1e-12)
    i_dd = int(np.argmax(dd))

    # 10 bp all-in stress on the same signal
    stress = {}
    stress_eq = {}
    for fee in (0.0008, 0.0010, 0.0020):
        rs = backtest(d, raw[lo:], args.vol, args.cap, args.band, args.vol_lookback,
                      fee=fee, funding=args.funding)
        ms = stats_of(rs["equity"])
        ms["liq"] = rs["liq"] >= 0
        stress[f"{fee*1e4:.0f}bp"] = ms
        stress_eq[f"{fee*1e4:.0f}bp"] = rs["equity"]

    bh = stats_of(d["close"].to_numpy(np.float64) / d["close"].iloc[0])

    print(f"ETH 4h fast momentum n={N_BARS} entry={ENTRY_Z}z exit={EXIT_Z}z "
          f"MA{MA_LEN} BTC-MA{args.btc_ma} vol={args.vol:.0%} cap={args.cap:g}x "
          f"band={args.band:.0%} vwin={args.vol_lookback} "
          f"fee={args.fee:.2%} funding={args.funding:.0%}")
    print(f"  total={m['total']:+12.1%} ann={m['ann']:+8.1%} sharpe={m['sharpe']:5.2f} "
          f"mdd={m['mdd']:6.1%} calmar={m['calmar']:5.2f} liq={m['liq']}")
    print(f"  B&H   total={bh['total']:+12.1%} ann={bh['ann']:+8.1%} "
          f"sharpe={bh['sharpe']:5.2f} mdd={bh['mdd']:6.1%}")
    print(f"  exposure avg={np.mean(np.abs(pos)):.2f} "
          f"p95={np.percentile(np.abs(pos), 95):.2f} max={np.max(np.abs(pos)):.2f} "
          f"closest margin={r['worst_margin']:.0f}x")
    print(f"  max dd {m['mdd']:.1%} on {d['open_time'].iloc[i_dd]:%Y-%m-%d}; "
          f"equity now {eq[-1] / peak[-1] - 1:+.1%} vs peak")
    print("  cost stress (full / 2022+):")
    yrs = d["open_time"].dt.year.to_numpy()
    for k, ms in stress.items():
        s22 = _sub(stress_eq[k], yrs >= 2022)
        print(f"    {k:>6s} ann={ms['ann']:+8.1%} mdd={ms['mdd']:6.1%} "
              f"sharpe={ms['sharpe']:5.2f} | 22+ {s22['ann']:+8.1%}/"
              f"{s22['mdd']:6.1%} liq={ms['liq']}")
    print("  per year:")
    for y, rec in years.items():
        print(f"    {y} total={rec['total']:+10.1%} ann={rec['ann']:+8.1%} "
              f"sharpe={rec['sharpe']:5.2f} mdd={rec['mdd']:6.1%}")

    payload = {
        "params": {"n_bars": N_BARS, "entry_z": ENTRY_Z, "exit_z": EXIT_Z,
                   "z_window": Z_WINDOW, "ma_len": MA_LEN, "btc_ma_len": args.btc_ma,
                   "vol_target": args.vol, "max_lev": args.cap, "band": args.band,
                   "vol_lookback": args.vol_lookback, "fee": args.fee,
                   "funding": args.funding, "start": args.start,
                   "bars_per_year": BARS_PER_YEAR, "long_only": True},
        "stats": m, "buy_hold": bh, "cost_stress": stress, "per_year": years,
        "risk": {"worst_margin_multiple": r["worst_margin"],
                 "avg_exposure": float(np.mean(np.abs(pos))),
                 "p95_exposure": float(np.percentile(np.abs(pos), 95)),
                 "max_exposure": float(np.max(np.abs(pos))),
                 "max_dd_date": f"{d['open_time'].iloc[i_dd]:%Y-%m-%d}",
                 "current_dd_from_peak": float(eq[-1] / peak[-1] - 1.0)},
        "series": {"dates": d["open_time"].dt.strftime("%Y-%m-%d").tolist(),
                   "equity": np.round(eq, 6).tolist(),
                   "buy_hold": np.round(
                       d["close"].to_numpy(np.float64) / d["close"].iloc[0], 6).tolist(),
                   "position": np.round(pos, 6).tolist()},
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")

    html = out.with_name(out.stem.replace("_results", "") + "_equity.html")
    sub = (f"4h 12-bar 动量 · 滞回进出 + 96 均线 · 只做多 · "
           f"vol {args.vol:.0%} · cap {args.cap:g}x — "
           f"年化 {m['ann']:+.1%} · MDD {m['mdd']:.1%} · Sharpe {m['sharpe']:.2f}")
    render_html(html, payload["series"]["dates"],
                {"trend": eq.tolist(), "buy_hold": payload["series"]["buy_hold"]},
                "gp_eth 4h 短线杠杆策略", sub, years,
                note=("* 2026 为 1-10 月部分年份。5 bp/边手续费 + 10%/年资金费，"
                      "含强平建模；10 bp 压测见 JSON 的 cost_stress。"),
                labels={"trend": "4h 短线杠杆", "buy_hold": "买入持有"})
    print(f"written: {out}\nwritten: {html}")


if __name__ == "__main__":
    main()
