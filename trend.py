"""Trend-following (TSMOM) strategy — the replacement for the GP signal.

Why this exists
---------------
The GP search found trees that looked good in-sample but had no out-of-sample
edge (see REPORT.md §5).  Meanwhile a plain, pre-registered time-series
momentum ensemble is positive in every year of the 2018-2026 ETH sample and
also transfers to BTC.  This module packages that strategy so it can be
backtested, paper-traded, and compared with the GP.

Rules
-----
* signal = average sign of 20/30/40/50/60-day returns (1-2 month momentum)
* optional confirmation: only take the ETH signal when BTC's signal agrees
  (BTC leads the crypto complex; the reverse filter does not work)
* position size = signal * (target_vol / realised 20-day vol), capped at 2x
* fees 5 bp per unit turnover, optional funding cost on |position|

Usage (from the package parent):
    py -m gp_eth.trend --btc ../cache_BTCUSDT_1d_3000.csv --start 2022-01-01
    py -m gp_eth.trend --vol 0.3 --long-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BARS_PER_YEAR, FEE, run_backtest

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

LOOKBACKS = (20, 30, 40, 50, 60)
VOL_TARGET = 0.40
MAX_LEV = 2.0
VOL_LOOKBACK = 20


# ------------------------------------------------------------------ signals
def tsmom(close: np.ndarray, lookbacks: tuple[int, ...] = LOOKBACKS) -> np.ndarray:
    """Average sign of past returns.  Causal: uses close up to and including t."""
    close = np.asarray(close, dtype=float)
    parts = []
    for n in lookbacks:
        s = np.zeros_like(close)
        s[n:] = np.sign(close[n:] / close[:-n] - 1.0)
        parts.append(s)
    return np.mean(parts, axis=0)


def realized_vol(ret: np.ndarray, n: int = VOL_LOOKBACK) -> np.ndarray:
    v = pd.Series(ret).rolling(n, min_periods=max(5, n // 2)).std(ddof=1).to_numpy()
    return v * np.sqrt(BARS_PER_YEAR)


def vol_scale(ret: np.ndarray, target: float, max_lev: float = MAX_LEV,
              n: int = VOL_LOOKBACK) -> np.ndarray:
    if target <= 0:
        return np.ones_like(ret)
    v = realized_vol(ret, n)
    scale = np.zeros_like(ret)
    ok = np.isfinite(v) & (v > 1e-6)
    scale[ok] = np.clip(target / v[ok], 0.0, max_lev)
    return scale


def target_position(df: pd.DataFrame, btc_df: pd.DataFrame | None = None,
                    lookbacks: tuple[int, ...] = LOOKBACKS,
                    vol_target: float = VOL_TARGET, max_lev: float = MAX_LEV,
                    long_only: bool = False) -> np.ndarray:
    """Fractional ETH position for every bar, decided at that bar's close."""
    close = df["close"].to_numpy(np.float64)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    raw = tsmom(close, lookbacks)
    if btc_df is not None:
        btc_close = btc_df["close"].to_numpy(np.float64)
        if len(btc_close) != len(close):
            raise ValueError("ETH and BTC frames must be aligned")
        braw = tsmom(btc_close, lookbacks)
        raw = np.where(np.sign(raw) == np.sign(braw), raw, 0.0)
    if long_only:
        raw = np.maximum(raw, 0.0)
    pos = raw * vol_scale(ret, vol_target, max_lev)
    # hard warmup: no position before the longest lookback and vol window
    warm = max(max(lookbacks), VOL_LOOKBACK)
    pos[:warm] = 0.0
    return pos


# ---------------------------------------------------------------- evaluation
def stats_of(close: np.ndarray, pos: np.ndarray, fee: float = FEE,
             funding: float = 0.0) -> dict:
    b = run_backtest(close, pos)
    s = b["strat"].copy()
    if fee != FEE:
        s -= (fee - FEE) * b["turnover"]
    if funding:
        s -= funding / BARS_PER_YEAR * np.abs(pos)
    eq = np.cumprod(1.0 + s)
    n = len(s)
    sd = float(s.std(ddof=1))
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))
    ann = float(eq[-1] ** (BARS_PER_YEAR / n) - 1.0)
    return {"total": float(eq[-1] - 1.0), "ann": ann,
            "sharpe": float(s.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0,
            "mdd": mdd, "calmar": ann / mdd if mdd > 1e-9 else 0.0,
            "turnover": float(b["turnover"].mean()),
            "exposure": float(np.mean(np.abs(pos))),
            "t_stat": (float(s.mean() / sd * np.sqrt(BARS_PER_YEAR))
                       * float(np.sqrt(n / BARS_PER_YEAR))) if sd > 1e-12 else 0.0,
            "days": n}


def per_year(df: pd.DataFrame, close: np.ndarray, pos: np.ndarray,
             fee: float, funding: float) -> dict:
    years = df["open_time"].dt.year.to_numpy()
    out = {}
    for y in sorted(set(years)):
        mask = years == y
        if mask.sum() < 20:
            continue
        out[str(y)] = stats_of(close[mask], pos[mask], fee, funding)
    return out


# ------------------------------------------------------------------- chart
def render_html(path: Path, dates: list[str], curves: dict[str, list[float]],
                title: str, subtitle: str, per_year: dict) -> None:
    W, H = 1040, 760
    ML, MR = 78, 28
    P1_TOP, P1_BOT = 96, 430
    P2_TOP, P2_BOT = 510, 640
    colors = {"trend": "#2563eb", "buy_hold": "#9ca3af"}
    labels = {"trend": "趋势策略", "buy_hold": "买入持有"}

    allv = [v for k in curves for v in curves[k]]
    vmin, vmax = min(allv), max(allv)
    pad = (vmax - vmin) * 0.06 or 0.1
    vmin, vmax = vmin - pad, vmax + pad
    dd = {}
    for k, eq in curves.items():
        peak, m, out = [], 0.0, []
        for v in eq:
            m = max(m, v)
            peak.append(m)
            out.append(1.0 - v / m)
        dd[k] = out
    dd_max = max(max(v) for v in dd.values()) * 1.08 or 0.1

    n = len(dates)

    def x(i: int) -> float:
        return ML + i / max(1, n - 1) * (W - ML - MR)

    def y1(v: float) -> float:
        return P1_BOT - (v - vmin) / (vmax - vmin) * (P1_BOT - P1_TOP)

    def y2(d: float) -> float:
        return P2_TOP + d / dd_max * (P2_BOT - P2_TOP)

    def poly(vals, yf) -> str:
        return " ".join(f"{x(i):.1f},{yf(v):.1f}" for i, v in enumerate(vals))

    p: list[str] = []
    add = p.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Segoe UI, system-ui, sans-serif">')
    add('<rect width="100%" height="100%" fill="#fff"/>')
    add(f'<text x="{ML}" y="30" font-size="20" font-weight="600" fill="#111827">{title}</text>')
    add(f'<text x="{ML}" y="52" font-size="12.5" fill="#4b5563">{subtitle}</text>')

    for frac in (0, .25, .5, .75, 1):
        v = vmin + frac * (vmax - vmin)
        yy = y1(v)
        add(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        add(f'<text x="{ML-10}" y="{yy+4:.1f}" font-size="11" fill="#6b7280" '
            f'text-anchor="end">{v:.2f}</text>')
    add(f'<text x="{ML}" y="{P1_TOP-16}" font-size="12" font-weight="600" fill="#374151">'
        '净值（初始 = 1.00）</text>')

    for frac in (0, .2, .4, .6, .8, 1):
        d = frac * dd_max
        yy = y2(d)
        add(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        add(f'<text x="{ML-10}" y="{yy+4:.1f}" font-size="11" fill="#6b7280" '
            f'text-anchor="end">-{d:.0%}</text>')
    add(f'<text x="{ML}" y="{P2_TOP-16}" font-size="12" font-weight="600" fill="#374151">'
        '回撤</text>')

    for i, d in enumerate(dates):
        if d[5:] == "01-01":
            add(f'<line x1="{x(i):.1f}" y1="{P1_TOP}" x2="{x(i):.1f}" y2="{P2_BOT}" '
                'stroke="#f3f4f6"/>')
            add(f'<text x="{x(i):.1f}" y="{P2_BOT+22}" font-size="11" fill="#6b7280" '
                f'text-anchor="middle">{d[:4]}</text>')

    for k in ("buy_hold", "trend"):
        width = 2.4 if k == "trend" else 1.3
        dash = "" if k == "trend" else ' stroke-dasharray="5 3"'
        add(f'<polyline points="{poly(curves[k], y1)}" fill="none" stroke="{colors[k]}" '
            f'stroke-width="{width}"{dash} stroke-linejoin="round"/>')
    for k in ("buy_hold", "trend"):
        add(f'<polyline points="{poly(dd[k], y2)}" fill="none" stroke="{colors[k]}" '
            f'stroke-width="{2.2 if k == "trend" else 1.2}" stroke-linejoin="round"/>')

    for i, k in enumerate(("trend", "buy_hold")):
        lx = ML + 220 + i * 190
        add(f'<line x1="{lx}" y1="76" x2="{lx+22}" y2="76" stroke="{colors[k]}" stroke-width="3"/>')
        add(f'<text x="{lx+30}" y="80" font-size="11.5" fill="#374151">{labels[k]}</text>')

    ty = 690
    add(f'<text x="{ML}" y="{ty-14}" font-size="12" font-weight="600" fill="#374151">分年表现</text>')
    for head, dx in (("年份", 0), ("策略", 80), ("Sharpe", 180), ("回撤", 260),
                     ("买入持有", 350)):
        add(f'<text x="{ML+dx}" y="{ty}" font-size="11.5" font-weight="600" fill="#4b5563">{head}</text>')
    for i, (yr, rec) in enumerate(per_year.items()):
        yy = ty + 20 + i * 18
        if i % 2 == 0:
            add(f'<rect x="{ML-6}" y="{yy-13}" width="{W-ML-MR+6}" height="18" fill="#f9fafb"/>')
        for val, dx, color in ((yr + ("*" if yr == "2026" else ""), 0, "#111827"),
                               (f'{rec["total"]:+.1%}', 80, "#2563eb"),
                               (f'{rec["sharpe"]:.2f}', 180, "#111827"),
                               (f'{rec["mdd"]:.1%}', 260, "#111827")):
            add(f'<text x="{ML+dx}" y="{yy}" font-size="11.5" fill="{color}">{val}</text>')
    add(f'<text x="{ML}" y="{H-10}" font-size="10.5" fill="#9ca3af">'
        '* 2026 为 1-10 月部分年份。5 bp/边手续费，不含资金费率。</text>')
    add("</svg>")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>gp_eth trend</title>"
            "</head><body style='margin:0;background:#fff'>" + "".join(p) + "</body></html>")
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parent / "cache_ETHUSDT_1d_3000.csv"))
    ap.add_argument("--btc", default=None, help="aligned BTC daily CSV for confirmation")
    ap.add_argument("--start", default=None)
    ap.add_argument("--vol", type=float, default=VOL_TARGET)
    ap.add_argument("--max-lev", type=float, default=MAX_LEV)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--funding", type=float, default=0.0)
    ap.add_argument("--out", default=str(OUT / "trend_results.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    btc = None
    if args.btc:
        btc = pd.read_csv(args.btc)
        btc["open_time"] = pd.to_datetime(btc["open_time"])
        btc = btc.sort_values("open_time").reset_index(drop=True)

    close = df["close"].to_numpy(np.float64)
    pos = target_position(df, btc, vol_target=args.vol, max_lev=args.max_lev,
                          long_only=args.long_only)
    lo = 0
    if args.start:
        lo = int(np.searchsorted(df["open_time"].to_numpy(), np.datetime64(args.start)))
    sl = slice(lo, None)
    m = stats_of(close[sl], pos[sl], args.fee, args.funding)
    bh = stats_of(close[sl], np.ones(len(close))[sl], args.fee, args.funding)
    years = per_year(df.iloc[lo:].reset_index(drop=True), close[sl], pos[sl],
                     args.fee, args.funding)

    print(f"ETH trend {LOOKBACKS} vol={args.vol:.0%} max_lev={args.max_lev:g} "
          f"confirm={'BTC' if btc is not None else 'off'} "
          f"long_only={args.long_only} fee={args.fee:.2%} funding={args.funding:.0%}")
    print(f"  total={m['total']:+8.1%} ann={m['ann']:+7.1%} sharpe={m['sharpe']:5.2f} "
          f"mdd={m['mdd']:6.1%} calmar={m['calmar']:5.2f} t={m['t_stat']:4.2f} "
          f"exposure={m['exposure']:.2f}")
    print(f"  B&H   total={bh['total']:+8.1%} ann={bh['ann']:+7.1%} "
          f"sharpe={bh['sharpe']:5.2f} mdd={bh['mdd']:6.1%}")
    print("  per year:")
    for y, r in years.items():
        print(f"    {y} total={r['total']:+8.1%} ann={r['ann']:+7.1%} "
              f"sharpe={r['sharpe']:5.2f} mdd={r['mdd']:6.1%}")

    eq = np.cumprod(1.0 + run_backtest(close[sl], pos[sl])["strat"])
    eq_bh = np.cumprod(1.0 + run_backtest(close[sl], np.ones(len(close))[sl])["strat"])
    dates = df["open_time"].iloc[lo:].dt.strftime("%Y-%m-%d").tolist()
    payload = {"params": {"lookbacks": list(LOOKBACKS), "vol_target": args.vol,
                          "max_lev": args.max_lev, "long_only": args.long_only,
                          "confirm": "btc" if btc is not None else None,
                          "fee": args.fee, "funding": args.funding,
                          "start": args.start},
               "stats": m, "buy_hold": bh, "per_year": years,
               "series": {"dates": dates, "trend": eq.round(6).tolist(),
                          "buy_hold": eq_bh.round(6).tolist()}}
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    html = out.with_name("trend_equity.html")
    sub = (f"TSMOM {'/'.join(map(str, LOOKBACKS))} · vol target {args.vol:.0%} · "
           f"max {args.max_lev:g}x · BTC confirm {'on' if btc is not None else 'off'} ｜ "
           f"年化 {m['ann']:+.1%} · MDD {m['mdd']:.1%} · Sharpe {m['sharpe']:.2f}")
    render_html(html, dates, {"trend": eq.tolist(), "buy_hold": eq_bh.tolist()},
                "gp_eth 趋势策略样本外净值", sub, years)
    print(f"written: {out}\nwritten: {html}")


if __name__ == "__main__":
    main()
