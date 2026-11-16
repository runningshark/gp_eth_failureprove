"""横盘突破（Range Breakout）策略

思路
----
在横盘（窄幅震荡）区域内，价格一旦向上突破区间高点 -> 做多；向下突破区间低点 -> 做空。
既做多也做空，符合"横盘向上突破做多，横盘向下突破做空"的要求。

规则（全因果，收盘决定，下一根K线执行）：
* 定义横盘区间：过去 N 根 K 线的最高价、最低价（用 shift(1) 避免当前K线泄露）
* 判断是否"横盘"：区间高度 / ATR(14) < threshold（区间相对波动率收窄）
* 突破确认：收盘价 > 区间高点 * (1 + breakout_buffer) -> 做多；收盘价 < 区间低点 * (1 - breakout_buffer) -> 做空
* 持有：突破后一直持有，直到收盘回到相反边界（多头 close < 区间低点 / 空头 close > 区间高点）或均线过滤失效
* 可选：成交量放大（vol > vol_ma * vol_mult）
* 趋势过滤（可选）：BTC/ETH 均线过滤避免极端逆势
* 仓位：1.5 / 实现波动率（240根）目标，最大杠杆 6x，带 band
* 成本：5bp/边，10%/年资金费，交叉保证金强平建模（同 aggressive/fast4h）

用法（先把目录命名为 gp_eth，再从父目录运行）：
    py -m gp_eth.range_breakout
    py -m gp_eth.range_breakout --range 24 --thresh 2.5 --long-short
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggressive import BARS_PER_YEAR, FUNDING, backtest, load, per_year, stats_of, atr
from .fast4h import sma
from .trend import render_html

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# 默认参数
RANGE_N = 24        # 4天横盘区间（24根4h）
ATR_N = 14
THRESH = 2.5        # 区间高度/ATR < 2.5 视为横盘。
                    # 实测该比值最小 2.11、中位数 4.96，原默认 0.7 在 2 万根 4h 上
                    # 满足率 0.0000%，因此原版默认参数下 0 笔交易。即使改成 2.5（约 1%
                    # 分位），2017–2026 也只有 ~12 笔；见 FAILURE_REPORT.md。
BREAKOUT_BUF = 0.001 # 0.1% 突破缓冲
VOL_MA_N = 20
VOL_MULT = 1.0      # 设为 0 则关闭量能确认
MA_LEN = 96         # 趋势过滤
BTC_MA_LEN = 96
VOL_TARGET = 1.5
MAX_LEV = 6.0
VOL_LOOKBACK = 240
BAND = 0.4
FEE = 0.0005


def raw_signal_rb(eth: pd.DataFrame, btc: pd.DataFrame | None = None,
                  range_n: int = RANGE_N, atr_n: int = ATR_N,
                  thresh: float = THRESH, buf: float = BREAKOUT_BUF,
                  vol_ma_n: int = VOL_MA_N, vol_mult: float = VOL_MULT,
                  ma_len: int = MA_LEN, btc_ma_len: int = BTC_MA_LEN,
                  long_short: bool = True) -> np.ndarray:
    c = eth["close"].to_numpy(np.float64)
    h = eth["high"].to_numpy(np.float64)
    l = eth["low"].to_numpy(np.float64)
    v = eth["volume"].to_numpy(np.float64)
    n = len(c)
    
    # 区间（用shift避免当前bar泄露）
    rh = pd.Series(h).rolling(range_n).max().shift(1).to_numpy()
    rl = pd.Series(l).rolling(range_n).min().shift(1).to_numpy()
    rr = (rh - rl) / np.maximum(c, 1e-12)
    
    # ATR
    atr_arr = atr(eth, atr_n)
    atr_r = atr_arr / np.maximum(c, 1e-12)
    
    # 是否横盘
    cons = rr < thresh * np.maximum(atr_r, 1e-12)
    
    # 成交量确认
    if vol_mult > 0 and vol_ma_n > 0:
        vm = pd.Series(v).rolling(vol_ma_n).mean().shift(1).to_numpy()
        vconf = v > vm * vol_mult
    else:
        vconf = np.ones(n, dtype=bool)
    
    raw = np.zeros(n)
    # 突破
    up_break = cons & vconf & (c > rh * (1.0 + buf))
    down_break = cons & vconf & (c < rl * (1.0 - buf))

    ma = sma(c, ma_len) if ma_len and ma_len > 0 else None
    if btc is not None and btc_ma_len and btc_ma_len > 0 and len(btc) == n:
        bclose = btc["close"].to_numpy(np.float64)
        bma = sma(bclose, btc_ma_len)
    else:
        bclose = bma = None

    # 持有逻辑（修复版）：突破后一直持有，直到
    #   (a) 收盘跌回相反的区间边界（多头: close < 区间低点；空头: close > 区间高点），
    #   (b) 均线过滤失效，或 (c) BTC 确认失效。
    # 原版只在"突破那一根"给出一帧信号，backtest 下一根就把仓位平掉，
    # 等于持仓只有一根 4h，根本没有"趋势"，这是它几乎不赚钱的直接原因。
    state = 0.0
    for t in range(n):
        if state > 0:
            if (rl[t] == rl[t] and c[t] < rl[t]) \
               or (ma is not None and not (c[t] > ma[t])) \
               or (bma is not None and not (bclose[t] > bma[t])):
                state = 0.0
        elif state < 0:
            if (rh[t] == rh[t] and c[t] > rh[t]) \
               or (ma is not None and not (c[t] < ma[t])) \
               or (bma is not None and not (bclose[t] < bma[t])):
                state = 0.0

        if state == 0.0:
            if up_break[t] and (ma is None or c[t] > ma[t]) and (bma is None or bclose[t] > bma[t]):
                state = 1.0
            elif long_short and down_break[t] and (ma is None or c[t] < ma[t]) and (bma is None or bclose[t] < bma[t]):
                state = -1.0
        raw[t] = state

    # warmup
    warm = max(range_n, atr_n, vol_ma_n, ma_len, btc_ma_len)
    raw[:warm] = 0.0
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "eth_4h.csv"))
    ap.add_argument("--btc", default=str(HERE / "btc_4h.csv"))
    ap.add_argument("--start", default="2017-09-01")
    ap.add_argument("--range", type=int, default=RANGE_N)
    ap.add_argument("--atr-n", type=int, default=ATR_N)
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--buf", type=float, default=BREAKOUT_BUF)
    ap.add_argument("--vol-ma", type=int, default=VOL_MA_N)
    ap.add_argument("--vol-mult", type=float, default=VOL_MULT)
    ap.add_argument("--ma", type=int, default=MA_LEN)
    ap.add_argument("--btc-ma", type=int, default=BTC_MA_LEN)
    ap.add_argument("--long-short", action="store_true", default=True)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--vol", type=float, default=VOL_TARGET)
    ap.add_argument("--cap", type=float, default=MAX_LEV)
    ap.add_argument("--band", type=float, default=BAND)
    ap.add_argument("--vol-lookback", type=int, default=VOL_LOOKBACK)
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--funding", type=float, default=FUNDING)
    ap.add_argument("--out", default=str(OUT / "range_breakout_results.json"))
    args = ap.parse_args()
    
    if args.long_only:
        args.long_short = False
    
    df = load(args.data)
    btc = load(args.btc) if args.btc else None
    raw = raw_signal_rb(df, btc, range_n=args.range, atr_n=args.atr_n,
                        thresh=args.thresh, buf=args.buf,
                        vol_ma_n=args.vol_ma, vol_mult=args.vol_mult,
                        ma_len=args.ma, btc_ma_len=args.btc_ma,
                        long_short=args.long_short)
    
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
    
    # cost stress
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
    
    side = "多空双向" if args.long_short else "仅做多"
    print(f"ETH 4h 横盘突破 | 区间={args.range}根 | ATR{args.atr_n} | 阈值={args.thresh} | 缓冲={args.buf:.3%} | {side}")
    print(f"MA{args.ma} BTC-MA{args.btc_ma} vol确认={args.vol_mult} vol={args.vol:.0%} cap={args.cap:g}x band={args.band:.0%}")
    print(f"  total={m['total']:+12.1%} ann={m['ann']:+8.1%} sharpe={m['sharpe']:5.2f} mdd={m['mdd']:6.1%} calmar={m['calmar']:5.2f} liq={m['liq']}")
    print(f"  B&H   total={bh['total']:+12.1%} ann={bh['ann']:+8.1%} sharpe={bh['sharpe']:5.2f} mdd={bh['mdd']:6.1%}")
    print(f"  exposure avg={np.mean(np.abs(pos)):.2f} p95={np.percentile(np.abs(pos),95):.2f} max={np.max(np.abs(pos)):.2f} closest margin={r['worst_margin']:.0f}x")
    print(f"  max dd {m['mdd']:.1%} on {d['open_time'].iloc[i_dd]:%Y-%m-%d}; equity now {eq[-1]/peak[-1]-1:+.1%} vs peak")
    print("  cost stress (full / 2022+):")
    yrs = d["open_time"].dt.year.to_numpy()
    for k, ms in stress.items():
        mask22 = yrs >= 2022
        e22 = stress_eq[k][mask22]
        s22 = stats_of(e22/e22[0]) if len(e22)>10 and e22[0]>0 else {"ann":-1,"mdd":1,"sharpe":0}
        print(f"    {k:>6s} ann={ms['ann']:+8.1%} mdd={ms['mdd']:6.1%} sharpe={ms['sharpe']:5.2f} | 22+ {s22['ann']:+8.1%}/{s22['mdd']:6.1%}")
    print("  per year:")
    for y, rec in years.items():
        print(f"    {y} total={rec['total']:+10.1%} ann={rec['ann']:+8.1%} sharpe={rec['sharpe']:5.2f} mdd={rec['mdd']:6.1%}")
    
    payload = {
        "params": {"range_n": args.range, "atr_n": args.atr_n, "thresh": args.thresh,
                   "buf": args.buf, "vol_ma": args.vol_ma, "vol_mult": args.vol_mult,
                   "ma_len": args.ma, "btc_ma_len": args.btc_ma, "long_short": args.long_short,
                   "vol_target": args.vol, "max_lev": args.cap, "band": args.band,
                   "vol_lookback": args.vol_lookback, "fee": args.fee, "funding": args.funding},
        "stats": m, "buy_hold": bh, "cost_stress": stress, "per_year": years,
        "risk": {"worst_margin_multiple": r["worst_margin"], "avg_exposure": float(np.mean(np.abs(pos))),
                 "p95_exposure": float(np.percentile(np.abs(pos),95)), "max_exposure": float(np.max(np.abs(pos))),
                 "max_dd_date": f"{d['open_time'].iloc[i_dd]:%Y-%m-%d}", "current_dd_from_peak": float(eq[-1]/peak[-1]-1.0)},
        "series": {"dates": d["open_time"].dt.strftime("%Y-%m-%d").tolist(),
                   "equity": np.round(eq,6).tolist(),
                   "buy_hold": np.round(d["close"].to_numpy(np.float64)/d["close"].iloc[0],6).tolist(),
                   "position": np.round(pos,6).tolist()},
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    
    html = out.with_name("range_breakout_equity.html")
    sub = f"4h 横盘突破（区间{args.range}根） | {side} | vol {args.vol:.0%} cap {args.cap:g}x — 年化 {m['ann']:+.1%} · MDD {m['mdd']:.1%} · Sharpe {m['sharpe']:.2f}"
    render_html(html, payload["series"]["dates"], {"trend": eq.tolist(), "buy_hold": payload["series"]["buy_hold"]},
                "gp_eth 4h 横盘突破策略", sub, years,
                note="* 2026为1-10月部分年份。5bp/边手续费+10%/年资金费，含强平建模。", labels={"trend": "横盘突破", "buy_hold": "买入持有"})
    print(f"written: {out}\nwritten: {html}")


if __name__ == "__main__":
    main()
