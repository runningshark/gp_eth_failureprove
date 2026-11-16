# -*- coding: utf-8 -*-
"""04 — 横盘突破：为什么默认参数是 0 笔交易，修好后又有多差?

三件事：
  1) 打印「24 根区间高度 / ATR14」的分位数，证明默认 thresh=0.7 的满足率是 0.0000%
  2) 打印原存档参数（thresh=10.0）为什么等价于「没有横盘过滤」
  3) 扫 21 个参数点，看修复后的真实表现

用法（目录必须命名为 gp_eth，在仓库根目录运行）：
    py verify/04_range_breakout_grid.py
"""
import statistics

import numpy as np
import pandas as pd

import _bootstrap  # noqa: E402,F401

from gp_eth.aggressive import atr, backtest, load, stats_of
from gp_eth.range_breakout import raw_signal_rb

ROOT = _bootstrap.ROOT


def main():
    eth = load(ROOT / "eth_4h.csv")
    btc = load(ROOT / "btc_4h.csv")
    c = eth["close"].to_numpy(float)
    h = eth["high"].to_numpy(float)
    lo = eth["low"].to_numpy(float)

    # 1) 分位数
    rh = pd.Series(h).rolling(24).max().shift(1).to_numpy()
    rl = pd.Series(lo).rolling(24).min().shift(1).to_numpy()
    ratio = ((rh - rl) / c) / (atr(eth, 14) / c)
    ratio = ratio[np.isfinite(ratio)]
    print("24 根区间高度 / ATR14 分位数：")
    print("  " + "  ".join(f"p{p}={np.percentile(ratio, p):.2f}"
                           for p in (0, 1, 5, 25, 50, 75, 95, 100)))
    print(f"  默认 thresh=0.7 的满足比例：{(ratio < 0.7).mean() * 100:.6f}%  → 默认 0 笔交易")
    print(f"  存档参数 thresh=10.0 的满足比例：{(ratio < 10.0).mean() * 100:.2f}%  "
          f"→ 「横盘」过滤恒为真\n")

    # 2) 原始默认参数（0.7）到底几笔
    raw0 = raw_signal_rb(eth, btc, range_n=24, thresh=0.7)
    print(f"原默认参数 (thresh=0.7) 非零信号数：{int(np.count_nonzero(raw0))}\n")

    # 3) 21 点参数面
    print(f"{'thresh':>6s} | {'range':>5s} | {'笔数':>6s} | {'全样本年化':>9s} | {'Sharpe':>6s} "
          f"| {'MDD':>7s} | {'敞口':>5s} | {'2022+':>8s}")
    print("-" * 78)
    rows = []
    for rn in (12, 24, 42):
        for th in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            raw = raw_signal_rb(eth, btc, range_n=rn, thresh=th)
            r = backtest(eth, raw, 1.5, 6.0, 0.4, 240, fee=0.0005, funding=0.10)
            m = stats_of(r["equity"])
            pos = r["position"]
            nz = np.abs(pos) > 1e-9
            trades = int(np.sum(nz[1:] & ~nz[:-1]))
            yrs = eth["open_time"].dt.year.to_numpy()
            e22 = r["equity"][yrs >= 2022]
            s22 = stats_of(e22 / e22[0]) if len(e22) > 10 and e22[0] > 0 else {"ann": 0.0}
            rows.append((th, rn, trades, m["ann"], m["sharpe"], m["mdd"],
                         float(np.mean(np.abs(pos))), s22["ann"]))
            print(f"{th:6.1f} | {rn:5d} | {trades:6d} | {m['ann']:+8.1%} | {m['sharpe']:6.2f} "
                  f"| {m['mdd']:7.1%} | {np.mean(np.abs(pos)):5.2f} | {s22['ann']:+8.1%}")

    neg22 = sum(1 for r in rows if r[7] < 0)
    flat22 = sum(1 for r in rows if r[7] == 0)
    print(f"\n21 个参数点：年化中位数 {statistics.median(r[3] for r in rows):+.1%}，"
          f"MDD 中位数 {statistics.median(r[5] for r in rows):.1%}，"
          f"2022+ 为负 {neg22} 个 / 为 0 {flat22} 个 / 为正 {len(rows) - neg22 - flat22} 个")
    print("结论：任何能产生有意义交易笔数的阈值都不再是「横盘」过滤；"
          "修好后仍是 90% 级别回撤。")


if __name__ == "__main__":
    main()
