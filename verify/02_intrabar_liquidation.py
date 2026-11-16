# -*- coding: utf-8 -*-
"""02 — 4h 杠杆回测的「无强平」是不是收盘价作弊?

原版 aggressive.backtest 只用收盘价判断强平。这里用**盘中最高/最低价**重做一遍：
若盘中权益跌破维持保证金，账户立即归零。

结论：不会强平。最差保证金仍有 47–92 倍。REPORT §5.9/5.10 的「无强平」成立。

用法（目录必须命名为 gp_eth，在仓库根目录运行）：
    py verify/02_intrabar_liquidation.py
"""
import sys

import numpy as np

import _bootstrap  # noqa: E402,F401

from gp_eth.aggressive import (BARS_PER_YEAR, FEE, FUNDING, MAINT_MARGIN,
                               load, raw_signal, vol_scale)
from gp_eth.fast4h import raw_signal as fast_signal
from gp_eth.range_breakout import raw_signal_rb

ROOT = _bootstrap.ROOT


def backtest_intrabar(df, raw, vol_target, max_lev, band, vol_n=60,
                      fee=FEE, funding=FUNDING, maint=MAINT_MARGIN):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    scale = vol_scale(c, vol_target, max_lev, vol_n)

    eq = np.ones(n)
    pos = 0.0
    liq = -1
    worst = np.inf

    for t in range(1, n):
        pnl = pos * (c[t] / c[t - 1] - 1.0) if pos != 0 else 0.0

        # ---- 盘中极值下的权益 ----
        if pos > 0:
            extreme = pos * (l[t] / c[t - 1] - 1.0)
        elif pos < 0:
            extreme = pos * (h[t] / c[t - 1] - 1.0)
        else:
            extreme = 0.0
        if pos != 0 and 1.0 + extreme <= maint * abs(pos):
            eq[t:] = 0.0
            return {"equity": eq, "liq": t, "worst": 0.0}

        step = 1.0 + pnl - funding / BARS_PER_YEAR * abs(pos)
        if step <= maint * abs(pos):
            eq[t:] = 0.0
            return {"equity": eq, "liq": t, "worst": 0.0}

        eq[t] = eq[t - 1] * step
        if abs(pos) > 1e-9:
            worst = min(worst, eq[t] / (abs(pos) * maint))

        s = np.sign(raw[t])
        target = s * scale[t]
        if target != 0.0 and pos != 0.0 and np.sign(target) == np.sign(pos):
            if abs(target - pos) < band * abs(pos):
                continue
        eq[t] *= 1.0 - abs(target - pos) * fee
        pos = target

    return {"equity": eq, "liq": liq, "worst": float(worst)}


def stats(eq):
    s = np.zeros(len(eq))
    s[1:] = eq[1:] / eq[:-1] - 1.0
    sd = s.std(ddof=1)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1 - eq / np.maximum(peak, 1e-12)))
    ann = eq[-1] ** (BARS_PER_YEAR / len(eq)) - 1 if eq[-1] > 0 else -1.0
    sharpe = s.mean() / sd * np.sqrt(BARS_PER_YEAR) if sd > 1e-12 else 0.0
    return ann, mdd, sharpe


def main():
    eth = load(ROOT / "eth_4h.csv")
    btc = load(ROOT / "btc_4h.csv")

    cases = [
        ("长线 vol1.5 cap10 band20%", raw_signal(eth, btc), 1.5, 10.0, 0.20, 60),
        ("长线 vol1.2 cap10 band20%", raw_signal(eth, btc), 1.2, 10.0, 0.20, 60),
        ("短线 vol1.5 cap4 band40%", fast_signal(eth, btc), 1.5, 4.0, 0.40, 240),
        ("短线 vol2.0 cap4 band40%", fast_signal(eth, btc), 2.0, 4.0, 0.40, 240),
        ("横盘突破 vol1.5 cap6", raw_signal_rb(eth, btc), 1.5, 6.0, 0.40, 240),
    ]
    print(f"{'case':26s} {'盘中强平':8s} {'年化':>8s} {'MDD':>7s} {'Sharpe':>7s} {'最差保证金':>8s}")
    ok = True
    for name, raw, vt, cap, band, vn in cases:
        r = backtest_intrabar(eth, raw, vt, cap, band, vn)
        ann, mdd, sh = stats(r["equity"])
        if abs(raw).sum() == 0:
            print(f"{name:26s} {'n/a':8s} {'无信号':>8s}")
            continue
        ok &= r["liq"] < 0
        print(f"{name:26s} {'无' if r['liq'] < 0 else '有':8s} {ann:+7.1%} {mdd:7.1%} "
              f"{sh:7.2f} {r['worst']:7.0f}x")

    # 成本压力：同一长线配置按 10bp/边 重跑（FAILURE_REPORT.md §4 的 +183%）
    r10 = backtest_intrabar(eth, raw_signal(eth, btc), 1.5, 10.0, 0.20, 60, fee=0.001)
    a10, m10, s10 = stats(r10["equity"])
    print(f"\n成本压力（长线 vol1.5 cap10，10bp/边）：ann {a10:+.1%}  mdd {m10:.1%}  sharpe {s10:.2f}")

    print("\n结论：" + ("所有配置盘中都不强平，'无强平' 是真的。" if ok
                       else "有配置在盘中被打穿，报告结论不成立。"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
