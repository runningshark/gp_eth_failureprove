# -*- coding: utf-8 -*-
"""03 — 趋势跟踪到底是不是 ETH/BTC 的运气?

REPORT §5.8 只验证了 ETH 和 BTC，却写「2018–2026 每一年都正」。
这里从 Gate.io 拉 16 个主流 USDT 日线，用 trend.py 的原函数跑同样规则，
报告横截面分布 + 等权组合 + 剔除 2026 后的结果。

结论：edge 是广泛存在的，但量级是 年化 14–17% / 回撤 30% / t≈1.5，
      不是单币 ETH 的 +28.3%。

用法（目录必须命名为 gp_eth，在仓库根目录运行；首次会联网拉数据）：
    py verify/03_multiasset_tsmom.py
"""
import time

import numpy as np
import pandas as pd
import requests

import _bootstrap  # noqa: E402,F401

from gp_eth.backtest import run_backtest
from gp_eth.trend import LOOKBACKS, MAX_LEV, VOL_TARGET, stats_of, tsmom, vol_scale

ROOT = _bootstrap.ROOT

CACHE = ROOT / "verify" / ".cache" / "gate_daily"
CACHE.mkdir(parents=True, exist_ok=True)

PAIRS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT",
         "DOGE_USDT", "AVAX_USDT", "LINK_USDT", "LTC_USDT", "DOT_USDT",
         "TRX_USDT", "ATOM_USDT", "NEAR_USDT", "BCH_USDT", "ETC_USDT"]
START = "2022-01-01"


def fetch(pair: str) -> pd.DataFrame | None:
    path = CACHE / f"{pair}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["open_time"])
        if len(df) > 500:
            return df

    rows, to = [], None
    for _ in range(12):                     # 12000 根上限
        params = {"currency_pair": pair, "interval": "1d", "limit": 1000}
        if to is not None:
            params["to"] = to
        r = requests.get("https://api.gateio.ws/api/v4/spot/candlesticks",
                         params=params, timeout=30)
        if r.status_code != 200:
            print(f"  {pair}: HTTP {r.status_code}")
            return None
        batch = r.json()
        if not batch:
            break
        rows = batch + rows
        if len(batch) < 1000:
            break
        to = int(batch[0][0]) - 1
        time.sleep(0.25)

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "quote_vol", "close", "high", "low",
                                     "open", "base_vol", "closed"])
    df["open_time"] = pd.to_datetime(df["ts"].astype(np.int64), unit="s")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df = (df.drop_duplicates("open_time").sort_values("open_time")
            .query("closed == 'true'").reset_index(drop=True))
    df[["open_time", "open", "high", "low", "close"]].to_csv(path, index=False)
    return df[["open_time", "open", "high", "low", "close"]]


def target_pos(df, btc_df=None):
    close = df["close"].to_numpy(float)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    raw = tsmom(close, LOOKBACKS)
    if btc_df is not None:
        braw = tsmom(btc_df["close"].to_numpy(float), LOOKBACKS)
        n = min(len(braw), len(close))
        mask = np.sign(raw[:n]) == np.sign(braw[:n])
        raw = np.concatenate([np.where(mask, raw[:n], 0.0), np.zeros(len(close) - n)])
    pos = raw * vol_scale(ret, VOL_TARGET, MAX_LEV)
    pos[:max(max(LOOKBACKS), 20)] = 0.0
    return pos


def bh(df):
    c = df["close"].to_numpy(float)
    n = len(c)
    r = c[1:] / c[:-1] - 1.0
    eq = c / c[0]
    return ((c[-1] / c[0]) ** (365.0 / n) - 1.0,
            float(np.max(1 - eq / np.maximum.accumulate(eq))),
            r.mean() / r.std(ddof=1) * np.sqrt(365))


def main():
    data = {}
    for p in PAIRS:
        try:
            d = fetch(p)
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {p}: {e}")
            continue
        if d is None or len(d) < 400:
            print(f"  skip {p}: 数据不足")
            continue
        data[p] = d
        print(f"  {p:10s} {len(d):5d} bars  {d.open_time.iloc[0].date()} -> {d.open_time.iloc[-1].date()}")

    btc = data.get("BTC_USDT")
    if btc is None:
        raise SystemExit("拿不到 BTC 数据")

    rows = []
    print(f"\n=== 全样本（各自起始日） 20/30/40/50/60 TSMOM, 40% vol, 2x cap, 5bp, BTC确认 ===")
    print(f"{'pair':10s} {'start':10s} {'ann':>8s} {'sharpe':>7s} {'mdd':>7s} "
          f"{'t':>6s} | {'BH ann':>8s} {'BH mdd':>7s}")
    for p, d in data.items():
        conf = None
        if p != "BTC_USDT":
            conf = btc.iloc[-len(d):].reset_index(drop=True) if len(btc) > len(d) else btc
        pos = target_pos(d, conf)
        s = stats_of(d["close"].to_numpy(float), pos)
        a, m, _ = bh(d)
        rows.append((p, s["ann"], s["sharpe"], s["mdd"], a))
        print(f"{p:10s} {str(d.open_time.iloc[0].date()):10s} {s['ann']:+7.1%} "
              f"{s['sharpe']:7.2f} {s['mdd']:7.1%} {s['t_stat']:6.2f} | {a:+7.1%} {m:7.1%}")

    anns = [r[1] for r in rows]
    print(f"\n横截面: 年化中位数 {np.median(anns):+.1%}, 正年化 {sum(a > 0 for a in anns)}/{len(rows)}, "
          f"跑赢 BH {sum(r[1] > r[4] for r in rows)}/{len(rows)}")

    # ---------------------------------------------------------------- 2022+
    print(f"\n=== {START} 起（与 REPORT §5.8 同区间）===")
    print(f"{'pair':10s} {'ann':>8s} {'sharpe':>7s} {'mdd':>7s} {'t':>6s} "
          f"| {'BH ann':>8s} | {'fund10%':>8s}")
    r22, st22 = [], {}
    for p, d in data.items():
        dd = d[d.open_time >= START].reset_index(drop=True)
        if len(dd) < 200:
            continue
        conf = None if p == "BTC_USDT" else btc[btc.open_time >= START].reset_index(drop=True)
        pos = target_pos(dd, conf)
        s = stats_of(dd["close"].to_numpy(float), pos)
        sf = stats_of(dd["close"].to_numpy(float), pos, funding=0.10)
        st22[p] = run_backtest(dd["close"].to_numpy(float), pos)["strat"]
        a, _, _ = bh(dd)
        r22.append((p, s["ann"], s["sharpe"], s["mdd"], a, sf["ann"], s["t_stat"]))
        print(f"{p:10s} {s['ann']:+7.1%} {s['sharpe']:7.2f} {s['mdd']:7.1%} {s['t_stat']:6.2f} "
              f"| {a:+7.1%} | {sf['ann']:+7.1%}")

    a22 = [r[1] for r in r22]
    print(f"\n横截面(2022+): 年化中位数 {np.median(a22):+.1%}, 正年化 {sum(a > 0 for a in a22)}/{len(r22)}, "
          f"Sharpe 中位数 {np.median([r[2] for r in r22]):.2f}, "
          f"加 10%/yr funding 后正年化 {sum(r[5] > 0 for r in r22)}/{len(r22)}, "
          f"t>2 的个数 {sum(1 for r in r22 if r[6] > 2)}")

    # ------------------------------------------------------------- 等权组合
    names = list(st22)
    R = np.vstack([st22[p] for p in names])
    port = R.mean(axis=0)
    dates = data[names[0]][data[names[0]].open_time >= START].reset_index(drop=True).open_time
    n = len(port)

    def stats(x):
        n = len(x)
        eq = np.cumprod(1 + x)
        sd = x.std(ddof=1)
        ann = eq[-1] ** (365.0 / n) - 1
        sh = x.mean() / sd * np.sqrt(365)
        mdd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
        return ann, sh, mdd, sh * np.sqrt(n / 365)

    print(f"\n=== 等权组合（{len(names)} 个品种）===")
    print(f"{'窗口':20s} {'ann':>8s} {'sharpe':>7s} {'mdd':>7s} {'t':>6s}")
    for label, mask in (("2022-2025 (ex-2026)", (dates.dt.year <= 2025).to_numpy()),
                        ("2022-2026 全部", np.ones(n, bool)),
                        ("仅 2026", (dates.dt.year == 2026).to_numpy())):
        a, s, m, t = stats(port[mask])
        print(f"{label:20s} {a:+7.1%} {s:7.2f} {m:7.1%} {t:6.2f}")

    cm = np.corrcoef(R)
    iu = np.triu_indices(len(names), 1)
    print(f"\n组合年化波动 {port.std(ddof=1) * np.sqrt(365):.1%}, "
          f"单品平均 {np.mean([R[i].std(ddof=1) * np.sqrt(365) for i in range(len(names))]):.1%}, "
          f"两两相关均值 {cm[iu].mean():.2f}")

    print("\n结论：趋势 edge 在 16 个品种上广泛为正，但年化中位数只有 +17%、t≈1.5，"
          "没有任何单币达到 t>2；组合把回撤压到 30%。")


if __name__ == "__main__":
    main()
