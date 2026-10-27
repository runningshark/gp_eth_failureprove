"""Regression tests for gp_eth.  Run from the Desktop:  py -m gp_eth.tests

These are deliberately cheap and target the bug classes that matter for a
trading tool: look-ahead, cost accounting, warmup, and live/backtest parity.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .backtest import (BARS_PER_YEAR, apply_risk_overlay, run_backtest,
                       summarize)
from .export_pine import to_pine
from .individual import to_position, tree_from_json
from .live_runner import overlay_step
from .primitives import _ema, _rsi, build_context


def _df(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.03, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open_time": pd.date_range("2020-01-01", periods=n, freq="D"),
                         "open": open_, "high": np.maximum(high, close),
                         "low": np.minimum(low, close), "close": close,
                         "volume": rng.lognormal(10, 0.5, n)})


def test_ema_seed():
    x = np.full(20, 5.0)
    assert np.allclose(_ema(x, 5), 5.0)


def test_rsi_bounds():
    x = np.linspace(10, 20, 100)
    r = _rsi(x, 14)
    assert np.all(np.isfinite(r)) and r.min() >= 0 and r.max() <= 100


def test_position_warmup_is_flat():
    raw = np.linspace(-1, 1, 500)
    pos = to_position(raw)
    assert np.all(pos[:199] == 0.0)
    assert np.any(pos[199:] != 0.0)


def test_no_lookahead_positions():
    df = _df(1200)
    for k in (400, 800):
        full = to_position(np.nan_to_num(_raw(df, "close"), nan=0.0))
        part = to_position(np.nan_to_num(_raw(df.iloc[:k], "close"), nan=0.0))
        assert np.array_equal(full[:k], part), f"future data changed past positions at k={k}"


def _raw(df: pd.DataFrame, op: str = "close") -> np.ndarray:
    ctx = build_context(df)
    return ctx[op]


def test_initial_fee():
    close = np.array([100.0, 110.0])
    pos = np.ones(2)
    m = summarize(close, pos)
    # one unit traded at t=0, then the 10% move
    assert abs(m["total"] - (1.10 * (1 - 0.0005) - 1)) < 1e-12


def test_annualization_is_365():
    n = 365
    close = 100 * np.cumprod(np.full(n, 1.001))
    pos = np.ones(n)
    m = summarize(close, pos)
    # Enter at bar 0's close, so only n-1 daily returns are earned; one fee.
    expected_total = (1 - 0.0005) * (1.001 ** (n - 1)) - 1
    assert abs(m["total"] - expected_total) < 1e-12
    assert abs(m["ann"] - m["total"]) < 1e-12  # 365 bars == one year
    assert BARS_PER_YEAR == 365.0


def test_overlay_threshold():
    close = np.full(30, 100.0)
    assert np.all(apply_risk_overlay(close, np.full(30, 0.01)) == 0.0)
    assert np.all(apply_risk_overlay(close, np.full(30, 0.10)) == 1.0)


def test_overlay_cooldown_is_five_bars():
    close = np.r_[np.full(10, 100.0), np.full(10, 90.0)]
    pos = np.ones(len(close))
    out = apply_risk_overlay(close, pos, hard_stop=0.07, trail=0.04, cooldown=5)
    # stop happens at t=10; re-entry is allowed at t=15 (5 full bars later)
    assert np.all(out[10:15] == 0.0)
    assert out[15] == 1.0


def test_live_overlay_matches_backtest():
    close = np.r_[np.linspace(100, 120, 40), np.linspace(120, 80, 30),
                  np.linspace(80, 130, 30), np.linspace(130, 100, 20)]
    pos = np.r_[np.ones(40), -np.ones(30), np.ones(30), -np.ones(20)]
    ref = apply_risk_overlay(close, pos)
    st = {"state": 0.0, "entry": 0.0, "peak": 0.0, "cd": 0,
          "last_bar_ts": None, "last_sig": 0.0}
    got = []
    for p, px in zip(pos, close):
        sig = 1.0 if p > 0.05 else (-1.0 if p < -0.05 else 0.0)
        st = overlay_step(st, sig, float(px), True, 0.07, 0.04, 5)
        got.append(st["state"])
    assert np.array_equal(ref, np.asarray(got))


def test_pine_export_has_cooldown():
    tree = ("close",)
    pine = to_pine(tree)
    assert "cd :=" in pine and "var int cd" in pine
    assert "strategy.close_all()" in pine
    assert "0.05" in pine


def test_tree_roundtrip():
    tree = ("mul", ("zscore", ("close",), 8), ("sub", ("high",), ("low",)))
    assert tree_from_json(json.loads(json.dumps(tree))) == tree


def test_constant_long_turnover():
    close = np.linspace(100, 120, 50)
    pos = np.ones(50)
    r = run_backtest(close, pos)
    assert abs(r["turnover"][0] - 1.0) < 1e-12
    assert np.all(r["turnover"][1:] == 0.0)


def test_trend_warmup_and_leverage_bounds():
    from .trend import LOOKBACKS, target_position
    df = _df(600)
    pos = target_position(df, vol_target=0.4, max_lev=2.0)
    assert np.all(pos[:max(LOOKBACKS)] == 0.0)
    assert np.max(np.abs(pos)) <= 2.0 + 1e-12
    assert np.all(np.isfinite(pos))


def test_trend_no_lookahead():
    from .trend import target_position
    df = _df(900)
    full = target_position(df)
    for k in (400, 700):
        part = target_position(df.iloc[:k])
        assert np.array_equal(full[:k], part), f"future data changed past trend positions at k={k}"


def test_trend_long_only():
    from .trend import target_position
    pos = target_position(_df(600), long_only=True)
    assert np.all(pos >= 0.0)


def test_trend_btc_confirmation_masks_opposite_signals():
    from .trend import target_position
    df = _df(600, seed=1)
    btc = _df(600, seed=2)
    pos = target_position(df, btc)
    assert np.max(np.abs(pos)) <= 2.0 + 1e-12
    assert np.all(np.isfinite(pos))
    # no position during the warmup even with BTC available
    assert np.all(pos[:60] == 0.0)


def main() -> None:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
