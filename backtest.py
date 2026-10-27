"""Backtest engine + multi-objective fitness + optional risk overlay."""

from __future__ import annotations

import numpy as np

FEE = 0.0005          # one-way taker fee on turnover
HARD_STOP = 0.07      # hard stop loss from entry (price move)
TRAIL_GIVEBACK = 0.04 # trailing stop give-back
BARS_PER_YEAR = 365.0 # crypto trades every day; 252 understates ann/Sharpe
SIGNAL_THRESHOLD = 0.05
COOLDOWN_BARS = 5


def run_backtest(close: np.ndarray, pos: np.ndarray) -> dict:
    n = len(close)
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0
    pos_l = np.zeros(n)
    pos_l[1:] = pos[:-1]                        # fill at bar t close, earn t+1
    turn = np.zeros(n)
    turn[0] = abs(pos[0])                       # opening trade also pays a fee
    turn[1:] = np.abs(pos[1:] - pos[:-1])
    strat = pos_l * ret - turn * FEE
    equity = np.cumprod(1.0 + strat)
    return {"equity": equity, "strat": strat, "turnover": turn, "ret": ret}


def summarize(close: np.ndarray, pos: np.ndarray) -> dict | None:
    r = run_backtest(close, pos)
    equity = r["equity"]
    if not np.all(np.isfinite(equity)) or equity[-1] <= 0:
        return None
    n = len(r["strat"])
    total = float(equity[-1] - 1.0)
    ann = float(equity[-1] ** (BARS_PER_YEAR / n) - 1.0) if equity[-1] > 0 else -1.0
    sd = float(r["strat"].std(ddof=1)) if n > 1 else 0.0
    sharpe = float(r["strat"].mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = float(np.max(1.0 - equity / np.maximum(peak, 1e-12)))
    calmar = ann / mdd if mdd > 1e-9 else 0.0
    flips = int(np.sum(np.sign(pos[1:]) != np.sign(pos[:-1])))
    return {
        "total": total, "ann": ann, "sharpe": sharpe, "mdd": mdd,
        "calmar": calmar, "turnover": float(r["turnover"].mean()),
        "flips": flips, "equity": equity, "pos": pos,
    }


def fitness(m: dict | None, complexity_weight: float = 0.015) -> float:
    if m is None:
        return -1e9
    s = m["sharpe"]
    s += 0.5 * float(np.clip(m["calmar"], -2.0, 5.0))
    s += 2.0 * float(np.log1p(max(m["total"], -0.8)))
    s -= 4.0 * max(0.0, m["mdd"] - 0.45)          # drawdown guard
    s -= 50.0 * max(0.0, m["turnover"] - 0.9)     # churn guard
    s -= complexity_weight * m.get("size", 0)     # parsimony pressure
    if m["flips"] < 4:
        s -= 3.0
    return float(s)


def apply_risk_overlay(close: np.ndarray, pos: np.ndarray,
                       hard_stop: float = HARD_STOP,
                       trail: float = TRAIL_GIVEBACK,
                       cooldown: int = COOLDOWN_BARS) -> np.ndarray:
    """Exit on hard stop / trailing give-back; re-enter after cooldown if signal persists.

    ``pos`` is the fractional model position.  A signal is only considered
    "on" when |pos| > SIGNAL_THRESHOLD, the same rule used by the live runner
    and the Pine export.  Using np.sign(pos) here would enter on numerically
    tiny positions and make live trading diverge from the backtest.
    """
    n = len(close)
    out = np.zeros(n)
    state = 0.0
    cd = 0
    entry = peak = 1.0
    for t in range(n):
        p = pos[t]
        sig = 1.0 if p > SIGNAL_THRESHOLD else (-1.0 if p < -SIGNAL_THRESHOLD else 0.0)
        just_stopped = False
        if state != 0.0:
            if sig != state:
                state = 0.0
            else:
                peak = max(peak, close[t]) if state > 0 else min(peak, close[t])
                exc = (close[t] / entry - 1.0) * state
                give = (1.0 - close[t] / peak) if state > 0 else (close[t] / peak - 1.0)
                if exc <= -hard_stop or give >= trail:
                    state = 0.0
                    cd = cooldown
                    just_stopped = True
        # Do not burn one cooldown bar on the stop bar itself.
        if cd > 0 and not just_stopped:
            cd -= 1
        if state == 0.0 and cd == 0 and sig != 0.0:
            state = sig
            entry = close[t]
            peak = close[t]
        out[t] = state
    return out
