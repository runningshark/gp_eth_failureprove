"""Translate an evolved expression tree into a TradingView Pine Script v6 strategy."""

from __future__ import annotations

from .backtest import (COOLDOWN_BARS, FEE, HARD_STOP, SIGNAL_THRESHOLD,
                       TRAIL_GIVEBACK)
from .primitives import BINARY, TS1, UNARY


def _pine(node: tuple) -> str:
    op = node[0]
    if op in ("close", "open", "high", "low", "volume"):
        return op
    if op == "ret":
        return "((close - nz(close[1], close)) / nz(close[1], close))"
    if op == "hlr":
        return "((high - low) / close)"
    if op in UNARY:
        inner = _pine(node[1])
        return f"-({inner})" if op == "neg" else f"math.abs({inner})"
    if op in BINARY:
        a, b = _pine(node[1]), _pine(node[2])
        if op == "add": return f"({a} + {b})"
        if op == "sub": return f"({a} - {b})"
        if op == "mul": return f"({a} * {b})"
        if op == "div": return f"(({b}) != 0.0 ? ({a}) / ({b}) : 0.0)"
        if op == "max": return f"math.max({a}, {b})"
        if op == "min": return f"math.min({a}, {b})"
        if op == "gt": return f"(({a}) > ({b}) ? 1.0 : 0.0)"
        return f"(({a}) < ({b}) ? 1.0 : 0.0)"
    if op in TS1:
        x = _pine(node[1])
        n = node[2]
        if op == "ema": return f"ta.ema({x}, {n})"
        if op == "sma": return f"ta.sma({x}, {n})"
        if op == "roc": return f"((({x}) - nz(({x})[{n}], {x})) / nz(({x})[{n}], {x}))"
        if op == "tsmax": return f"ta.highest({x}, {n})"
        if op == "tsmin": return f"ta.lowest({x}, {n})"
        if op == "stddev": return f"nz(ta.stdev({x}, {n}), 0.0)"
        if op == "zscore":
            return f"((({x}) - ta.sma({x}, {n})) / max(nz(ta.stdev({x}, {n}), 0.0), 1e-12))"
        return f"ta.rsi({x}, {n})"
    raise ValueError(op)


def to_pine(tree: tuple, name: str = "GP-Evolved ETH") -> str:
    raw = _pine(tree)
    return f"""//@version=6
strategy("{name}", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity,
     default_qty_value=100, commission_type=strategy.commission.percent, commission_value={FEE * 100:.3f},
     process_orders_on_close=true, calc_on_every_tick=false)

// ---- evolved expression ----
raw = {raw}
z = (raw - ta.sma(raw, 200)) / max(nz(ta.stdev(raw, 200), 0.0), 1e-12)
pos = math.tanh(nz(z))
pos := math.sum(pos, 3) / 3.0

// ---- risk overlay: hard stop -7% + trailing give-back 4% + 5-bar cooldown ----
// Same state machine as backtest.apply_risk_overlay and live_runner.overlay_step.
var float entry = na
var float peak = na
var float state = 0.0
var int cd = 0
sig = pos > {SIGNAL_THRESHOLD} ? 1.0 : pos < -{SIGNAL_THRESHOLD} ? -1.0 : 0.0
stopped = false
if state != 0
    if sig != state
        state := 0.0
    else
        peak := state > 0 ? math.max(peak, close) : math.min(peak, close)
        exc = (close / entry - 1.0) * state
        give = state > 0 ? 1.0 - close / peak : close / peak - 1.0
        if exc <= -{HARD_STOP} or give >= {TRAIL_GIVEBACK}
            state := 0.0
            cd := {COOLDOWN_BARS}
            stopped := true
if cd > 0 and not stopped
    cd := cd - 1
if state == 0 and cd == 0 and sig != 0
    state := sig
    entry := close
    peak := close

plot(pos, "pos", color=color.gray, display=display.data_window)
plotshape(state > 0 and state[1] <= 0, style=shape.labelup, location=location.belowbar,
     color=color.new(color.teal, 0), text="L", size=size.tiny)
plotshape(state < 0 and state[1] >= 0, style=shape.labeldown, location=location.abovebar,
     color=color.new(color.maroon, 0), text="S", size=size.tiny)

if state > 0
    strategy.entry("L", strategy.long)
else if state < 0
    strategy.entry("S", strategy.short)
else
    strategy.close_all()
"""
