"""Primitive set for GP-evolved ETH strategies.

Tree node forms:
  ('close'|'open'|'high'|'low'|'volume'|'ret'|'hlr',)          terminals
  ('neg'|'abs', x)                                              unary
  ('add'|'sub'|'mul'|'div'|'max'|'min'|'gt'|'lt', a, b)        binary
  ('ema'|'sma'|'roc'|'tsmax'|'tsmin'|'stddev'|'zscore'|'rsi', x, n)  time-series
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

PERIODS = (2, 3, 5, 8, 13, 21, 34, 55, 89, 144)
TERMINALS = ("close", "open", "high", "low", "volume", "ret", "hlr")
UNARY = ("neg", "abs")
BINARY = ("add", "sub", "mul", "div", "max", "min", "gt", "lt")
TS1 = ("ema", "sma", "roc", "tsmax", "tsmin", "stddev", "zscore", "rsi")


def build_context(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1.0
    hlr = (h - lo) / np.maximum(c, 1e-12)
    return {"close": c, "open": o, "high": h, "low": lo,
            "volume": v, "ret": ret, "hlr": hlr}


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > 1e-12, a / np.where(np.abs(b) > 1e-12, b, 1.0), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).mean().to_numpy()


def _stddev(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=2).std(ddof=0).fillna(0.0).to_numpy()


def _roc(x: np.ndarray, n: int) -> np.ndarray:
    xs = np.empty_like(x)
    xs[:] = np.nan
    xs[n:] = x[:-n]
    out = _safe_div(x - xs, np.where(np.isnan(xs), 1.0, xs))
    return np.where(np.isnan(xs), 0.0, out)


def _tsmax(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).max().to_numpy()


def _tsmin(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).min().to_numpy()


def _zscore(x: np.ndarray, n: int) -> np.ndarray:
    m = pd.Series(x).rolling(n, min_periods=2).mean().to_numpy()
    s = _stddev(x, n)
    return np.nan_to_num(_safe_div(x - m, np.where(s < 1e-12, 1.0, s)), nan=0.0)


def _rsi(x: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(x, prepend=x[0])
    up = _ema(np.where(d > 0, d, 0.0), n)
    dn = _ema(np.where(d < 0, -d, 0.0), n)
    return 100.0 * _safe_div(up, up + dn)


def eval_tree(node: tuple, ctx: dict[str, np.ndarray]) -> np.ndarray:
    op = node[0]
    if op in TERMINALS:
        return ctx[op]
    if op in UNARY:
        x = eval_tree(node[1], ctx)
        return -x if op == "neg" else np.abs(x)
    if op in BINARY:
        a = eval_tree(node[1], ctx)
        b = eval_tree(node[2], ctx)
        if op == "add": return a + b
        if op == "sub": return a - b
        if op == "mul": return a * b
        if op == "div": return _safe_div(a, b)
        if op == "max": return np.maximum(a, b)
        if op == "min": return np.minimum(a, b)
        if op == "gt": return np.where(a > b, 1.0, 0.0)
        return np.where(a < b, 1.0, 0.0)
    if op in TS1:
        x = eval_tree(node[1], ctx)
        n = node[2]
        if op == "ema": return _ema(x, n)
        if op == "sma": return _sma(x, n)
        if op == "roc": return _roc(x, n)
        if op == "tsmax": return _tsmax(x, n)
        if op == "tsmin": return _tsmin(x, n)
        if op == "stddev": return _stddev(x, n)
        if op == "zscore": return _zscore(x, n)
        return _rsi(x, n)
    raise ValueError(f"unknown op {op}")
