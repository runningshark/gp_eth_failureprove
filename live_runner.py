"""GP ETH 日线择时 - 币安部署入口。

默认 paper 模拟盘(独立资金池, 不影响 follow_bot 的 paper_state)。
实盘需要同时满足: config.py 填了 API_KEY/API_SECRET 且 TRADING["ARMED"]=True 且 MODE="live"。

用法:
  py -m gp_eth.live_runner --once            # 算最新信号 + 对齐仓位, 跑一次退出
  py -m gp_eth.live_runner --loop            # 常驻(盘中每 POLL_SECONDS 检查止损/新收盘)
  py -m gp_eth.live_runner --once --live     # 显式请求实盘(仍要过 ARMED 闸门)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # 允许从任意 cwd 导入根目录 config

RUN_LOG = HERE / "out" / "run_log.json"
STATE_FILE = HERE / "gp_live_state.json"
PAPER_FILE = HERE / "gp_paper_state.json"
EQUITY_LOG = HERE / "gp_equity_log.json"

MIRROR = "https://data-api.binance.vision"
FAPI = "https://fapi.binance.com"
TAKER_FEE = 0.0005

DEFAULTS = {"SYMBOL": "ETHUSDT", "NOTIONAL_USDT": 100, "LEVERAGE": 1,
            "POLL_SECONDS": 300, "HARD_STOP": 0.07, "TRAIL": 0.04,
            "COOLDOWN_BARS": 5}


# ------------------------------------------------------------------ config

def load_config():
    cfg = dict(DEFAULTS)
    proxies = None
    api_key = api_secret = ""
    armed = False
    mode = "paper"
    try:
        import config as root_cfg
        cfg.update(getattr(root_cfg, "GP", {}))
        proxies = getattr(root_cfg, "PROXIES", None)
        api_key = getattr(root_cfg, "API_KEY", "") or ""
        api_secret = getattr(root_cfg, "API_SECRET", "") or ""
        tr = getattr(root_cfg, "TRADING", {}) or {}
        armed = bool(tr.get("ARMED", False))
        mode = str(tr.get("MODE", "paper")).lower()
    except ImportError:
        pass
    return cfg, proxies, api_key, api_secret, armed, mode


# ------------------------------------------------------------------ market data

def _http_get(url, params, proxies, timeout=10):
    for px in (None, proxies):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             proxies=px if px else None)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    raise ConnectionError(f"行情不可用: {url}")


def fetch_daily(symbol: str, limit: int, proxies) -> pd.DataFrame:
    now_ms = time.time() * 1000
    rows = None
    for base in (MIRROR + "/api/v3/klines", FAPI + "/fapi/v1/klines"):
        try:
            rows = _http_get(base, {"symbol": symbol, "interval": "1d", "limit": limit}, proxies)
            break
        except ConnectionError:
            continue
    if rows is None:
        raise ConnectionError("ETH 日线拉取失败")
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[df["close_time"].astype(float) <= now_ms].reset_index(drop=True)  # 剔除未收盘K线
    return df


def fetch_price(symbol: str, proxies) -> float:
    for url in (MIRROR + "/api/v3/ticker/price",
                FAPI + "/fapi/v1/ticker/price"):
        try:
            d = _http_get(url, {"symbol": symbol}, proxies, timeout=6)
            return float(d["price"])
        except ConnectionError:
            continue
    raise ConnectionError("最新价拉取失败")


# ------------------------------------------------------------------ signal

def load_tree() -> tuple:
    from .individual import tree_from_json
    log = json.loads(RUN_LOG.read_text(encoding="utf-8"))
    if "tree" not in log:
        raise RuntimeError("run_log.json 缺少 tree 字段, 请先重跑 py -m gp_eth.run")
    return tree_from_json(log["tree"])


def compute_signal(tree, df: pd.DataFrame) -> tuple[float, float]:
    from .individual import to_position
    from .primitives import build_context, eval_tree
    ctx = build_context(df)
    raw = np.nan_to_num(eval_tree(tree, ctx), nan=0.0, posinf=0.0, neginf=0.0)
    pos = to_position(raw)
    p = float(pos[-1])
    sig = 1.0 if p > 0.05 else (-1.0 if p < -0.05 else 0.0)
    return sig, p


# ------------------------------------------------------------------ risk overlay state machine

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"state": 0.0, "entry": 0.0, "peak": 0.0, "cd": 0,
            "last_bar_ts": None, "last_sig": 0.0}


def save_state(st: dict) -> None:
    STATE_FILE.write_text(json.dumps(st, indent=1), encoding="utf-8")


def overlay_step(st: dict, sig: float, price: float, new_bar: bool,
                 hard_stop: float, trail: float, cooldown: int) -> dict:
    """与 backtest.apply_risk_overlay 同构的单步状态机。
    new_bar=True 时走收盘逻辑(含 cd 递减/重进); 盘中只查止损与追踪。"""
    if new_bar:
        just_stopped = False
        if st["state"] != 0.0:
            if sig != st["state"]:
                st["state"] = 0.0
            else:
                st["peak"] = max(st["peak"], price) if st["state"] > 0 else min(st["peak"], price)
                exc = (price / st["entry"] - 1.0) * st["state"]
                give = (1.0 - price / st["peak"]) if st["state"] > 0 else (price / st["peak"] - 1.0)
                if exc <= -hard_stop or give >= trail:
                    st["state"] = 0.0
                    st["cd"] = cooldown
                    just_stopped = True
        if st["cd"] > 0 and not just_stopped:
            st["cd"] -= 1
        if st["state"] == 0.0 and st["cd"] == 0 and sig != 0.0:
            st["state"] = sig
            st["entry"] = price
            st["peak"] = price
    else:
        if st["state"] != 0.0:
            st["peak"] = max(st["peak"], price) if st["state"] > 0 else min(st["peak"], price)
            exc = (price / st["entry"] - 1.0) * st["state"]
            give = (1.0 - price / st["peak"]) if st["state"] > 0 else (price / st["peak"] - 1.0)
            if exc <= -hard_stop or give >= trail:
                st["state"] = 0.0
                st["cd"] = cooldown
    return st


# ------------------------------------------------------------------ execution

class PaperDesk:
    """独立模拟盘(ETHUSDT 单品种), taker 0.05%, 不碰 follow_bot 的 paper_state。"""

    def __init__(self, price_fn):
        self.price_fn = price_fn
        if PAPER_FILE.exists():
            self.s = json.loads(PAPER_FILE.read_text(encoding="utf-8"))
        else:
            self.s = {"balance": 1000.0, "pos": None, "history": []}

    def _save(self):
        PAPER_FILE.write_text(json.dumps(self.s, indent=1), encoding="utf-8")

    def position(self, symbol):
        p = self.s.get("pos")
        if not p:
            return None
        return {"amt": p["qty"] * (1 if p["side"] == "BUY" else -1), "entry": p["entry"]}

    def market(self, symbol, side, qty, price):
        fee = qty * price * TAKER_FEE
        self.s["balance"] -= fee
        p = self.s.get("pos")
        if p and ((p["side"] == "BUY") != (side == "BUY")):  # 反向 → 先平
            self._close(symbol, price)
            p = None
        if side == "BUY" and (p is None or p["side"] == "BUY") or \
           side == "SELL" and (p is None or p["side"] == "SELL"):
            if p:
                tot = p["qty"] + qty
                p["entry"] = (p["entry"] * p["qty"] + price * qty) / tot
                p["qty"] = tot
            else:
                self.s["pos"] = {"side": side, "qty": qty, "entry": price,
                                 "ts": time.time()}
        self._save()

    def _close(self, symbol, price):
        p = self.s.get("pos")
        if not p:
            return
        sign = 1 if p["side"] == "BUY" else -1
        pnl = (price - p["entry"]) * sign * p["qty"] - p["qty"] * price * TAKER_FEE
        self.s["balance"] += pnl
        self.s["history"].append({"side": p["side"], "qty": p["qty"],
                                  "entry": p["entry"], "exit": price,
                                  "pnl": round(pnl, 2), "ts": time.time()})
        self.s["pos"] = None
        self._save()

    def close_all(self, symbol, price):
        if self.s.get("pos"):
            self._close(symbol, price)

    def equity(self, symbol=None, price=None):
        price = price or self.price_fn(symbol)
        eq = self.s["balance"]
        p = self.s.get("pos")
        if p:
            sign = 1 if p["side"] == "BUY" else -1
            eq += (price - p["entry"]) * sign * p["qty"]
        return eq


class LiveDesk:
    def __init__(self, key, secret, proxies, symbol, leverage):
        import executor
        self.ex = executor
        self.key, self.secret, self.proxies = key, secret, proxies
        try:
            executor.set_margin_type(symbol, "CROSSED", key, secret, proxies)
        except Exception:
            pass
        try:
            executor.set_leverage(symbol, leverage, key, secret, proxies)
        except Exception as e:
            print(f"[live] set_leverage 失败: {e}")

    def position(self, symbol):
        p = self.ex.get_position(symbol, self.key, self.secret, self.proxies)
        amt = float(p.get("positionAmt", 0)) if p else 0.0
        if abs(amt) < 1e-12:
            return None
        return {"amt": amt, "entry": float(p.get("entryPrice", 0))}

    def market(self, symbol, side, qty, price):
        self.ex.place_order(symbol, side, qty, order_type="MARKET",
                            key=self.key, secret=self.secret, proxies=self.proxies)

    def close_all(self, symbol, price):
        p = self.position(symbol)
        if not p:
            return
        side = "SELL" if p["amt"] > 0 else "BUY"
        self.ex.place_order(symbol, side, abs(p["amt"]), order_type="MARKET",
                            reduce_only=True, key=self.key, secret=self.secret,
                            proxies=self.proxies)

    def equity(self, symbol=None, price=None):
        bal = self.ex.get_futures_balance(self.key, self.secret, self.proxies)
        total = bal[1] if bal else 0.0
        try:
            p = self.ex.get_position(symbol, self.key, self.secret, self.proxies)
            if p:
                total += float(p.get("unRealizedProfit", p.get("unrealizedProfit", 0)))
        except Exception:
            pass
        return total


def log_equity(eq: float, price: float, st: dict, live: bool) -> None:
    """权益快照: 有变化即时落点; 无变化时每30分钟补一点; 最多5000条。"""
    try:
        now = time.time()
        items = (json.loads(EQUITY_LOG.read_text(encoding="utf-8"))
                 if EQUITY_LOG.exists() else [])
        eq = round(float(eq), 2)
        if items:
            last = items[-1]
            changed = abs(last["equity"] - eq) >= 0.01
            if not changed and now - last["ts"] < 1800:
                return
        items.append({"ts": int(now), "equity": eq,
                      "price": round(float(price), 2),
                      "state": int(st["state"]), "live": bool(live)})
        EQUITY_LOG.write_text(json.dumps(items[-5000:], separators=(",", ":")),
                              encoding="utf-8")
    except Exception:
        pass


def sync_position(desk, symbol, target: int, notional: float, price: float) -> str:
    cur = desk.position(symbol)
    cur_dir = 0
    if cur and abs(cur["amt"]) > 1e-12:
        cur_dir = 1 if cur["amt"] > 0 else -1
    if cur_dir == int(target):
        return "hold"
    if cur_dir != 0:
        desk.close_all(symbol, price)
    if target != 0:
        qty = notional / price
        qty = float(np.floor(qty * 1e6) / 1e6)
        if qty > 0:
            desk.market(symbol, "BUY" if target > 0 else "SELL", qty, price)
            return "open_long" if target > 0 else "open_short"
    return "flat"


# ------------------------------------------------------------------ main

def cycle(tree, cfg, proxies, desk, live: bool) -> str:
    symbol = cfg["SYMBOL"]
    df = fetch_daily(symbol, 500, proxies)
    sig, pos_val = compute_signal(tree, df)
    st = load_state()
    bar_ts = int(df["close_time"].iloc[-1])
    new_bar = st.get("last_bar_ts") != bar_ts

    if new_bar:
        price = float(df["close"].iloc[-1])
        st = overlay_step(st, sig, price, True, cfg["HARD_STOP"], cfg["TRAIL"],
                          cfg["COOLDOWN_BARS"])
        st["last_bar_ts"] = bar_ts
    else:
        price = fetch_price(symbol, proxies)
        st = overlay_step(st, sig, price, False,
                          cfg["HARD_STOP"], cfg["TRAIL"], cfg["COOLDOWN_BARS"])
    st["last_sig"] = sig
    save_state(st)

    target = int(st["state"])
    action = sync_position(desk, symbol, target, cfg["NOTIONAL_USDT"], price)
    tag = "LIVE" if live else "PAPER"
    now = datetime.now(timezone.utc).strftime("%m-%d %H:%M")
    eq = desk.equity(symbol, price) if hasattr(desk, "equity") else float("nan")
    log_equity(eq, price, st, live)
    msg = (f"[{now} UTC] {tag} sig={sig:+.0f} pos={pos_val:+.3f} state={target:+.0f} "
           f"cd={st['cd']} price={price:.2f} action={action} equity={eq:.2f}U "
           f"bar_closed={new_bar}")
    print(msg)
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="跑一次信号同步后退出")
    ap.add_argument("--loop", action="store_true", help="常驻轮询")
    ap.add_argument("--live", action="store_true", help="请求实盘(仍需 ARMED 与密钥)")
    args = ap.parse_args()

    cfg, proxies, api_key, api_secret, armed, mode = load_config()
    tree = load_tree()
    print(f"symbol={cfg['SYMBOL']} notional={cfg['NOTIONAL_USDT']}U lev={cfg['LEVERAGE']} "
          f"hard_stop={cfg['HARD_STOP']:.0%} trail={cfg['TRAIL']:.0%}")

    live = bool(args.live and armed and api_key and api_secret and mode == "live")
    if args.live and not live:
        reason = []
        if not (api_key and api_secret):
            reason.append("未配置API密钥")
        if not armed:
            reason.append("ARMED=False")
        if mode != "live":
            reason.append(f"MODE={mode}")
        print(f"[gate] 实盘被拒绝: {', '.join(reason)} → 使用 paper")

    if live:
        desk = LiveDesk(api_key, api_secret, proxies, cfg["SYMBOL"], cfg["LEVERAGE"])
    else:
        desk = PaperDesk(lambda s: fetch_price(s, proxies))

    while True:
        try:
            cycle(tree, cfg, proxies, desk, live)
        except Exception as e:
            print(f"[error] {e}")
        if args.once or not args.loop:
            break
        time.sleep(cfg["POLL_SECONDS"])


if __name__ == "__main__":
    main()
