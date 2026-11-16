"""GP ETH 策略上位机: Flask 监控面板。

用法:  py -m gp_eth.dashboard [--port 8787]
打开:  http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

from . import live_runner as lr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV_CACHE = HERE / "cache_ETHUSDT_1d_3000.csv"

app = Flask(__name__)
_charts_cache = {"ts": 0.0, "data": None}
CHART_TTL = 30.0

# 权益曲线单独缓存(更短), 避免每次都重算全年回测
_equity_cache = {"ts": 0.0, "data": None}
EQUITY_TTL = 15.0


# ------------------------------------------------------------------ data

def _load_history() -> pd.DataFrame:
    if CSV_CACHE.exists():
        df = pd.read_csv(CSV_CACHE, parse_dates=["open_time"]).sort_values("open_time")
        return df.reset_index(drop=True)
    cfg, proxies, *_ = lr.load_config()
    return lr.fetch_daily(cfg["SYMBOL"], 800, proxies)


def _charts_data() -> dict:
    now = time.time()
    if _charts_cache["data"] and now - _charts_cache["ts"] < CHART_TTL:
        return _charts_cache["data"]

    tree = lr.load_tree()
    df = _load_history()
    from .backtest import run_backtest
    from .individual import to_position
    from .primitives import build_context, eval_tree

    ctx = build_context(df)
    raw = np.nan_to_num(eval_tree(tree, ctx), nan=0.0, posinf=0.0, neginf=0.0)
    pos = to_position(raw)
    r = run_backtest(ctx["close"], pos)
    hold = np.cumprod(1.0 + np.r_[0.0, np.diff(ctx["close"]) / ctx["close"][:-1]])

    sig = np.where(pos > 0.05, 1.0, np.where(pos < -0.05, -1.0, 0.0))
    markers = []
    prev = 0.0
    for i, s in enumerate(sig):
        if s != 0.0 and s != prev:
            markers.append({"date": str(df["open_time"].iloc[i].date()),
                            "dir": int(s),
                            "price": float(ctx["close"][i])})
        prev = s

    log = json.loads((HERE / "out" / "run_log.json").read_text(encoding="utf-8"))

    data = {
        "dates": [str(d.date()) for d in df["open_time"]],
        "close": [round(float(x), 2) for x in ctx["close"]],
        "equity": [round(float(x), 4) for x in r["equity"]],
        "hold": [round(float(x), 4) for x in hold],
        "markers": markers,
        "split": "2024-01-01",
        "evolution": {"best": log.get("gen_best_scores", []),
                      "mean": log.get("gen_mean_scores", [])},
        "expr": log.get("expr", ""),
        "results": log.get("results", {}),
        "seed": log.get("seed"),
        "gens": log.get("gens"),
        "robust": log.get("robust", False),
    }
    _charts_cache["ts"] = now
    _charts_cache["data"] = data
    return data


# ------------------------------------------------------------------ api

@app.get("/api/status")
def api_status():
    cfg, proxies, api_key, api_secret, armed, mode = lr.load_config()
    tree = lr.load_tree()
    df = lr.fetch_daily(cfg["SYMBOL"], 500, proxies)
    sig, pos_val = lr.compute_signal(tree, df)
    price = float(df["close"].iloc[-1])
    st = lr.load_state()
    live_ok = bool(armed and api_key and api_secret and mode == "live")

    paper = None
    if lr.PAPER_FILE.exists():
        s = json.loads(lr.PAPER_FILE.read_text(encoding="utf-8"))
        p = s.get("pos")
        eq = s["balance"]
        if p:
            sign = 1 if p["side"] == "BUY" else -1
            eq += (price - p["entry"]) * sign * p["qty"]
        paper = {
            "balance": round(s["balance"], 2),
            "equity": round(eq, 2),
            "pos": ({"side": p["side"], "qty": p["qty"], "entry": p["entry"],
                     "upnl": round((price - p["entry"]) * (1 if p["side"] == "BUY" else -1) * p["qty"], 2)}
                    if p else None),
            "trades": list(reversed(s.get("history", [])[-12:])),
        }
    return jsonify({
        "price": price,
        "sig": sig,
        "pos": round(pos_val, 4),
        "state": int(st["state"]),
        "entry": st["entry"],
        "peak": st["peak"],
        "cd": st["cd"],
        "last_bar": df["open_time"].iloc[-1].strftime("%Y-%m-%d"),
        "mode": "LIVE" if live_ok else "PAPER",
        "cfg": {"symbol": cfg["SYMBOL"], "notional": cfg["NOTIONAL_USDT"],
                "lev": cfg["LEVERAGE"], "hard_stop": cfg["HARD_STOP"],
                "trail": cfg["TRAIL"]},
        "paper": paper,
        "ts": time.time(),
    })


@app.get("/api/charts")
def api_charts():
    return jsonify(_charts_data())


@app.get("/api/equity")
def api_equity():
    now = time.time()
    if _equity_cache["data"] is not None and now - _equity_cache["ts"] < EQUITY_TTL:
        return jsonify(_equity_cache["data"])
    from datetime import datetime as _dt
    items = []
    if lr.EQUITY_LOG.exists():
        try:
            for x in json.loads(lr.EQUITY_LOG.read_text(encoding="utf-8")):
                items.append({"t": _dt.fromtimestamp(x["ts"]).strftime("%m-%d %H:%M"),
                              "e": x["equity"], "state": x["state"],
                              "live": x.get("live", False)})
        except Exception:
            pass
    _equity_cache["ts"] = now
    _equity_cache["data"] = items
    return jsonify(items)


@app.post("/api/sync")
def api_sync():
    try:
        cfg, proxies, api_key, api_secret, armed, mode = lr.load_config()
        tree = lr.load_tree()
        live = bool(armed and api_key and api_secret and mode == "live")
        if live:
            desk = lr.LiveDesk(api_key, api_secret, proxies,
                               cfg["SYMBOL"], cfg["LEVERAGE"])
        else:
            desk = lr.PaperDesk(lambda s: lr.fetch_price(s, proxies))
        msg = lr.cycle(tree, cfg, proxies, desk, live)
        _charts_cache["ts"] = 0.0  # 强制下次刷新图表
        return jsonify({"ok": True, "log": msg})
    except Exception as e:
        return jsonify({"ok": False, "log": str(e)}), 500


@app.post("/api/reset_paper")
def api_reset_paper():
    if request.json and request.json.get("confirm"):
        lr.PAPER_FILE.unlink(missing_ok=True)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "log": "需要 confirm=true"}), 400


# ------------------------------------------------------------------ page

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>GP-Eth 上位机</title>
<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
<script>if (typeof echarts === 'undefined') document.write('<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"><\\/script>');</script>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { background: #0d1117; color: #c9d1d9; font: 13px/1.5 "Segoe UI", "Microsoft YaHei", sans-serif; padding: 14px; }
  h1 { font-size: 17px; color: #e6edf3; margin-bottom: 10px; }
  h1 small { color: #8b949e; font-weight: normal; font-size: 12px; margin-left: 8px; }
  .cards { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; min-width: 130px; }
  .card .k { color: #8b949e; font-size: 11px; }
  .card .v { font-size: 19px; font-weight: 600; margin-top: 2px; }
  .long { color: #3fb950; } .short { color: #f85149; } .flat { color: #8b949e; }
  .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
  .panel h2 { font-size: 13px; color: #e6edf3; margin-bottom: 6px; }
  .chart { width: 100%; height: 300px; }
  .expr { font-family: Consolas, monospace; font-size: 11px; color: #a5d6ff; word-break: break-all; background: #0d1117; padding: 8px; border-radius: 6px; border: 1px solid #30363d; }
  button { background: #238636; color: #fff; border: 0; border-radius: 6px; padding: 6px 14px; font-size: 12px; cursor: pointer; margin-right: 8px; }
  button.gray { background: #30363d; }
  button:disabled { opacity: .5; cursor: wait; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: normal; } td:first-child, th:first-child { text-align: left; }
  .logline { font-family: Consolas, monospace; font-size: 12px; color: #d2a8ff; margin: 6px 0; }
  .two { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; }
  @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
  .muted { color: #8b949e; font-size: 11px; }
</style>
</head>
<body>
<h1>GP-Eth 择时上位机 <small id="subtitle">加载中…</small></h1>

<div class="cards">
  <div class="card"><div class="k">ETH 现价</div><div class="v" id="price">–</div></div>
  <div class="card"><div class="k">模型信号</div><div class="v" id="sig">–</div></div>
  <div class="card"><div class="k">风控状态</div><div class="v" id="state">–</div></div>
  <div class="card"><div class="k">入场 / 峰值</div><div class="v" id="entrypeak">–</div></div>
  <div class="card"><div class="k">冷却</div><div class="v" id="cd">–</div></div>
  <div class="card"><div class="k">Paper 权益</div><div class="v" id="equity">–</div></div>
  <div class="card"><div class="k">模式</div><div class="v" id="mode">–</div></div>
</div>

<div style="margin-bottom:12px">
  <button id="btnSync" onclick="sync()">立即同步</button>
  <button class="gray" onclick="loadStatus()">刷新状态</button>
  <button class="gray" onclick="loadCharts(true)">重算图表</button>
  <span class="muted" id="lastRefresh"></span>
</div>
<div class="logline" id="log"></div>

<div class="grid">
  <div class="panel">
    <h2>ETH 日线 · 信号标记 <span class="muted">(红=开空 / 绿=开多)</span></h2>
    <div id="chPrice" class="chart"></div>
  </div>
  <div class="two">
    <div class="panel">
      <h2>净值曲线 <span class="muted">(蓝=GP策略 / 灰=买入持有, 对数轴)</span></h2>
      <div id="chEquity" class="chart"></div>
    </div>
    <div class="panel">
      <h2>进化曲线 <span class="muted">(fitness)</span></h2>
      <div id="chEvo" class="chart" style="height:300px"></div>
    </div>
  </div>
  <div class="panel">
    <h2>实盘权益曲线 <span class="muted">(模拟/实盘账户真实权益落点 · 起始 1000U)</span></h2>
    <div id="chPaper" class="chart" style="height:240px"></div>
  </div>
  <div class="two">
    <div class="panel">
      <h2>最优表达式 <span class="muted" id="meta"></span></h2>
      <div class="expr" id="expr">–</div>
      <h2 style="margin-top:10px">Walk-forward 成绩</h2>
      <table id="tblRes"><tr><th></th><th>总收益</th><th>年化</th><th>Sharpe</th><th>Calmar</th><th>MDD</th></tr></table>
    </div>
    <div class="panel">
      <h2>最近成交 (paper)</h2>
      <table id="tblTrades"><tr><th>方向</th><th>数量</th><th>入场</th><th>出场</th><th>PnL</th></tr></table>
      <div style="margin-top:8px"><button class="gray" onclick="resetPaper()">重置模拟盘</button></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const fmtPct = x => (x>=0?'+':'') + (x*100).toFixed(1) + '%';
let chPrice, chEquity, chEvo, chPaper;

function initCharts() {
  chPrice = echarts.init($('chPrice'), 'dark');
  chEquity = echarts.init($('chEquity'), 'dark');
  chEvo = echarts.init($('chEvo'), 'dark');
  chPaper = echarts.init($('chPaper'), 'dark');
  window.onresize = () => { chPrice.resize(); chEquity.resize(); chEvo.resize(); chPaper.resize(); };
}

const base = {
  backgroundColor: 'transparent',
  textStyle: { color: '#8b949e' },
  grid: { left: 60, right: 20, top: 30, bottom: 40 },
  tooltip: { trigger: 'axis', backgroundColor: '#1c2128', borderColor: '#30363d' },
  xAxis: { type: 'category', axisLine: { lineStyle: { color: '#30363d' } } },
  yAxis: { type: 'value', splitLine: { lineStyle: { color: '#21262d' } } },
};

async function loadStatus() {
  const s = await (await fetch('/api/status')).json();
  $('price').textContent = s.price.toLocaleString(undefined, {maximumFractionDigits: 1});
  const sigEl = $('sig');
  sigEl.textContent = s.sig > 0 ? '做多 LONG' : s.sig < 0 ? '做空 SHORT' : '观望 FLAT';
  sigEl.className = 'v ' + (s.sig > 0 ? 'long' : s.sig < 0 ? 'short' : 'flat');
  const st = $('state');
  st.textContent = s.state > 0 ? '持多' : s.state < 0 ? '持空' : '空仓';
  st.className = 'v ' + (s.state > 0 ? 'long' : s.state < 0 ? 'short' : 'flat');
  $('entrypeak').textContent = (s.entry ? s.entry.toFixed(1) : '–') + ' / ' + (s.peak ? s.peak.toFixed(1) : '–');
  $('cd').textContent = s.cd > 0 ? s.cd + ' 根' : '无';
  if (s.paper) {
    $('equity').textContent = s.paper.equity.toLocaleString() + ' U';
    const t = s.paper.trades.map(t =>
      `<tr><td>${t.side==='BUY'?'多':'空'}</td><td>${t.qty}</td><td>${t.entry.toFixed(1)}</td><td>${t.exit.toFixed(1)}</td><td style="color:${t.pnl>=0?'#3fb950':'#f85149'}">${fmtPct(t.pnl/ (t.entry*t.qty))}</td></tr>`).join('');
    $('tblTrades').innerHTML = '<tr><th>方向</th><th>数量</th><th>入场</th><th>出场</th><th>PnL</th></tr>' + (t || '<tr><td colspan=5 class=muted>暂无</td></tr>');
  }
  $('mode').textContent = s.mode;
  $('mode').className = 'v ' + (s.mode === 'LIVE' ? 'short' : 'flat');
  $('subtitle').textContent = `${s.cfg.symbol} · ${s.cfg.notional}U · x${s.cfg.lev} · 止损${(s.cfg.hard_stop*100).toFixed(0)}% / 回撤${(s.cfg.trail*100).toFixed(0)}% · 最新日线 ${s.last_bar}`;
  $('lastRefresh').textContent = '状态刷新于 ' + new Date().toLocaleTimeString();
}

async function loadCharts(force) {
  const d = await (await fetch('/api/charts')).json();
  $('expr').textContent = d.expr;
  $('meta').textContent = `seed=${d.seed} gens=${d.gens}${d.robust ? ' robust' : ''}`;

  const recent = Math.min(d.dates.length, 800);
  const dates = d.dates.slice(-recent), close = d.close.slice(-recent);
  const mk = d.markers.filter(m => dates.includes(m.date));
  chPrice.setOption(Object.assign({}, base, {
    xAxis: { ...base.xAxis, data: dates },
    yAxis: { ...base.yAxis, scale: true },
    series: [
      { name: 'ETH', type: 'line', data: close, showSymbol: false,
        lineStyle: { color: '#58a6ff', width: 1.4 }, areaStyle: { color: 'rgba(88,166,255,.08)' } },
      { name: '开多', type: 'scatter', symbolSize: 9,
        data: mk.filter(m=>m.dir>0).map(m=>[m.date, m.price]),
        itemStyle: { color: '#3fb950' } },
      { name: '开空', type: 'scatter', symbolSize: 9,
        data: mk.filter(m=>m.dir<0).map(m=>[m.date, m.price]),
        itemStyle: { color: '#f85149' } },
    ],
    legend: { textStyle: { color: '#8b949e' } },
  }));

  chEquity.setOption(Object.assign({}, base, {
    xAxis: { ...base.xAxis, data: d.dates },
    yAxis: { ...base.yAxis, type: 'log', min: 'dataMin' },
    series: [
      { name: 'GP策略', type: 'line', data: d.equity, showSymbol: false, lineStyle: { color: '#58a6ff', width: 1.5 } },
      { name: '买入持有', type: 'line', data: d.hold, showSymbol: false, lineStyle: { color: '#8b949e', width: 1.2 } },
    ],
    markLine: undefined,
    legend: { textStyle: { color: '#8b949e' } },
  }));

  chEvo.setOption(Object.assign({}, base, {
    xAxis: { ...base.xAxis, data: d.evolution.best.map((_, i) => i) },
    yAxis: base.yAxis,
    series: [
      { name: '每代最优', type: 'line', data: d.evolution.best, showSymbol: false, lineStyle: { color: '#3fb950', width: 1.5 } },
      { name: '种群均值', type: 'line', data: d.evolution.mean, showSymbol: false, lineStyle: { color: '#d29922', width: 1 } },
    ],
    legend: { textStyle: { color: '#8b949e' } },
  }));

  const r = d.results;
  let rows = '';
  for (const [k, label] of [['train','训练 2018-2023'],['test','测试 2024-2026']]) {
    const plain = r[k] && r[k].plain, risk = r[k] && r[k].risk;
    if (!plain) continue;
    for (const [tag, m] of [['裸信号', plain], ['+风控', risk]]) {
      if (!m) continue;
      rows += `<tr><td>${label} ${tag}</td><td>${fmtPct(m.total)}</td><td>${fmtPct(m.ann)}</td><td>${m.sharpe.toFixed(2)}</td><td>${m.calmar.toFixed(2)}</td><td style="color:#f85149">-${(m.mdd*100).toFixed(1)}%</td></tr>`;
    }
  }
  $('tblRes').innerHTML = '<tr><th></th><th>总收益</th><th>年化</th><th>Sharpe</th><th>Calmar</th><th>MDD</th></tr>' + rows;
}

async function loadEquity() {
  const el = await (await fetch('/api/equity')).json();
  if (!el.length) {
    chPaper.clear();
    chPaper.setOption({ graphic: [{ type: 'text', left: 'center', top: 'middle',
      style: { text: '暂无权益快照 — 点击「立即同步」开始记录', fill: '#8b949e', fontSize: 13 } }] });
    return;
  }
  chPaper.setOption(Object.assign({}, base, {
    xAxis: { ...base.xAxis, data: el.map(x => x.t) },
    yAxis: { ...base.yAxis, scale: true, splitLine: { lineStyle: { color: '#21262d' } } },
    tooltip: { ...base.tooltip, formatter: params => {
      const i = params[0].dataIndex, x = el[i];
      return `${x.t}<br/>权益 <b>${x.e}</b> U　状态 ${x.state > 0 ? '持多' : x.state < 0 ? '持空' : '空仓'}${x.live ? '　<span style=color:#f85149>LIVE</span>' : ''}`;
    }},
    series: [{
      name: '账户权益', type: 'line', data: el.map(x => x.e), showSymbol: el.length < 60, symbolSize: 5,
      lineStyle: { color: '#d29922', width: 1.6 }, itemStyle: { color: '#d29922' },
      areaStyle: { color: 'rgba(210,153,34,.12)' },
      markLine: { symbol: 'none', data: [{ yAxis: 1000, label: { formatter: '起始 1000U', color: '#8b949e' } }],
        lineStyle: { color: '#8b949e', type: 'dashed' } },
    }],
    legend: { show: false },
  }), true);
}

async function sync() {
  const b = $('btnSync'); b.disabled = true; b.textContent = '同步中…';
  try {
    const r = await (await fetch('/api/sync', { method: 'POST' })).json();
    $('log').textContent = r.log || '';
    await loadStatus(); await loadEquity(); await loadCharts();
  } finally { b.disabled = false; b.textContent = '立即同步'; }
}

async function resetPaper() {
  if (!confirm('确认重置模拟盘? 权益回到 1000U, 清空持仓与历史。')) return;
  const r = await (await fetch('/api/reset_paper', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({confirm: true}) })).json();
  if (r.ok) { await loadStatus(); await loadEquity(); await loadCharts(); }
}

initCharts();
loadStatus(); loadCharts(); loadEquity();
setInterval(loadStatus, 30000);    // 状态卡片每30秒
setInterval(loadEquity, 30000);    // 权益曲线每30秒(轻量接口)
setInterval(loadCharts, 300000);   // 全年回测/进化图每5分钟(重算慢)
</script>
</body>
</html>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    print(f"GP-Eth 上位机: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
