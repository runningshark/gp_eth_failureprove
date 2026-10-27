"""Render the rolling walk-forward equity/drawdown chart as a dependency-free SVG.

Reads out/walkforward_results.json (written by walkforward.py) and writes a
self-contained HTML page.  No matplotlib / Pillow needed.

Usage (from the package parent):
    py -m gp_eth.plot_results [--data ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

COLORS = {
    "median": "#2563eb",
    "median_risk": "#dc2626",
    "buy_hold": "#6b7280",
    "ma10_50": "#059669",
}
LABELS = {
    "median": "GP 中位数集成",
    "median_risk": "GP + 风险 overlay",
    "buy_hold": "买入持有",
    "ma10_50": "MA 10/50",
}

W, H = 1040, 920
ML, MR = 78, 28
P1_TOP, P1_BOT = 96, 452
P2_TOP, P2_BOT = 540, 736


def render(data: dict) -> str:
    dates = data["series"]["dates"]
    n = len(dates)
    series = data["series"]
    meta = data["ensemble"]
    pre = data["ex_2026"]

    eq_all = [v for k in series if k != "dates" for v in series[k]]
    vmin, vmax = min(eq_all), max(eq_all)
    pad = (vmax - vmin) * 0.06
    vmin, vmax = vmin - pad, vmax + pad

    dd = {}
    for k in ("median", "median_risk", "buy_hold"):
        eq = series[k]
        peak, m, out = [], 0.0, []
        for v in eq:
            m = max(m, v)
            peak.append(m)
            out.append(1.0 - v / m)
        dd[k] = out
    dd_max = max(max(v) for v in dd.values()) * 1.08

    def x(i: int) -> float:
        return ML + i / max(1, n - 1) * (W - ML - MR)

    def y1(v: float) -> float:
        return P1_BOT - (v - vmin) / (vmax - vmin) * (P1_BOT - P1_TOP)

    def y2(d: float) -> float:
        return P2_TOP + d / dd_max * (P2_BOT - P2_TOP)

    def poly(vals, yf) -> str:
        return " ".join(f"{x(i):.1f},{yf(v):.1f}" for i, v in enumerate(vals))

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    add = []
    add.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
               f'viewBox="0 0 {W} {H}" font-family="Segoe UI, system-ui, sans-serif">')
    add.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    add.append(f'<text x="{ML}" y="30" font-size="20" font-weight="600" fill="#111827">'
               'gp_eth 滚动样本外 2022-01 → 2026-10</text>')
    add.append(
        f'<text x="{ML}" y="52" font-size="12.5" fill="#4b5563">'
        f'5 seeds · fold fitness · 中位数集成 ｜ 年化 {meta["median"]["ann"]:+.1%} · '
        f'MDD {meta["median"]["mdd"]:.1%} · Calmar {meta["median"]["calmar"]:.2f} · '
        f't {meta["median"]["t_stat"]:.2f} ｜ 剔除 2026：年化 {pre["median"]["ann"]:+.1%} · '
        f'MDD {pre["median"]["mdd"]:.1%}</text>')

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = vmin + frac * (vmax - vmin)
        yy = y1(v)
        add.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" '
                   'stroke="#e5e7eb" stroke-width="1"/>')
        add.append(f'<text x="{ML-10}" y="{yy+4:.1f}" font-size="11" fill="#6b7280" '
                   f'text-anchor="end">{v:.2f}</text>')
    add.append(f'<text x="{ML}" y="{P1_TOP-16}" font-size="12" font-weight="600" '
               'fill="#374151">净值（初始 = 1.00）</text>')

    i26 = next((i for i, d in enumerate(dates) if d >= "2026-01-01"), n - 1)
    add.append(f'<line x1="{x(i26):.1f}" y1="{P1_TOP}" x2="{x(i26):.1f}" y2="{P2_BOT}" '
               'stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 4"/>')
    add.append(f'<text x="{x(i26)+6:.1f}" y="{P2_BOT-6}" font-size="11" '
               'fill="#6b7280">2026 起</text>')

    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        d = frac * dd_max
        yy = y2(d)
        add.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" '
                   'stroke="#e5e7eb" stroke-width="1"/>')
        add.append(f'<text x="{ML-10}" y="{yy+4:.1f}" font-size="11" fill="#6b7280" '
                   f'text-anchor="end">-{d:.0%}</text>')
    add.append(f'<text x="{ML}" y="{P2_TOP-16}" font-size="12" font-weight="600" '
               'fill="#374151">回撤</text>')

    for i, d in enumerate(dates):
        if d[5:] == "01-01":
            add.append(f'<line x1="{x(i):.1f}" y1="{P1_TOP}" x2="{x(i):.1f}" y2="{P1_BOT}" '
                       'stroke="#f3f4f6" stroke-width="1"/>')
            add.append(f'<text x="{x(i):.1f}" y="{P2_BOT+22}" font-size="11" '
                       f'fill="#6b7280" text-anchor="middle">{d[:4]}</text>')

    for k in ("buy_hold", "ma10_50", "median_risk", "median"):
        width = 2.4 if k == "median" else (1.6 if k == "median_risk" else 1.3)
        dash = "" if k in ("median", "median_risk") else ' stroke-dasharray="5 3"'
        add.append(f'<polyline points="{poly(series[k], y1)}" fill="none" '
                   f'stroke="{COLORS[k]}" stroke-width="{width}"{dash} stroke-linejoin="round"/>')

    for k in ("buy_hold", "median_risk", "median"):
        width = 2.2 if k == "median" else 1.4
        add.append(f'<polyline points="{poly(dd[k], y2)}" fill="none" '
                   f'stroke="{COLORS[k]}" stroke-width="{width}" stroke-linejoin="round"/>')

    for i, k in enumerate(("median", "median_risk", "buy_hold", "ma10_50")):
        lx = ML + 180 + i * 185
        add.append(f'<line x1="{lx}" y1="76" x2="{lx+22}" y2="76" '
                   f'stroke="{COLORS[k]}" stroke-width="3"/>')
        add.append(f'<text x="{lx+30}" y="80" font-size="11.5" fill="#374151">'
                   f'{esc(LABELS[k])}</text>')

    ty = 800
    add.append(f'<text x="{ML}" y="{ty-14}" font-size="12" font-weight="600" '
               'fill="#374151">分年表现</text>')
    for head, dx in (("年份", 0), ("策略", 90), ("Sharpe", 180), ("回撤", 260),
                     ("+overlay", 350), ("B&amp;H", 450)):
        add.append(f'<text x="{ML+dx}" y="{ty}" font-size="11.5" font-weight="600" '
                   f'fill="#4b5563">{head}</text>')
    for i, (yr, rec) in enumerate(data["years"].items()):
        yy = ty + 20 + i * 18
        if i % 2 == 0:
            add.append(f'<rect x="{ML-6}" y="{yy-13}" width="{W-ML-MR+6}" height="18" '
                       'fill="#f9fafb"/>')
        m, r, b = rec["ensemble_median"], rec["ensemble_median_risk"], rec["buy_hold"]
        label = f'{yr}{"*" if yr == "2026" else ""}'
        for val, dx, color in (
            (label, 0, "#111827"),
            (f'{m["total"]:+.1%}', 90, "#2563eb"),
            (f'{m["sharpe"]:.2f}', 180, "#111827"),
            (f'{m["mdd"]:.1%}', 260, "#111827"),
            (f'{r["total"]:+.1%}', 350, "#dc2626"),
            (f'{b["total"]:+.1%}', 450, "#6b7280"),
        ):
            add.append(f'<text x="{ML+dx}" y="{yy}" font-size="11.5" fill="{color}">{val}</text>')
    add.append(f'<text x="{ML}" y="{H-8}" font-size="10.5" fill="#9ca3af">'
               '* 2026 为 1-10 月部分年份。收益不含资金费率；手续费 5 bp/边。</text>')
    add.append("</svg>")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>gp_eth OOS</title></head>"
            "<body style='margin:0;background:#fff'>" + "".join(add) + "</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "walkforward_results.json"))
    ap.add_argument("--out", default=str(OUT / "walkforward_equity.html"))
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if "series" not in data:
        raise SystemExit("no 'series' in results; re-run walkforward.py first")
    html = render(data)
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"written: {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
