# -*- coding: utf-8 -*-
"""01 — GP 留出到底是多少?

复现 REPORT.md §5.1:
  * 原始树 (out/run_log.json):      训练 +8697.7%，留出 -5.3% / Sharpe 0.17 / MDD 66.0%
  * fixed_seed42 树:                训练 +19966%， 留出 -58.4% / MDD 72.2%

并对比「实盘一致预热（全历史算 z-score 再切片）」与「切分点重置预热」。

用法（目录必须命名为 gp_eth，在仓库根目录运行）：
    py verify/01_gp_holdout.py
"""
import json
import sys

import numpy as np

import _bootstrap  # noqa: E402,F401  把仓库注册为 gp_eth 包

from gp_eth.backtest import apply_risk_overlay, summarize
from gp_eth.experiments import load_data
from gp_eth.individual import to_position, tree_from_json
from gp_eth.primitives import build_context, eval_tree

ROOT = _bootstrap.ROOT


def positions(df, tree):
    ctx = build_context(df)
    pos = to_position(np.nan_to_num(eval_tree(tree, ctx), nan=0.0, posinf=0.0, neginf=0.0))
    return ctx, pos


def main():
    df = load_data(ROOT / "cache_ETHUSDT_1d_3000.csv")
    test_mask = (df["open_time"] >= "2024-01-01").to_numpy()
    print(f"数据 {len(df)} 根, 训练 {int((~test_mask).sum())}, 留出 {int(test_mask.sum())}\n")

    cases = [
        ("原始树 run_log.json", "run_log.json", -0.053, 86.977),
        ("seed42 run_log.fixed_seed42.json", "run_log.fixed_seed42.json", -0.584, 199.663),
    ]
    ok = True
    for label, fname, expect_test, expect_train in cases:
        log = json.load(open(ROOT / "out" / fname, encoding="utf-8"))
        tree = tree_from_json(log["tree"])

        # A. 实盘一致：全历史预热，再切片
        ctx, pos = positions(df, tree)
        close = ctx["close"]
        train = summarize(close[~test_mask], pos[~test_mask])
        test = summarize(close[test_mask], pos[test_mask])
        risk = summarize(close[test_mask],
                         apply_risk_overlay(close[test_mask], pos[test_mask]))

        # B. 切分点重置预热（原 run_log 的口径）
        te = df[test_mask].reset_index(drop=True)
        ctx_te = build_context(te)
        pos_te = to_position(np.nan_to_num(eval_tree(tree, ctx_te), nan=0.0, posinf=0.0, neginf=0.0))
        reset = summarize(ctx_te["close"], pos_te)

        print(f"--- {label}")
        print(f"    expr          : {log['expr'][:88]}...")
        print(f"    训练 实盘一致  : total={train['total']:+10.1%} ann={train['ann']:+7.1%} "
              f"sharpe={train['sharpe']:.2f}")
        print(f"    留出 实盘一致  : total={test['total']:+7.1%} sharpe={test['sharpe']:.2f} "
              f"mdd={test['mdd']:.1%}")
        print(f"    留出 +风控     : total={risk['total']:+7.1%} sharpe={risk['sharpe']:.2f}")
        print(f"    留出 重置预热  : total={reset['total']:+7.1%} sharpe={reset['sharpe']:.2f}")

        hit_test = abs(test["total"] - expect_test) < 0.005
        hit_train = abs(train["total"] - expect_train) < 0.5
        ok &= hit_test and hit_train
        print(f"    预期 留出 {expect_test:+.1%} → {'PASS' if hit_test else 'FAIL'}；"
              f"训练 {expect_train:+.1f}% → {'PASS' if hit_train else 'FAIL'}")
        if label.startswith("原始"):
            hit_risk = abs(risk["total"] - 0.048) < 0.005
            ok &= hit_risk
            print(f"    +风控预期 +4.8% → {'PASS' if hit_risk else 'FAIL'}")
        print()

    print("结论：实盘一致口径下，两棵树的留出都是负的；"
          "报告里的 +5.6% 无法在一致口径下复现。")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
