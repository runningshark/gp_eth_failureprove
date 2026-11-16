"""统一入口：目录不必叫 gp_eth，也能跑所有策略与实验。

Python 的模块名不能以数字开头，而本仓库目录叫 `1_gp_eth_failureprove`，
所以 `py -m gp_eth.trend` 这种写法在原始目录名下用不了。这个脚本把仓库
注册成 `gp_eth` 包，然后转发到对应模块的 `main()`。

用法（在仓库根目录，也就是本文件所在目录）：

    py main.py                      # 列出所有命令
    py main.py tests                # 回归测试（27 个）
    py main.py trend --start 2022-01-01
    py main.py fast4h
    py main.py aggressive
    py main.py range_breakout
    py main.py benchmarks
    py main.py run --pop 100 --gens 40
    py main.py experiments --seeds 0-9 --modes full,folds
    py main.py walkforward
    py main.py null_calibration
    py main.py cross_asset
    py main.py dashboard
    py main.py plot_results
    py main.py live_runner

独立复核脚本不走这里，直接 `py verify/xx.py`。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMMANDS = {
    "tests": "回归测试（27 个）",
    "trend": "日线趋势跟踪（TSMOM）",
    "fast4h": "4h 短线动量",
    "aggressive": "4h 长线杠杆",
    "range_breakout": "横盘突破（已修复阈值与持有逻辑）",
    "benchmarks": "基线对比（B&H / MA / 波动目标）",
    "run": "跑一次 GP 演化 + 留出",
    "experiments": "多种子 / 多目标对比",
    "walkforward": "滚动 walk-forward",
    "null_calibration": "零假设校准（块 bootstrap）",
    "cross_asset": "跨品种检验",
    "dashboard": "结果面板",
    "plot_results": "画权益曲线",
    "live_runner": "paper / live 信号生成",
}


def register_package() -> None:
    """把本目录注册成 gp_eth 包，与目录名无关。"""
    spec = importlib.util.spec_from_file_location(
        "gp_eth",
        HERE / "__init__.py",
        submodule_search_locations=[str(HERE)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp_eth"] = module
    spec.loader.exec_module(module)


def usage() -> None:
    print(__doc__.strip().split("用法", 1)[0].strip())
    print("\n可用命令（py main.py <命令> [参数]）：\n")
    width = max(len(k) for k in COMMANDS)
    for name, desc in COMMANDS.items():
        print(f"  {name:{width}s}  {desc}")
    print("\n例：py main.py trend --start 2022-01-01")


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "-l", "--list"):
        usage()
        return 0

    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"未知命令：{name}\n")
        usage()
        return 1

    register_package()
    module = importlib.import_module(f"gp_eth.{name}")
    sys.argv = [f"main.py {name}", *rest]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
