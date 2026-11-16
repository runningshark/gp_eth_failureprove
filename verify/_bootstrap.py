# -*- coding: utf-8 -*-
"""把本仓库当作 `gp_eth` 包导入，无论目录叫什么名字。

Python 的模块名不能以数字开头，而本仓库目录可能叫 `1_gp_eth_failureprove`。
这里直接给这个目录注册一个名为 `gp_eth` 的包，于是 `from gp_eth.xxx import ...`
在任何目录名下都能工作。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 仓库根目录
sys.path.insert(0, str(ROOT.parent))                   # 便于 `import gp_eth` 直接命中

try:
    import gp_eth  # noqa: F401
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "gp_eth",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp_eth"] = module
    spec.loader.exec_module(module)
