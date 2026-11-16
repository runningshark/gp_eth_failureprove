# -*- coding: utf-8 -*-
"""00 — 跑一遍包自带的回归测试（27 个）。

用法（在仓库根目录运行）：
    py verify/00_package_tests.py
"""
import _bootstrap  # noqa: F401

from gp_eth import tests

if __name__ == "__main__":
    tests.main()
