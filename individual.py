"""Tree genome: random generation, mutation, crossover, formatting."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .primitives import BINARY, PERIODS, TERMINALS, TS1, UNARY

MAX_DEPTH = 6


def _rand_terminal(rng: random.Random) -> tuple:
    return (rng.choice(TERMINALS),)


def _rand_ts(rng: random.Random, depth: int, full: bool) -> tuple:
    n = rng.choice(PERIODS)
    return (rng.choice(TS1), _grow(rng, depth + 1, full), n)


def _grow(rng: random.Random, depth: int = 0, full: bool = False) -> tuple:
    if depth >= MAX_DEPTH - 1 or (not full and depth >= 2 and rng.random() < 0.3):
        return _rand_terminal(rng)
    r = rng.random()
    if r < 0.25:
        return _rand_terminal(rng)
    if r < 0.50:
        return (rng.choice(UNARY), _grow(rng, depth + 1, full))
    if r < 0.70:
        return _rand_ts(rng, depth, full)
    return (rng.choice(BINARY), _grow(rng, depth + 1, full), _grow(rng, depth + 1, full))


def random_tree(rng: random.Random) -> tuple:
    return _grow(rng, full=rng.random() < 0.5)


def nodes(tree: tuple) -> list[tuple]:
    out = [tree]
    for child in tree[1:]:
        if isinstance(child, tuple):
            out.extend(nodes(child))
    return out


def count_nodes(tree: tuple) -> int:
    return 1 + sum(count_nodes(c) for c in tree[1:] if isinstance(c, tuple))


def get_subtrees(tree: tuple) -> list[tuple]:
    return nodes(tree)


def replace_subtree(tree: tuple, target: tuple, new: tuple) -> tuple:
    if tree is target:
        return new
    kids = tuple(replace_subtree(c, target, new) if isinstance(c, tuple) else c
                 for c in tree[1:])
    return (tree[0],) + kids


def find_nodes(tree: tuple, pred) -> list[tuple]:
    return [n for n in nodes(tree) if pred(n)]


def param_mutate(tree: tuple, rng: random.Random) -> tuple:
    ts_nodes = find_nodes(tree, lambda n: n[0] in TS1)
    if not ts_nodes or rng.random() < 0.35:
        return swap_op(tree, rng)
    node = rng.choice(ts_nodes)
    others = [p for p in PERIODS if p != node[2]]
    new_node = (node[0], node[1], rng.choice(others))
    return replace_subtree(tree, node, new_node)


def swap_op(tree: tuple, rng: random.Random) -> tuple:
    candidates = find_nodes(
        tree,
        lambda n: (n[0] in TS1) or (n[0] in BINARY) or (n[0] in UNARY),
    )
    if not candidates:
        return tree
    node = rng.choice(candidates)
    if node[0] in TS1:
        pool = TS1
    elif node[0] in BINARY:
        pool = BINARY
    else:
        pool = UNARY
    new_op = rng.choice(pool)
    if new_op == node[0]:
        return tree
    return replace_subtree(tree, node, (new_op,) + node[1:])


def subtree_mutate(tree: tuple, rng: random.Random) -> tuple:
    victim = rng.choice(get_subtrees(tree))
    return replace_subtree(tree, victim, _grow(rng, depth=rng.randint(0, 2)))


def hoist_mutate(tree: tuple, rng: random.Random) -> tuple:
    kids = [n for n in get_subtrees(tree) if n is not tree]
    if not kids:
        return tree
    return rng.choice(kids)


def mutate(tree: tuple, rng: random.Random) -> tuple:
    r = rng.random()
    if r < 0.45:
        return param_mutate(tree, rng)
    if r < 0.75:
        return subtree_mutate(tree, rng)
    if r < 0.90:
        return swap_op(tree, rng)
    return hoist_mutate(tree, rng)


def crossover(a: tuple, b: tuple, rng: random.Random) -> tuple:
    kids_a = get_subtrees(a)
    kids_b = get_subtrees(b)
    if len(kids_a) < 2 or len(kids_b) < 2:
        return a
    victim = rng.choice(kids_a[1:])
    donor = rng.choice(kids_b[1:])
    child = replace_subtree(a, victim, donor)
    if count_nodes(child) > 60:
        return a
    return child


def to_expr(tree: tuple) -> str:
    op = tree[0]
    if len(tree) == 1:
        return op
    if op in UNARY:
        return f"{op}({to_expr(tree[1])})"
    if op in BINARY:
        return f"{op}({to_expr(tree[1])}, {to_expr(tree[2])})"
    return f"{op}({to_expr(tree[1])}, {tree[2]})"


def tree_from_json(obj) -> tuple:
    if isinstance(obj, list):
        return tuple(tree_from_json(x) for x in obj)
    return obj


def to_position(raw: np.ndarray, smooth: int = 3, warmup: int = 200) -> np.ndarray:
    """Rolling zscore of the raw tree output -> tanh -> bounded position [-1,1].

    The first ``warmup - 1`` bars are forced flat: a 2-observation z-score is
    numerically extreme and would otherwise create huge fake positions at the
    start of every evaluation window (and would not match Pine, where the
    200-bar SMA is na during warmup).
    """
    from .primitives import _zscore
    z = _zscore(raw, warmup)
    if warmup > 1:
        z[: warmup - 1] = 0.0
    pos = np.tanh(z)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        # 'full' convolution is causal here: y[t] uses pos[t], pos[t-1], ...
        pos = np.convolve(pos, k, mode="full")[: len(pos)]
    return np.clip(np.nan_to_num(pos, nan=0.0), -1.0, 1.0)
