"""Evolution engine: tournament selection, elitism, crossover/mutation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .backtest import fitness, summarize
from .individual import count_nodes, crossover, mutate, random_tree, to_expr, to_position
from .primitives import build_context, eval_tree


@dataclass
class Individual:
    tree: tuple
    score: float = -1e9
    metrics: dict | None = None

    @property
    def expr(self) -> str:
        return to_expr(self.tree)

    @property
    def size(self) -> int:
        return count_nodes(self.tree)


@dataclass
class EvolutionLog:
    gen: list = field(default_factory=list)
    best: list = field(default_factory=list)
    mean: list = field(default_factory=list)


def evaluate(ind: Individual, ctx: dict, robust: bool = False) -> float:
    try:
        raw = eval_tree(ind.tree, ctx)
        if not np.all(np.isfinite(raw)):
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        pos = to_position(raw)
        if robust:
            n = len(pos)
            mid = n // 2
            m1 = summarize(ctx["close"][:mid], pos[:mid])
            m2 = summarize(ctx["close"][mid:], pos[mid:])
            if m1 is None or m2 is None:
                raise ValueError("window failed")
            s1, s2 = fitness(m1), fitness(m2)
            ind.metrics = {k: 0.5 * (m1[k] + m2[k])
                           for k in ("total", "ann", "sharpe", "mdd", "calmar",
                                     "turnover", "flips")}
            ind.metrics["size"] = ind.size
            ind.score = min(s1, s2) + 0.25 * 0.5 * (s1 + s2)
        else:
            raw_metrics = summarize(ctx["close"], pos)
            ind.metrics = raw_metrics
            if ind.metrics is not None:
                ind.metrics["size"] = ind.size
            ind.score = fitness(ind.metrics)
    except Exception:
        ind.metrics = None
        ind.score = -1e9
    return ind.score


def evolve(df, pop_size=100, generations=30, tournament_k=5,
           seed=42, verbose=True, robust=False) -> tuple[Individual, EvolutionLog]:
    ctx = build_context(df)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    pop = [Individual(random_tree(rng)) for _ in range(pop_size)]
    for ind in pop:
        evaluate(ind, ctx, robust=robust)

    log = EvolutionLog()
    best_ever = max(pop, key=lambda i: i.score)

    for gen in range(generations):
        scores = [i.score for i in pop]
        ranked = sorted(pop, key=lambda i: i.score, reverse=True)
        elite = ranked[: max(2, pop_size // 20)]
        log.gen.append(gen)
        log.best.append(ranked[0].score)
        log.mean.append(float(np.mean(scores)))
        if verbose:
            b = ranked[0]
            m = b.metrics or {}
            print(f"[gen {gen:02d}] best={b.score:8.3f} "
                  f"sharpe={m.get('sharpe', float('nan')):6.2f} "
                  f"calmar={m.get('calmar', float('nan')):6.2f} "
                  f"mdd={m.get('mdd', float('nan')):6.2f} "
                  f"ret={m.get('total', float('nan')):8.1%} size={b.size}")

        next_pop = [Individual(tree=i.tree, score=i.score, metrics=i.metrics) for i in elite]
        while len(next_pop) < pop_size:
            p1 = _tournament(pop, tournament_k, rng)
            r = rng.random()
            if r < 0.55:
                p2 = _tournament(pop, tournament_k, rng)
                child_tree = crossover(p1.tree, p2.tree, rng)
                if rng.random() < 0.6:
                    child_tree = mutate(child_tree, rng)
            else:
                child_tree = mutate(p1.tree, rng)
            child = Individual(tree=child_tree)
            evaluate(child, ctx, robust=robust)
            next_pop.append(child)
        pop = next_pop

        cur_best = max(pop, key=lambda i: i.score)
        if cur_best.score > best_ever.score:
            best_ever = Individual(tree=cur_best.tree, score=cur_best.score,
                                   metrics=cur_best.metrics)

    if verbose:
        print(f"\nBEST: {best_ever.expr}")
        m = best_ever.metrics
        print(f"  sharpe={m['sharpe']:.2f} calmar={m['calmar']:.2f} "
              f"mdd={m['mdd']:.2%} total={m['total']:.1%} flips={m['flips']}")
    return best_ever, log


def _tournament(pop, k, rng) -> Individual:
    picks = [rng.choice(pop) for _ in range(k)]
    return max(picks, key=lambda i: i.score)
