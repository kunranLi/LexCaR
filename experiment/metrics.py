"""Ranking metrics for LexCaR vs LLM+MCP experiments."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def _as_list(ids: Iterable[str]) -> List[str]:
    return [str(x) for x in ids]


def precision_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    g: Set[str] = set(_as_list(gold))
    hit = sum(1 for x in _as_list(pred)[:k] if x in g)
    return hit / float(k)


def recall_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = _as_list(gold)
    if not g:
        return 0.0
    gset = set(g)
    hit = sum(1 for x in _as_list(pred)[:k] if x in gset)
    return hit / float(len(g))


def average_precision(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    gset = set(_as_list(gold))
    if not gset:
        return 0.0
    hit = 0
    total = 0.0
    for i, doc in enumerate(_as_list(pred)[:k], start=1):
        if doc in gset:
            hit += 1
            total += hit / float(i)
    return total / float(min(len(gset), k))


def ndcg_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    gset = set(_as_list(gold))
    if not gset or k <= 0:
        return 0.0

    def dcg(seq: Sequence[str]) -> float:
        s = 0.0
        for i, doc in enumerate(_as_list(seq)[:k], start=1):
            rel = 1.0 if doc in gset else 0.0
            s += rel / math.log2(i + 1)
        return s

    ideal = dcg(list(gset)[:k])
    if ideal <= 0:
        return 0.0
    return dcg(pred) / ideal


def overlap_count(a: Sequence[str], b: Sequence[str]) -> int:
    return len(set(_as_list(a)) & set(_as_list(b)))
