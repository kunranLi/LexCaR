"""Efficiency + controllability: LexCaR vs full LLM+MCP (3-round session).

Protocol
--------
Round 1 (shared): Stage1 MCP retrieval + Stage2 issue discovery.
Round 2: first focus shift (legal issue A).
Round 3: second focus shift (issue B) or an exclusion rule.

Each shift:
  LexCaR  — edit criteria / filters and re-rank the *same* pool (0 MCP).
  LLM+MCP — append the new focus to the query and re-run Stage1.

Auto Target Set (no human labels): for each shift, score the Round-1 pool with
Jev on focus relevance; Top-G = gold for that round.

Cost accounting
---------------
Counts MCP tool calls, LLM chat calls, and Jev judge calls, then converts to
estimated CNY with configurable unit prices (defaults are comparison proxies).

Usage
-----
  python -m experiment.run_efficiency_controllability --qid "00000,00002" --pool-size 15
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config
from clients.agentic_retriever import AgenticRetriever
from clients.dmx_client import DmxClient
from experiment.metrics import (
    average_precision,
    ndcg_at_k,
    overlap_count,
    precision_at_k,
    recall_at_k,
)
from stages.session import SessionState
from stages.stage1_candidate_pool import Stage1CandidatePool
from stages.stage2_experience import Stage2Experience
from stages.stage3_user_follow import Stage3UserFollow

console = Console()


# --------------------------------------------------------------------------- counters / cost
class CountingClient(DmxClient):
    """Wraps DmxClient and counts Jev / LLM HTTP calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.jev_calls = 0
        self.llm_calls = 0

    def reset_counts(self) -> None:
        self.jev_calls = 0
        self.llm_calls = 0

    def snapshot(self) -> Dict[str, int]:
        return {"jev_calls": int(self.jev_calls), "llm_calls": int(self.llm_calls)}

    def jev_judge(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self.jev_calls += 1
        return super().jev_judge(*args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        self.llm_calls += 1
        return super().chat(*args, **kwargs)


@dataclass
class UnitPrices:
    # MCP: PKULaw ~¥0.25/call estimate; LLM/Jev: amortized per call at DMXAPI scale
    mcp: float = 0.25
    llm: float = 0.005
    jev: float = 0.002

    def cost(self, mcp: int = 0, llm: int = 0, jev: int = 0) -> float:
        return round(
            mcp * self.mcp + llm * self.llm + jev * self.jev,
            4,
        )


@dataclass
class RoundCost:
    seconds: float = 0.0
    mcp_calls: int = 0
    jev_calls: int = 0
    llm_calls: int = 0
    cost_cny: float = 0.0
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_parts(
        *,
        seconds: float,
        mcp: int,
        jev: int,
        llm: int,
        prices: UnitPrices,
        note: str,
    ) -> "RoundCost":
        return RoundCost(
            seconds=round(seconds, 2),
            mcp_calls=int(mcp),
            jev_calls=int(jev),
            llm_calls=int(llm),
            cost_cny=prices.cost(mcp=mcp, llm=llm, jev=jev),
            note=note,
        )

    def __add__(self, other: "RoundCost") -> "RoundCost":
        return RoundCost(
            seconds=round(self.seconds + other.seconds, 2),
            mcp_calls=self.mcp_calls + other.mcp_calls,
            jev_calls=self.jev_calls + other.jev_calls,
            llm_calls=self.llm_calls + other.llm_calls,
            cost_cny=round(self.cost_cny + other.cost_cny, 4),
            note=(self.note + "|" + other.note).strip("|"),
        )


def empty_cost() -> RoundCost:
    return RoundCost()


def round_cost_from_dict(d: Dict[str, Any] | None) -> RoundCost:
    d = d or {}
    return RoundCost(
        seconds=float(d.get("seconds") or 0.0),
        mcp_calls=int(d.get("mcp_calls") or 0),
        jev_calls=int(d.get("jev_calls") or 0),
        llm_calls=int(d.get("llm_calls") or 0),
        cost_cny=float(d.get("cost_cny") or 0.0),
        note=str(d.get("note") or ""),
    )


def shift_from_dict(d: Dict[str, Any]) -> ShiftResult:
    return ShiftResult(
        round_id=int(d.get("round_id") or 0),
        focus_name=str(d.get("focus_name") or ""),
        mode=str(d.get("mode") or ""),
        gold_ids=[str(x) for x in (d.get("gold_ids") or [])],
        gold_size=int(d.get("gold_size") or 0),
        label_cost=round_cost_from_dict(d.get("label_cost")),
        lexcar=round_cost_from_dict(d.get("lexcar")),
        mcp=round_cost_from_dict(d.get("mcp")),
        before_metrics=dict(d.get("before_metrics") or {}),
        lexcar_metrics=dict(d.get("lexcar_metrics") or {}),
        mcp_metrics=dict(d.get("mcp_metrics") or {}),
        mcp_pool_recall=float(d.get("mcp_pool_recall") or 0.0),
        focus_lift_lexcar=float(d.get("focus_lift_lexcar") or 0.0),
        focus_lift_mcp=float(d.get("focus_lift_mcp") or 0.0),
        before_top=[str(x) for x in (d.get("before_top") or [])],
        lexcar_top=[str(x) for x in (d.get("lexcar_top") or [])],
        mcp_top=[str(x) for x in (d.get("mcp_top") or [])],
    )


def query_result_from_payload(payload: Dict[str, Any]) -> QueryResult:
    return QueryResult(
        qid=str(payload.get("qid") or ""),
        pool_size=int(payload.get("pool_size") or 0),
        round1=round_cost_from_dict(payload.get("round1")),
        shifts=[shift_from_dict(s) for s in (payload.get("shifts") or [])],
        lexcar_total=round_cost_from_dict(payload.get("lexcar_total")),
        mcp_total=round_cost_from_dict(payload.get("mcp_total")),
        annotation_total=round_cost_from_dict(payload.get("annotation_total")),
        error=str(payload.get("error") or ""),
    )


def load_results_from_dir(out_dir: Path) -> List[QueryResult]:
    rows: List[QueryResult] = []
    for path in sorted(out_dir.glob("result_*.json")):
        rows.append(query_result_from_payload(json.loads(path.read_text(encoding="utf-8"))))
    return rows


# --------------------------------------------------------------------------- helpers
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def load_queries(qids: Sequence[str] | None, n: int, max_chars: int) -> List[Dict[str, str]]:
    path = config.LECARDV2_DIR / "query.json"
    wanted = {str(x) for x in (qids or [])}
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id") or obj.get("qid") or "")
            text = (obj.get("fact") or obj.get("query") or "").strip()
            if not text:
                continue
            if wanted and qid not in wanted:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            out.append({"qid": qid, "query": text})
            if not wanted and len(out) >= n:
                break
            if wanted and len(out) >= len(wanted):
                break
    if wanted and len(out) < len(wanted):
        missing = wanted - {x["qid"] for x in out}
        raise SystemExit(f"Missing qids in query.json: {sorted(missing)}")
    return out


def _focus_from_fact(fact: Dict[str, Any], *, as_exclude: bool = False) -> Dict[str, Any]:
    name = str(fact.get("name") or fact.get("id") or "legal issue").strip()
    question = str(fact.get("question") or name).strip()
    if as_exclude:
        intent = (
            f"排除与「{name}」明显相关、或裁判说理围绕该问题展开的案例。"
            f"参考：{question}"
        )
        mode = "exclude"
        focus_q = (
            "仅根据候选案例的事实与裁判说理判断：该候选案例是否实质涉及或详细讨论"
            f"以下法律问题「{name}」？（{question}）"
            "若有明确分析则回答是（应排除），否则回答否。"
        )
    else:
        intent = (
            f"更关注与「{name}」相关的类案，优先保留裁判说理中实质讨论该问题的案例。"
            f"判定问题参考：{question}"
        )
        mode = "add_criterion"
        focus_q = (
            "仅根据候选案例的事实与裁判说理判断：该候选案例是否实质涉及或详细讨论"
            f"以下法律问题「{name}」？（问题表述：{question}）"
            "若裁判理由对该问题有明确分析则回答是，仅顺带提及或完全无关则回答否。"
        )
    return {
        "name": name,
        "question": question,
        "verdict": fact.get("verdict"),
        "established_prob": fact.get("established_prob"),
        "sufficiency_prob": fact.get("sufficiency_prob"),
        "user_intent": intent,
        "mode": mode,
        "focus_jev_question": focus_q,
    }


def pick_foci(session: SessionState, n: int = 2) -> List[Dict[str, Any]]:
    """Pick up to n distinct Stage2 issues to simulate successive focus shifts."""
    facts = list(session.confirmed_facts or [])
    if not facts:
        raise RuntimeError("Stage2 produced no confirmed_facts; cannot auto-label focus.")

    def rank_key(item: Dict[str, Any]) -> Tuple[int, float, float]:
        verdict = str(item.get("verdict") or "").lower()
        v_rank = {"yes": 0, "unknown": 1, "no": 2}.get(verdict, 3)
        return (
            v_rank,
            -float(item.get("sufficiency_prob") or 0.0),
            -float(item.get("established_prob") or 0.0),
        )

    facts_sorted = sorted(facts, key=rank_key)
    foci: List[Dict[str, Any]] = []
    seen = set()
    for fact in facts_sorted:
        focus = _focus_from_fact(fact, as_exclude=False)
        key = focus["name"]
        if key in seen:
            continue
        seen.add(key)
        foci.append(focus)
        if len(foci) >= n:
            break

    # If only one addable issue, Round 3 becomes an exclusion from a "no"/weak issue.
    if len(foci) < n:
        for fact in sorted(facts, key=rank_key, reverse=True):
            focus = _focus_from_fact(fact, as_exclude=True)
            if focus["name"] in seen and focus["mode"] == "exclude":
                # still allow exclude of a previously used name with different mode
                pass
            if any(f["name"] == focus["name"] and f["mode"] == "exclude" for f in foci):
                continue
            foci.append(focus)
            if len(foci) >= n:
                break

    while len(foci) < n and facts_sorted:
        # Duplicate last with exclude mode as last resort.
        foci.append(_focus_from_fact(facts_sorted[-1], as_exclude=True))
    return foci[:n]


def score_pool_focus_jev(
    client: CountingClient,
    query: str,
    pool: Sequence[Dict[str, Any]],
    focus_question: str,
    max_workers: int,
) -> Dict[str, float]:
    def one(case: Dict[str, Any]) -> Tuple[str, float]:
        state = config.build_jev_compare_state(query, case)
        answers = client.jev_judge(
            state=state,
            questions={"focus": {"type": "noul", "instructions": focus_question}},
        )
        return str(case["doc_id"]), float(answers["focus"].probability)

    scores: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, c) for c in pool]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Auto-label Target Set"):
            doc_id, prob = fut.result()
            scores[doc_id] = prob
    return scores


def build_target_set(
    pool: Sequence[Dict[str, Any]],
    jev_scores: Dict[str, float],
    gold_k: int,
    min_prob: float,
    *,
    invert: bool = False,
) -> Dict[str, Any]:
    """invert=True: gold = low focus score (for exclusion rounds)."""
    ranked = sorted(
        (
            {
                "doc_id": str(c["doc_id"]),
                "jev_focus": float(jev_scores.get(str(c["doc_id"]), 0.0)),
            }
            for c in pool
        ),
        key=lambda x: x["jev_focus"],
        reverse=not invert,
    )
    if invert:
        above = [x for x in ranked if x["jev_focus"] <= (1.0 - min_prob)]
    else:
        above = [x for x in ranked if x["jev_focus"] >= min_prob]
    chosen = above[:gold_k] if len(above) >= max(3, gold_k // 2) else ranked[:gold_k]
    return {
        "gold_ids": [x["doc_id"] for x in chosen],
        "gold_detail": chosen,
        "min_prob": min_prob,
        "invert": invert,
        "method": "jev_focus_topk",
    }


def top_ids(ranking: Sequence[Dict[str, Any]], k: int) -> List[str]:
    return [str(x.get("doc_id")) for x in ranking[:k] if x.get("doc_id")]


def eval_against_gold(pred_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> Dict[str, float]:
    return {
        "P@K": round(precision_at_k(pred_ids, gold_ids, k), 4),
        "R@K": round(recall_at_k(pred_ids, gold_ids, k), 4),
        "AP@K": round(average_precision(pred_ids, gold_ids, k), 4),
        "nDCG@K": round(ndcg_at_k(pred_ids, gold_ids, k), 4),
        "overlap": overlap_count(pred_ids[:k], gold_ids),
    }


def mean_focus_in_topk(ids: Sequence[str], jev_scores: Dict[str, float], k: int) -> float:
    vals = [float(jev_scores.get(str(i), 0.0)) for i in ids[:k]]
    return sum(vals) / float(len(vals) or 1)


def mcp_tool_calls(session: SessionState) -> int:
    meta = session.stage1_meta or {}
    if meta.get("tool_calls") is not None:
        return int(meta["tool_calls"])
    plans = meta.get("plans") or []
    total = 0
    for round_ in plans:
        total += len(round_.get("calls") or [])
    return total


def run_stage1(
    query: str,
    *,
    pool_size: int,
    max_rounds: int,
    llm: CountingClient,
    prices: UnitPrices,
) -> Tuple[SessionState, RoundCost]:
    llm.reset_counts()
    t0 = time.perf_counter()
    retriever = AgenticRetriever(pool_size=pool_size, max_rounds=max_rounds, llm=llm)
    stage1 = Stage1CandidatePool(
        pool_size=pool_size, max_rounds=max_rounds, retriever=retriever
    )
    session = stage1.run(query)
    counts = llm.snapshot()
    cost = RoundCost.from_parts(
        seconds=time.perf_counter() - t0,
        mcp=mcp_tool_calls(session),
        jev=counts["jev_calls"],
        llm=counts["llm_calls"],
        prices=prices,
        note="stage1",
    )
    return session, cost


@dataclass
class ShiftResult:
    round_id: int
    focus_name: str
    mode: str
    gold_ids: List[str]
    gold_size: int
    label_cost: RoundCost
    lexcar: RoundCost
    mcp: RoundCost
    before_metrics: Dict[str, float]
    lexcar_metrics: Dict[str, float]
    mcp_metrics: Dict[str, float]
    mcp_pool_recall: float
    focus_lift_lexcar: float
    focus_lift_mcp: float
    before_top: List[str]
    lexcar_top: List[str]
    mcp_top: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "focus_name": self.focus_name,
            "mode": self.mode,
            "gold_ids": self.gold_ids,
            "gold_size": self.gold_size,
            "label_cost": self.label_cost.as_dict(),
            "lexcar": self.lexcar.as_dict(),
            "mcp": self.mcp.as_dict(),
            "before_metrics": self.before_metrics,
            "lexcar_metrics": self.lexcar_metrics,
            "mcp_metrics": self.mcp_metrics,
            "mcp_pool_recall": self.mcp_pool_recall,
            "focus_lift_lexcar": self.focus_lift_lexcar,
            "focus_lift_mcp": self.focus_lift_mcp,
            "before_top": self.before_top,
            "lexcar_top": self.lexcar_top,
            "mcp_top": self.mcp_top,
        }


@dataclass
class QueryResult:
    qid: str
    pool_size: int
    round1: RoundCost = field(default_factory=empty_cost)
    shifts: List[ShiftResult] = field(default_factory=list)
    lexcar_total: RoundCost = field(default_factory=empty_cost)
    mcp_total: RoundCost = field(default_factory=empty_cost)
    annotation_total: RoundCost = field(default_factory=empty_cost)
    error: str = ""


def _focused_query(base: str, foci_so_far: Sequence[Dict[str, Any]]) -> str:
    parts = [base, "", "【本次检索重点（多轮焦点）】"]
    for i, f in enumerate(foci_so_far, start=1):
        tag = "排除" if f["mode"] == "exclude" else "关注"
        parts.append(f"{i}. [{tag}] {f['name']} — {f['question']}")
    return "\n".join(parts)


def run_one(
    item: Dict[str, str],
    *,
    pool_size: int,
    max_rounds: int,
    top_k: int,
    gold_k: int,
    min_prob: float,
    dim_weight: float,
    n_shifts: int,
    prices: UnitPrices,
    out_dir: Path,
) -> QueryResult:
    qid = item["qid"]
    query = item["query"]
    client = CountingClient(
        jev_timeout=config.JEV_TIMEOUT_STAGE2,
        llm_timeout=config.LLM_TIMEOUT,
    )
    result = QueryResult(qid=qid, pool_size=0)

    try:
        # ---- Round 1: Stage1 + Stage2 (shared) ----
        session, s1_cost = run_stage1(
            query, pool_size=pool_size, max_rounds=max_rounds, llm=client, prices=prices
        )
        client.reset_counts()
        t1 = time.perf_counter()
        stage2 = Stage2Experience(client=client)
        session = stage2.run(session)
        s2 = client.snapshot()
        s2_cost = RoundCost.from_parts(
            seconds=time.perf_counter() - t1,
            mcp=0,
            jev=s2["jev_calls"],
            llm=s2["llm_calls"],
            prices=prices,
            note="stage2",
        )
        result.round1 = s1_cost + s2_cost
        result.round1.note = "stage1+stage2"
        result.pool_size = len(session.candidate_pool)
        session.save(out_dir / f"session_{qid}_round1.json")

        foci = pick_foci(session, n=n_shifts)
        stage3 = Stage3UserFollow(client=client, top_k=top_k)
        session = stage3.ensure_dimensions(session)
        for dim in session.active_dimensions:
            if str(dim.get("source") or "") == "preset":
                dim["weight"] = 0.15

        # Baseline ranking before any shift.
        before_ids = top_ids(session.stage2_ranked or session.current_ranking, top_k)
        foci_applied: List[Dict[str, Any]] = []

        for idx, focus in enumerate(foci):
            round_id = idx + 2  # Round 2, Round 3, ...
            console.rule(f"{qid} Round {round_id}: {focus['mode']} · {focus['name']}")
            foci_applied.append(focus)

            # Auto-label Target Set on the *original* pool (fair controllability).
            client.reset_counts()
            t_label = time.perf_counter()
            jev_focus = score_pool_focus_jev(
                client,
                session.query,
                session.candidate_pool,
                focus["focus_jev_question"],
                max_workers=config.JEV_MAX_WORKERS,
            )
            label_snap = client.snapshot()
            label_cost = RoundCost.from_parts(
                seconds=time.perf_counter() - t_label,
                mcp=0,
                jev=label_snap["jev_calls"],
                llm=label_snap["llm_calls"],
                prices=prices,
                note=f"auto_label_r{round_id}",
            )
            invert = focus["mode"] == "exclude"
            target = build_target_set(
                session.candidate_pool,
                jev_focus,
                gold_k=gold_k,
                min_prob=min_prob,
                invert=invert,
            )
            gold_ids = target["gold_ids"]
            (out_dir / f"gold_{qid}_r{round_id}.json").write_text(
                json.dumps(
                    {"qid": qid, "round_id": round_id, "focus": focus, "target": target},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            before_metrics = eval_against_gold(before_ids, gold_ids, top_k)
            base_focus_mean = mean_focus_in_topk(before_ids, jev_focus, top_k)

            # LexCaR shift on same pool
            client.reset_counts()
            t_lex = time.perf_counter()
            if focus["mode"] == "exclude":
                session = stage3.add_filter(
                    session, user_intent=focus["user_intent"], rerank=True
                )
            else:
                session = stage3.add_dimension(
                    session,
                    user_intent=focus["user_intent"],
                    weight=dim_weight,
                    name=focus["name"],
                    rerank=True,
                )
            lex_snap = client.snapshot()
            lex_cost = RoundCost.from_parts(
                seconds=time.perf_counter() - t_lex,
                mcp=0,
                jev=lex_snap["jev_calls"],
                llm=lex_snap["llm_calls"],
                prices=prices,
                note=f"lexcar_r{round_id}",
            )
            lex_ids = top_ids(session.current_ranking, top_k)
            lex_metrics = eval_against_gold(lex_ids, gold_ids, top_k)
            lex_lift = round(mean_focus_in_topk(lex_ids, jev_focus, top_k) - base_focus_mean, 4)
            if invert:
                lex_lift = -lex_lift
            session.save(out_dir / f"session_{qid}_lexcar_r{round_id}.json")

            # LLM+MCP full re-run with cumulative foci
            focused_q = _focused_query(query, foci_applied)
            session_b, mcp_cost = run_stage1(
                focused_q,
                pool_size=pool_size,
                max_rounds=max_rounds,
                llm=client,
                prices=prices,
            )
            mcp_cost.note = f"mcp_r{round_id}"
            new_pool = list(session_b.candidate_pool or [])
            new_pool.sort(
                key=lambda c: (
                    float(c.get("retrieval_score") or 0.0),
                    -int(c.get("retrieval_rank") or 10**9),
                ),
                reverse=True,
            )
            mcp_ids = top_ids(new_pool, top_k)
            mcp_metrics = eval_against_gold(mcp_ids, gold_ids, top_k)
            new_id_set = {str(c["doc_id"]) for c in new_pool}
            pool_recall = round(
                len(set(gold_ids) & new_id_set) / float(len(gold_ids) or 1), 4
            )
            mcp_lift = round(
                mean_focus_in_topk(mcp_ids, jev_focus, top_k) - base_focus_mean, 4
            )
            if invert:
                mcp_lift = -mcp_lift
            session_b.save(out_dir / f"session_{qid}_mcp_r{round_id}.json")

            result.shifts.append(
                ShiftResult(
                    round_id=round_id,
                    focus_name=focus["name"],
                    mode=focus["mode"],
                    gold_ids=gold_ids,
                    gold_size=len(gold_ids),
                    label_cost=label_cost,
                    lexcar=lex_cost,
                    mcp=mcp_cost,
                    before_metrics=before_metrics,
                    lexcar_metrics=lex_metrics,
                    mcp_metrics=mcp_metrics,
                    mcp_pool_recall=pool_recall,
                    focus_lift_lexcar=lex_lift,
                    focus_lift_mcp=mcp_lift,
                    before_top=list(before_ids),
                    lexcar_top=lex_ids,
                    mcp_top=mcp_ids,
                )
            )
            before_ids = lex_ids  # next shift's "before" = LexCaR after previous edit

        # Totals: both paths share Round 1; then add their own Round 2..N
        lex_extra = empty_cost()
        mcp_extra = empty_cost()
        ann = empty_cost()
        for sh in result.shifts:
            lex_extra = lex_extra + sh.lexcar
            mcp_extra = mcp_extra + sh.mcp
            ann = ann + sh.label_cost
        result.lexcar_total = result.round1 + lex_extra
        result.lexcar_total.note = "lexcar_path_R1_to_R3"
        result.mcp_total = result.round1 + mcp_extra
        result.mcp_total.note = "mcp_path_R1_to_R3"
        result.annotation_total = ann
        result.annotation_total.note = "auto_label_only_not_in_system_totals"

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        console.print(f"[red]Query {qid} failed:[/] {exc}")

    payload = {
        "qid": result.qid,
        "pool_size": result.pool_size,
        "round1": result.round1.as_dict(),
        "shifts": [s.as_dict() for s in result.shifts],
        "lexcar_total": result.lexcar_total.as_dict(),
        "mcp_total": result.mcp_total.as_dict(),
        "annotation_total": result.annotation_total.as_dict(),
        "unit_prices_cny": asdict(prices),
        "error": result.error,
    }
    (out_dir / f"result_{qid}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def aggregate(results: Sequence[QueryResult], prices: UnitPrices) -> Dict[str, Any]:
    ok = [r for r in results if not r.error and r.shifts]
    if not ok:
        return {"n": 0, "unit_prices_cny": asdict(prices)}

    def avg(vals: Sequence[float]) -> float:
        return round(sum(vals) / float(len(vals) or 1), 4)

    shift_n = max(len(r.shifts) for r in ok)
    per_round: Dict[str, Any] = {}
    for rid in range(2, shift_n + 2):
        slices = [s for r in ok for s in r.shifts if s.round_id == rid]
        if not slices:
            continue
        per_round[f"R{rid}"] = {
            "avg_lexcar_sec": avg([s.lexcar.seconds for s in slices]),
            "avg_mcp_sec": avg([s.mcp.seconds for s in slices]),
            "avg_lexcar_mcp": avg([float(s.lexcar.mcp_calls) for s in slices]),
            "avg_mcp_mcp": avg([float(s.mcp.mcp_calls) for s in slices]),
            "avg_lexcar_llm": avg([float(s.lexcar.llm_calls) for s in slices]),
            "avg_mcp_llm": avg([float(s.mcp.llm_calls) for s in slices]),
            "avg_lexcar_jev": avg([float(s.lexcar.jev_calls) for s in slices]),
            "avg_mcp_jev": avg([float(s.mcp.jev_calls) for s in slices]),
            "avg_lexcar_cost_cny": avg([s.lexcar.cost_cny for s in slices]),
            "avg_mcp_cost_cny": avg([s.mcp.cost_cny for s in slices]),
            "avg_P@K_lexcar": avg([s.lexcar_metrics.get("P@K", 0.0) for s in slices]),
            "avg_P@K_mcp": avg([s.mcp_metrics.get("P@K", 0.0) for s in slices]),
            "avg_nDCG_lexcar": avg([s.lexcar_metrics.get("nDCG@K", 0.0) for s in slices]),
            "avg_nDCG_mcp": avg([s.mcp_metrics.get("nDCG@K", 0.0) for s in slices]),
            "avg_mcp_pool_recall": avg([s.mcp_pool_recall for s in slices]),
        }

    return {
        "n": len(ok),
        "n_shifts": shift_n,
        "unit_prices_cny": asdict(prices),
        "avg_round1": {
            "seconds": avg([r.round1.seconds for r in ok]),
            "mcp": avg([float(r.round1.mcp_calls) for r in ok]),
            "llm": avg([float(r.round1.llm_calls) for r in ok]),
            "jev": avg([float(r.round1.jev_calls) for r in ok]),
            "cost_cny": avg([r.round1.cost_cny for r in ok]),
        },
        "avg_path_total": {
            "lexcar_seconds": avg([r.lexcar_total.seconds for r in ok]),
            "mcp_seconds": avg([r.mcp_total.seconds for r in ok]),
            "latency_saving_sec": avg(
                [r.mcp_total.seconds - r.lexcar_total.seconds for r in ok]
            ),
            "lexcar_refine_seconds": avg(
                [sum(s.lexcar.seconds for s in r.shifts) for r in ok]
            ),
            "mcp_refine_seconds": avg(
                [sum(s.mcp.seconds for s in r.shifts) for r in ok]
            ),
            "refine_latency_saving_sec": avg(
                [
                    sum(s.mcp.seconds for s in r.shifts)
                    - sum(s.lexcar.seconds for s in r.shifts)
                    for r in ok
                ]
            ),
            "lexcar_mcp": avg([float(r.lexcar_total.mcp_calls) for r in ok]),
            "mcp_mcp": avg([float(r.mcp_total.mcp_calls) for r in ok]),
            "lexcar_llm": avg([float(r.lexcar_total.llm_calls) for r in ok]),
            "mcp_llm": avg([float(r.mcp_total.llm_calls) for r in ok]),
            "lexcar_jev": avg([float(r.lexcar_total.jev_calls) for r in ok]),
            "mcp_jev": avg([float(r.mcp_total.jev_calls) for r in ok]),
            "lexcar_cost_cny": avg([r.lexcar_total.cost_cny for r in ok]),
            "mcp_cost_cny": avg([r.mcp_total.cost_cny for r in ok]),
            "cost_saving_cny": avg(
                [r.mcp_total.cost_cny - r.lexcar_total.cost_cny for r in ok]
            ),
            "mcp_calls_saved": avg(
                [float(r.mcp_total.mcp_calls - r.lexcar_total.mcp_calls) for r in ok]
            ),
        },
        "per_shift_round": per_round,
        "note": (
            "Path totals = Round1 (shared) + Round2..N system cost (wall-clock seconds included). "
            "refine_*_seconds = Round2..N only (interactive wait after the pool exists). "
            "Auto-label Jev is reported separately and not added to path totals."
        ),
    }


def write_summary(results: Sequence[QueryResult], out_dir: Path, prices: UnitPrices) -> None:
    agg = aggregate(results, prices)
    rows: List[Dict[str, Any]] = []
    for r in results:
        base = {
            "qid": r.qid,
            "pool": r.pool_size,
            "r1_sec": r.round1.seconds,
            "r1_mcp": r.round1.mcp_calls,
            "r1_llm": r.round1.llm_calls,
            "r1_jev": r.round1.jev_calls,
            "r1_cost": r.round1.cost_cny,
            "lex_total_mcp": r.lexcar_total.mcp_calls,
            "lex_total_llm": r.lexcar_total.llm_calls,
            "lex_total_jev": r.lexcar_total.jev_calls,
            "lex_total_cost": r.lexcar_total.cost_cny,
            "lex_total_sec": r.lexcar_total.seconds,
            "mcp_total_mcp": r.mcp_total.mcp_calls,
            "mcp_total_llm": r.mcp_total.llm_calls,
            "mcp_total_jev": r.mcp_total.jev_calls,
            "mcp_total_cost": r.mcp_total.cost_cny,
            "mcp_total_sec": r.mcp_total.seconds,
            "lex_refine_sec": round(sum(s.lexcar.seconds for s in r.shifts), 2),
            "mcp_refine_sec": round(sum(s.mcp.seconds for s in r.shifts), 2),
            "latency_saving_sec": round(
                r.mcp_total.seconds - r.lexcar_total.seconds, 2
            ),
            "refine_latency_saving_sec": round(
                sum(s.mcp.seconds for s in r.shifts)
                - sum(s.lexcar.seconds for s in r.shifts),
                2,
            ),
            "cost_saving": round(r.mcp_total.cost_cny - r.lexcar_total.cost_cny, 4),
            "error": r.error,
        }
        for s in r.shifts:
            rid = s.round_id
            base[f"r{rid}_focus"] = s.focus_name
            base[f"r{rid}_mode"] = s.mode
            base[f"r{rid}_lex_mcp"] = s.lexcar.mcp_calls
            base[f"r{rid}_lex_llm"] = s.lexcar.llm_calls
            base[f"r{rid}_lex_jev"] = s.lexcar.jev_calls
            base[f"r{rid}_lex_cost"] = s.lexcar.cost_cny
            base[f"r{rid}_lex_sec"] = s.lexcar.seconds
            base[f"r{rid}_mcp_mcp"] = s.mcp.mcp_calls
            base[f"r{rid}_mcp_llm"] = s.mcp.llm_calls
            base[f"r{rid}_mcp_jev"] = s.mcp.jev_calls
            base[f"r{rid}_mcp_cost"] = s.mcp.cost_cny
            base[f"r{rid}_mcp_sec"] = s.mcp.seconds
            base[f"r{rid}_P_lex"] = s.lexcar_metrics.get("P@K", "")
            base[f"r{rid}_P_mcp"] = s.mcp_metrics.get("P@K", "")
            base[f"r{rid}_poolR"] = s.mcp_pool_recall
        rows.append(base)

    csv_path = out_dir / "summary.csv"
    if rows:
        # union of keys for stable header
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    md = [
        "# LexCaR vs LLM+MCP — 3-Round Efficiency, Cost & Controllability",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Protocol",
        "",
        "1. **Round 1 (shared):** Stage1 agentic MCP + Stage2 issue discovery.",
        "2. **Round 2:** focus shift A (add ranking criterion).",
        "3. **Round 3:** focus shift B (second criterion, or exclusion rule).",
        "4. Each shift: LexCaR reuses the pool; LLM+MCP re-runs Stage1 with cumulative foci.",
        "5. Auto Target Set per shift via Jev focus scoring on the Round-1 pool.",
        "",
        "## Unit prices (CNY / call, configurable)",
        "",
        f"- MCP tool call: ¥{prices.mcp}",
        f"- LLM call: ¥{prices.llm}",
        f"- Jev call: ¥{prices.jev}",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(agg, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Per-query path totals (cost + wall-clock latency)",
        "",
        "| qid | Lex MCP/LLM/Jev / ¥ / sec | LLM+MCP MCP/LLM/Jev / ¥ / sec | ¥ saved | sec saved (path) | refine sec Lex→MCP |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in results:
        lex_ref = round(sum(s.lexcar.seconds for s in r.shifts), 2)
        mcp_ref = round(sum(s.mcp.seconds for s in r.shifts), 2)
        md.append(
            f"| {r.qid} | {r.lexcar_total.mcp_calls}/{r.lexcar_total.llm_calls}/"
            f"{r.lexcar_total.jev_calls} / ¥{r.lexcar_total.cost_cny} / {r.lexcar_total.seconds}s | "
            f"{r.mcp_total.mcp_calls}/{r.mcp_total.llm_calls}/{r.mcp_total.jev_calls} "
            f"/ ¥{r.mcp_total.cost_cny} / {r.mcp_total.seconds}s | "
            f"{round(r.mcp_total.cost_cny - r.lexcar_total.cost_cny, 4)} | "
            f"{round(r.mcp_total.seconds - r.lexcar_total.seconds, 2)} | "
            f"{lex_ref}s → {mcp_ref}s |"
        )

    md.extend(["", "## Per-shift (controllability + latency)", ""])
    md.append(
        "| qid | round | focus | mode | Lex P@K | MCP P@K | poolR | Lex sec | MCP sec | Lex ¥ | MCP ¥ |"
    )
    md.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        for s in r.shifts:
            md.append(
                f"| {r.qid} | R{s.round_id} | {s.focus_name} | {s.mode} | "
                f"{s.lexcar_metrics.get('P@K', '')} | {s.mcp_metrics.get('P@K', '')} | "
                f"{s.mcp_pool_recall} | {s.lexcar.seconds} | {s.mcp.seconds} | "
                f"{s.lexcar.cost_cny} | {s.mcp.cost_cny} |"
            )

    if any(r.error for r in results):
        md.extend(["", "## Errors", ""])
        for r in results:
            if r.error:
                md.append(f"- `{r.qid}`: {r.error}")

    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = Table(title="Path totals (R1+R2+R3): cost + latency")
    for col in (
        "qid",
        "Lex ¥",
        "MCP ¥",
        "¥ saved",
        "Lex sec",
        "MCP sec",
        "sec saved",
        "refine Lex/MCP sec",
    ):
        table.add_column(col)
    for r in results:
        lex_ref = round(sum(s.lexcar.seconds for s in r.shifts), 2)
        mcp_ref = round(sum(s.mcp.seconds for s in r.shifts), 2)
        table.add_row(
            r.qid,
            str(r.lexcar_total.cost_cny),
            str(r.mcp_total.cost_cny),
            str(round(r.mcp_total.cost_cny - r.lexcar_total.cost_cny, 4)),
            str(r.lexcar_total.seconds),
            str(r.mcp_total.seconds),
            str(round(r.mcp_total.seconds - r.lexcar_total.seconds, 2)),
            f"{lex_ref}/{mcp_ref}",
        )
    console.print(table)
    console.print(f"[green]Wrote[/] {csv_path}")
    console.print(f"[green]Wrote[/] {out_dir / 'summary.md'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3, help="Number of LeCaRD queries (ignored if --qid)")
    p.add_argument("--qid", type=str, default="", help="Comma-separated query ids")
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--max-rounds", type=int, default=2, help="Stage1 agent retrieval rounds")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--gold-k", type=int, default=8, help="Target Set size per shift")
    p.add_argument("--min-prob", type=float, default=0.55, help="Jev focus threshold")
    p.add_argument("--dim-weight", type=float, default=0.7, help="Weight for new criterion")
    p.add_argument("--n-shifts", type=int, default=2, help="Focus shifts after Round1 (2 => R2+R3)")
    p.add_argument("--query-max-chars", type=int, default=1800)
    p.add_argument("--price-mcp", type=float, default=0.25, help="CNY per MCP tool call (PKULaw estimate)")
    p.add_argument("--price-llm", type=float, default=0.005, help="CNY per LLM call (DMXAPI gpt-4o-mini estimate)")
    p.add_argument("--price-jev", type=float, default=0.002, help="CNY per Jev call (DMXAPI jev ~6k tokens)")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default experiment/results/exp_eff_ctrl_<timestamp>)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip qids that already have result_<qid>.json in --out",
    )
    p.add_argument(
        "--reaggregate-only",
        action="store_true",
        help="Only rebuild summary/figures inputs from existing result_*.json in --out",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel query workers (each query still uses Jev thread pool inside)",
    )
    return p


def main() -> None:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = build_argparser().parse_args()
    prices = UnitPrices(mcp=args.price_mcp, llm=args.price_llm, jev=args.price_jev)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_root = Path(__file__).resolve().parent / "results"
    out_dir = Path(args.out) if args.out else (default_root / f"exp_eff_ctrl_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reaggregate_only:
        results = load_results_from_dir(out_dir)
        console.print(f"[bold]Reaggregate[/] n={len(results)} → {out_dir}")
        write_summary(results, out_dir, prices)
        return

    qids = [x.strip() for x in args.qid.split(",") if x.strip()] or None
    queries = load_queries(qids, n=args.n, max_chars=args.query_max_chars)
    if args.skip_existing:
        before = len(queries)
        queries = [q for q in queries if not (out_dir / f"result_{q['qid']}.json").exists()]
        console.print(f"[dim]skip-existing: {before - len(queries)} done, {len(queries)} remaining[/]")

    workers = max(1, int(args.workers))
    console.print(
        f"[bold]Experiment[/] n={len(queries)} workers={workers} pool={args.pool_size} "
        f"shifts={args.n_shifts} (=> {1 + args.n_shifts} rounds) "
        f"prices MCP/LLM/Jev=¥{prices.mcp}/{prices.llm}/{prices.jev} → {out_dir}"
    )

    def _run(item: Dict[str, str]) -> str:
        console.rule(f"qid={item['qid']}")
        run_one(
            item,
            pool_size=args.pool_size,
            max_rounds=args.max_rounds,
            top_k=args.top_k,
            gold_k=args.gold_k,
            min_prob=args.min_prob,
            dim_weight=args.dim_weight,
            n_shifts=args.n_shifts,
            prices=prices,
            out_dir=out_dir,
        )
        return item["qid"]

    if workers <= 1 or len(queries) <= 1:
        for item in queries:
            _run(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, item): item["qid"] for item in queries}
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    fut.result()
                    console.print(f"[green]done[/] qid={qid}")
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]worker failed[/] qid={qid}: {exc}")

    # Always summarize *all* results present in out_dir (old + new).
    results = load_results_from_dir(out_dir)
    write_summary(results, out_dir, prices)


if __name__ == "__main__":
    main()
