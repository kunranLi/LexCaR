"""Stage 2：初步经验沉淀。

1) 对候选池用预设三维 Jev Boolean 加权打分，取 Top-10；
2) 轻量 LLM 读取 Top-10 + 用户 query，自动推导结构化问题；
3) Jev 依据用户 query 逐一回答，输出「要件成立概率」+「信息充分性概率」。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config
from clients.dmx_client import DmxClient
from stages.session import SessionState

console = Console()

# 判决主文之后的执行、告知事项，不能用来做判决前预判。
_POST_JUDGMENT = re.compile(
    r"判决执行|刑期|折抵|上诉|缴纳罚金|判决生效|罚金于|剥夺政治权利|权利告知|诉讼费"
)
# 问句需要点明法律要件，不能只是生活事实。
_LEGAL_ELEMENT = re.compile(
    r"要件|罪的|罪中|罪所|案由|手段|目的|加重|从宽|从重|减轻|自首|坦白|立功|"
    r"退赔|退赃|谅解|责任|违约|侵权|违法|职权|过错|因果关系|"
    r"既遂|未遂|中止|主犯|从犯|入户|数额巨大|不能反抗|丧失反抗|自动投案|如实供述"
)
_VAGUE_ELEMENT = re.compile(
    r"构成要件|犯罪情节|从轻或从重|从轻、从重|可从轻或从重|证据充分|证据是否确实|"
    r"量刑适当|辩护意见|争议焦点|关键事实|法律关系|裁判结果是否|处理基础|案件性质|"
    r"是否符合.{0,16}条件"
)


class Stage2Experience:
    def __init__(
        self,
        client: Optional[DmxClient] = None,
        dimensions: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.client = client or DmxClient(
            jev_timeout=config.JEV_TIMEOUT_STAGE2,
            llm_timeout=config.LLM_TIMEOUT,
        )
        self.dimensions = dimensions or config.STAGE2_DIMENSIONS
        self.top_k = top_k or config.TOP_K_AFTER_STAGE2
        self.max_workers = max_workers or config.JEV_MAX_WORKERS
        self.checklist_top_n = config.STAGE2_CHECKLIST_TOP_N
        self.case_brief_max_chars = config.STAGE2_CASE_BRIEF_MAX_CHARS
        self.fact_prob_threshold = config.STAGE2_FACT_PROB_THRESHOLD

    # ----------------------------------------------------------- public API
    def run(self, session: SessionState) -> SessionState:
        if not session.candidate_pool:
            raise RuntimeError("候选池为空，请先运行 Stage1")

        console.print("[bold cyan]Stage2.1[/] 三维 Jev Boolean 重排候选池 ...")
        ranked = self.rerank_with_preset_dimensions(session.query, session.candidate_pool)
        session.stage2_ranked = ranked[: self.top_k]
        session.current_ranking = [
            {
                "doc_id": x["doc_id"],
                "score": x["weighted_score"],
                "rank": i + 1,
                "dim_scores": x["dim_scores"],
            }
            for i, x in enumerate(session.stage2_ranked)
        ]
        self._print_top(session.stage2_ranked)

        console.print("[bold cyan]Stage2.2[/] 轻量 LLM 推导本案结构化判断问题 ...")
        questions = self.derive_structured_questions(session.query, session.stage2_ranked)
        console.print(f"推导出 {len(questions)} 个结构化问题")

        console.print("[bold cyan]Stage2.3[/] Jev 确认事实（成立概率 + 信息充分性）...")
        facts = self.confirm_facts(session.query, questions)
        session.confirmed_facts = facts
        # 将预设维度写入 active_dimensions，供 Stage3 继续迭代
        session.active_dimensions = [
            {
                "id": d["id"],
                "name": d["name"],
                "weight": d["weight"],
                "question": d["question"],
                "source": "preset",
            }
            for d in self.dimensions
        ]
        self._print_facts(facts)
        path = session.save()
        console.print(f"[bold green]Stage2 完成[/] 已保存: {path}")
        return session

    # ----------------------------------------------------- 2.1 preset rerank
    def rerank_with_preset_dimensions(
        self,
        query: str,
        pool: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        questions = {
            d["id"]: {"type": "noul", "instructions": d["question"]}
            for d in self.dimensions
        }
        weight_map = {d["id"]: float(d["weight"]) for d in self.dimensions}

        def score_one(case: Dict[str, Any]) -> Dict[str, Any]:
            state = config.build_jev_compare_state(query, case)
            answers = self.client.jev_judge(state=state, questions=questions)
            dim_scores = {qid: ans.probability for qid, ans in answers.items()}
            weighted = sum(dim_scores.get(k, 0.0) * w for k, w in weight_map.items())
            out = dict(case)
            out["dim_scores"] = dim_scores
            out["weighted_score"] = weighted
            return out

        scored: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(score_one, c): c["doc_id"] for c in pool}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Jev 三维打分"):
                scored.append(fut.result())

        scored.sort(key=lambda x: x["weighted_score"], reverse=True)
        for i, item in enumerate(scored, start=1):
            item["stage2_rank"] = i
        return scored

    # ------------------------------------------- 2.2 derive structured qs
    def derive_structured_questions(
        self,
        query: str,
        top_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        # query 保留参数以兼容调用方；预判清单不再喂入用户案情
        _ = query
        n = min(self.checklist_top_n, len(top_cases))
        case_briefs = []
        for i, c in enumerate(top_cases[:n], start=1):
            adjudication = config.extract_adjudication_text(
                c, self.case_brief_max_chars
            )
            title = (c.get("title") or "").strip()
            head = f"{i}. [{c['doc_id']}]"
            if title:
                head += f" {title}"
            case_briefs.append(f"{head}\n裁判摘要:{adjudication}")

        system = config.PROMPT_LLM_CHECKLIST_SYSTEM
        user = config.PROMPT_LLM_CHECKLIST_USER.format(
            top_k=n,
            case_briefs="\n\n".join(case_briefs),
            example_json=config.PROMPT_LLM_CHECKLIST_EXAMPLE_JSON,
            min_questions=config.STAGE2_CHECKLIST_MIN_QUESTIONS,
            max_questions=config.STAGE2_CHECKLIST_MAX_QUESTIONS,
        )
        normalized = self._request_checklist(system, user)
        kept = [q for q in normalized if _is_atomic_element(q)]
        rejected = [q for q in normalized if not _is_atomic_element(q)]
        minimum = config.STAGE2_CHECKLIST_MIN_QUESTIONS
        if rejected or len(kept) < minimum:
            lines = [f"- {q['name']}：{q['question']}" for q in rejected]
            kept_lines = [f"- {q['name']}：{q['question']}" for q in kept]
            retry_user = (
                user
                + "\n\n需要拆开或删除的复合问题：\n"
                + ("\n".join(lines) if lines else "无")
                + "\n已经合格、不要重复的问题：\n"
                + ("\n".join(kept_lines) if kept_lines else "无")
                + f"\n把没有点明法律要件的问题改写成说理中的要件，删掉判决主文和执行事项。"
                + f"每个问题只能有一个要件，问句里要写出要件名称。"
                + f"合计至少 {minimum} 个。"
            )
            retried = self._request_checklist(system, retry_user)
            extra = [q for q in retried if _is_atomic_element(q)]
            kept = _dedupe_questions(kept + extra)
        chosen = kept
        if not chosen:
            chosen = _normalize_questions(list(config.PROMPT_LLM_CHECKLIST_FALLBACK))
        return _renumber([q for q in chosen if q["question"]])[
            : config.STAGE2_CHECKLIST_MAX_QUESTIONS
        ]

    def _request_checklist(self, system: str, user: str) -> List[Dict[str, str]]:
        try:
            data = self.client.chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]要件抽取失败[/]: {exc}")
            return []
        questions = data.get("questions") if isinstance(data, dict) else data
        if not isinstance(questions, list):
            return []
        return _normalize_questions(questions)

    # ------------------------------------------------ 2.3 confirm facts
    def confirm_facts(
        self,
        query: str,
        questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """对每个结构化问题，用两个 noul 分别估成立概率与信息充分性。"""
        # 一次请求打包所有问题，降低调用次数；指令模板来自 .env
        jev_questions: Dict[str, Dict[str, Any]] = {}
        for q in questions:
            qid = q["id"]
            jev_questions[f"{qid}__established"] = {
                "type": "noul",
                "instructions": config.PROMPT_JEV_FACT_ESTABLISHED.format(
                    question=q["question"]
                ),
            }
            jev_questions[f"{qid}__sufficiency"] = {
                "type": "noul",
                "instructions": config.PROMPT_JEV_FACT_SUFFICIENCY.format(
                    question=q["question"]
                ),
            }

        state = config.PROMPT_JEV_FACT_STATE.format(
            query=query[: config.QUERY_TEXT_MAX_CHARS]
        )
        answers = self.client.jev_judge(state=state, questions=jev_questions)

        facts: List[Dict[str, Any]] = []
        for q in questions:
            qid = q["id"]
            est = answers.get(f"{qid}__established")
            suf = answers.get(f"{qid}__sufficiency")
            established = bool(
                est and est.probability >= self.fact_prob_threshold
            )
            info_sufficient = bool(
                suf and suf.probability >= self.fact_prob_threshold
            )
            # 信息充分且成立→成立；充分但不成立→不成立；信息不充分→未知
            # 使用 ASCII 标记：Windows GBK 控制台无法编码 ✓/✗，会导致 Stage2 中断
            if not info_sufficient:
                verdict = "unknown"
                mark = "?"
            elif established:
                verdict = "yes"
                mark = "Y"
            else:
                verdict = "no"
                mark = "N"
            facts.append(
                {
                    "id": qid,
                    "name": q["name"],
                    "question": q["question"],
                    "rationale": q.get("rationale", ""),
                    "established_prob": est.probability if est else 0.0,
                    "sufficiency_prob": suf.probability if suf else 0.0,
                    "established": established,
                    "info_sufficient": info_sufficient,
                    "verdict": verdict,
                    "mark": mark,
                }
            )
        return facts

    # ------------------------------------------------------------- helpers
    def _print_top(self, ranked: List[Dict[str, Any]]) -> None:
        table = Table(title="Stage2 Top 相关案例")
        table.add_column("Rank", justify="right")
        table.add_column("DocID")
        table.add_column("Score", justify="right")
        table.add_column("标题/摘要")
        table.add_column("三维分")
        for i, c in enumerate(ranked[: self.top_k], start=1):
            dims = c.get("dim_scores") or {}
            dim_str = ", ".join(f"{k}:{v:.2f}" for k, v in dims.items())
            title = (c.get("title") or "").strip()
            if not title:
                title = ((c.get("fact") or "")[:40] + "...") if c.get("fact") else "-"
            table.add_row(
                str(i),
                str(c.get("doc_id")),
                f"{c.get('weighted_score', 0):.3f}",
                title[:48],
                dim_str,
            )
        try:
            console.print(table)
        except UnicodeEncodeError:
            pass

    @staticmethod
    def _print_facts(facts: List[Dict[str, Any]]) -> None:
        table = Table(title="确认事实（Y成立 / N不成立 / ?信息不足）")
        table.add_column("判定")
        table.add_column("ID")
        table.add_column("问题")
        table.add_column("成立P", justify="right")
        table.add_column("充分P", justify="right")
        for f in facts:
            table.add_row(
                f.get("mark") or "?",
                f["id"],
                f.get("question") or f.get("name") or "",
                f"{f['established_prob']:.2f}",
                f"{f['sufficiency_prob']:.2f}",
            )
        try:
            console.print(table)
        except UnicodeEncodeError:
            pass


def _normalize_questions(questions: List[Any]) -> List[Dict[str, str]]:
    normalized = []
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue
        normalized.append(
            {
                "id": str(q.get("id") or f"q{i}"),
                "name": str(q.get("name") or f"问题{i}").strip(),
                "question": str(q.get("question") or q.get("text") or "").strip(),
                "rationale": str(q.get("rationale") or "").strip(),
            }
        )
    return [q for q in normalized if q["question"]]


def _is_vague_element(question: Dict[str, str]) -> bool:
    text = f"{question.get('name') or ''}{question.get('question') or ''}"
    return bool(_VAGUE_ELEMENT.search(text))


def _is_compound_element(question: Dict[str, str]) -> bool:
    """一个问句里用连词或逗号绑了第二个事实。"""
    text = question.get("question") or ""
    if text.count("是否") > 1:
        return True
    if re.search(r"[，；]|并且|以及|同时|而且|且|而(?!是|已|非)", text):
        return True
    return bool(re.search(r"并(?!非|列)", text))


def _is_atomic_element(question: Dict[str, str]) -> bool:
    text = f"{question.get('name') or ''}{question.get('question') or ''}"
    if not question.get("question"):
        return False
    if _is_vague_element(question) or _is_compound_element(question):
        return False
    if _POST_JUDGMENT.search(text):
        return False
    return bool(_LEGAL_ELEMENT.search(text))


def _dedupe_questions(questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for question in questions:
        key = re.sub(r"\s+", "", question.get("question") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(question)
    return out


def _renumber(questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    numbered = []
    for i, question in enumerate(questions, start=1):
        item = dict(question)
        item["id"] = f"q{i}"
        numbered.append(item)
    return numbered
