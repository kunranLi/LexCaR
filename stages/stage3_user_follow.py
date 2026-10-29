"""Stage 3：用户跟随。

在同一候选池内：
- 新增/批量上传检索要件（维度）并动态调整权重公式
- 排除不想要的案例类型 → Jev Boolean 过滤
不重新做 dense retrieval。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import config
from clients.dmx_client import DmxClient
from stages.session import SessionState

console = Console()


class Stage3UserFollow:
    def __init__(
        self,
        client: Optional[DmxClient] = None,
        max_workers: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.client = client or DmxClient(
            jev_timeout=config.JEV_TIMEOUT_STAGE3,
            llm_timeout=config.LLM_TIMEOUT,
        )
        self.max_workers = max_workers or config.JEV_MAX_WORKERS
        self.top_k = top_k if top_k is not None else config.STAGE3_TOP_K

    # ----------------------------------------------------------- public API
    def ensure_dimensions(self, session: SessionState) -> SessionState:
        """无维度时回退到 Stage2 预设三维。"""
        if not session.active_dimensions:
            session.active_dimensions = [
                {**d, "source": "preset"} for d in config.STAGE2_DIMENSIONS
            ]
        return session

    def print_status(self, session: SessionState) -> None:
        """实时展示当前检索要件、归一化权重与打分公式。"""
        self.ensure_dimensions(session)
        dims = session.active_dimensions
        filters = session.filter_rules
        norms = self.normalized_weights(dims)

        table = Table(title="当前检索要件（维度）与权重")
        table.add_column("#", justify="right")
        table.add_column("ID")
        table.add_column("名称")
        table.add_column("原始w", justify="right")
        table.add_column("归一化w", justify="right")
        table.add_column("来源")
        table.add_column("Jev 问题")
        for i, d in enumerate(dims, start=1):
            did = str(d["id"])
            table.add_row(
                str(i),
                did,
                str(d.get("name") or ""),
                f"{float(d.get('weight', 0)):.3f}",
                f"{norms.get(did, 0.0):.3f}",
                str(d.get("source") or "-"),
                str(d.get("question") or "")[:72],
            )
        console.print(table)

        if filters:
            ft = Table(title="当前排除过滤条件")
            ft.add_column("ID")
            ft.add_column("名称")
            ft.add_column("阈值", justify="right")
            ft.add_column("Jev 问题")
            for fr in filters:
                ft.add_row(
                    str(fr.get("id")),
                    str(fr.get("name") or ""),
                    f"{float(fr.get('threshold', config.STAGE3_FILTER_THRESHOLD)):.2f}",
                    str(fr.get("question") or "")[:72],
                )
            console.print(ft)
        else:
            console.print("[dim]当前无排除过滤条件[/]")

        terms = [
            f"{norms[str(d['id'])]:.3f}*P({d.get('name') or d['id']})" for d in dims
        ]
        console.print(
            "[bold]权重公式[/]  score = sum w_i * P_i  (P_i = Jev noul)"
        )
        console.print("  score = " + (" + ".join(terms) if terms else "0"))

    @staticmethod
    def normalized_weights(dims: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        weight_sum = sum(float(d.get("weight", 0)) for d in dims) or 1.0
        return {str(d["id"]): float(d.get("weight", 0)) / weight_sum for d in dims}

    def add_dimension(
        self,
        session: SessionState,
        user_intent: str,
        weight: Optional[float] = None,
        name: Optional[str] = None,
        *,
        rerank: bool = True,
    ) -> SessionState:
        """根据用户自然语言意图，新增一个 Jev Boolean 维度。"""
        if weight is None:
            weight = config.STAGE3_DEFAULT_DIM_WEIGHT
        self.ensure_dimensions(session)
        console.print(f"[bold cyan]Stage3[/] 新增维度意图: {user_intent}")
        dim = self._intent_to_dimension(session, user_intent, weight=weight, name=name)
        dim = self._upsert_dimension(session, dim)
        console.print(f"生成问题: {dim['question']} (weight={dim['weight']})")
        self.print_status(session)
        if rerank:
            return self.rerank(session, note=f"add_dimension:{dim['id']}")
        session.save()
        return session

    def add_filter(
        self,
        session: SessionState,
        user_intent: str,
        threshold: Optional[float] = None,
        *,
        rerank: bool = True,
    ) -> SessionState:
        """根据用户排除意图，新增过滤条件（Jev 判定为 True 则剔除）。"""
        if threshold is None:
            threshold = config.STAGE3_FILTER_THRESHOLD
        console.print(f"[bold cyan]Stage3[/] 排除意图: {user_intent}")
        rule = self._intent_to_filter(session, user_intent, threshold=threshold)
        session.filter_rules.append(rule)
        console.print(f"过滤问题: {rule['question']} (threshold={threshold})")
        self.print_status(session)
        if rerank:
            return self.rerank(session, note=f"add_filter:{rule['id']}")
        session.save()
        return session

    def adjust_weight(
        self,
        session: SessionState,
        dim_id: str,
        weight: float,
        *,
        rerank: bool = True,
    ) -> SessionState:
        return self.set_weights(session, {dim_id: weight}, rerank=rerank)

    def set_weights(
        self,
        session: SessionState,
        weights: Dict[str, float],
        *,
        rerank: bool = True,
    ) -> SessionState:
        """批量设置维度原始权重（键为名称，重排时自动归一化）。"""
        self.ensure_dimensions(session)
        resolved: Dict[str, float] = {}
        missing: List[str] = []
        for key, w in weights.items():
            ids = self._resolve_dimension_keys(session, [key])
            if not ids:
                missing.append(str(key))
                continue
            for did in ids:
                resolved[did] = float(w)
        if missing:
            raise KeyError(f"未找到维度名称: {', '.join(missing)}")
        id_map = {str(d["id"]): d for d in session.active_dimensions}
        label_map = {str(d["id"]): str(d.get("name") or d["id"]) for d in session.active_dimensions}
        for dim_id, w in resolved.items():
            id_map[dim_id]["weight"] = float(w)
        note = "set_weights:" + ",".join(
            f"{label_map.get(k, k)}={resolved[k]}" for k in resolved
        )
        console.print(f"[bold cyan]Stage3[/] 已更新权重: {note}")
        self.print_status(session)
        if rerank:
            return self.rerank(session, note=note)
        session.save()
        return session

    def remove_dimensions(
        self,
        session: SessionState,
        names: Sequence[str],
        *,
        rerank: bool = True,
    ) -> SessionState:
        """批量删除检索要件（按名称；名称相同时全部删除）。"""
        drop_ids = set(self._resolve_dimension_keys(session, names))
        if not drop_ids:
            raise KeyError(f"未找到维度名称: {', '.join(str(x) for x in names)}")
        labels = [
            str(d.get("name") or d["id"])
            for d in session.active_dimensions
            if str(d["id"]) in drop_ids
        ]
        before = len(session.active_dimensions)
        session.active_dimensions = [
            d for d in session.active_dimensions if str(d["id"]) not in drop_ids
        ]
        removed = before - len(session.active_dimensions)
        if not session.active_dimensions:
            console.print("[yellow]维度已清空，将回退到预设三维[/]")
            self.ensure_dimensions(session)
        console.print(
            f"[bold cyan]Stage3[/] 已删除 {removed} 个维度: {', '.join(labels)}"
        )
        self.print_status(session)
        if rerank:
            return self.rerank(
                session, note=f"remove_dimensions:{','.join(labels)}"
            )
        session.save()
        return session

    def remove_filters(
        self,
        session: SessionState,
        names: Sequence[str],
        *,
        rerank: bool = True,
    ) -> SessionState:
        """批量删除排除过滤条件（按名称）。"""
        drop_ids = set(self._resolve_filter_keys(session, names))
        if not drop_ids:
            raise KeyError(f"未找到过滤名称: {', '.join(str(x) for x in names)}")
        labels = [
            str(f.get("name") or f["id"])
            for f in session.filter_rules
            if str(f["id"]) in drop_ids
        ]
        before = len(session.filter_rules)
        session.filter_rules = [
            f for f in session.filter_rules if str(f["id"]) not in drop_ids
        ]
        removed = before - len(session.filter_rules)
        console.print(
            f"[bold cyan]Stage3[/] 已删除 {removed} 条过滤: {', '.join(labels)}"
        )
        self.print_status(session)
        if rerank:
            return self.rerank(session, note=f"remove_filters:{','.join(labels)}")
        session.save()
        return session

    @staticmethod
    def _resolve_dimension_keys(
        session: SessionState, keys: Sequence[str]
    ) -> List[str]:
        """按名称解析维度 id（优先精确匹配名称；找不到再试 id）。"""
        want = [str(x).strip() for x in keys if str(x).strip()]
        found: List[str] = []
        seen = set()
        for key in want:
            matched = [
                str(d["id"])
                for d in session.active_dimensions
                if str(d.get("name") or "") == key
            ]
            if not matched:
                matched = [
                    str(d["id"])
                    for d in session.active_dimensions
                    if str(d["id"]) == key
                ]
            for did in matched:
                if did not in seen:
                    seen.add(did)
                    found.append(did)
        return found

    @staticmethod
    def _resolve_filter_keys(
        session: SessionState, keys: Sequence[str]
    ) -> List[str]:
        """按名称解析过滤 id（优先名称，再试 id）。"""
        want = [str(x).strip() for x in keys if str(x).strip()]
        found: List[str] = []
        seen = set()
        for key in want:
            matched = [
                str(f["id"])
                for f in session.filter_rules
                if str(f.get("name") or "") == key
            ]
            if not matched:
                matched = [
                    str(f["id"])
                    for f in session.filter_rules
                    if str(f["id"]) == key
                ]
            for fid in matched:
                if fid not in seen:
                    seen.add(fid)
                    found.append(fid)
        return found

    def apply_dimensions(
        self,
        session: SessionState,
        items: Sequence[Dict[str, Any]],
        *,
        mode: str = "merge",
        rerank: bool = True,
    ) -> SessionState:
        """批量上传/合并检索要件。

        mode:
          - merge：按 id 覆盖或追加
          - replace：用本次列表整体替换当前维度

        每项可含 id/name/weight/question；若只有 intent/user_intent，则走 LLM 生成问题。
        """
        if mode not in {"merge", "replace"}:
            raise ValueError("mode 只能是 merge 或 replace")
        if mode == "replace":
            session.active_dimensions = []
        else:
            self.ensure_dimensions(session)

        applied: List[str] = []
        for raw in items:
            dim = self._coerce_dimension_item(session, raw)
            dim = self._upsert_dimension(session, dim)
            applied.append(str(dim["id"]))

        console.print(
            f"[bold cyan]Stage3[/] 批量{'替换' if mode == 'replace' else '合并'} "
            f"{len(applied)} 个维度: {', '.join(applied)}"
        )
        self.print_status(session)
        if rerank:
            return self.rerank(session, note=f"apply_dimensions:{mode}:{len(applied)}")
        session.save()
        return session

    def apply_dimensions_from_file(
        self,
        session: SessionState,
        path: Union[str, Path],
        *,
        mode: str = "merge",
        rerank: bool = True,
    ) -> SessionState:
        """从 JSON 文件批量上传维度（数组，或 {{"dimensions":[...]}}）。"""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("dimensions") or data.get("items") or []
            mode = str(data.get("mode") or mode)
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("维度文件须为 JSON 数组或含 dimensions 字段的对象")
        if not isinstance(items, list) or not items:
            raise ValueError(f"维度文件为空或格式错误: {path}")
        console.print(f"[bold cyan]Stage3[/] 从文件加载维度: {path}")
        return self.apply_dimensions(session, items, mode=mode, rerank=rerank)

    def rerank(self, session: SessionState, note: str = "rerank") -> SessionState:
        if not session.candidate_pool:
            raise RuntimeError("候选池为空，请先运行 Stage1")
        self.ensure_dimensions(session)
        self.print_status(session)

        console.print(
            f"[bold cyan]Stage3[/] 池内重排 | 维度={len(session.active_dimensions)} "
            f"| 过滤={len(session.filter_rules)}"
        )

        dims = session.active_dimensions
        filters = session.filter_rules
        norm_weights = self.normalized_weights(dims)

        dim_questions = {
            d["id"]: {"type": "noul", "instructions": d["question"]} for d in dims
        }
        filter_questions = {
            f["id"]: {"type": "noul", "instructions": f["question"]} for f in filters
        }
        all_questions = {**dim_questions, **filter_questions}

        def score_one(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            state = config.build_jev_compare_state(session.query, case)
            answers = self.client.jev_judge(state=state, questions=all_questions)

            # 过滤：若「应排除」概率 >= threshold，则丢弃
            for fr in filters:
                prob = answers[fr["id"]].probability if fr["id"] in answers else 0.0
                if prob >= float(
                    fr.get("threshold", config.STAGE3_FILTER_THRESHOLD)
                ):
                    return None

            dim_scores = {
                d["id"]: answers[d["id"]].probability if d["id"] in answers else 0.0
                for d in dims
            }
            score = sum(dim_scores[k] * norm_weights[k] for k in norm_weights)
            out = {
                "doc_id": case["doc_id"],
                "charges": case.get("charges") or [],
                "fact": case.get("fact") or "",
                "result": case.get("result") or "",
                "dim_scores": dim_scores,
                "weighted_score": score,
                "retrieval_score": case.get("retrieval_score", case.get("dense_score")),
                "norm_weights": dict(norm_weights),
            }
            return out

        scored: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(score_one, c) for c in session.candidate_pool]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="池内 Jev 重排"):
                item = fut.result()
                if item is not None:
                    scored.append(item)

        scored.sort(key=lambda x: x["weighted_score"], reverse=True)
        for i, item in enumerate(scored, start=1):
            item["rank"] = i

        session.current_ranking = scored
        session.stage3_history.append(
            {
                "note": note,
                "dimensions": [dict(d) for d in session.active_dimensions],
                "norm_weights": dict(norm_weights),
                "filters": [dict(f) for f in session.filter_rules],
                "top_ids": [x["doc_id"] for x in scored[: self.top_k]],
            }
        )
        self._print_ranking(scored[: self.top_k])
        path = session.save()
        console.print(f"[bold green]Stage3 完成[/] 剩余候选={len(scored)}，已保存: {path}")
        return session

    # ---------------------------------------------------- intent → jev q
    def _intent_to_dimension(
        self,
        session: SessionState,
        user_intent: str,
        weight: float,
        name: Optional[str],
    ) -> Dict[str, Any]:
        existing = [d["question"] for d in session.active_dimensions]
        existing_block = "- " + "\n- ".join(existing[:10]) if existing else "（无）"
        system = config.PROMPT_LLM_STAGE3_DIM_SYSTEM
        user = config.PROMPT_LLM_STAGE3_DIM_USER.format(
            query=session.query[: config.STAGE3_LLM_QUERY_MAX_CHARS],
            user_intent=user_intent,
            existing_questions=existing_block,
            example_json=config.PROMPT_LLM_STAGE3_DIM_EXAMPLE_JSON,
        )
        data = self.client.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        dim_id = str(data.get("id") or f"dim_{len(session.active_dimensions)+1}")
        return {
            "id": dim_id,
            "name": name or str(data.get("name") or user_intent[:20]),
            "weight": float(weight),
            "question": str(
                data.get("question")
                or config.PROMPT_JEV_STAGE3_DIM_FALLBACK.format(user_intent=user_intent)
            ),
            "source": "user",
            "user_intent": user_intent,
        }

    def _intent_to_filter(
        self,
        session: SessionState,
        user_intent: str,
        threshold: float,
    ) -> Dict[str, Any]:
        system = config.PROMPT_LLM_STAGE3_FILTER_SYSTEM
        user = config.PROMPT_LLM_STAGE3_FILTER_USER.format(
            query=session.query[: config.STAGE3_LLM_QUERY_MAX_CHARS],
            user_intent=user_intent,
            example_json=config.PROMPT_LLM_STAGE3_FILTER_EXAMPLE_JSON,
        )
        data = self.client.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return {
            "id": str(data.get("id") or f"flt_{len(session.filter_rules)+1}"),
            "name": str(data.get("name") or user_intent[:20]),
            "question": str(
                data.get("question")
                or config.PROMPT_JEV_STAGE3_FILTER_FALLBACK.format(
                    user_intent=user_intent
                )
            ),
            "threshold": float(threshold),
            "source": "user",
            "user_intent": user_intent,
        }

    def _coerce_dimension_item(
        self, session: SessionState, raw: Dict[str, Any]
    ) -> Dict[str, Any]:
        """把上传项规范为维度 dict；缺 question 时用 intent 调 LLM。"""
        intent = str(raw.get("user_intent") or raw.get("intent") or "").strip()
        question = str(raw.get("question") or "").strip()
        weight = float(
            raw["weight"]
            if raw.get("weight") is not None
            else config.STAGE3_DEFAULT_DIM_WEIGHT
        )
        name = raw.get("name")
        if question:
            dim_id = str(
                raw.get("id")
                or f"dim_{len(session.active_dimensions) + 1}"
            )
            return {
                "id": dim_id,
                "name": str(name or dim_id),
                "weight": weight,
                "question": question,
                "source": str(raw.get("source") or "upload"),
                "user_intent": intent or question,
            }
        if not intent:
            raise ValueError(
                "维度项须提供 question，或提供 intent/user_intent 供 LLM 生成"
            )
        return self._intent_to_dimension(
            session, user_intent=intent, weight=weight, name=str(name) if name else None
        )

    @staticmethod
    def _upsert_dimension(
        session: SessionState, dim: Dict[str, Any]
    ) -> Dict[str, Any]:
        """按 id 覆盖已有维度，否则追加。"""
        dim_id = str(dim["id"])
        for i, old in enumerate(session.active_dimensions):
            if str(old["id"]) == dim_id:
                merged = dict(old)
                merged.update(dim)
                session.active_dimensions[i] = merged
                return merged
        session.active_dimensions.append(dim)
        return dim

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _print_ranking(ranked: List[Dict[str, Any]]) -> None:
        table = Table(title="Stage3 当前 Top 排序")
        table.add_column("Rank", justify="right")
        table.add_column("DocID")
        table.add_column("Score", justify="right")
        table.add_column("事实摘要")
        for c in ranked:
            fact = (c.get("fact") or "")[:48]
            table.add_row(
                str(c.get("rank")),
                str(c.get("doc_id")),
                f"{c.get('weighted_score', 0):.3f}",
                fact + ("..." if len(c.get("fact") or "") > 48 else ""),
            )
        console.print(table)
