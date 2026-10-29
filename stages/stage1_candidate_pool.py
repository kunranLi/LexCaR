"""Stage 1：候选池搭建。

用户输入案情 → 大模型按案例检索技能拆查询 → MCP 案例库召回指定条数。
"""

from __future__ import annotations

from typing import Callable, Optional

from rich.console import Console
from rich.table import Table

import config
from clients.agentic_retriever import AgenticRetriever
from stages.session import SessionState

console = Console()


class Stage1CandidatePool:
    def __init__(
        self,
        pool_size: Optional[int] = None,
        max_rounds: Optional[int] = None,
        retriever: Optional[AgenticRetriever] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.pool_size = pool_size or config.CANDIDATE_POOL_SIZE
        self.max_rounds = max_rounds or config.STAGE1_AGENT_MAX_ROUNDS
        self.retriever = retriever or AgenticRetriever(
            pool_size=self.pool_size,
            max_rounds=self.max_rounds,
            on_progress=on_progress,
        )
        if on_progress is not None and retriever is not None:
            self.retriever._on_progress = on_progress

    def run(
        self,
        query: str,
        session: Optional[SessionState] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> SessionState:
        """按案情检索，把指定条数的候选写入会话。"""
        if on_progress is not None:
            self.retriever._on_progress = on_progress
        result = self.retriever.search(query)
        pool = result.cases

        session = session or SessionState.create(query=query)
        session.query = query
        session.source = result.source
        session.candidate_pool = pool
        session.stage1_meta = result.meta
        session.current_ranking = [
            {
                "doc_id": item["doc_id"],
                "score": item.get("retrieval_score"),
                "rank": item.get("retrieval_rank"),
            }
            for item in pool
        ]
        path = session.save()
        _print_pool(pool)
        console.print(
            f"[bold green]Stage1 完成[/] 候选池大小={len(pool)}，已保存: {path}"
        )
        return session


def _print_pool(pool: list) -> None:
    table = Table(title="候选池", show_lines=False)
    table.add_column("#", justify="right", width=4)
    table.add_column("案号/ID", overflow="fold")
    table.add_column("标题", overflow="fold")
    table.add_column("来源", width=16)
    for case in pool[:10]:
        table.add_row(
            str(case.get("retrieval_rank") or ""),
            str(case.get("doc_id") or ""),
            str(case.get("title") or "")[:40],
            str(case.get("source_db") or ""),
        )
    if len(pool) > 10:
        table.caption = f"仅展示前 10 条，共 {len(pool)} 条"
    console.print(table)
