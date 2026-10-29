"""LexCaR three-stage pipeline entrypoint.

Usage examples:
  python pipeline.py --stage all --query "..."
  python pipeline.py --stage 1 --query "..."
  python pipeline.py --stage 2 --session outputs/session_xxx.json
  python pipeline.py --stage 3 --session outputs/session_xxx.json --add-dim "关注合同解除后的返还" --exclude "不要行政案件"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

import config
from stages.session import SessionState
from stages.stage1_candidate_pool import Stage1CandidatePool
from stages.stage2_experience import Stage2Experience
from stages.stage3_user_follow import Stage3UserFollow

console = Console()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LexCaR: user-centered legal case retrieval with editable ranking criteria"
    )
    p.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which stage to run (all=1→2→3)",
    )
    p.add_argument("--query", type=str, default="", help="User case description (required for Stage1/all)")
    p.add_argument("--qid", type=str, default="", help="Use a query id from local query.json as the case text")
    p.add_argument("--session", type=str, default="", help="Path to an existing session JSON")
    p.add_argument("--pool-size", type=int, default=None, help="Stage1 recall pool size")
    p.add_argument("--add-dim", type=str, default="", help="Stage3: add a ranking dimension (natural language)")
    p.add_argument("--dim-weight", type=float, default=0.3, help="Weight for the newly added dimension")
    p.add_argument("--exclude", type=str, default="", help="Stage3: exclude case type (natural language)")
    p.add_argument("--demo", action="store_true", help="Use the built-in sample case description")
    return p


_DEMO_QUERY = (
    "被告人张某潜入他人住宅盗窃现金约一万元，案发后退还部分赃款，是否可从轻？"
)


def resolve_query(args: argparse.Namespace) -> str:
    if args.query and args.query.strip():
        return args.query.strip()
    if args.demo:
        return _DEMO_QUERY
    if args.qid:
        text = _read_local_query(args.qid)
        if text:
            return text
        raise SystemExit(f"未在 query.json 中找到 qid={args.qid}")
    raise SystemExit("请通过 --query 提供案情，或用 --qid / --demo。")


def _read_local_query(qid: str) -> str:
    path = config.LECARDV2_DIR / "query.json"
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            current = str(obj.get("id") or obj.get("qid") or "")
            if current == str(qid):
                return (obj.get("fact") or obj.get("query") or "").strip()
    return ""


def main() -> None:
    args = build_argparser().parse_args()

    session: SessionState | None = None
    if args.session:
        session = SessionState.load(Path(args.session))
        console.print(f"已加载会话: {args.session}")

    # -------- Stage 1 --------
    if args.stage in ("1", "all"):
        query = resolve_query(args)
        stage1 = Stage1CandidatePool(pool_size=args.pool_size)
        session = stage1.run(query=query, session=session)
        if args.stage == "1":
            return

    assert session is not None, "Stage2/3 需要先有会话（--session 或先跑 Stage1）"

    # -------- Stage 2 --------
    if args.stage in ("2", "all"):
        stage2 = Stage2Experience()
        session = stage2.run(session)
        if args.stage == "2":
            return

    # -------- Stage 3 --------
    if args.stage in ("3", "all"):
        stage3 = Stage3UserFollow()
        # When running the built-in criminal demo end-to-end, demo one add-dim and one exclude
        use_demo_follow = args.demo and args.stage == "all" and not args.add_dim and not args.exclude
        add_dim = args.add_dim or (
            "退还赃款/退赃情节对本案很关键，优先保留有退赃的类案" if use_demo_follow else ""
        )
        exclude = args.exclude or ("不要入户盗窃的案例" if use_demo_follow else "")

        if add_dim:
            session = stage3.add_dimension(
                session, user_intent=add_dim, weight=args.dim_weight
            )
        if exclude:
            session = stage3.add_filter(session, user_intent=exclude)

        if not add_dim and not exclude:
            # Rerank once with the current dimensions only
            session = stage3.rerank(session, note="manual_rerank")

    console.print(f"[bold]会话文件[/]: outputs/session_{session.session_id}.json")


if __name__ == "__main__":
    main()
