"""LexCaR 三阶段流程入口。

用法示例：
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
        description="LexCaR：以用户为中心、可编辑排序标准的法律案例检索系统"
    )
    p.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="运行哪一阶段（all=1→2→3）",
    )
    p.add_argument("--query", type=str, default="", help="用户案情描述（Stage1/all 必需）")
    p.add_argument("--qid", type=str, default="", help="使用本地 query.json 中的 query id 作为案情")
    p.add_argument("--session", type=str, default="", help="已有会话 JSON 路径")
    p.add_argument("--pool-size", type=int, default=None, help="Stage1 召回条数")
    p.add_argument("--add-dim", type=str, default="", help="Stage3：新增判断维度（自然语言）")
    p.add_argument("--dim-weight", type=float, default=0.3, help="新增维度权重")
    p.add_argument("--exclude", type=str, default="", help="Stage3：排除案例类型（自然语言）")
    p.add_argument("--demo", action="store_true", help="使用内置样例案情")
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
        # 内置刑事样例跑全流程时，演示一次新增要件和排除
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
            # 仅按当前维度重排一次
            session = stage3.rerank(session, note="manual_rerank")

    console.print(f"[bold]会话文件[/]: outputs/session_{session.session_id}.json")


if __name__ == "__main__":
    main()
