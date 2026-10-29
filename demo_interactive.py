"""Interactive demo: advance stage by stage; Stage3 can view/batch-edit criteria and weights."""

from __future__ import annotations

import json

from rich.console import Console
from rich.prompt import Confirm, Prompt

import config
from stages.stage1_candidate_pool import Stage1CandidatePool
from stages.stage2_experience import Stage2Experience
from stages.stage3_user_follow import Stage3UserFollow

console = Console()


def _parse_weight_map(raw: str) -> dict:
    """Parse weights: a JSON object, or id=0.3,id2=0.5."""
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("权重 JSON 须为对象 {id: weight}")
        return {str(k): float(v) for k, v in data.items()}
    out = {}
    for part in raw.replace("；", ",").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"无法解析权重项: {part}（格式 id=0.3）")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _parse_name_list(raw: str) -> list:
    """Comma/semicolon-separated name list."""
    return [
        x.strip()
        for x in raw.replace("；", ",").replace(";", ",").split(",")
        if x.strip()
    ]


def _stage3_loop(session, stage3: Stage3UserFollow) -> None:
    stage3.ensure_dimensions(session)
    stage3.print_status(session)

    while True:
        console.print(
            "\n命令: [bold]s[/]=查看要件/公式  [bold]d[/]=新增要件  "
            "[bold]x[/]=删除要件  [bold]w[/]=修改权重  "
            "[bold]f[/]=排除类型  [bold]xf[/]=删除过滤  "
            "[bold]r[/]=重排  [bold]q[/]=退出"
        )
        cmd = Prompt.ask(
            "选择",
            choices=["s", "d", "x", "w", "f", "xf", "r", "q"],
            default="s",
        )
        if cmd == "q":
            break
        if cmd == "s":
            stage3.print_status(session)
            continue
        try:
            if cmd == "d":
                intent = Prompt.ask("描述你关心的检索要件", default="关注与本案争议焦点直接相关的情节")
                weight = float(
                    Prompt.ask("权重", default=str(config.STAGE3_DEFAULT_DIM_WEIGHT))
                )
                do_rerank = Confirm.ask("立即按新公式重排？", default=True)
                session = stage3.add_dimension(
                    session, user_intent=intent, weight=weight, rerank=do_rerank
                )
            elif cmd == "x":
                stage3.print_status(session)
                names = _parse_name_list(
                    Prompt.ask("要删除的要件名称（逗号分隔，与表中「名称」一致）")
                )
                if not names:
                    console.print("[yellow]未输入名称[/]")
                    continue
                do_rerank = Confirm.ask("立即按新公式重排？", default=True)
                session = stage3.remove_dimensions(session, names, rerank=do_rerank)
            elif cmd == "w":
                stage3.print_status(session)
                console.print(
                    "输入示例: [dim]案由或法律关系是否相近=0.5,关键情节是否重合=0.3[/] "
                    "或 JSON [dim]{\"案由或法律关系是否相近\":0.5}[/]"
                )
                raw = Prompt.ask("修改权重（按名称）")
                weights = _parse_weight_map(raw)
                if not weights:
                    console.print("[yellow]未解析到权重[/]")
                    continue
                do_rerank = Confirm.ask("立即按新公式重排？", default=True)
                session = stage3.set_weights(session, weights, rerank=do_rerank)
            elif cmd == "f":
                intent = Prompt.ask("描述要排除的类型", default="不要与本案案由明显无关的")
                do_rerank = Confirm.ask("立即重排并应用过滤？", default=True)
                session = stage3.add_filter(session, user_intent=intent, rerank=do_rerank)
            elif cmd == "xf":
                stage3.print_status(session)
                names = _parse_name_list(
                    Prompt.ask("要删除的过滤名称（逗号分隔，与表中「名称」一致）")
                )
                if not names:
                    console.print("[yellow]未输入名称[/]")
                    continue
                do_rerank = Confirm.ask("立即重排？", default=True)
                session = stage3.remove_filters(session, names, rerank=do_rerank)
            elif cmd == "r":
                session = stage3.rerank(session)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]操作失败[/]: {exc}")


def _sample_query() -> str:
    path = config.LECARDV2_DIR / "query.json"
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            return (obj.get("fact") or obj.get("query") or "").strip()
    return ""


def main() -> None:
    console.print(
        "[bold]LexCaR[/]\n"
        "基于可编辑要件与决策模型的类案检索重排系统\n"
    )

    default_q = _sample_query()
    if default_q and Confirm.ask("使用本地样例案情？", default=False):
        console.print(f"[dim]案情预览[/]: {default_q[:160]}...")
        query = default_q
    else:
        query = Prompt.ask("请输入案情描述")

    console.rule("Stage 1 候选池搭建")
    pool_size = int(
        Prompt.ask("召回条数", default=str(config.CANDIDATE_POOL_SIZE))
    )
    stage1 = Stage1CandidatePool(pool_size=pool_size)
    session = stage1.run(query=query)

    if not Confirm.ask("继续 Stage 2？", default=True):
        return

    # Stage 2
    console.rule("Stage 2 初步经验沉淀")
    stage2 = Stage2Experience()
    session = stage2.run(session)

    console.print("\n[bold]确认事实摘要[/]  Y成立  N不成立  ?信息不足")
    for f in session.confirmed_facts:
        flag = f.get("mark") or (
            "Y"
            if f.get("established")
            else ("N" if f.get("info_sufficient") else "?")
        )
        label = {
            "yes": "成立",
            "no": "不成立",
            "unknown": "信息不足",
        }.get(f.get("verdict") or "", "")
        question = f.get("question") or f.get("name") or ""
        console.print(
            f"  {flag} {question}"
            + (f" → {label}" if label else "")
            + f": 成立={f['established_prob']:.2f}, 充分={f['sufficiency_prob']:.2f}"
        )

    if not Confirm.ask("进入 Stage 3 用户跟随？", default=True):
        return

    # Stage 3 loop
    console.rule("Stage 3 用户跟随（池内迭代 · 动态权重公式）")
    stage3 = Stage3UserFollow()
    _stage3_loop(session, stage3)

    console.print(f"\n会话已保存: outputs/session_{session.session_id}.json")


if __name__ == "__main__":
    main()
