"""Stage1 agentic retrieval: LLM plans queries via skills; MCP recall + dedupe."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console

import config
from clients.dmx_client import DmxClient
from clients.mcp_http import McpHttpClient, McpHttpError, content_to_text
from clients.mcp_registry import McpServerSpec, load_mcp_servers
from clients.skill_registry import SkillSpec, load_skills

console = Console()

_CASE_NO_RE = re.compile(r"[（(][12][0-9]{3}[）)][^\s，。；;、]{2,40}号")

_FIELD_ALIASES = {
    "doc_id": (
        "doc_id", "case_no", "caseNo", "case_number", "caseNumber",
        "anhao", "案号", "ah", "case_id", "caseId", "caseflag", "CaseFlag",
    ),
    "title": ("title", "case_title", "caseTitle", "标题", "案件名称", "name"),
    "court": (
        "court", "court_name", "courtName", "courthouse", "courthouse_name",
        "courthouseName", "法院", "审理法院", "终审法院",
    ),
    "fact": (
        "fact", "facts", "案情", "基本案情", "事实", "case_fact",
        "fact_summary", "summary", "摘要", "案件事实", "ascertain", "本院查明",
    ),
    "reason": (
        "reason", "法院认为", "本院认为", "裁判理由", "court_opinion",
        "reasoning", "judge_reason", "opinion", "裁判要旨", "identified",
    ),
    "result": (
        "result", "裁判结果", "判决结果", "judgment", "verdict",
        "judgement", "refereeResult",
    ),
    "fulltext": (
        "fulltext", "content", "text", "全文", "snippet", "裁判片段",
        "abstract", "highlight", "hitContent", "hit_content",
    ),
    "url": ("url", "link", "href", "链接", "detail_url", "doc_url", "detailUrl"),
    "charges": (
        "charges", "charge", "案由", "cause", "case_cause", "caseCause",
        "cause_of_action", "causeOfAction", "category", "Category",
    ),
    "articles": ("articles", "法条", "laws", "legal_basis", "引用法条"),
}


@dataclass
class AgenticSearchResult:
    cases: List[Dict[str, Any]]
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)


class AgenticRetriever:
    """Retrieve by user case text; recall size is pool_size."""

    def __init__(
        self,
        pool_size: Optional[int] = None,
        llm: Optional[DmxClient] = None,
        max_rounds: Optional[int] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.pool_size = pool_size or config.CANDIDATE_POOL_SIZE
        self.max_rounds = max_rounds or config.STAGE1_AGENT_MAX_ROUNDS
        self.llm = llm or DmxClient(llm_timeout=config.LLM_TIMEOUT)
        self._servers = load_mcp_servers()
        self._skills = load_skills()
        self._dead: set[str] = set()
        self._calls = 0
        self._on_progress = on_progress

    def _progress(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if self._on_progress is not None:
            try:
                self._on_progress(text)
            except Exception:  # noqa: BLE001
                pass

    def search(self, query: str) -> AgenticSearchResult:
        if not self._servers:
            raise RuntimeError(
                "未配置可用的案例库 MCP。请检查 mcp/ 下的 mcpServers，并在 .env 中填写对应令牌。"
            )
        query = (query or "").strip()
        if len(query) < 4:
            raise RuntimeError("案情过短，无法检索。")

        self._discover_tools()
        if not any(server.tools for server in self._servers):
            raise RuntimeError("已连接的 MCP 没有发现可用工具。")
        selected = self._select_skills(query)
        skill_text = _skill_text(selected)
        tool_text = _tool_catalog(self._servers)

        seen: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        plans: List[Dict[str, Any]] = []
        used_calls: List[str] = []

        stage1_msg = (
            f"Stage1 agent retrieval · target {self.pool_size}"
            f" (MCP: {', '.join(server.name for server in self._servers)})"
        )
        console.print(f"[bold cyan]Stage1[/] agent retrieval · target {self.pool_size}"
                      f" (MCP: {', '.join(server.name for server in self._servers)})")
        self._progress(stage1_msg)
        calls = self._plan(query, skill_text, tool_text)
        for round_i in range(1, self.max_rounds + 1):
            if not calls:
                calls = self._fallback_calls(query, used_calls)
            plans.append({"calls": calls})
            round_msg = f"Round {round_i} · {len(calls)} tool call(s)"
            console.print(f"[cyan]Round {round_i}[/] · {len(calls)} tool call(s)")
            self._progress(round_msg)
            hits, round_errors = self._execute(calls)
            errors.extend(round_errors)
            for call in calls:
                used_calls.append(_call_key(call))
            for hit in hits:
                key = _dedupe_key(hit)
                if not key or key in seen:
                    continue
                seen[key] = hit
                if len(seen) >= self.pool_size:
                    break
            pool_msg = f"Candidate pool {len(seen)} / {self.pool_size}"
            console.print(f"  {pool_msg}")
            self._progress(pool_msg)
            if len(seen) >= self.pool_size:
                break
            if self._all_dead():
                break
            if round_i >= self.max_rounds or self._calls >= config.STAGE1_MAX_TOOL_CALLS:
                break
            calls = self._plan(
                query,
                skill_text,
                tool_text,
                already=used_calls,
                found=len(seen),
                titles=[c.get("title") or c.get("doc_id") or "" for c in list(seen.values())[:8]],
            )

        cases = list(seen.values())[: self.pool_size]
        for rank, case in enumerate(cases, start=1):
            score = round(max(0.01, 1.0 - (rank - 1) / max(self.pool_size, 1)), 4)
            case["retrieval_rank"] = rank
            case["retrieval_score"] = score
            case["dense_rank"] = rank
            case["dense_score"] = score

        if not cases:
            detail = "；".join(errors[:4]) or "案例库没有返回可用案例"
            raise RuntimeError(f"智能体检索未获得候选。{detail}")

        used_sources: List[str] = []
        for case in cases:
            src = str(case.get("source_db") or "")
            if src and src not in used_sources:
                used_sources.append(src)
        source = "agentic:" + "+".join(
            used_sources or [server.name for server in self._servers]
        )
        if errors:
            console.print("[yellow]部分检索调用未成功[/]")
            for err in errors[:6]:
                console.print(f"  [dim]{err[:180]}[/]")
        return AgenticSearchResult(
            cases=cases,
            source=source,
            meta={
                "pool_size": self.pool_size,
                "returned": len(cases),
                "plans": plans,
                "skills": [skill.name for skill in selected],
                "tool_calls": self._calls,
                "errors": errors[:12],
                "dead_backends": sorted(self._dead),
            },
        )

    def _discover_tools(self) -> None:
        alive: List[McpServerSpec] = []
        for server in self._servers:
            try:
                server.tools = server.client.list_tools()
                console.print(f"  [dim]{server.name}[/] tools discovered: {len(server.tools)}")
                self._progress(f"{server.name} tools discovered")
                if server.tools:
                    alive.append(server)
            except McpHttpError as exc:
                self._note_dead(server.name, exc)
        self._servers = alive

    def _select_skills(self, query: str) -> List[SkillSpec]:
        if not self._skills:
            console.print("[yellow]未发现技能[/] 请在 skills/<名称>/SKILL.md 添加")
            return []
        catalog = "\n".join(
            f"- {skill.name}: {skill.description}" for skill in self._skills
        )
        messages = [
            {"role": "system", "content": config.PROMPT_STAGE1_SKILL_SELECT_SYSTEM},
            {
                "role": "user",
                "content": config.PROMPT_STAGE1_SKILL_SELECT_USER.format(
                    query=query[: config.QUERY_TEXT_MAX_CHARS],
                    catalog=catalog,
                    example_json=config.PROMPT_STAGE1_SKILL_SELECT_EXAMPLE_JSON,
                ),
            },
        ]
        names: List[str] = []
        try:
            data = self.llm.chat_json(
                messages, temperature=config.STAGE1_PLAN_TEMPERATURE
            )
            raw = data.get("skills") if isinstance(data, dict) else []
            if isinstance(raw, list):
                names = [str(item).strip() for item in raw if str(item).strip()]
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]技能选择失败，改用全部技能[/]: {exc}")
            names = [skill.name for skill in self._skills]
        by_name = {skill.name: skill for skill in self._skills}
        chosen = [by_name[name] for name in names if name in by_name]
        if not chosen:
            chosen = list(self._skills)
        skill_msg = "Selected skills: " + ", ".join(skill.name for skill in chosen)
        console.print(skill_msg)
        self._progress(skill_msg)
        return chosen

    def _plan(
        self,
        query: str,
        skill_text: str,
        tool_text: str,
        already: Optional[List[str]] = None,
        found: int = 0,
        titles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        followup = ""
        if already:
            followup = config.PROMPT_STAGE1_PLAN_FOLLOWUP.format(
                already=json.dumps(already, ensure_ascii=False),
                found=found,
                target=self.pool_size,
                titles=json.dumps(titles or [], ensure_ascii=False),
            )
        messages = [
            {
                "role": "system",
                "content": config.PROMPT_STAGE1_PLAN_SYSTEM
                + "\n\n"
                + (skill_text or "没有额外技能。只根据案情调用案例检索工具。"),
            },
            {
                "role": "user",
                "content": config.PROMPT_STAGE1_PLAN_USER.format(
                    query=query[: config.QUERY_TEXT_MAX_CHARS],
                    tools=tool_text,
                    example_json=config.PROMPT_STAGE1_PLAN_EXAMPLE_JSON,
                    max_calls=config.STAGE1_MAX_TOOL_CALLS,
                    semantic_count=config.STAGE1_SEMANTIC_QUERY_COUNT,
                    keyword_count=config.STAGE1_KEYWORD_QUERY_COUNT,
                    keyword_min_terms=config.STAGE1_KEYWORD_MIN_TERMS,
                    keyword_max_terms=config.STAGE1_KEYWORD_MAX_TERMS,
                    followup=followup,
                ),
            },
        ]
        try:
            data = self.llm.chat_json(
                messages, temperature=config.STAGE1_PLAN_TEMPERATURE
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]查询规划失败，改用案情原文检索[/]: {exc}")
            data = {}
        return _coerce_calls(data, self._servers, already or [], self.pool_size)

    def _execute(self, calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        hits: List[Dict[str, Any]] = []
        errors: List[str] = []
        tools = _tool_index(self._servers)
        for call in calls:
            if self._calls >= config.STAGE1_MAX_TOOL_CALLS or self._filled(hits):
                break
            server = next((item for item in self._servers if item.name == call["server"]), None)
            if server is None or server.name in self._dead:
                continue
            schema = tools.get((call["server"], call["tool"]), {})
            arguments = _clamp_arguments(call.get("arguments") or {}, schema, self.pool_size)
            self._invoke(
                server.name,
                server.client,
                call["tool"],
                arguments,
                hits,
                errors,
                query_used=_query_preview(arguments),
            )
        return hits, errors

    def _fallback_calls(
        self,
        query: str,
        already: List[str],
    ) -> List[Dict[str, Any]]:
        calls = []
        text = query[: config.STAGE1_FALLBACK_QUERY_CHARS]
        for server in self._servers:
            if server.name in self._dead:
                continue
            tool = _best_case_tool(server.tools)
            if tool is None:
                continue
            arguments = _fallback_arguments(tool, text, self.pool_size)
            call = {"server": server.name, "tool": tool.get("name") or "", "arguments": arguments}
            if _call_key(call) in set(already):
                continue
            calls.append(call)
        return calls

    def _note_dead(self, name: str, exc: McpHttpError) -> None:
        if exc.fatal:
            self._dead.add(name)
        console.print(f"  [yellow]{name} 工具发现失败[/]: {str(exc)[:160]}")

    def _filled(self, hits: List[Dict[str, Any]]) -> bool:
        keys = {_dedupe_key(hit) for hit in hits}
        keys.discard("")
        return len(keys) >= self.pool_size

    def _invoke(
        self,
        backend: str,
        client: McpHttpClient,
        tool_name: str,
        arguments: Dict[str, Any],
        hits: List[Dict[str, Any]],
        errors: List[str],
        query_used: str,
    ) -> None:
        if self._calls >= config.STAGE1_MAX_TOOL_CALLS:
            return
        self._calls += 1
        label = f"{backend}/{tool_name}"
        call_msg = f"{label} {query_used[:60]}".strip()
        console.print(f"  [dim]{label}[/] {query_used[:60]}")
        self._progress(call_msg)
        try:
            result = client.call_tool(tool_name, arguments)
            parsed = cases_from_tool_result(result, source_db=backend, query_used=query_used)
            if not parsed:
                text = content_to_text(result)
                parsed = self._extract_with_llm(text, source_db=backend, query_used=query_used)
            hits.extend(parsed)
            console.print(f"    [green]parsed {len(parsed)} case(s)[/]")
            self._progress(f"Parsed {len(parsed)} case(s)")
        except McpHttpError as exc:
            errors.append(f"{label}: {exc}")
            if exc.fatal:
                self._dead.add(backend)
                console.print(f"    [yellow]{backend} stopped[/]: {str(exc)[:120]}")
                self._progress(f"{backend} stopped further calls: {str(exc)[:120]}")
            else:
                console.print(f"    [yellow]skip[/]: {str(exc)[:120]}")
                self._progress(f"Skipped: {str(exc)[:120]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            console.print(f"    [yellow]skip[/]: {exc}")
            self._progress(f"Skipped: {exc}")

    def _extract_with_llm(
        self,
        text: str,
        *,
        source_db: str,
        query_used: str,
    ) -> List[Dict[str, Any]]:
        text = (text or "").strip()
        if len(text) < 40 or _error_payload(text):
            return []
        messages = [
            {
                "role": "system",
                "content": config.PROMPT_STAGE1_EXTRACT_SYSTEM,
            },
            {"role": "user", "content": text[: config.STAGE1_EXTRACT_MAX_CHARS]},
        ]
        try:
            data = self.llm.chat_json(
                messages, temperature=config.STAGE1_EXTRACT_TEMPERATURE
            )
        except Exception:
            return []
        raw_cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(raw_cases, list):
            return []
        grounded = []
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            case = normalize_case(item, source_db=source_db, query_used=query_used)
            if not case:
                continue
            doc_id = case.get("doc_id") or ""
            title = case.get("title") or ""
            if (doc_id and doc_id in text) or (len(title) >= 8 and title in text):
                grounded.append(case)
        return grounded

    def _all_dead(self) -> bool:
        names = {server.name for server in self._servers}
        return bool(names) and self._dead >= names


def cases_from_tool_result(
    result: Dict[str, Any],
    *,
    source_db: str,
    query_used: str,
) -> List[Dict[str, Any]]:
    structured = result.get("structuredContent")
    if isinstance(structured, (dict, list)):
        raw_items: List[Dict[str, Any]] = []
        _collect_case_dicts(structured, raw_items)
        cases = [
            case
            for item in raw_items
            if (case := normalize_case(item, source_db=source_db, query_used=query_used))
        ]
        if cases:
            return cases

    text = content_to_text(result)
    if _error_payload(text):
        raise McpHttpError(text[:500], fatal=True)
    parsed = _try_json(text)
    raw_items = []
    if parsed is not None:
        _collect_case_dicts(parsed, raw_items)
    cases = []
    for item in raw_items:
        case = normalize_case(item, source_db=source_db, query_used=query_used)
        if case:
            cases.append(case)
    if cases:
        return cases
    return _parse_text_cases(text, source_db=source_db, query_used=query_used)


def normalize_case(
    raw: Dict[str, Any],
    *,
    source_db: str,
    query_used: str,
) -> Optional[Dict[str, Any]]:
    if _is_error_dict(raw):
        message = str(raw.get("message") or raw.get("error") or raw)
        raise McpHttpError(message[:500], fatal=_error_payload(message))
    doc_id = _first_str(raw, _FIELD_ALIASES["doc_id"])
    title = _first_str(raw, _FIELD_ALIASES["title"])
    court = _first_str(raw, _FIELD_ALIASES["court"])
    fact = _first_str(raw, _FIELD_ALIASES["fact"])
    reason = _first_str(raw, _FIELD_ALIASES["reason"])
    result = _first_str(raw, _FIELD_ALIASES["result"])
    fulltext = _first_str(raw, _FIELD_ALIASES["fulltext"])
    url = _first_str(raw, _FIELD_ALIASES["url"])
    charges = _as_str_list(_first_any(raw, _FIELD_ALIASES["charges"]))
    articles = _as_str_list(_first_any(raw, _FIELD_ALIASES["articles"]))
    if not doc_id:
        found = _CASE_NO_RE.search(" ".join(x for x in (title, fact, fulltext) if x))
        if found:
            doc_id = found.group(0)
    if not fulltext:
        fulltext = "\n".join(part for part in (fact, reason, result) if part)
    if not fact and fulltext:
        fact = fulltext
    if reason == "" and fulltext:
        for marker in ("本院认为", "本院评判如下", "经审查认为"):
            idx = fulltext.find(marker)
            if idx >= 0:
                reason = fulltext[idx:]
                break
    if not any((doc_id, title, fact, reason, result)):
        return None
    if not any((fact, reason, result, fulltext)):
        return None
    return {
        "doc_id": doc_id or title or url,
        "title": title,
        "court": court,
        "charges": charges,
        "fact": fact,
        "reason": reason,
        "result": result,
        "fulltext": fulltext,
        "articles": articles,
        "url": url,
        "source_db": source_db,
        "query_used": query_used,
    }


def _parse_text_cases(text: str, *, source_db: str, query_used: str) -> List[Dict[str, Any]]:
    matches = list(_CASE_NO_RE.finditer(text or ""))
    if not matches:
        return []
    cases = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        raw = {
            "案号": match.group(0),
            "标题": _labeled(chunk, "标题", "案件名称"),
            "法院": _labeled(chunk, "法院", "审理法院"),
            "案由": _labeled(chunk, "案由", "罪名"),
            "案情": _labeled(chunk, "案情", "基本案情", "事实"),
            "本院认为": _labeled(chunk, "本院认为", "裁判理由", "裁判要旨"),
            "裁判结果": _labeled(chunk, "裁判结果", "判决结果"),
            "全文": chunk[: config.STAGE1_TEXT_CHUNK_CHARS],
            "链接": _labeled(chunk, "链接", "url", "URL"),
        }
        case = normalize_case(raw, source_db=source_db, query_used=query_used)
        if case:
            cases.append(case)
    return cases


def _labeled(text: str, *labels: str) -> str:
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*[:：]\s*(.+)", text)
        if match:
            return match.group(1).strip().split("\n", 1)[0].strip()
    return ""


def _collect_case_dicts(node: Any, out: List[Dict[str, Any]]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_case_dicts(item, out)
        return
    if not isinstance(node, dict):
        return
    if _is_error_dict(node):
        message = str(node.get("message") or node.get("error") or node)
        raise McpHttpError(message[:500], fatal=_error_payload(message))
    nested_lists = [
        value
        for value in node.values()
        if isinstance(value, list) and any(isinstance(item, dict) for item in value)
    ]
    if nested_lists and not _has_case_body(node):
        for value in nested_lists:
            _collect_case_dicts(value, out)
        return
    if _looks_like_case(node):
        out.append(node)
        return
    for value in node.values():
        if isinstance(value, (list, dict)):
            _collect_case_dicts(value, out)


def _has_case_body(node: Dict[str, Any]) -> bool:
    body_names = (
        _FIELD_ALIASES["doc_id"]
        + _FIELD_ALIASES["fact"]
        + _FIELD_ALIASES["reason"]
        + _FIELD_ALIASES["result"]
        + _FIELD_ALIASES["fulltext"]
    )
    return bool(_folded_keys(node) & {name.lower() for name in body_names})


def _looks_like_case(node: Dict[str, Any]) -> bool:
    keys = _folded_keys(node)
    interesting = {name.lower() for names in _FIELD_ALIASES.values() for name in names}
    return bool(keys & interesting)


def _folded_keys(node: Dict[str, Any]) -> set:
    return {str(key).lower() for key in node}


def _is_error_dict(node: Dict[str, Any]) -> bool:
    keys = _folded_keys(node)
    if not keys or not keys <= {"code", "message", "error", "msg", "required_scope", "detail"}:
        return False
    blob = json.dumps(node, ensure_ascii=False)
    return _error_payload(blob) or "error" in keys


def _error_payload(text: str) -> bool:
    lowered = (text or "").lower()
    if _looks_like_case_text(text) and "案号" in text:
        return False
    markers = ("quota_exhausted", "积分不足", "次数已用完", "请充值", "请补充额度")
    return any(marker in lowered for marker in markers)


def _looks_like_case_text(text: str) -> bool:
    return "判决" in text or "本院认为" in text


def _try_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _coerce_calls(
    data: Any,
    servers: List[McpServerSpec],
    already: List[str],
    pool_size: int,
) -> List[Dict[str, Any]]:
    raw_calls = data.get("calls") if isinstance(data, dict) else None
    if not isinstance(raw_calls, list):
        return []
    known = _tool_index(servers)
    used = set(already)
    calls: List[Dict[str, Any]] = []
    semantic_n = 0
    keyword_n = 0
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        server = str(item.get("server") or "").strip()
        tool = str(item.get("tool") or "").strip()
        schema = known.get((server, tool))
        if schema is None:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        arguments = _clamp_arguments(arguments, schema, pool_size)
        call = {"server": server, "tool": tool, "arguments": arguments}
        key = _call_key(call)
        if key in used:
            continue
        kind = _call_kind(tool, arguments)
        if kind == "keyword" and keyword_n >= config.STAGE1_KEYWORD_QUERY_COUNT:
            continue
        if kind != "keyword" and semantic_n >= config.STAGE1_SEMANTIC_QUERY_COUNT:
            continue
        if kind == "keyword":
            keyword_n += 1
        else:
            semantic_n += 1
        used.add(key)
        calls.append(call)
        if len(calls) >= config.STAGE1_MAX_TOOL_CALLS:
            break
    return calls


def _call_kind(tool: str, arguments: Dict[str, Any]) -> str:
    blob = tool.lower() + " " + " ".join(str(k).lower() for k in arguments)
    if "keyword" in blob or "fulltext" in arguments:
        return "keyword"
    return "semantic"


def _call_key(call: Dict[str, Any]) -> str:
    arguments = call.get("arguments") or {}
    return f"{call.get('server')}/{call.get('tool')}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def _tool_index(servers: List[McpServerSpec]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for server in servers:
        for tool in server.tools:
            name = str(tool.get("name") or "")
            if name:
                index[(server.name, name)] = tool.get("inputSchema") or {}
    return index


def _clamp_arguments(
    arguments: Dict[str, Any],
    schema: Dict[str, Any],
    pool_size: int,
) -> Dict[str, Any]:
    properties = schema.get("properties") or {}
    cleaned: Dict[str, Any] = {}
    source = arguments
    if properties:
        source = {key: value for key, value in arguments.items() if key in properties}
    for key, value in source.items():
        if value is None or value == "":
            continue
        spec = properties.get(key) or {}
        if _schema_type(spec) == "integer" or key in {"size", "limit", "top_k", "page_size"}:
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            minimum = _schema_bound(spec, "minimum")
            maximum = _schema_bound(spec, "maximum")
            if minimum is not None:
                value = max(value, minimum)
            if maximum is not None:
                value = min(value, maximum)
            if key in {"size", "limit", "top_k", "page_size"}:
                value = min(value, max(1, pool_size))
        enum = _enum_values(spec)
        if enum:
            fitted = _fit_enum(value, enum)
            if fitted is None:
                continue
            value = fitted
        cleaned[key] = value
    return cleaned


def _query_preview(arguments: Dict[str, Any]) -> str:
    for key in ("question", "text", "query", "fulltext", "title"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(arguments, ensure_ascii=False)


def _best_case_tool(tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ranked = [tool for tool in tools if _case_tool_rank(tool) < 50]
    if not ranked:
        return None
    ranked.sort(key=_case_tool_rank)
    return ranked[0]


def _case_tool_rank(tool: Dict[str, Any]) -> int:
    name = str(tool.get("name") or "").lower()
    description = str(tool.get("description") or "")
    if name == "search_case":
        return 0
    if "case" in name and "keyword" not in name:
        return 1
    if "keyword" in name and "case" in name:
        return 2
    if "案例" in description or "裁判" in description:
        return 3
    return 99


def _fallback_arguments(tool: Dict[str, Any], text: str, pool_size: int) -> Dict[str, Any]:
    schema = tool.get("inputSchema") or {}
    properties = schema.get("properties") or {}
    arguments: Dict[str, Any] = {}
    for key in ("question", "text", "query", "fulltext"):
        if key in properties or not properties:
            arguments[key] = text
            break
    if "output_format" in properties:
        enum = _enum_values(properties["output_format"])
        arguments["output_format"] = "json" if "json" in enum or not enum else enum[0]
    if "size" in properties:
        maximum = int(properties["size"].get("maximum") or 20)
        arguments["size"] = min(maximum, max(1, pool_size))
    return _clamp_arguments(arguments, schema, pool_size)


def _tool_catalog(servers: List[McpServerSpec]) -> str:
    blocks = []
    for server in servers:
        lines = [f"服务 {server.name}:"]
        for tool in server.tools:
            lines.append(
                f"- {tool.get('name')}: {str(tool.get('description') or '').strip()}"
            )
            lines.append("  参数: " + _schema_brief(tool.get("inputSchema") or {}))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _schema_brief(schema: Dict[str, Any]) -> str:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not properties:
        return "无"
    parts = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        kind = _schema_type(spec) or "any"
        extra = []
        if name in required:
            extra.append("必填")
        enum = _enum_values(spec)
        if enum:
            extra.append("只能是 " + "/".join(str(item) for item in enum))
        maximum = _schema_bound(spec, "maximum")
        if maximum is not None:
            extra.append(f"最大 {maximum}")
        desc = str(spec.get("description") or "").strip()
        suffix = f"（{'，'.join(extra)}）" if extra else ""
        parts.append(f"{name}: {kind}{suffix}" + (f" {desc}" if desc else ""))
    return "；".join(parts)


def _schema_type(spec: Dict[str, Any]) -> str:
    if spec.get("type"):
        return str(spec["type"])
    for key in ("anyOf", "oneOf"):
        for item in spec.get(key) or []:
            if isinstance(item, dict) and item.get("type") and item.get("type") != "null":
                return str(item["type"])
    return ""


def _schema_bound(spec: Dict[str, Any], name: str) -> Optional[int]:
    if name in spec:
        return int(spec[name])
    for key in ("anyOf", "oneOf"):
        for item in spec.get(key) or []:
            if isinstance(item, dict) and name in item:
                return int(item[name])
    return None


def _enum_values(spec: Dict[str, Any]) -> List[Any]:
    values: List[Any] = []
    if isinstance(spec.get("enum"), list):
        values.extend(spec["enum"])
    for key in ("anyOf", "oneOf"):
        for item in spec.get(key) or []:
            if isinstance(item, dict):
                values.extend(_enum_values(item))
    seen = set()
    unique = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _fit_enum(value: Any, options: List[Any]) -> Any:
    if value in options:
        return value
    text = str(value).strip()
    if text in options:
        return text
    hits = [
        option
        for option in options
        if isinstance(option, str)
        and (option.startswith(text) or text.startswith(option) or text in option)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _skill_text(skills: List[SkillSpec]) -> str:
    parts = []
    for skill in skills:
        parts.append(f"# 技能 {skill.name}\n{skill.description}\n\n{skill.body}")
    return "\n\n".join(parts)


def _dedupe_key(case: Dict[str, Any]) -> str:
    doc_id = re.sub(r"\s+", "", str(case.get("doc_id") or ""))
    if len(doc_id) >= 6:
        return "id:" + doc_id
    title = re.sub(r"\s+", "", str(case.get("title") or ""))
    if len(title) >= 6:
        return "title:" + title
    return ""


def _first_any(raw: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    folded = {str(key).lower(): key for key in raw}
    for name in names:
        key = folded.get(name.lower())
        if key is None:
            continue
        value = raw[key]
        if value not in (None, "", [], {}):
            return value
    return None


def _first_str(raw: Dict[str, Any], names: Tuple[str, ...]) -> str:
    value = _first_any(raw, names)
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return ""
    return str(value).strip()


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[、,，;；/|]", text)
    return [part.strip() for part in parts if part.strip()]
