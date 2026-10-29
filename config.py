"""Global config loaded from environment variables / .env."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _resolve_path(value: str, default: Path) -> Path:
    p = Path(value) if value else default
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    return p


def _env_text(key: str, default: str) -> str:
    """Read multi-line text; supports \\n escapes in .env."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.replace("\\n", "\n").replace("\\t", "\t")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


# Optional: pin HF cache under the project tree instead of C:\\Users\\...\\.cache
_hf_home = os.getenv("HF_HOME", "").strip()
if _hf_home:
    hf_home = _resolve_path(_hf_home, ROOT_DIR / "models" / ".hf_home")
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))

DMX_API_KEY = os.getenv("DMX_API_KEY", "")
DMX_BASE_URL = os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn").rstrip("/")

JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.13.0")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# HTTP timeout (seconds) and retries: shared by LLM; Jev may override per stage
DMX_MAX_RETRIES = _env_int("DMX_MAX_RETRIES", 3)
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 90)
JEV_TIMEOUT = _env_int("JEV_TIMEOUT", 90)
JEV_TIMEOUT_STAGE2 = _env_int("JEV_TIMEOUT_STAGE2", JEV_TIMEOUT)
JEV_TIMEOUT_STAGE3 = _env_int("JEV_TIMEOUT_STAGE3", JEV_TIMEOUT)

LECARDV2_DIR = _resolve_path(
    os.getenv("LECARDV2_DIR", ""), ROOT_DIR / "data" / "lecardv2"
)

# --- Stage1: agentic retrieval (LLM + case-retrieval skills + MCP) ---
CANDIDATE_POOL_SIZE = _env_int("CANDIDATE_POOL_SIZE", 50)
STAGE1_AGENT_MAX_ROUNDS = _env_int("STAGE1_AGENT_MAX_ROUNDS", 3)
STAGE1_MAX_TOOL_CALLS = _env_int("STAGE1_MAX_TOOL_CALLS", 16)
STAGE1_MCP_TIMEOUT = _env_int("STAGE1_MCP_TIMEOUT", 60)

# Temperature for query planning / field extraction from raw case text
STAGE1_PLAN_TEMPERATURE = _env_float("STAGE1_PLAN_TEMPERATURE", 0.2)
STAGE1_EXTRACT_TEMPERATURE = _env_float("STAGE1_EXTRACT_TEMPERATURE", 0.0)
# Max semantic / keyword queries per planning round
STAGE1_SEMANTIC_QUERY_COUNT = _env_int("STAGE1_SEMANTIC_QUERY_COUNT", 4)
STAGE1_KEYWORD_QUERY_COUNT = _env_int("STAGE1_KEYWORD_QUERY_COUNT", 4)
STAGE1_KEYWORD_MIN_TERMS = _env_int("STAGE1_KEYWORD_MIN_TERMS", 2)
STAGE1_KEYWORD_MAX_TERMS = _env_int("STAGE1_KEYWORD_MAX_TERMS", 6)
STAGE1_SEMANTIC_MIN_CHARS = _env_int("STAGE1_SEMANTIC_MIN_CHARS", 8)
STAGE1_KEYWORD_MIN_CHARS = _env_int("STAGE1_KEYWORD_MIN_CHARS", 2)
STAGE1_EXTRACT_MAX_CHARS = _env_int("STAGE1_EXTRACT_MAX_CHARS", 12000)
STAGE1_TEXT_CHUNK_CHARS = _env_int("STAGE1_TEXT_CHUNK_CHARS", 4000)
STAGE1_FALLBACK_QUERY_CHARS = _env_int("STAGE1_FALLBACK_QUERY_CHARS", 400)
STAGE1_CAUSE_MAX_CHARS = _env_int("STAGE1_CAUSE_MAX_CHARS", 40)
STAGE1_FOCUS_MAX_CHARS = _env_int("STAGE1_FOCUS_MAX_CHARS", 200)

PROMPT_STAGE1_SKILL_SELECT_SYSTEM = _env_text(
    "PROMPT_STAGE1_SKILL_SELECT_SYSTEM",
    "你是法律案例检索智能体。根据案情，从技能目录里选择本次检索要遵循的技能。"
    "用户新增加的技能，只要和类案检索、检索范围、检索式或案例筛选有关，就应当选上。"
    "只输出 JSON。",
)
# Placeholders: {query} {catalog} {example_json}
PROMPT_STAGE1_SKILL_SELECT_USER = _env_text(
    "PROMPT_STAGE1_SKILL_SELECT_USER",
    "【用户案情】\n{query}\n\n"
    "【技能目录】\n{catalog}\n\n"
    "输出 JSON，格式示例：\n{example_json}",
)
PROMPT_STAGE1_SKILL_SELECT_EXAMPLE_JSON = _env_text(
    "PROMPT_STAGE1_SKILL_SELECT_EXAMPLE_JSON",
    '{"skills":["case-retrieval"]}',
)
PROMPT_STAGE1_PLAN_SYSTEM = _env_text(
    "PROMPT_STAGE1_PLAN_SYSTEM",
    "你是法律案例检索智能体的查询规划器。严格遵循已调用的技能，"
    "只根据给出的 MCP 工具 schema 发起工具调用。"
    "案件类型以用户案情为准，可以是刑事、民事、行政或其他类型；"
    "工具若有类型参数，必须使用 schema 中的全称，不要默认刑事案件。"
    "不要编写任何案例、案号或裁判要旨。只输出 JSON。",
)
# Placeholders: {query} {tools} {example_json} {max_calls} {semantic_count} {keyword_count} {keyword_min_terms} {keyword_max_terms} {followup}
PROMPT_STAGE1_PLAN_USER = _env_text(
    "PROMPT_STAGE1_PLAN_USER",
    "【用户案情】\n{query}\n\n"
    "【可用 MCP 工具】\n{tools}\n\n"
    "请输出工具调用 JSON，格式示例：\n{example_json}\n\n"
    "server 和 tool 必须来自上面的列表，arguments 只能使用该工具列出的参数。"
    "本轮最多 {max_calls} 次调用。"
    "语义检索最多 {semantic_count} 次，关键词检索最多 {keyword_count} 次。"
    "关键词每条 {keyword_min_terms} 到 {keyword_max_terms} 个词。"
    "示例只说明 JSON 结构，检索词和案件类型必须来自用户案情，不要照抄示例。"
    "{followup}",
)
PROMPT_STAGE1_PLAN_EXAMPLE_JSON = _env_text(
    "PROMPT_STAGE1_PLAN_EXAMPLE_JSON",
    '{"calls":[{"server":"pkulaw-semantic","tool":"search_case",'
    '"arguments":{"text":"房屋买卖合同解除后已付购房款应否返还","case_type":"民事案件","size":10}}]}',
)
# Placeholders: {already} {found} {target} {titles}
PROMPT_STAGE1_PLAN_FOLLOWUP = _env_text(
    "PROMPT_STAGE1_PLAN_FOLLOWUP",
    "\n\n已经调用过的工具：{already}\n"
    "目前去重后只有 {found} 条，目标 {target} 条。\n"
    "已有标题示例：{titles}\n"
    "请换角度再给一批不重复的工具调用。",
)
PROMPT_STAGE1_EXTRACT_SYSTEM = _env_text(
    "PROMPT_STAGE1_EXTRACT_SYSTEM",
    "你只从给定的案例库原文中抽取案例字段，禁止补充原文没有的案号、事实或裁判。"
    "charges 填写原文中的案由或罪名，民事和行政案件写案由。"
    "输出 JSON："
    '{"cases":[{"doc_id":"","title":"","court":"","charges":[],'
    '"fact":"","reason":"","result":"","fulltext":"","url":"","articles":[]}]}',
)

# --- Stage2 ---
TOP_K_AFTER_STAGE2 = _env_int("TOP_K_AFTER_STAGE2", 10)
JEV_MAX_WORKERS = _env_int("JEV_MAX_WORKERS", 8)
# Jev compare budgets: fact summary + adjudication summary (segmented; no whole-text head truncate)
CASE_FACT_MAX_CHARS = _env_int("CASE_FACT_MAX_CHARS", 1500)
CASE_ADJUDICATION_MAX_CHARS = _env_int("CASE_ADJUDICATION_MAX_CHARS", 4000)
# Legacy: if CASE_TEXT_MAX_CHARS is still set, use it only as a soft total-length cap
CASE_TEXT_MAX_CHARS = _env_int(
    "CASE_TEXT_MAX_CHARS",
    CASE_FACT_MAX_CHARS + CASE_ADJUDICATION_MAX_CHARS,
)
QUERY_TEXT_MAX_CHARS = _env_int("QUERY_TEXT_MAX_CHARS", 2500)
# Stage2.2: precedent briefs for LLM / per-brief truncate / atomic question count
STAGE2_CHECKLIST_TOP_N = _env_int("STAGE2_CHECKLIST_TOP_N", 10)
STAGE2_CASE_BRIEF_MAX_CHARS = _env_int("STAGE2_CASE_BRIEF_MAX_CHARS", 600)
STAGE2_CHECKLIST_MIN_QUESTIONS = _env_int("STAGE2_CHECKLIST_MIN_QUESTIONS", 12)
STAGE2_CHECKLIST_MAX_QUESTIONS = _env_int("STAGE2_CHECKLIST_MAX_QUESTIONS", 18)
# Stage2.3: thresholds for established / information-sufficiency
STAGE2_FACT_PROB_THRESHOLD = _env_float("STAGE2_FACT_PROB_THRESHOLD", 0.5)

# --- Stage3 ---
STAGE3_TOP_K = _env_int("STAGE3_TOP_K", 10)
STAGE3_DEFAULT_DIM_WEIGHT = _env_float("STAGE3_DEFAULT_DIM_WEIGHT", 0.3)
STAGE3_FILTER_THRESHOLD = _env_float("STAGE3_FILTER_THRESHOLD", 0.5)
STAGE3_LLM_QUERY_MAX_CHARS = _env_int("STAGE3_LLM_QUERY_MAX_CHARS", 1200)

# Stage2.1 / Stage3 preset 3-D Jev dimensions (JSON array: id/name/weight/question)
_DEFAULT_STAGE2_DIMENSIONS = [
    {
        "id": "similar_cause",
        "name": "案由或法律关系是否相近",
        "weight": 0.5,
        "question": "仅根据本案与候选案例的事实及裁判说理判断：二者的案由、罪名或法律关系是否相同或相近？刑事案件比较罪名，民事案件比较案由与请求权，行政案件比较被诉行政行为。案件大类不同则不相近。",
    },
    {
        "id": "key_facts_overlap",
        "name": "关键情节是否重合",
        "weight": 0.3,
        "question": "候选案例的关键事实与本案是否重合？只比较本案实际涉及的情节，如行为方式、合同履行、行政行为、损害后果、数额、过错、退赔或前科。",
    },
    {
        "id": "outcome_reference",
        "name": "裁判结果是否有参考价值",
        "weight": 0.2,
        "question": "候选案例的裁判结果对本案是否有参考价值？刑事看定罪量刑，民事看责任承担与给付，行政看合法性判断与处理方式。",
    },
]
try:
    STAGE2_DIMENSIONS: List[Dict[str, Any]] = json.loads(
        _env_text(
            "PROMPT_JEV_STAGE2_DIMENSIONS_JSON",
            json.dumps(_DEFAULT_STAGE2_DIMENSIONS, ensure_ascii=False),
        )
    )
    if not isinstance(STAGE2_DIMENSIONS, list) or not STAGE2_DIMENSIONS:
        STAGE2_DIMENSIONS = _DEFAULT_STAGE2_DIMENSIONS
except json.JSONDecodeError:
    STAGE2_DIMENSIONS = _DEFAULT_STAGE2_DIMENSIONS

# ---------------------------------------------------------------------------
# Stage2 prompts: LLM checklist generation + Jev structured fact checks (overridable via .env)
# See comments on each constant for placeholders
# ---------------------------------------------------------------------------

# LLM system: extract elements only from high-relevance precedent adjudication text
PROMPT_LLM_CHECKLIST_SYSTEM = _env_text(
    "PROMPT_LLM_CHECKLIST_SYSTEM",
    "你是中国裁判要件抽取助手。先例可能是刑事、民事、行政或其他案件。"
    "只从裁判说理中抽取法院作出裁判主文之前用来认定的法律要件，写成是否问句。"
    "每个问句都要明确指向一个法律要件，写出要件名称，不能只问生活事实。"
    "例如写「是否具有抢劫罪的非法占有目的」「是否以麻醉方法作为抢劫手段」，"
    "不要写「是否使用麻醉药物」「是否有认罪态度」「是否情节严重」。"
    "一个问题只能有一个要件，目的、手段、场所、结果、从宽情节必须拆开，"
    "不能用逗号或「并」「且」「以及」写在同一问。"
    "不要抽取判决主文之后的执行和告知事项，例如刑期起算、羁押折抵、上诉权利、"
    "罚金缴纳期限、剥夺政治权利的宣告、判决生效后如何履行。"
    "禁止总括问题，例如「是否符合某罪的构成要件」「是否存在从轻或从重情节」"
    "「证据是否充分」「量刑是否适当」「辩护意见是否被采纳」。"
    "民事写具体责任要件，行政写具体合法性要件，不要改写成罪名或量刑。"
    "不要抽取说理没有讨论的要件来凑数，不要依赖或臆造用户案情；只输出 JSON。",
)

# LLM user: {top_k} {case_briefs} {example_json} {min_questions} {max_questions} (no user query)
PROMPT_LLM_CHECKLIST_USER = _env_text(
    "PROMPT_LLM_CHECKLIST_USER",
    "【高相关先例裁判摘要 Top-{top_k}】\n{case_briefs}\n\n"
    "请仅依据上述裁判摘要，提取裁判要件清单，输出 JSON，格式示例：\n"
    "{example_json}\n\n"
    "要求：生成 {min_questions}～{max_questions} 个问题，id 用 q1/q2...。"
    "只列本院认为里作为定案依据的法律要件，摘要里每出现一个独立要件就单列一问。"
    "不要合并，也不要为了凑数编造说理里没有的要件，不要抽取判决主文和执行事项。"
    "每个问题只有一个要件，问句里必须出现该要件的法律名称。"
    "不要用逗号、分号、「并」「且」「以及」「同时」连接第二个要件。"
    "name 用该要件的短名。"
    "示例只说明写法，不要照抄示例中的案情；摘要不是抢劫时，不要输出入户、麻醉、退赔。"
    "不要出现对「用户案情」的引用。",
)

# LLM output-format example (embedded in user prompt)
PROMPT_LLM_CHECKLIST_EXAMPLE_JSON = _env_text(
    "PROMPT_LLM_CHECKLIST_EXAMPLE_JSON",
    json.dumps(
        {
            "questions": [
                {
                    "id": "q1",
                    "name": "非法占有目的",
                    "question": "是否具有抢劫罪的非法占有目的？",
                    "rationale": "说理将非法占有目的作为抢劫罪的主观要件",
                },
                {
                    "id": "q2",
                    "name": "入户加重",
                    "question": "是否属于入户抢劫这一加重情节？",
                    "rationale": "说理将入户作为抢劫的加重情节",
                },
                {
                    "id": "q3",
                    "name": "麻醉手段",
                    "question": "是否以麻醉方法作为抢劫手段？",
                    "rationale": "说理将药物麻醉作为抢劫的手段要件",
                },
                {
                    "id": "q4",
                    "name": "不能反抗",
                    "question": "是否达到抢劫手段所要求的不能反抗？",
                    "rationale": "说理将不能反抗作为麻醉手段的结果状态",
                },
                {
                    "id": "q5",
                    "name": "劫取财物",
                    "question": "是否具备当场劫取财物这一结果要件？",
                    "rationale": "说理将当场取得财物作为抢劫的结果要件",
                },
                {
                    "id": "q6",
                    "name": "自动投案",
                    "question": "是否具备自首所要求的自动投案？",
                    "rationale": "说理将自动投案作为自首的构成事实",
                },
                {
                    "id": "q7",
                    "name": "如实供述",
                    "question": "是否具备自首所要求的如实供述主要犯罪事实？",
                    "rationale": "说理将如实供述作为自首中独立于自动投案的事实",
                },
                {
                    "id": "q8",
                    "name": "退赔违法所得",
                    "question": "是否具有退赔这一从宽情节？",
                    "rationale": "说理将退赔作为量刑从宽情节",
                },
            ]
        },
        ensure_ascii=False,
        indent=2,
    ),
)

# Fallback checklist when LLM fails (JSON array)
_DEFAULT_FALLBACK = [
    {
        "id": "q1",
        "name": "案由或法律关系",
        "question": "裁判是否围绕与该类案件相同的案由、罪名或法律关系展开？",
        "rationale": "先例说理通常先确定案件性质",
    },
    {
        "id": "q2",
        "name": "争议焦点",
        "question": "裁判是否处理了该类案件的主要争议焦点？",
        "rationale": "先例说理围绕争议焦点展开",
    },
    {
        "id": "q3",
        "name": "关键事实",
        "question": "裁判是否依赖与该类案件相似的关键事实？",
        "rationale": "先例将关键事实作为认定基础",
    },
    {
        "id": "q4",
        "name": "裁判结果",
        "question": "裁判结果的处理方式对该类案件是否具有参考意义？",
        "rationale": "先例结论体现可参考的处理方式",
    },
]
try:
    PROMPT_LLM_CHECKLIST_FALLBACK: List[Dict[str, Any]] = json.loads(
        _env_text(
            "PROMPT_LLM_CHECKLIST_FALLBACK_JSON",
            json.dumps(_DEFAULT_FALLBACK, ensure_ascii=False),
        )
    )
    if not isinstance(PROMPT_LLM_CHECKLIST_FALLBACK, list):
        PROMPT_LLM_CHECKLIST_FALLBACK = _DEFAULT_FALLBACK
except json.JSONDecodeError:
    PROMPT_LLM_CHECKLIST_FALLBACK = _DEFAULT_FALLBACK

# --- Jev: Stage2.3 structured fact checks against the query checklist ---
# state placeholders: {query}
PROMPT_JEV_FACT_STATE = _env_text(
    "PROMPT_JEV_FACT_STATE",
    "【本案案情】\n{query}\n\n请严格依据上述案情作答，不要臆造未出现的事实。",
)

# Jev noul: probability that the element is established; {question}
PROMPT_JEV_FACT_ESTABLISHED = _env_text(
    "PROMPT_JEV_FACT_ESTABLISHED",
    "仅根据本案案情判断：{question} 若信息不足以判断，请倾向于给出较低概率。",
)

# Jev noul: probability that the query text has enough info; {question}
PROMPT_JEV_FACT_SUFFICIENCY = _env_text(
    "PROMPT_JEV_FACT_SUFFICIENCY",
    "判断本案案情文本中，是否已提供足够信息来可靠回答：{question}",
)

# --- Jev: compare context for Stage2.1 / Stage3 in-pool rerank ---
# Placeholders: {query} {doc_id} {body}
PROMPT_JEV_COMPARE_STATE = _env_text(
    "PROMPT_JEV_COMPARE_STATE",
    "【本案案情】\n{query}\n\n"
    "【候选案例】\n"
    "案例ID: {doc_id}\n"
    "{body}",
)

# Stage3: LLM rewrites user intent into a Jev Boolean dimension question
PROMPT_LLM_STAGE3_DIM_SYSTEM = _env_text(
    "PROMPT_LLM_STAGE3_DIM_SYSTEM",
    "你是法律检索系统中的问题生成器。把用户的检索偏好改写成一个清晰的 "
    "Boolean（是/否）Jev 判定问题，用于比较「本案」与「候选案例」是否匹配该偏好。"
    "只输出 JSON。",
)
# Placeholders: {query} {user_intent} {existing_questions} {example_json}
PROMPT_LLM_STAGE3_DIM_USER = _env_text(
    "PROMPT_LLM_STAGE3_DIM_USER",
    "【本案案情】\n{query}\n\n"
    "【用户新增检索重点】\n{user_intent}\n\n"
    "【已有维度问题】\n{existing_questions}\n\n"
    "输出 JSON，格式示例：\n{example_json}",
)
PROMPT_LLM_STAGE3_DIM_EXAMPLE_JSON = _env_text(
    "PROMPT_LLM_STAGE3_DIM_EXAMPLE_JSON",
    '{"id":"dim_focus","name":"争议焦点重合","question":"候选案例是否与用户关注的争议焦点直接相关？"}',
)
# Fallback Jev dimension question when LLM fails; placeholder: {user_intent}
PROMPT_JEV_STAGE3_DIM_FALLBACK = _env_text(
    "PROMPT_JEV_STAGE3_DIM_FALLBACK",
    "候选案例是否体现用户关注的情节：{user_intent}？",
)

# Stage3: LLM rewrites exclude intent into a Jev Boolean filter question
PROMPT_LLM_STAGE3_FILTER_SYSTEM = _env_text(
    "PROMPT_LLM_STAGE3_FILTER_SYSTEM",
    "你是法律检索系统中的过滤条件生成器。"
    "把用户的排除偏好改写成一个 Boolean 问题："
    "若对候选案例回答「是」，则该案例应被排除。"
    "只输出 JSON。",
)
# Placeholders: {query} {user_intent} {example_json}
PROMPT_LLM_STAGE3_FILTER_USER = _env_text(
    "PROMPT_LLM_STAGE3_FILTER_USER",
    "【本案案情】\n{query}\n\n"
    "【用户排除偏好】\n{user_intent}\n\n"
    "输出 JSON，格式示例：\n{example_json}",
)
PROMPT_LLM_STAGE3_FILTER_EXAMPLE_JSON = _env_text(
    "PROMPT_LLM_STAGE3_FILTER_EXAMPLE_JSON",
    '{"id":"flt_unrelated","name":"排除无关案件","question":"该候选案例是否属于用户声明要排除的案件类型？"}',
)
# Fallback Jev filter question when LLM fails; placeholder: {user_intent}
PROMPT_JEV_STAGE3_FILTER_FALLBACK = _env_text(
    "PROMPT_JEV_STAGE3_FILTER_FALLBACK",
    "该候选案例是否属于用户不想要的类型：{user_intent}？",
)


def extract_adjudication_text(case: Dict[str, Any], max_chars: int) -> str:
    """Extract adjudication text: prefer reason+result; else from court-opinion marker; else tail."""
    reason = (case.get("reason") or "").strip()
    result = (case.get("result") or "").strip()
    if reason or result:
        text = "\n".join(p for p in [reason, result] if p)
        return text[:max_chars]

    full = (
        case.get("fulltext") or case.get("text") or case.get("fact") or ""
    ).strip()
    if not full:
        return ""
    for marker in ("本院认为", "本院评判如下", "经审查认为", "本院意见如下"):
        idx = full.find(marker)
        if idx >= 0:
            return full[idx : idx + max_chars]
    if len(full) > max_chars:
        return full[-max_chars:]
    return full


def build_case_body_for_jev(
    case: Dict[str, Any],
    *,
    fact_max: Optional[int] = None,
    adj_max: Optional[int] = None,
    total_max: Optional[int] = None,
) -> str:
    """Fact summary (head) + adjudication summary (court opinion / tail) so reasoning reaches the model."""
    fact_max = CASE_FACT_MAX_CHARS if fact_max is None else fact_max
    adj_max = CASE_ADJUDICATION_MAX_CHARS if adj_max is None else adj_max
    total_max = CASE_TEXT_MAX_CHARS if total_max is None else total_max

    fact = (case.get("fact") or case.get("fulltext") or case.get("text") or "").strip()
    fact_part = fact[:fact_max] if fact else ""
    adj_part = extract_adjudication_text(case, adj_max)

    # Drop adjudication if it is fully contained in the fact summary (avoid short-doc duplication)
    if fact_part and adj_part and adj_part in fact_part:
        adj_part = ""

    def _pack(fact_s: str, adj_s: str) -> str:
        chunks: List[str] = []
        if fact_s:
            chunks.append(f"【事实摘要】\n{fact_s}")
        if adj_s:
            chunks.append(f"【裁判摘要】\n{adj_s}")
        return "\n\n".join(chunks)

    body = _pack(fact_part, adj_part)
    if not body:
        return ""
    if len(body) <= total_max:
        return body
    # When over total cap, keep adjudication first, then compress facts
    if adj_part:
        overhead = len(_pack("", adj_part))
        remain = max(0, total_max - overhead - 2)
        if remain >= 32:
            return _pack(fact_part[:remain], adj_part)[:total_max]
        return f"【裁判摘要】\n{adj_part}"[:total_max]
    return body[:total_max]


def build_jev_compare_state(query: str, case: Dict[str, Any]) -> str:
    """Build shared query–candidate compare state for Stage2.1 / Stage3."""
    body = build_case_body_for_jev(case)
    kwargs: Dict[str, Any] = {
        "query": query[:QUERY_TEXT_MAX_CHARS],
        "doc_id": case.get("doc_id") or "",
        "body": body,
    }
    # Compat: older .env templates may still contain {charges}
    if "{charges}" in PROMPT_JEV_COMPARE_STATE:
        kwargs["charges"] = "、".join(case.get("charges") or []) or ""
    return PROMPT_JEV_COMPARE_STATE.format(**kwargs)


def require_api_key() -> str:
    if not DMX_API_KEY or DMX_API_KEY.startswith("sk-your"):
        raise RuntimeError(
            "请先在 .env 中设置 DMX_API_KEY（可复制 .env.example）。"
            "密钥获取：https://www.dmxapi.cn"
        )
    return DMX_API_KEY
