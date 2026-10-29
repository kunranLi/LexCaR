"""LeCaRDv2 数据加载。

支持三种数据源（按优先级）：
1. 官方本地目录：query.json + candidates/*.json（或 candidates.jsonl）
2. HuggingFace mteb/LeCaRDv2 子集（便于快速体验）
3. 内置 demo 样例（无网 / 无数据时跑通流水线）

官方仓库：https://github.com/THUIR/LeCaRDv2
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import config

_TITLE_RE = re.compile(
    r"^(?P<title>.{2,120}?(?:刑事|民事|行政)?(?:判决书|裁定书|调解书|决定书))"
)


def extract_case_title(text: str, max_len: int = 80) -> str:
    """从文书正文开头抽取案名标题（HF 子集 title 字段原本为空）。"""
    text = (text or "").strip()
    if not text:
        return ""
    m = _TITLE_RE.match(text)
    if m:
        return m.group("title").strip()[:max_len]
    cut = re.split(r"[：:\n]", text, maxsplit=1)[0].strip()
    return cut[: min(40, max_len)]


@dataclass
class CaseRecord:
    doc_id: str
    title: str = ""
    fact: str = ""
    reason: str = ""
    result: str = ""
    fulltext: str = ""
    charges: List[str] = field(default_factory=list)
    articles: List[Union[int, str]] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    def retrieval_text(self, max_chars: Optional[int] = None) -> str:
        """用于稠密检索的文本：事实优先，辅以理由/结果。"""
        parts = [p for p in [self.fact, self.reason, self.result, self.fulltext] if p]
        text = "\n".join(parts) if parts else ""
        if max_chars and len(text) > max_chars:
            return text[:max_chars]
        return text

    def jev_state_text(
        self,
        query_text: str,
        max_case_chars: Optional[int] = None,
        max_query_chars: Optional[int] = None,
    ) -> str:
        """构造 Jev 共享上下文：本案 + 候选案例（事实摘要+裁判摘要）。"""
        _ = max_case_chars  # 长度由 config 分段预算控制
        max_query_chars = max_query_chars or config.QUERY_TEXT_MAX_CHARS
        return config.build_jev_compare_state(
            query_text[:max_query_chars],
            {
                "doc_id": self.doc_id,
                "title": self.title,
                "charges": self.charges,
                "fact": self.fact,
                "reason": self.reason,
                "result": self.result,
                "fulltext": self.fulltext,
            },
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CaseRecord":
        fulltext = data.get("fulltext") or data.get("qw") or data.get("text") or ""
        fact = data.get("fact") or ""
        title = (data.get("title") or "").strip()
        if not title:
            title = extract_case_title(fact or fulltext)
        return cls(
            doc_id=str(data.get("doc_id") or data.get("pid") or data.get("_id") or ""),
            title=title,
            fact=fact,
            reason=data.get("reason") or "",
            result=data.get("result") or "",
            fulltext=fulltext,
            charges=list(data.get("charges") or data.get("charge") or []),
            articles=list(data.get("articles") or data.get("article") or []),
            meta={k: v for k, v in data.items() if k not in {
                "doc_id", "pid", "_id", "title", "fact", "reason", "result",
                "fulltext", "qw", "text", "charges", "charge", "articles", "article",
            }},
        )


class LeCaRDv2Corpus:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir or config.LECARDV2_DIR)
        self.queries: Dict[str, Dict] = {}
        self.cases: Dict[str, CaseRecord] = {}
        self.source: str = "empty"

    # ---------------------------------------------------------------- load
    def load(self, prefer_hf: bool = False, max_candidates: Optional[int] = None) -> "LeCaRDv2Corpus":
        if not prefer_hf and self._has_local():
            self._load_local(max_candidates=max_candidates)
            return self
        try:
            self._load_hf(max_candidates=max_candidates)
            return self
        except Exception as hf_err:  # noqa: BLE001
            if self._has_local():
                self._load_local(max_candidates=max_candidates)
                return self
            print(f"[WARN] HuggingFace 加载失败: {hf_err}，改用内置 demo 样例。")
            self._load_demo()
            return self

    def _has_local(self) -> bool:
        if (self.data_dir / "query.json").exists():
            return True
        if (self.data_dir / "candidates").exists():
            return True
        if (self.data_dir / "candidates.jsonl").exists():
            return True
        if (self.data_dir / "corpus.jsonl").exists():
            return True
        return False

    def _load_local(self, max_candidates: Optional[int] = None) -> None:
        self.source = f"local:{self.data_dir}"
        query_path = self.data_dir / "query.json"
        if query_path.exists():
            self.queries = {}
            with query_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = str(obj.get("id") or obj.get("qid"))
                    self.queries[qid] = {
                        "qid": qid,
                        "query": obj.get("query") or "",
                        "fact": obj.get("fact") or "",
                    }

        # candidates 目录（每文件一个 json）或 jsonl
        cand_dir = self.data_dir / "candidates"
        jsonl_path = self.data_dir / "candidates.jsonl"
        corpus_jsonl = self.data_dir / "corpus.jsonl"

        loaded = 0
        if cand_dir.exists() and cand_dir.is_dir():
            files = sorted(cand_dir.glob("*.json"))
            for fp in files:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                rec = CaseRecord.from_dict(obj)
                if not rec.doc_id:
                    rec.doc_id = fp.stem
                self.cases[rec.doc_id] = rec
                loaded += 1
                if max_candidates and loaded >= max_candidates:
                    break
        elif jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec = CaseRecord.from_dict(obj)
                    if not rec.doc_id or not (rec.fact or rec.fulltext):
                        continue
                    self.cases[rec.doc_id] = rec
                    loaded += 1
                    if max_candidates and loaded >= max_candidates:
                        break
        elif corpus_jsonl.exists():
            with corpus_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # HF/MTEB 风格：_id / title / text
                    doc_id = str(obj.get("_id") or obj.get("id") or obj.get("pid") or "")
                    text = obj.get("text") or obj.get("fact") or ""
                    if not doc_id or not text:
                        continue
                    title = (obj.get("title") or "").strip() or extract_case_title(text)
                    rec = CaseRecord(
                        doc_id=doc_id,
                        title=title,
                        fact=text,
                        fulltext=text,
                        charges=list(obj.get("charge") or []),
                    )
                    self.cases[rec.doc_id] = rec
                    loaded += 1
                    if max_candidates and loaded >= max_candidates:
                        break
        else:
            raise FileNotFoundError(
                f"在 {self.data_dir} 未找到 candidates / candidates.jsonl / corpus.jsonl"
            )

    def _load_hf(self, max_candidates: Optional[int] = None) -> None:
        """加载 HuggingFace mteb/LeCaRDv2（子集，适合 demo）。"""
        from datasets import load_dataset

        self.source = "huggingface:mteb/LeCaRDv2"
        corpus_ds = load_dataset("mteb/LeCaRDv2", "corpus", split="test")
        query_ds = load_dataset("mteb/LeCaRDv2", "queries", split="test")

        self.cases = {}
        for i, row in enumerate(corpus_ds):
            doc_id = str(row.get("_id") or row.get("id"))
            text = row.get("text") or ""
            title = (row.get("title") or "").strip() or extract_case_title(text)
            self.cases[doc_id] = CaseRecord(
                doc_id=doc_id, title=title, fact=text, fulltext=text
            )
            if max_candidates and i + 1 >= max_candidates:
                break

        self.queries = {}
        for row in query_ds:
            qid = str(row.get("_id") or row.get("id"))
            self.queries[qid] = {
                "qid": qid,
                "query": row.get("text") or "",
                "fact": row.get("text") or "",
            }

    def _load_demo(self) -> None:
        """内置极小样例，保证无数据也能跑通 Stage 流程。"""
        self.source = "builtin-demo"
        demo_cases = [
            {
                "doc_id": "demo_001",
                "fact": "被告人甲入户盗窃他人财物价值8000元，案发后部分退赃。",
                "reason": "本院认为被告人甲构成盗窃罪，入户盗窃应从重处罚。",
                "result": "判处有期徒刑一年六个月，并处罚金。",
                "charges": ["盗窃罪"],
            },
            {
                "doc_id": "demo_002",
                "fact": "被告人乙在商场扒窃手机一部，价值3000元，系初犯。",
                "reason": "被告人乙构成盗窃罪，情节较轻。",
                "result": "判处拘役四个月，缓刑六个月。",
                "charges": ["盗窃罪"],
            },
            {
                "doc_id": "demo_003",
                "fact": "被告人丙诈骗被害人转账5万元，承诺高额回报。",
                "reason": "被告人丙以非法占有为目的，虚构事实骗取财物，构成诈骗罪。",
                "result": "判处有期徒刑二年，并处罚金。",
                "charges": ["诈骗罪"],
            },
            {
                "doc_id": "demo_004",
                "fact": "被告人丁多次盗窃电动车，累计金额2万元，有前科。",
                "reason": "被告人丁构成盗窃罪，多次盗窃且有前科。",
                "result": "判处有期徒刑二年六个月。",
                "charges": ["盗窃罪"],
            },
            {
                "doc_id": "demo_005",
                "fact": "被告人戊入户盗窃未得手即被发现逃离，未造成实质损失。",
                "reason": "盗窃未遂，可比照既遂从轻处罚。",
                "result": "判处拘役三个月。",
                "charges": ["盗窃罪"],
            },
            {
                "doc_id": "demo_006",
                "fact": "被告人己盗窃公司财物后主动退还全部赃款并取得谅解。",
                "reason": "退赃退赔并谅解，可从轻处罚。",
                "result": "判处有期徒刑八个月，缓刑一年。",
                "charges": ["盗窃罪"],
            },
            {
                "doc_id": "demo_007",
                "fact": "被告人庚故意伤害致人轻伤二级，赔偿后取得谅解。",
                "reason": "构成故意伤害罪，有赔偿谅解情节。",
                "result": "判处拘役五个月，缓刑六个月。",
                "charges": ["故意伤害罪"],
            },
            {
                "doc_id": "demo_008",
                "fact": "被告人辛在夜间撬门入户盗窃现金1.2万元，部分挥霍。",
                "reason": "入户盗窃，社会危害性较大。",
                "result": "判处有期徒刑二年。",
                "charges": ["盗窃罪"],
            },
        ]
        self.cases = {c["doc_id"]: CaseRecord.from_dict(c) for c in demo_cases}
        self.queries = {
            "demo_q1": {
                "qid": "demo_q1",
                "query": "被告人张某潜入他人住宅盗窃现金约一万元，案发后退还部分赃款，是否可从轻？",
                "fact": "被告人张某潜入他人住宅盗窃现金约一万元，案发后退还部分赃款。",
            }
        }

    # ---------------------------------------------------------------- utils
    def get_query_text(self, qid: Optional[str] = None, raw_query: Optional[str] = None) -> str:
        if raw_query:
            return raw_query
        if not self.queries:
            raise ValueError("语料中无 query")
        if qid is None:
            qid = next(iter(self.queries))
        q = self.queries[str(qid)]
        return q.get("fact") or q.get("query") or ""

    def iter_for_index(self, max_chars: int = 4000) -> Iterable[tuple]:
        for doc_id, case in self.cases.items():
            yield doc_id, case.retrieval_text(max_chars=max_chars)

    def subset(self, doc_ids: Iterable[str]) -> Dict[str, CaseRecord]:
        return {str(i): self.cases[str(i)] for i in doc_ids if str(i) in self.cases}
