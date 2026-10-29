"""Session state: build the candidate pool once; later stages read/write the same JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


@dataclass
class SessionState:
    session_id: str
    query: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source: str = ""
    # Stage1
    candidate_pool: List[Dict[str, Any]] = field(default_factory=list)
    stage1_meta: Dict[str, Any] = field(default_factory=dict)
    # Stage2
    stage2_ranked: List[Dict[str, Any]] = field(default_factory=list)
    confirmed_facts: List[Dict[str, Any]] = field(default_factory=list)
    # Stage3
    active_dimensions: List[Dict[str, Any]] = field(default_factory=list)
    filter_rules: List[Dict[str, Any]] = field(default_factory=list)
    stage3_history: List[Dict[str, Any]] = field(default_factory=list)
    current_ranking: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (config.OUTPUT_DIR / f"session_{self.session_id}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SessionState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def create(cls, query: str, session_id: Optional[str] = None) -> "SessionState":
        sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(session_id=sid, query=query)
