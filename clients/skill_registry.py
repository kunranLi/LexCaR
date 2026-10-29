"""Load skills from skills/. Adding a SKILL.md with a description makes it selectable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config

_NAME_RE = re.compile(r"(?m)^name:\s*(.+?)\s*$")
_DESC_RE = re.compile(r"(?m)^description:\s*(.+?)\s*$")


@dataclass
class SkillSpec:
    name: str
    description: str
    body: str
    path: Path


def load_skills(directory: Optional[Path] = None) -> List[SkillSpec]:
    """Read skills/<name>/SKILL.md. Directories whose names start with _ are skipped."""
    folder = directory or (config.ROOT_DIR / "skills")
    if not folder.is_dir():
        return []
    skills: List[SkillSpec] = []
    seen = set()
    for path in sorted(folder.glob("*/SKILL.md")):
        if path.parent.name.startswith("_"):
            continue
        skill = _parse(path)
        if skill.name in seen:
            continue
        seen.add(skill.name)
        skills.append(skill)
    return skills


def _parse(path: Path) -> SkillSpec:
    text = path.read_text(encoding="utf-8")
    name = path.parent.name
    description = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            front, body = parts[1], parts[2]
            found_name = _field(front, _NAME_RE)
            found_desc = _field(front, _DESC_RE)
            if found_name:
                name = found_name
            description = found_desc
    body = body.strip()
    if not description:
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                description = line
                break
    return SkillSpec(name=name, description=description, body=body, path=path)


def _field(front: str, pattern: re.Pattern) -> str:
    match = pattern.search(front)
    if not match:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()
