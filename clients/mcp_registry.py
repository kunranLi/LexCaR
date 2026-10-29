"""Load standard mcpServers config. Tools are discovered via tools/list after connect."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

import config
from clients.mcp_http import McpHttpClient

console = Console()

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HTTP_TYPES = {"", "http", "streamablehttp", "streamable_http", "sse"}


@dataclass
class McpServerSpec:
    name: str
    url: str
    client: McpHttpClient
    tools: List[Dict[str, Any]] = field(default_factory=list)


def load_mcp_servers(directory: Optional[Path] = None) -> List[McpServerSpec]:
    """Merge mcpServers from mcp/*.json. Files whose names start with _ are skipped."""
    folder = directory or (config.ROOT_DIR / "mcp")
    if not folder.is_dir():
        return []
    servers: List[McpServerSpec] = []
    seen = set()
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("_"):
            continue
        for spec in _load_file(path):
            if spec.name in seen:
                console.print(f"[yellow]忽略重复的 MCP 服务[/] {spec.name}（{path.name}）")
                continue
            seen.add(spec.name)
            servers.append(spec)
    return servers


def _load_file(path: Path) -> List[McpServerSpec]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP 配置无法解析: {path.name}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("mcpServers"), dict):
        raise RuntimeError(f"{path.name} 需要包含 mcpServers 对象")
    specs: List[McpServerSpec] = []
    for name, raw in data["mcpServers"].items():
        spec = _parse_server(str(name), raw, path.name)
        if spec is not None:
            specs.append(spec)
    return specs


def _parse_server(name: str, raw: Any, filename: str) -> Optional[McpServerSpec]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"{filename} 中 {name} 必须是对象")
    transport = str(raw.get("type") or "").strip().lower()
    url = _subst(str(raw.get("url") or "")).strip()
    if raw.get("command") and not url:
        console.print(f"[yellow]跳过 {name}[/]：当前只连接带 url 的 Streamable HTTP MCP")
        return None
    if transport not in _HTTP_TYPES:
        console.print(f"[yellow]跳过 {name}[/]：不支持的 type={transport}")
        return None
    if not url:
        raise RuntimeError(f"{filename} 中 {name} 缺少 url")
    headers = _headers(raw.get("headers") or {})
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    if _missing_secret(raw, auth):
        console.print(f"[yellow]跳过 {name}[/]：请求头中的环境变量为空")
        return None
    client = McpHttpClient(
        url,
        headers=headers,
        timeout=config.STAGE1_MCP_TIMEOUT,
        name=name,
    )
    return McpServerSpec(name=name, url=url, client=client)


def _headers(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    headers = {}
    for key, value in raw.items():
        headers[str(key)] = _subst(str(value))
    return headers


def _missing_secret(raw: Dict[str, Any], authorization: str) -> bool:
    """Skip connecting when a ${TOKEN} is referenced but the env var is empty."""
    blob = json.dumps(raw.get("headers") or {}, ensure_ascii=False)
    names = _ENV_RE.findall(blob)
    if not names:
        return False
    if any(not os.getenv(name, "").strip() for name in names):
        return True
    return authorization.strip() in {"Bearer", "Bearer "}


def _subst(value: str) -> str:
    return _ENV_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
