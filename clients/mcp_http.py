"""MCP Streamable HTTP client: handshake then tools/call."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import requests


class McpHttpError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class McpHttpClient:
    """Connect to one Streamable HTTP MCP server and invoke tools."""

    def __init__(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        name: str = "mcp",
    ) -> None:
        self.url = url.strip()
        self.name = name
        self.timeout = timeout
        self._id = 0
        self._session_id: Optional[str] = None
        self._initialized = False
        self._protocol = "2025-03-26"
        self._extra_headers = {
            str(key): str(value)
            for key, value in (headers or {}).items()
            if str(value).strip()
        }
        self._http = requests.Session()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        message = self._rpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        if message.get("error"):
            err = message["error"]
            text = err.get("message") if isinstance(err, dict) else str(err)
            raise McpHttpError(f"{self.name}/{tool_name}: {text}", fatal=_is_fatal(text))
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpHttpError(f"{self.name}/{tool_name}: 返回缺少 result")
        if result.get("isError"):
            text = content_to_text(result) or f"{self.name}/{tool_name} 调用失败"
            raise McpHttpError(text, fatal=_is_fatal(text))
        text = content_to_text(result)
        if _is_fatal(text) and not _looks_like_cases(text):
            raise McpHttpError(text[:500], fatal=True)
        return result

    def list_tools(self) -> list:
        """MCP tools/list; returns tool name, description, and inputSchema."""
        self._ensure_initialized()
        tools: list = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            message = self._rpc("tools/list", params)
            if message.get("error"):
                err = message["error"]
                text = err.get("message") if isinstance(err, dict) else str(err)
                raise McpHttpError(f"{self.name} tools/list: {text}", fatal=_is_fatal(text))
            result = message.get("result") or {}
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        last_err: Optional[Exception] = None
        for version in ("2025-03-26", "2024-11-05"):
            self._protocol = version
            self._session_id = None
            try:
                message = self._rpc(
                    "initialize",
                    {
                        "protocolVersion": version,
                        "capabilities": {},
                        "clientInfo": {"name": "lexderanker", "version": "1.0"},
                    },
                )
                if message.get("error"):
                    err = message["error"]
                    text = err.get("message") if isinstance(err, dict) else str(err)
                    raise McpHttpError(f"{self.name} 初始化失败: {text}", fatal=_is_fatal(text))
                self._notify("notifications/initialized")
                self._initialized = True
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if isinstance(exc, McpHttpError) and exc.fatal:
                    raise
        raise McpHttpError(f"{self.name} 初始化失败: {last_err}", fatal=True)

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._id += 1
        return self._post(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        )

    def _notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method})

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self._protocol,
        }
        headers.update(self._extra_headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            resp = self._http.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise McpHttpError(f"{self.name} 网络错误: {exc}") from exc
        session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        if resp.status_code in (202, 204) or not (resp.text or "").strip():
            return {}
        if resp.status_code >= 400:
            text = (resp.text or "")[:500]
            raise McpHttpError(
                f"{self.name} HTTP {resp.status_code}: {text}",
                fatal=resp.status_code in (401, 403) or _is_fatal(text),
            )
        return _parse_http_body(resp)


def content_to_text(result: Dict[str, Any]) -> str:
    parts = []
    for block in result.get("content") or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("text"):
                parts.append(str(block["text"]))
            elif block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
    if parts:
        return "\n".join(parts).strip()
    if isinstance(result.get("structuredContent"), (dict, list)):
        return json.dumps(result["structuredContent"], ensure_ascii=False)
    return ""


def _parse_http_body(resp: requests.Response) -> Dict[str, Any]:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    text = resp.text or ""
    if "text/event-stream" in ctype or text.lstrip().startswith("event:") or text.lstrip().startswith("data:"):
        messages = _parse_sse(text)
        if not messages:
            return {}
        for message in reversed(messages):
            if isinstance(message, dict) and ("result" in message or "error" in message):
                return message
        last = messages[-1]
        return last if isinstance(last, dict) else {}
    try:
        data = resp.json()
    except ValueError as exc:
        raise McpHttpError(f"MCP 返回无法解析: {text[:300]}") from exc
    if isinstance(data, list):
        for message in reversed(data):
            if isinstance(message, dict) and ("result" in message or "error" in message):
                return message
        return {}
    if isinstance(data, dict):
        return data
    raise McpHttpError(f"MCP 返回类型异常: {type(data).__name__}")


def _parse_sse(text: str) -> list:
    messages = []
    # Split only on CR/LF. splitlines() would break on Unicode newlines inside JSON strings.
    for block in re.split(r"\r\n\r\n|\n\n|\r\r", text):
        data_lines = []
        for line in re.split(r"\r\n|\n|\r", block):
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines).strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return messages


def _is_fatal(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "quota",
        "insufficient",
        "unauthorized",
        "forbidden",
        "积分不足",
        "次数已用完",
        "额度",
        "请充值",
        "请补充",
        "invalid api key",
        "invalid token",
    )
    return any(marker in lowered for marker in markers)


def _looks_like_cases(text: str) -> bool:
    """Do not treat quota-like wording inside case text as an auth failure."""
    return "案号" in text or "本院认为" in text or "判决书" in text
