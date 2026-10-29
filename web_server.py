"""LexCaR local web server.

Run from this directory with ``python web_server.py``.  Existing session files
are read directly; new Stage 1 searches use the persistent search cache.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Windows GBK consoles raise UnicodeEncodeError on chars like ✓; reconfigure stdout/stderr safely
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
from clients.skill_registry import load_skills
from stages.session import SessionState
from stages.stage1_candidate_pool import Stage1CandidatePool
from stages.stage2_experience import Stage2Experience
from stages.stage3_user_follow import Stage3UserFollow

WEB_DIR = config.ROOT_DIR / "web"
_MCP_DIR = config.ROOT_DIR / "mcp"
_SKILLS_DIR = config.ROOT_DIR / "skills"
_ENV_PATH = config.ROOT_DIR / ".env"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ENV_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class SafeStream:
    """When the process has no valid console handle on Windows (e.g. IDE / scheduled task),
    rich / tqdm / print writing to stdout can raise ``OSError [Errno 22] Invalid argument``
    and abort an in-flight web request. This wrapper swallows write failures so console
    output cannot break the service.
    """

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except OSError:
            return len(data)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_safe_stdio() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = SafeStream(sys.stdout)
    sys.stderr = SafeStream(sys.stderr)


def _session_files() -> list[Path]:
    return sorted(config.OUTPUT_DIR.glob("session_*.json"), reverse=True)


def _load(path: Path) -> SessionState:
    return SessionState.load(path)


def _summary(session: SessionState, path: Path) -> dict:
    return {
        "session_id": session.session_id,
        "query": session.query,
        "created_at": session.created_at,
        "source": session.source,
        "candidate_count": len(session.candidate_pool),
        "ranking_count": len(session.current_ranking),
        "dimension_count": len(session.active_dimensions),
        "filter_count": len(session.filter_rules),
        "path": str(path.name),
    }


def _session_payload(session: SessionState, path: Path) -> dict:
    # Stage2 writes a slim current_ranking (doc_id/score/rank/dim_scores);
    # the UI table needs case metadata — enrich from the candidate pool and normalize weighted-score keys.
    pool_map = {str(item.get("doc_id")): item for item in session.candidate_pool or []}
    ranking = []
    for item in session.current_ranking or []:
        entry = dict(item)
        base = pool_map.get(str(entry.get("doc_id")))
        if base:
            for key in ("title", "court", "charges", "fact", "reason", "result", "retrieval_score"):
                if entry.get(key) is None and base.get(key) is not None:
                    entry[key] = base.get(key)
        # Stage1 also uses a score field; only treat score as weighted after Stage2/3
        # (dim_scores or confirmed elements), so the UI does not skip Stage2 as already reranked.
        if entry.get("weighted_score") is None and entry.get("score") is not None:
            if entry.get("dim_scores") is not None or session.confirmed_facts:
                entry["weighted_score"] = entry["score"]
        ranking.append(entry)
    payload = {
        "session": _summary(session, path),
        "query": session.query,
        "source": session.source,
        "stage1_meta": session.stage1_meta,
        "candidate_pool": session.candidate_pool,
        "stage2_ranked": session.stage2_ranked,
        "confirmed_facts": session.confirmed_facts,
        "active_dimensions": session.active_dimensions,
        "filter_rules": session.filter_rules,
        "stage3_history": session.stage3_history,
        "current_ranking": ranking,
    }
    return payload


# ---- MCP / Skill config management (frontend-customizable retrieval) ----

def _safe_name(name: object) -> str:
    name = str(name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError("名称只能由字母/数字开头，含字母、数字、下划线、连字符，1-64 位")
    return name


def _mcp_entries() -> list[dict]:
    items: list[dict] = []
    if not _MCP_DIR.is_dir():
        return items
    for path in sorted(_MCP_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for name, raw in (data.get("mcpServers") or {}).items():
            if not isinstance(raw, dict):
                continue
            headers = raw.get("headers") or {}
            has_auth = any(str(key).lower() == "authorization" for key in headers)
            items.append({
                "name": str(name),
                "url": str(raw.get("url") or ""),
                "auth": has_auth,
                "file": path.name,
            })
    return items


def _skill_entries() -> list[dict]:
    return [
        {"name": spec.name, "description": spec.description, "dir": spec.path.parent.name}
        for spec in load_skills()
    ]


def _safe_file_stem(name: object, default: str = "mcp") -> str:
    stem = Path(str(name or "").strip()).stem or default
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_")
    if not stem or not _NAME_RE.match(stem):
        stem = default
    return stem[:64]


def _parse_mcp_document(payload: dict) -> dict:
    """Accept either a full mcpServers document or {json/text, filename}."""
    if isinstance(payload.get("mcpServers"), dict):
        parsed: dict = {"mcpServers": payload["mcpServers"]}
    else:
        raw = payload.get("json")
        if raw is None:
            raw = payload.get("text")
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                raise ValueError("MCP JSON cannot be empty")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid MCP JSON: {exc}") from exc
        elif isinstance(raw, dict):
            parsed = raw
        else:
            raise ValueError("Submit a standard mcpServers JSON document")

    if not isinstance(parsed, dict) or not isinstance(parsed.get("mcpServers"), dict):
        raise ValueError("JSON must contain a top-level mcpServers object")
    servers = parsed["mcpServers"]
    if not servers:
        raise ValueError("mcpServers is empty")
    for name, raw in servers.items():
        try:
            _safe_name(name)
        except ValueError as exc:
            raise ValueError(f"Invalid MCP server name '{name}': {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"MCP server '{name}' must be an object")
        url = str(raw.get("url") or "").strip()
        if not re.match(r"^https?://", url):
            raise ValueError(
                f"MCP server '{name}' needs an http(s) url (Streamable HTTP only)"
            )
    return {"mcpServers": servers}


def _write_mcp_document(document: dict, filename: str | None = None) -> str:
    _MCP_DIR.mkdir(parents=True, exist_ok=True)
    servers = document["mcpServers"]
    stem = _safe_file_stem(filename)
    if stem == "mcp" and len(servers) == 1:
        stem = _safe_file_stem(next(iter(servers.keys())), default="mcp")
    path = _MCP_DIR / f"{stem}.json"
    # Avoid clobbering unrelated multi-server files that already hold the same name
    for entry in _mcp_entries():
        if entry["name"] in servers and entry["file"] != path.name:
            raise ValueError(
                f"MCP service '{entry['name']}' already exists in {entry['file']}; "
                "delete it first or use a different server name"
            )
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path.name


def _install_skill_zip(zip_bytes: bytes) -> list[str]:
    """Install skill folders from a zip. Each skill needs <name>/SKILL.md."""
    if len(zip_bytes) > 5 * 1024 * 1024:
        raise ValueError("Skill zip must be <= 5 MB")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid zip archive") from exc

    # Map skill_dir -> relative path of SKILL.md inside the zip
    skill_md: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        if name.startswith("__MACOSX/") or "/." in f"/{name}":
            continue
        parts = [p for p in name.split("/") if p]
        if len(parts) >= 2 and parts[-1].lower() == "skill.md":
            skill_dir = parts[-2]
            skill_md[skill_dir] = "/".join(parts[:-1]) + "/" + parts[-1]
        elif len(parts) == 1 and parts[0].lower() == "skill.md":
            # Bare SKILL.md at zip root → folder name from archive filename later
            skill_md[""] = parts[0]

    if not skill_md:
        raise ValueError(
            "Zip must contain at least one SKILL.md "
            "(e.g. my-skill/SKILL.md)"
        )

    installed: list[str] = []
    for folder_name, md_path in skill_md.items():
        if folder_name == "":
            raise ValueError(
                "Place SKILL.md inside a named folder, e.g. my-skill/SKILL.md"
            )
        if folder_name.startswith("_"):
            raise ValueError(
                f"Skill folder '{folder_name}' cannot start with _ "
                "(those are treated as disabled/examples)"
            )
        try:
            safe = _safe_name(folder_name)
        except ValueError as exc:
            raise ValueError(f"Invalid skill folder '{folder_name}': {exc}") from exc

        target = _SKILLS_DIR / safe
        if target.exists() and not (target / "SKILL.md").is_file():
            raise ValueError(f"Skill path conflict: {safe}")
        target.mkdir(parents=True, exist_ok=True)

        # Extract only files under this skill's directory prefix
        prefix = md_path.rsplit("/", 1)[0] + "/"
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = info.filename.replace("\\", "/")
            if member.startswith("__MACOSX/") or "/." in f"/{member}":
                continue
            if not (member == md_path or member.startswith(prefix)):
                continue
            rel = member[len(prefix) :] if member.startswith(prefix) else Path(member).name
            if not rel or ".." in Path(rel).parts:
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(info))
        if not (target / "SKILL.md").is_file():
            raise ValueError(f"Failed to install SKILL.md for '{safe}'")
        installed.append(safe)
    return installed


def _delete_mcp(name: str) -> bool:
    removed = False
    for path in sorted(_MCP_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        servers = data.get("mcpServers") or {}
        if name not in servers:
            continue
        servers.pop(name)
        removed = True
        if servers:
            path.write_text(json.dumps({"mcpServers": servers}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        else:
            # Empty file: disable by adding a _ prefix (loader ignores; rename back to restore)
            try:
                path.rename(path.with_name(f"_disabled_{path.name}"))
            except OSError:
                path.write_text(json.dumps({"mcpServers": {}}, ensure_ascii=False), encoding="utf-8")
    return removed


def _delete_skill(name: str) -> bool:
    source = _SKILLS_DIR / name
    if not (source / "SKILL.md").is_file():
        return False
    target = _SKILLS_DIR / f"_{name}"
    index = 1
    while target.exists():
        target = _SKILLS_DIR / f"_{name}_{index}"
        index += 1
    source.rename(target)
    return True


def _mask_api_key(key: str) -> str:
    key = str(key or "").strip()
    if not key or key.startswith("sk-your"):
        return ""
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:4]}...{key[-4:]}"


def _system_settings() -> dict:
    key = str(config.DMX_API_KEY or "").strip()
    key_set = bool(key) and not key.startswith("sk-your")
    return {
        "dmx_api_key_set": key_set,
        "dmx_api_key_mask": _mask_api_key(key),
        "dmx_base_url": str(config.DMX_BASE_URL or ""),
        "llm_model": str(config.LLM_MODEL or ""),
        "jev_model": str(config.JEV_MODEL or ""),
        "pool_size": int(config.CANDIDATE_POOL_SIZE),
        "max_rounds": int(config.STAGE1_AGENT_MAX_ROUNDS),
        "top_k": int(config.TOP_K_AFTER_STAGE2),
    }


def _upsert_env(updates: dict[str, str]) -> None:
    """Create or update keys in project .env; preserve unrelated lines."""
    lines: list[str] = []
    if _ENV_PATH.is_file():
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        match = _ENV_KEY_RE.match(line.strip()) if line.strip() and not line.lstrip().startswith("#") else None
        if match and match.group(1) in updates:
            key = match.group(1)
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    _ENV_PATH.write_text(text, encoding="utf-8")


def _apply_system_settings(payload: dict) -> dict:
    updates: dict[str, str] = {}

    if "dmx_api_key" in payload:
        key = str(payload.get("dmx_api_key") or "").strip()
        # Empty / unchanged mask → keep existing key
        if key and key != config.DMX_API_KEY and "..." not in key and "…" not in key and not key.startswith("••••"):
            updates["DMX_API_KEY"] = key

    if "dmx_base_url" in payload:
        base = str(payload.get("dmx_base_url") or "").strip().rstrip("/")
        if not base:
            raise ValueError("Base URL cannot be empty")
        if not re.match(r"^https?://", base):
            raise ValueError("Base URL must start with http:// or https://")
        updates["DMX_BASE_URL"] = base

    if "llm_model" in payload:
        model = str(payload.get("llm_model") or "").strip()
        if not model:
            raise ValueError("LLM model cannot be empty")
        updates["LLM_MODEL"] = model

    if "jev_model" in payload:
        model = str(payload.get("jev_model") or "").strip()
        if not model:
            raise ValueError("Jev model cannot be empty")
        updates["JEV_MODEL"] = model

    if "pool_size" in payload:
        try:
            pool_size = int(payload.get("pool_size"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Candidate pool size must be an integer") from exc
        if pool_size < 5 or pool_size > 500:
            raise ValueError("Candidate pool size must be between 5 and 500")
        updates["CANDIDATE_POOL_SIZE"] = str(pool_size)

    if "max_rounds" in payload:
        try:
            max_rounds = int(payload.get("max_rounds"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Retrieval rounds must be an integer") from exc
        if max_rounds < 1 or max_rounds > 20:
            raise ValueError("Retrieval rounds must be between 1 and 20")
        updates["STAGE1_AGENT_MAX_ROUNDS"] = str(max_rounds)

    if "top_k" in payload:
        try:
            top_k = int(payload.get("top_k"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Top-K must be an integer") from exc
        if top_k < 1 or top_k > 100:
            raise ValueError("Top-K must be between 1 and 100")
        # Stage2 / Stage3 share the same target Top-K
        updates["TOP_K_AFTER_STAGE2"] = str(top_k)
        updates["STAGE3_TOP_K"] = str(top_k)

    if not updates:
        return _system_settings()

    _upsert_env(updates)
    for key, value in updates.items():
        os.environ[key] = value
        if key == "DMX_API_KEY":
            config.DMX_API_KEY = value
        elif key == "DMX_BASE_URL":
            config.DMX_BASE_URL = value
        elif key == "LLM_MODEL":
            config.LLM_MODEL = value
        elif key == "JEV_MODEL":
            config.JEV_MODEL = value
        elif key == "CANDIDATE_POOL_SIZE":
            config.CANDIDATE_POOL_SIZE = int(value)
        elif key == "STAGE1_AGENT_MAX_ROUNDS":
            config.STAGE1_AGENT_MAX_ROUNDS = int(value)
        elif key == "TOP_K_AFTER_STAGE2":
            config.TOP_K_AFTER_STAGE2 = int(value)
        elif key == "STAGE3_TOP_K":
            config.STAGE3_TOP_K = int(value)
    return _system_settings()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self, max_bytes: int = 65536) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > max_bytes:
            raise ValueError("请求体无效")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是对象")
        return payload

    def _serve_file(self, path: Path, content_type: str) -> bool:
        try:
            body = path.read_bytes()
        except OSError:
            return False
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _write_ndjson(self, payload: dict) -> None:
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(line)
        self.wfile.flush()

    def _search_stream(self, query: str, *, pool_size: int, max_rounds: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def on_progress(message: str) -> None:
            self._write_ndjson({"type": "status", "text": str(message)})

        try:
            session = Stage1CandidatePool(
                pool_size=pool_size,
                max_rounds=max_rounds,
                on_progress=on_progress,
            ).run(query, on_progress=on_progress)
            path = config.OUTPUT_DIR / f"session_{session.session_id}.json"
            self._write_ndjson(
                {"type": "done", "payload": _session_payload(session, path)}
            )
        except Exception as exc:  # noqa: BLE001
            self._write_ndjson({"type": "error", "error": str(exc)})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/api/health":
            self._json(200, {"sessions": len(_session_files()), "cache": "persistent"})
            return
        if parsed.path == "/api/sessions":
            records = []
            for path in _session_files():
                try:
                    records.append(_summary(_load(path), path))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            self._json(200, {"items": records})
            return
        if parsed.path == "/api/demo":
            paths = _session_files()
            if not paths:
                self._json(404, {"error": "暂无可演示的会话"})
                return
            complete_paths = []
            for path in paths:
                try:
                    session = _load(path)
                    if session.stage2_ranked and session.confirmed_facts:
                        complete_paths.append((len(session.active_dimensions), path, session))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            if complete_paths:
                _, path, session = max(complete_paths, key=lambda item: item[0])
                self._json(200, _session_payload(session, path))
                return
            for path in paths:
                try:
                    session = _load(path)
                    if session.stage2_ranked or session.confirmed_facts or session.active_dimensions:
                        self._json(200, _session_payload(session, path))
                        return
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            try:
                self._json(200, _session_payload(_load(paths[0]), paths[0]))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(422, {"error": f"Demo data unreadable: {exc}"})
            return
        if parsed.path == "/api/session":
            name = params.get("id", [""])[0]
            path = config.OUTPUT_DIR / Path(name).name
            if not name or not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            try:
                self._json(200, _session_payload(_load(path), path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(422, {"error": f"Session unreadable: {exc}"})
            return
        if parsed.path == "/api/session/download":
            name = params.get("id", [""])[0]
            path = config.OUTPUT_DIR / Path(name).name
            if not name or not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            try:
                payload = _session_payload(_load(path), path)
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(422, {"error": f"Session unreadable: {exc}"})
                return
            fname = Path(name).stem + "_result.json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{fname}"',
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/config":
            self._json(
                200,
                {
                    "system": _system_settings(),
                    "mcp": _mcp_entries(),
                    "skills": _skill_entries(),
                },
            )
            return
        if parsed.path in ("/", "/index.html"):
            try:
                body = (WEB_DIR / "index.html").read_bytes()
            except OSError:
                self.send_error(500, "web/index.html not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Serve images under /figures/ and /assets/
        for prefix, dirname in (("/figures/", "figures"), ("/assets/", "assets")):
            if not parsed.path.startswith(prefix):
                continue
            root = (config.ROOT_DIR / dirname).resolve()
            rel = parsed.path[len(prefix) :]
            target = (root / rel).resolve()
            if (
                rel
                and target.is_file()
                and (target == root or root in target.parents)
            ):
                ctype = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".svg": "image/svg+xml",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }.get(target.suffix.lower(), "application/octet-stream")
                if self._serve_file(target, ctype):
                    return
            self.send_error(404)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        # Skill zip payloads are base64 JSON and may exceed the default 64 KB limit.
        max_bytes = 8 * 1024 * 1024 if parsed.path == "/api/config/skill" else 65536
        try:
            payload = self._read_json(max_bytes=max_bytes)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return

        if parsed.path == "/api/search":
            query = str(payload.get("query") or "").strip()
            if len(query) < 4:
                self._json(400, {"error": "The case description needs at least 4 characters"})
                return
            try:
                pool_size = int(payload.get("pool_size") or config.CANDIDATE_POOL_SIZE)
                max_rounds = int(payload.get("max_rounds") or config.STAGE1_AGENT_MAX_ROUNDS)
                pool_size = max(5, min(500, pool_size))
                max_rounds = max(1, min(20, max_rounds))
            except (TypeError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
                return
            stream = bool(payload.get("stream"))
            if stream:
                self._search_stream(query, pool_size=pool_size, max_rounds=max_rounds)
                return
            try:
                session = Stage1CandidatePool(
                    pool_size=pool_size,
                    max_rounds=max_rounds,
                ).run(query)
                path = config.OUTPUT_DIR / f"session_{session.session_id}.json"
                self._json(200, _session_payload(session, path))
            except Exception as exc:
                self._json(502, {"error": str(exc)})
            return

        if parsed.path == "/api/stage2":
            name = str(payload.get("session_id") or "")
            path = config.OUTPUT_DIR / Path(name).name
            if not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            try:
                session = _load(path)
                session = Stage2Experience().run(session)
                self._json(200, _session_payload(session, path))
            except Exception as exc:
                self._json(502, {"error": str(exc)})
            return

        if parsed.path == "/api/filters/clear":
            name = str(payload.get("session_id") or "")
            path = config.OUTPUT_DIR / Path(name).name
            if not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            session = _load(path)
            removed = len(session.filter_rules)
            session.filter_rules = []
            session.save()
            self._json(200, dict(_session_payload(session, path), removed=removed))
            return

        if parsed.path == "/api/filters/remove":
            name = str(payload.get("session_id") or "")
            path = config.OUTPUT_DIR / Path(name).name
            if not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            target = str(payload.get("name") or "")
            session = _load(path)
            before = len(session.filter_rules)
            session.filter_rules = [
                rule for rule in session.filter_rules
                if target not in {str(rule.get("name") or ""), str(rule.get("id") or "")}
            ]
            if len(session.filter_rules) == before:
                self._json(404, {"error": f"Filter rule not found: {target}"})
                return
            session.save()
            self._json(200, _session_payload(session, path))
            return

        # Only adjust formula / criteria / exclude rules; do not trigger Jev rerank
        if parsed.path == "/api/criteria":
            name = str(payload.get("session_id") or "")
            path = config.OUTPUT_DIR / Path(name).name
            if not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            try:
                session = _load(path)
                stage3 = Stage3UserFollow()
                weights = payload.get("weights") or {}
                if weights:
                    session = stage3.set_weights(
                        session,
                        {str(key): float(value) for key, value in weights.items()},
                        rerank=False,
                    )
                dimension_intent = str(payload.get("dimension") or "").strip()
                if dimension_intent:
                    session = stage3.add_dimension(
                        session, dimension_intent, rerank=False
                    )
                filter_intent = str(payload.get("filter") or "").strip()
                if filter_intent:
                    session = stage3.add_filter(
                        session, filter_intent, rerank=False
                    )
                self._json(200, _session_payload(session, path))
            except KeyError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(502, {"error": str(exc)})
            return

        if parsed.path == "/api/dimensions/remove":
            name = str(payload.get("session_id") or "")
            path = config.OUTPUT_DIR / Path(name).name
            if not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            target = str(payload.get("name") or "").strip()
            if not target:
                self._json(400, {"error": "Missing dimension name"})
                return
            try:
                session = _load(path)
                session = Stage3UserFollow().remove_dimensions(
                    session, [target], rerank=False
                )
                self._json(200, _session_payload(session, path))
            except KeyError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                self._json(502, {"error": str(exc)})
            return

        # In-pool rerank only (using already-applied criteria and rules)
        if parsed.path == "/api/rerank":
            name = str(payload.get("session_id") or "")
            path = config.OUTPUT_DIR / Path(name).name
            if not path.is_file():
                self._json(404, {"error": "Session not found"})
                return
            try:
                session = _load(path)
                session = Stage3UserFollow().rerank(session, note="web_rerank")
                self._json(200, _session_payload(session, path))
            except Exception as exc:
                self._json(502, {"error": str(exc)})
            return

        if parsed.path == "/api/config/system":
            try:
                system = _apply_system_settings(payload)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except OSError as exc:
                self._json(500, {"error": f"Failed to write .env: {exc}"})
                return
            self._json(200, {"ok": True, "system": system})
            return

        if parsed.path == "/api/config/mcp":
            try:
                document = _parse_mcp_document(payload)
                saved = _write_mcp_document(document, filename=str(payload.get("filename") or ""))
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except OSError as exc:
                self._json(500, {"error": f"写入配置失败：{exc}"})
                return
            self._json(
                200,
                {
                    "ok": True,
                    "file": saved,
                    "servers": list(document["mcpServers"].keys()),
                },
            )
            return

        if parsed.path == "/api/config/skill":
            raw_b64 = str(payload.get("zip_base64") or payload.get("zip") or "").strip()
            if not raw_b64:
                self._json(400, {"error": "Upload a skill zip (field zip_base64)"})
                return
            if raw_b64.startswith("data:") and "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                zip_bytes = base64.b64decode(raw_b64, validate=False)
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": f"Invalid base64 zip: {exc}"})
                return
            try:
                installed = _install_skill_zip(zip_bytes)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except OSError as exc:
                self._json(500, {"error": f"写入技能失败：{exc}"})
                return
            self._json(200, {"ok": True, "skills": installed})
            return

        self.send_error(404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/api/session":
            name = params.get("id", [""])[0]
            path = config.OUTPUT_DIR / Path(name).name
            if (
                not name
                or not path.is_file()
                or not path.name.startswith("session_")
                or path.suffix.lower() != ".json"
            ):
                self._json(404, {"error": "Session not found"})
                return
            try:
                path.unlink()
            except OSError as exc:
                self._json(500, {"error": f"Failed to delete session: {exc}"})
                return
            self._json(200, {"ok": True, "deleted": path.name})
            return
        if parsed.path == "/api/config/mcp":
            name = params.get("name", [""])[0]
            if _delete_mcp(name):
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "MCP service not found"})
            return
        if parsed.path == "/api/config/skill":
            name = params.get("name", [""])[0]
            try:
                name = _safe_name(name)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            if _delete_skill(name):
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "Skill not found"})
            return
        self.send_error(404)


def main() -> None:
    _install_safe_stdio()
    parser = argparse.ArgumentParser(description="USLeCaR Web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"USLeCaR: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
