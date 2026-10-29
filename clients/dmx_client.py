"""DMXAPI client: Jev TypeSafe judgment + light LLM chat.

Docs:
- Gateway overview: https://www.dmxapi.cn/rmb
- Jev: https://doc.dmxapi.cn/jev-1.13.0-typesafe-structured-judgment.html
- Chat: https://www.dmxapi.cn/v1/chat/completions
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests

import config


@dataclass
class JevAnswer:
    """Structured answer for one Jev question."""

    question_id: str
    type: str
    noul: Optional[float] = None
    choice: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def probability(self) -> float:
        """Return the yes/established probability for weighted rerank."""
        if self.noul is not None:
            return float(self.noul)
        if self.probabilities and self.choice is not None:
            return float(self.probabilities.get(self.choice, 0.0))
        if self.score is not None:
            return float(self.score)
        return 0.0


class DmxClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        jev_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        timeout: Optional[int] = None,
        jev_timeout: Optional[int] = None,
        llm_timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or config.require_api_key()
        self.base_url = (base_url or config.DMX_BASE_URL).rstrip("/")
        self.jev_model = jev_model or config.JEV_MODEL
        self.llm_model = llm_model or config.LLM_MODEL
        # timeout is legacy; prefer explicit jev_timeout / llm_timeout
        default_timeout = timeout if timeout is not None else config.JEV_TIMEOUT
        self.jev_timeout = (
            jev_timeout if jev_timeout is not None else default_timeout
        )
        self.llm_timeout = (
            llm_timeout if llm_timeout is not None else config.LLM_TIMEOUT
        )
        self.timeout = self.jev_timeout  # compat if callers still read .timeout
        self.max_retries = (
            max_retries if max_retries is not None else config.DMX_MAX_RETRIES
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------ Jev
    def jev_judge(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, JevAnswer]:
        """Call jev-1.13.0 TypeSafe API; answer multiple structured questions in one request."""
        url = f"{self.base_url}/typesafe/v1/systemone"
        payload = {
            "model": model or self.jev_model,
            "state": state,
            "questions": questions,
        }
        data = self._post_json(url, payload, timeout=self.jev_timeout)
        answers: Dict[str, JevAnswer] = {}
        for qid, ans in (data.get("answers") or {}).items():
            answers[qid] = JevAnswer(
                question_id=qid,
                type=ans.get("type", ""),
                noul=ans.get("noul"),
                choice=ans.get("choice"),
                score=ans.get("score"),
                confidence=ans.get("confidence"),
                probabilities=ans.get("probabilities"),
                raw=ans,
            )
        return answers

    def jev_noul(self, state: Union[str, Dict[str, Any]], instructions: str) -> float:
        """Convenience: single Boolean (noul) judgment; returns probability in [0, 1]."""
        answers = self.jev_judge(
            state=state,
            questions={"q": {"type": "noul", "instructions": instructions}},
        )
        return answers["q"].probability

    # ------------------------------------------------------------------ LLM
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """OpenAI-compatible Chat Completions."""
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model or self.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data = self._post_json(url, payload, timeout=self.llm_timeout)
        return data["choices"][0]["message"]["content"]

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Any:
        """Require JSON from the model and parse it into a Python object."""
        # Prefer json_object; fall back to plain chat + manual parse when unsupported
        try:
            content = self.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            content = self.chat(messages=messages, model=model, temperature=temperature)
        return self._extract_json(content)

    # --------------------------------------------------------------- helpers
    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        req_timeout = timeout if timeout is not None else self.timeout
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=req_timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:800]}"
                    )
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"DMXAPI 请求失败（已重试 {self.max_retries} 次）: {last_err}")

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Strip ```json / ``` fences
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise
