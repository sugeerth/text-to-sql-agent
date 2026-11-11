"""Text-to-SQL agent: one LLM call on the happy path, then a code-driven
guard -> execute -> repair loop.

Design: the LLM does the reasoning (understand the question, write SQL,
interpret database errors); deterministic code owns safety and control
(read-only guard, execution, retry limit, timeouts). This keeps P50 latency
at a single model call while still letting the model observe real database
feedback and repair its own mistakes.
"""

import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from openai import OpenAI

from src.db import DEFAULT_DB, connect, query
from src.prompts import build_system_prompt


def _env(*names: str) -> Optional[str]:
    """First non-empty environment variable among `names`."""
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


@dataclass
class AgentConfig:
    # Any OpenAI-compatible endpoint: a hosted provider, OpenAI, or a local
    # server (vLLM, SGLang, Ollama). OPENAI_* names work as fallbacks.
    model: str = field(default_factory=lambda: _env("LLM_MODEL") or "")
    base_url: Optional[str] = field(default_factory=lambda: _env("LLM_BASE_URL", "OPENAI_BASE_URL"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "OPENAI_API_KEY") or "")
    temperature: float = 0.0
    max_tokens: int = 600
    max_repairs: int = 2      # LLM calls per question: 1 + up to 2 repairs
    # Optional, for reasoning models: "none" = no hidden thinking. Measured ~1s replies
    # vs 10-40s with reasoning on, and hidden reasoning can starve max_tokens.
    reasoning_effort: Optional[str] = field(default_factory=lambda: _env("LLM_REASONING_EFFORT"))
    request_timeout: float = 25.0  # serverless tail latency can spike on cold shards
    history_turns: int = 6    # user/assistant exchanges kept for follow-ups


@dataclass
class AgentResult:
    question: str
    sql: str = ""
    df: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    attempts: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.df is not None


class SqlAgent:
    """Converts natural-language questions to executed SQL results."""

    def __init__(self, db_path: str = DEFAULT_DB, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        # Schema context is built once from a normal connection...
        bootstrap = connect(db_path)
        self.system_prompt = build_system_prompt(bootstrap)
        bootstrap.close()
        # ...but generated SQL only ever runs on a READ-ONLY connection.
        self.conn = connect(db_path, read_only=True)
        self.client = OpenAI(
            base_url=self.config.base_url,
            # Placeholder keeps offline paths (self-test, imports) working;
            # real calls fail with a clear auth error if the key is absent.
            api_key=self.config.api_key or "LLM_API_KEY-not-set",
            timeout=self.config.request_timeout,
            max_retries=2,
        )
        self.history: list[dict] = []  # finished user/assistant turns only

    # ---------------------------------------------------------------- public

    def ask(self, question: str) -> AgentResult:
        """generate -> guard -> execute; on failure, feed the error back to the
        model and retry (max_repairs times). Repair chatter never enters
        conversation history — only the final working SQL does."""
        result = AgentResult(question=question)
        start = time.perf_counter()
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.history
            + [{"role": "user", "content": question}]
        )

        for attempt in range(1 + self.config.max_repairs):
            result.attempts = attempt + 1
            try:
                raw, usage = self._generate(messages)
                if usage:
                    result.prompt_tokens += usage.prompt_tokens
                    result.completion_tokens += usage.completion_tokens
            except Exception as e:  # network/timeout — not repairable by the model
                result.error = f"Model request failed: {e}"
                break

            result.sql = self._extract_sql(raw)
            error = self._guard(result.sql)
            if error is None:
                try:
                    result.df = query(self.conn, result.sql)
                    result.error = None
                    break
                except (sqlite3.Error, pd.errors.DatabaseError) as e:
                    error = str(e)

            result.error = error
            # Show the model its SQL and the concrete failure, ask for a fix.
            messages += [
                {"role": "assistant", "content": result.sql},
                {"role": "user",
                 "content": f"That query failed: {error}\nReturn a corrected query."},
            ]

        result.latency_s = time.perf_counter() - start
        if result.ok:
            self._remember(question, result)
        return result

    def reset(self) -> None:
        self.history.clear()

    # --------------------------------------------------------------- internal

    def _generate(self, messages: list[dict]):
        if not self.config.model:
            raise RuntimeError("LLM_MODEL is not set (see .env.example)")
        extra = {}
        if self.config.reasoning_effort:
            extra["extra_body"] = {"reasoning_effort": self.config.reasoning_effort}
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **extra,
        )
        return response.choices[0].message.content or "", response.usage

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Pull SQL out of the model reply: strip any reasoning trace, prefer a
        ```sql fence, fall back to the whole reply."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        fence = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
        return (fence.group(1) if fence else text).strip()

    @staticmethod
    def _guard(sql: str) -> Optional[str]:
        """Read-only whitelist: exactly one statement, starting with SELECT or
        WITH. Runs before execution; the read-only connection is defense in
        depth behind it."""
        stripped = re.sub(r"--[^\n]*", "", sql)                       # line comments
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)  # block comments
        stripped = stripped.strip().rstrip(";").strip()
        if not stripped:
            return "Empty query."
        if ";" in stripped:
            return "Multiple SQL statements are not allowed."
        if not re.match(r"(?i)^(SELECT|WITH)\b", stripped):
            return "Only read-only SELECT (or WITH ... SELECT) queries are allowed."
        return None

    def _remember(self, question: str, result: AgentResult) -> None:
        """Store the finished turn: SQL plus a small result preview, so
        follow-ups like 'now only for Brazil' have real context."""
        preview = result.df.head(3).to_string(index=False) if len(result.df) else "(0 rows)"
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant",
             "content": f"```sql\n{result.sql}\n```\nResult preview:\n{preview}"},
        ]
        max_messages = self.config.history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
