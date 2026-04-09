"""Claude Code CLI subprocess adapter for Hermes Agent.

Enables inference via the locally-installed ``claude`` CLI tool, letting
users with a Claude Pro/Max subscription run Hermes without a separate API
key.  The CLI manages its own auth (OAuth via claude.ai); Hermes just drives
it as a subprocess.

Why this exists vs the native ``anthropic`` provider
-----------------------------------------------------
The native ``anthropic`` provider calls ``api.anthropic.com`` directly and
requires API credits.  Claude Pro/Max subscriptions don't grant API access —
they grant CLI access.  This adapter bridges the two by shelling out to the
``claude`` binary for each completion request, which honours the subscription
billing path.

Limitations
-----------
- Tool calls use JSON-in-prompt encoding (no native streaming tool events).
- Streaming is not implemented; each call blocks until the CLI exits.
- Multi-turn history is flattened into a single prompt via ``<conversation>``
  XML tags — token usage numbers may differ from the native API.

Usage::

    from agent.claude_cli_adapter import ClaudeCliAdapter, is_claude_cli_available

    if is_claude_cli_available():
        adapter = ClaudeCliAdapter()
        resp = adapter.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}],
            model="claude-sonnet-4-6",
        )
        print(resp.choices[0].message.content)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CLAUDE_CLI_NAME = "claude"
DEFAULT_CLAUDE_CLI_MODEL = "claude-sonnet-4-6"
DEFAULT_TIMEOUT_SECONDS = 120

# ── Tool-call injection ───────────────────────────────────────────────────────
# When tools are present Hermes needs structured tool-call objects back.
# We ask Claude to emit a sentinel JSON block that we can reliably parse.

_TOOL_CALL_SYSTEM_INJECT = """
When you need to call a tool, respond ONLY with this exact JSON structure — no surrounding prose, no markdown fences:

{"tool_call": {"name": "<tool_name>", "arguments": {<argument_object>}}}

After you receive the tool result continue responding normally.
The available tools are listed below.
""".strip()

_TOOL_CALL_RE = re.compile(r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


# ── Availability helpers ──────────────────────────────────────────────────────

def is_claude_cli_available() -> bool:
    """Return True if the ``claude`` CLI is installed and on PATH."""
    return shutil.which(CLAUDE_CLI_NAME) is not None


def get_claude_cli_path() -> Optional[str]:
    """Return the full path to the ``claude`` binary, or None if not found."""
    return shutil.which(CLAUDE_CLI_NAME)


# ── Message formatting ────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    """Flatten an OpenAI content field (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                parts.append(f"[Tool result: {inner}]")
        return "\n".join(parts)
    return str(content) if content else ""


def _format_tool_definitions(tools: List[Dict]) -> str:
    """Produce a compact JSON summary of tool definitions for the system prompt."""
    simplified = [
        {
            "name": t.get("function", {}).get("name", ""),
            "description": t.get("function", {}).get("description", ""),
            "parameters": t.get("function", {}).get("parameters", {}),
        }
        for t in tools
        if isinstance(t, dict)
    ]
    return json.dumps(simplified, indent=2)


def build_prompt_and_system(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
) -> Tuple[Optional[str], str]:
    """Convert OpenAI-style messages to ``(system_prompt, user_prompt)``.

    Multi-turn history is embedded in the user prompt inside
    ``<conversation_history>`` XML tags so Claude can follow prior context.
    Tool definitions, when provided, are appended to the system prompt with
    instructions on how to emit a parseable tool-call block.

    Args:
        messages: OpenAI-style message list.
        tools:    OpenAI tool definitions (optional).

    Returns:
        Tuple of ``(system_prompt_or_None, user_prompt_string)``.
    """
    system_parts: List[str] = []
    turns: List[Dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_text(msg.get("content", ""))

        if role == "system":
            system_parts.append(content)
            continue

        if role == "assistant":
            # Embed any tool_calls the assistant made as text so the next turn
            # has context about what was requested.
            tc_list = msg.get("tool_calls") or []
            if tc_list:
                tc_summary = json.dumps([
                    {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    }
                    for tc in tc_list
                ], indent=2)
                content = (content + f"\n[Tool calls: {tc_summary}]").strip()

        if role == "tool":
            tool_name = msg.get("name", "tool")
            turns.append({"role": "tool_result", "name": tool_name, "content": content})
            continue

        turns.append({"role": role, "content": content})

    # Build system prompt string
    system_prompt: Optional[str] = "\n\n".join(p for p in system_parts if p) or None

    if tools:
        tool_defs = _format_tool_definitions(tools)
        tool_block = f"{_TOOL_CALL_SYSTEM_INJECT}\n\n<tools>\n{tool_defs}\n</tools>"
        system_prompt = (system_prompt + "\n\n" + tool_block) if system_prompt else tool_block

    if not turns:
        return system_prompt, ""

    if len(turns) == 1:
        return system_prompt, turns[0]["content"]

    # Multi-turn: embed history, append last turn as the new prompt
    _role_label = {"user": "Human", "assistant": "Assistant", "tool_result": "Tool"}
    history_lines: List[str] = []
    for turn in turns[:-1]:
        label = _role_label.get(turn["role"], turn["role"].capitalize())
        name = turn.get("name", "")
        prefix = f"{label} ({name})" if name else label
        history_lines.append(f"{prefix}: {turn['content']}")

    last = turns[-1]
    last_label = _role_label.get(last["role"], last["role"].capitalize())
    history = "\n".join(history_lines)
    prompt = (
        f"<conversation_history>\n{history}\n</conversation_history>\n\n"
        f"{last_label}: {last['content']}"
    )
    return system_prompt, prompt


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> Optional[Dict]:
    """Extract the first ``{"tool_call": {...}}`` block from the model output."""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        return obj.get("tool_call")
    except (json.JSONDecodeError, AttributeError):
        return None


def _make_completion_response(
    text: str,
    model: str,
    tools_present: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Any:
    """Wrap a CLI text response in an OpenAI-compatible ChatCompletion object."""
    tool_calls = None
    finish_reason = "stop"

    if tools_present:
        tc = _parse_tool_call(text)
        if tc:
            call_id = f"call_{int(time.time() * 1000)}"
            tool_calls = [SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(
                    name=str(tc.get("name", "")),
                    arguments=(
                        tc["arguments"]
                        if isinstance(tc.get("arguments"), str)
                        else json.dumps(tc.get("arguments", {}))
                    ),
                ),
            )]
            text = ""
            finish_reason = "tool_calls"

    message = SimpleNamespace(
        role="assistant",
        content=text,
        tool_calls=tool_calls,
        refusal=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=finish_reason,
        logprobs=None,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        id=f"claude-cli-{int(time.time() * 1000)}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


# ── Namespace wrappers ────────────────────────────────────────────────────────

class _CompletionsAdapter:
    def __init__(self, adapter: "ClaudeCliAdapter") -> None:
        self._adapter = adapter

    def create(
        self,
        *,
        messages: List[Dict],
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        timeout: Optional[float] = None,
        **_kwargs: Any,
    ) -> Any:
        return self._adapter._invoke(
            messages=messages,
            model=model or self._adapter.model,
            tools=tools,
            timeout=timeout,
        )


class _ChatNamespace:
    def __init__(self, adapter: "ClaudeCliAdapter") -> None:
        self.completions = _CompletionsAdapter(adapter)


# ── Public adapter class ──────────────────────────────────────────────────────

class ClaudeCliAdapter:
    """Subprocess-backed completions adapter using the local ``claude`` CLI.

    Exposes a ``chat.completions.create`` interface compatible with the OpenAI
    Python SDK so it acts as a drop-in replacement in Hermes's provider chain.

    Tool calls are handled via JSON-in-prompt encoding: tool definitions are
    injected into the system prompt and Claude's response is scanned for a
    ``{"tool_call": {...}}`` JSON block.  This is reliable with Claude models
    but lacks native streaming tool events.

    Args:
        model:           Default model slug passed to ``--model``.
        default_timeout: Subprocess wall-clock timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_CLI_MODEL,
        default_timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        cli_path = get_claude_cli_path()
        if not cli_path:
            raise FileNotFoundError(
                "The 'claude' CLI is not installed. "
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )
        self._cli_path = cli_path
        self.model = model
        self._default_timeout = default_timeout
        self.chat = _ChatNamespace(self)

    # OpenAI client duck-type attributes used by Hermes's auxiliary framework
    @property
    def api_key(self) -> str:
        return "claude-cli"

    @property
    def base_url(self) -> str:
        return "subprocess://claude"

    def _invoke(
        self,
        messages: List[Dict],
        model: str,
        tools: Optional[List[Dict]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        system_prompt, user_prompt = build_prompt_and_system(messages, tools)

        if not user_prompt.strip():
            raise ValueError("claude-cli: no user message found in conversation")

        cmd = [
            self._cli_path,
            "-p", user_prompt,
            "--output-format", "json",
            "--model", model,
        ]
        if system_prompt:
            cmd += ["--system", system_prompt]

        logger.debug(
            "claude-cli: invoking %s model=%s tools=%d",
            self._cli_path, model, len(tools or []),
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self._default_timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude CLI timed out after {timeout or self._default_timeout}s"
            ) from exc

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "(no stderr)"
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {stderr}"
            )

        raw = proc.stdout.strip()
        text = raw
        prompt_tokens = 0
        completion_tokens = 0

        try:
            data = json.loads(raw)
            if data.get("is_error"):
                raise RuntimeError(
                    f"claude CLI error: {data.get('result', 'unknown error')}"
                )
            text = data.get("result", "")
            usage = data.get("usage", {})
            prompt_tokens = int(usage.get("input_tokens", 0))
            completion_tokens = int(usage.get("output_tokens", 0))
        except json.JSONDecodeError:
            # Older CLI builds may not emit JSON — treat stdout as plain text.
            text = raw

        return _make_completion_response(
            text,
            model,
            tools_present=bool(tools),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# ── Async shim (runs sync adapter in a thread pool) ───────────────────────────

class _AsyncCompletionsAdapter:
    def __init__(self, sync: _CompletionsAdapter) -> None:
        self._sync = sync

    async def create(self, **kwargs: Any) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncChatNamespace:
    def __init__(self, sync_chat: _ChatNamespace) -> None:
        self.completions = _AsyncCompletionsAdapter(sync_chat.completions)


class AsyncClaudeCliAdapter:
    """Async wrapper over :class:`ClaudeCliAdapter` for use in async contexts."""

    def __init__(self, sync_adapter: ClaudeCliAdapter) -> None:
        self._sync = sync_adapter
        self.chat = _AsyncChatNamespace(sync_adapter.chat)
        self.api_key = sync_adapter.api_key
        self.base_url = sync_adapter.base_url
