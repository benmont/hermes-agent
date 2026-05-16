"""Anthropic Messages API transport.

Delegates to the existing adapter functions in agent/anthropic_adapter.py.
This transport owns format conversion and normalization — NOT client lifecycle.
"""

import json as _json
import re as _re
from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse

# Matches <tool_use>...</tool_use> blocks emitted by the model in text-tools mode.
_TOOL_USE_BLOCK_RE = _re.compile(r"<tool_use>\s*(.*?)\s*</tool_use>", _re.DOTALL)


def _parse_embedded_tool_calls(text: str) -> tuple:
    """Extract <tool_use>...</tool_use> JSON blocks from assistant text.

    Returns (list[ToolCall], pre_text) where pre_text is the text that
    appeared BEFORE the first <tool_use> block.  Text after </tool_use>
    tags is discarded — the model is instructed to stop there, and any
    continuation is typically a hallucinated tool result.
    """
    from agent.transports.types import build_tool_call

    calls = []
    pre_text = ""
    first_match = True

    for m in _TOOL_USE_BLOCK_RE.finditer(text):
        if first_match:
            pre_text = text[: m.start()].strip()
            first_match = False
        try:
            data = _json.loads(m.group(1))
            name = str(data.get("name") or "")
            input_val = data.get("input") or {}
            tc_id = str(data.get("id") or f"tc_{len(calls):03d}")
            calls.append(build_tool_call(id=tc_id, name=name, arguments=input_val))
        except (_json.JSONDecodeError, KeyError, TypeError):
            pass  # malformed block — skip silently

    # If no calls found, return original text unchanged
    remaining = pre_text if calls else text
    return calls, remaining


class AnthropicTransport(ProviderTransport):
    """Transport for api_mode='anthropic_messages'.

    Wraps the existing functions in anthropic_adapter.py behind the
    ProviderTransport ABC.  Each method delegates — no logic is duplicated.
    """

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to Anthropic (system, messages) tuple.

        kwargs:
            base_url: Optional[str] — affects thinking signature handling.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        base_url = kwargs.get("base_url")
        return convert_messages_to_anthropic(messages, base_url=base_url)

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Anthropic input_schema format."""
        from agent.anthropic_adapter import convert_tools_to_anthropic

        return convert_tools_to_anthropic(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Anthropic messages.create() kwargs.

        Calls convert_messages and convert_tools internally.

        params (all optional):
            max_tokens: int
            reasoning_config: dict | None
            tool_choice: str | None
            is_oauth: bool
            preserve_dots: bool
            context_length: int | None
            base_url: str | None
            fast_mode: bool
            drop_context_1m_beta: bool
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        return build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=params.get("max_tokens", 16384),
            reasoning_config=params.get("reasoning_config"),
            tool_choice=params.get("tool_choice"),
            is_oauth=params.get("is_oauth", False),
            preserve_dots=params.get("preserve_dots", False),
            context_length=params.get("context_length"),
            base_url=params.get("base_url"),
            fast_mode=params.get("fast_mode", False),
            drop_context_1m_beta=params.get("drop_context_1m_beta", False),
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Anthropic response to NormalizedResponse.

        Parses content blocks (text, thinking, tool_use), maps stop_reason
        to OpenAI finish_reason, and collects reasoning_details in provider_data.
        """
        import json
        from agent.anthropic_adapter import _to_plain_data
        from agent.transports.types import ToolCall

        strip_tool_prefix = kwargs.get("strip_tool_prefix", False)
        _MCP_PREFIX = "mcp_"

        text_parts = []
        reasoning_parts = []
        reasoning_details = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                reasoning_parts.append(block.thinking)
                block_dict = _to_plain_data(block)
                if isinstance(block_dict, dict):
                    reasoning_details.append(block_dict)
            elif block.type == "tool_use":
                name = block.name
                if strip_tool_prefix and name.startswith(_MCP_PREFIX):
                    name = name[len(_MCP_PREFIX):]
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=name,
                        arguments=json.dumps(block.input),
                    )
                )

        # Text-tools mode (HERMES_OAUTH_TEXT_TOOLS): the model outputs tool calls
        # as <tool_use>...</tool_use> text blocks instead of native tool_use
        # content blocks.  Parse them when strip_tool_prefix is set (OAuth path)
        # and no native tool_use blocks were found in the response.
        if strip_tool_prefix and not tool_calls and text_parts:
            full_text = "\n".join(text_parts)
            parsed_calls, remaining_text = _parse_embedded_tool_calls(full_text)
            if parsed_calls:
                tool_calls = parsed_calls
                text_parts = [remaining_text] if remaining_text.strip() else []

        finish_reason = self._STOP_REASON_MAP.get(response.stop_reason, "stop")
        # Promote finish_reason to tool_calls when we parsed embedded tool calls
        # (the API returned end_turn since no native tools were used).
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        provider_data = {}
        if reasoning_details:
            provider_data["reasoning_details"] = reasoning_details

        return NormalizedResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            usage=None,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Anthropic response structure is valid.

        An empty content list is legitimate when ``stop_reason == "end_turn"``
        — the model's canonical way of signalling "nothing more to add" after
        a tool turn that already delivered the user-facing text. Treating it
        as invalid falsely retries a completed response.
        """
        if response is None:
            return False
        content_blocks = getattr(response, "content", None)
        if not isinstance(content_blocks, list):
            return False
        if not content_blocks:
            return getattr(response, "stop_reason", None) == "end_turn"
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract Anthropic cache_read and cache_creation token counts."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        if cached or written:
            return {"cached_tokens": cached, "creation_tokens": written}
        return None

    # Promote the adapter's canonical mapping to module level so it's shared
    _STOP_REASON_MAP = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "refusal": "content_filter",
        "model_context_window_exceeded": "length",
    }

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Anthropic stop_reason to OpenAI finish_reason."""
        return self._STOP_REASON_MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("anthropic_messages", AnthropicTransport)
