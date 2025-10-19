"""Harmony prompt renderer and parser utilities."""

from __future__ import annotations

import re
from typing import Optional

SYSTEM_TEMPLATE = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-10-19

Reasoning: {reasoning}

{system_extra}
# Valid channels: final. Channel must be included for every message.<|end|>
"""

DEVELOPER_TEMPLATE = """<|start|>developer<|message|># Instructions
{instructions}

# Response Formats

## {format_name}
{json_schema}<|end|>
"""

USER_TEMPLATE = "<|start|>user<|message|>{user}<|end|>\n<|start|>assistant"

_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(?P<body>.*?)(?:<\|return\|>|<\|end\|>)",
    flags=re.DOTALL,
)


def render_harmony_prompt(
    developer_instructions: str,
    response_format_name: str,
    response_format_json_schema: str,
    user_prompt: str,
    reasoning: str = "medium",
    system_extra: str = "",
) -> str:
    """Render a single Harmony-formatted prompt suitable for /v1/completions."""
    extra = system_extra.strip()
    if extra:
        extra = extra + "\n"
    sys_prompt = SYSTEM_TEMPLATE.format(
        reasoning=reasoning.strip(),
        system_extra=extra,
    )
    dev_prompt = DEVELOPER_TEMPLATE.format(
        instructions=developer_instructions.strip(),
        format_name=response_format_name.strip(),
        json_schema=response_format_json_schema.strip(),
    )
    user_block = USER_TEMPLATE.format(user=user_prompt.strip())
    return f"{sys_prompt}{dev_prompt}{user_block}"


def extract_final(text: str) -> Optional[str]:
    """Extract the assistant final channel from a Harmony completion, if present."""
    match = _FINAL_RE.search(text)
    if not match:
        return None
    return match.group("body").strip()
