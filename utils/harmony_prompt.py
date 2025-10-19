"""Harmony prompt rendering utilities leveraging the openai_harmony library."""

from __future__ import annotations

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)


_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def render_harmony_prompt(
    system_prompt: str,
    developer_prompt: str,
    user_prompt: str,
) -> str:
    """Render a Harmony-formatted prompt using openai_harmony."""

    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_prompt.strip()),
        Message.from_role_and_content(Role.DEVELOPER, developer_prompt.strip()),
        Message.from_role_and_content(Role.USER, user_prompt.strip()),
    ]
    conversation = Conversation.from_messages(messages)
    rendered = _ENCODING.render_conversation_for_completion(conversation, Role.ASSISTANT)
    prompt = rendered if isinstance(rendered, str) else _ENCODING.decode_tokens(rendered)

    if "<|start|>assistant" in prompt:
        if not prompt.rstrip().endswith("<|channel|>final<|message|>"):
            prompt = f"{prompt.rstrip()}<|channel|>final<|message|>"
    else:
        prompt = f"{prompt.rstrip()}\n<|start|>assistant<|channel|>final<|message|>"
    return prompt


def extract_final(text: str) -> str | None:
    """Extract the assistant final-channel payload using openai_harmony."""

    messages = _ENCODING.parse_messages_from_completion_text(text, Role.ASSISTANT)
    for msg in messages:
        if msg.role == Role.ASSISTANT and msg.channel == "final":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content.strip()
    return None
