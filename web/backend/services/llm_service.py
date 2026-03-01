"""LLM provider abstraction — Anthropic and OpenAI streaming."""

from __future__ import annotations

import base64
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from ..config import Settings


async def _stream_anthropic(prompt: str, images: list[dict],
                            settings: Settings) -> AsyncGenerator[str, None]:
    """Stream from Anthropic Messages API."""
    try:
        import anthropic
    except ImportError:
        yield "[ERROR] anthropic package not installed. Run: pip install anthropic"
        return

    api_key = settings.llm_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        yield "[ERROR] No Anthropic API key configured."
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build content blocks
    content: list[dict] = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.get("media_type", "image/png"),
                "data": img["data"],
            },
        })
    content.append({"type": "text", "text": prompt})

    async with client.messages.stream(
        model=settings.llm_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_openai(prompt: str, images: list[dict],
                         settings: Settings) -> AsyncGenerator[str, None]:
    """Stream from OpenAI Chat Completions API."""
    try:
        import openai
    except ImportError:
        yield "[ERROR] openai package not installed. Run: pip install openai"
        return

    api_key = settings.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        yield "[ERROR] No OpenAI API key configured."
        return

    client = openai.AsyncOpenAI(api_key=api_key)

    content: list[dict] = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img.get('media_type', 'image/png')};base64,{img['data']}",
            },
        })
    content.append({"type": "text", "text": prompt})

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def stream_analysis(prompt: str, image_paths: list[Path],
                          settings: Settings) -> AsyncGenerator[str, None]:
    """Stream LLM analysis with images.

    Reads images from disk, base64-encodes them, and streams the response
    from the configured provider.
    """
    images: list[dict] = []
    for p in image_paths:
        if p.exists() and p.stat().st_size < 20_000_000:
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            suffix = p.suffix.lower()
            media = "image/png" if suffix == ".png" else "image/jpeg"
            images.append({"data": data, "media_type": media})

    if settings.llm_provider == "openai":
        gen = _stream_openai(prompt, images, settings)
    else:
        gen = _stream_anthropic(prompt, images, settings)

    async for token in gen:
        yield token
