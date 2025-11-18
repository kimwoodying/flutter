from __future__ import annotations

from functools import lru_cache

from chatbot.config import get_settings


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    settings = get_settings()
    with open(settings.prompt_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()
