# chatbot/config.py
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py가 있는 위치: chat-django/chatbot/config.py
BASE_DIR = Path(__file__).resolve().parent  # => chat-django/chatbot


class Settings(BaseSettings):
    """
    Global application settings loaded from .env or environment variables.
    RAG / Gemini / Groq에서 공통으로 사용하는 설정들.
    """

    # ---- 🔑 API Keys ----
    gemini_api_key: str  # 반드시 .env에 설정
    groq_api_key: str | None = None  # 없으면 Failover 시 Groq은 건너뜀

    # Groq 모델 이름 (있다면)
    groq_model: str = "llama-3.1-8b-instant"

    # ---- 🧠 Embedding model ----
    embedding_model: str = "jhgan/ko-sroberta-multitask"

    # ---- 📂 Data / Vector store paths ----
    # data 디렉토리: chat-django/chatbot/data
    data_dir: Path = BASE_DIR / "data"

    # FAISS 인덱스 / 메타데이터 경로 (Path 타입)
    faiss_index_path: Path = data_dir / "faiss.index"
    metadata_path: Path = data_dir / "metadata.json"

    # ---- 🔍 RAG 검색 / 성능 옵션 ----
    top_k: int = 3                   # 검색해서 가져올 최대 결과 수
    max_context_chars: int = 1200    # LLM에 넘길 컨텍스트 전체 길이 제한(문자 수)

    # 어떤 LLM을 1순위로 쓸지: "gemini" 또는 "groq"
    primary_llm: str = "gemini"

    # ---- Optional ----
    database_url: str | None = None

    # ---- Pydantic Settings Config ----
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Django DEBUG, SECRET_KEY 등은 무시
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Ensure settings are loaded once (singleton behavior)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
