# chatbot/services/gemini_client.py
from __future__ import annotations

import hashlib
import logging
import re
import time

import httpx

from chatbot.config import get_settings
from chatbot.models import ChatCache

logger = logging.getLogger(__name__)


# ---------- 공통: 컨텍스트 정리 ----------
def format_context(text: str) -> str:
    """컨텍스트에서 불필요한 포맷 제거."""
    if not text:
        return ""

    cleaned = text
    cleaned = re.sub(r"(참고자료|출처)", "", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)      # 마크다운 제목
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)     # 리스트 기호
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def clean_response(text: str) -> str:
    """최종 응답 마무리 정리."""
    if not text:
        return ""

    text = text.replace("**", "").replace("__", "").strip()
    # 너무 기묘하게 끝나면 마침표 하나 붙여주기
    if text and text[-1] not in {".", "!", "?", "~", "다"}:
        text += "."
    return text.strip()


# ---------- LLM 저수준 호출: Gemini ----------
def _call_gemini(system_prompt: str, user_message: str, temperature: float) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY가 설정되어 있지 않습니다.")
        return ""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    headers = {"Content-Type": "application/json"}
    params = {"key": settings.gemini_api_key}
    body = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": user_message},
                ]
            }
        ],
        "generationConfig": {"temperature": temperature},
    }

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(url, params=params, headers=headers, json=body)
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("Gemini API error %s: %s", status, exc.response.text)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return ""
        except httpx.RequestError as exc:
            logger.error("Gemini request error: %s", exc)
            if attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return ""

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        return ""
    text = parts[0].get("text")
    return text.strip() if isinstance(text, str) else ""


# ---------- LLM 저수준 호출: Groq(OpenAI 호환) ----------
def _call_groq(system_prompt: str, user_message: str, temperature: float) -> str:
    settings = get_settings()
    if not getattr(settings, "groq_api_key", None):
        return ""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }
    body = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
    }

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("Groq API error %s: %s", status, exc.response.text)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return ""
        except httpx.RequestError as exc:
            logger.error("Groq request error: %s", exc)
            if attempt < max_attempts - 1:
                time.sleep(1 + attempt)
                continue
            return ""

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content")
    return content.strip() if isinstance(content, str) else ""


# ---------- LLM Failover 래퍼 ----------
def call_llm_with_failover(system_prompt: str, user_message: str, temperature: float) -> str:
    """
    PRIMARY_LLM(env) 기준으로 우선 LLM 선택.
    - primary_llm = "gemini" → Gemini 먼저, 실패 시 Groq
    - primary_llm = "groq"   → Groq 먼저, 실패 시 Gemini
    """
    settings = get_settings()
    primary = (getattr(settings, "primary_llm", "gemini") or "gemini").lower()

    def use_gemini() -> str:
        return _call_gemini(system_prompt, user_message, temperature)

    def use_groq() -> str:
        return _call_groq(system_prompt, user_message, temperature)

    if primary == "groq":
        first, second = use_groq, use_gemini
    else:
        first, second = use_gemini, use_groq

    result = first()
    if result:
        return result

    logger.warning("1차 LLM 실패, 백업 LLM으로 시도합니다. primary=%s", primary)
    result = second()
    return result or ""


# ---------- DB 캐시 유틸 ----------
def _make_cache_key(query: str, context: str) -> str:
    raw = (query.strip() + "||" + context.strip()).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _get_cached_response(query: str, context: str) -> str | None:
    key = _make_cache_key(query, context)
    cache = ChatCache.objects.filter(query_hash=key).first()
    if cache:
        cache.hit_count += 1
        cache.save(update_fields=["hit_count"])
        logger.info("💾 DB 캐시 HIT (%s)", cache.query_hash)
        return cache.response
    return None


def _save_cache_response(query: str, context: str, response: str) -> None:
    key = _make_cache_key(query, context)
    try:
        ChatCache.objects.create(
            query_hash=key,
            query=query,
            context=context,
            response=response,
        )
        logger.info("💾 DB 캐시 저장 (%s)", key)
    except Exception as exc:
        logger.error("캐시 저장 중 오류: %s", exc)


# ---------- RAG + 패턴별 말투 + DB 캐싱 ----------
def call_gemini_with_rag(query: str, retrieved_docs: list) -> str:
    """
    RAG 기반 응답 생성 + 질문 패턴별 톤 조절 + DB 기반 캐싱.
    """

    # 1) 컨텍스트 합치기
    parts: list[str] = []
    for d in retrieved_docs:
        text = None
        if isinstance(d, dict):
            text = d.get("text") or d.get("snippet") or d.get("page_content")
        elif isinstance(d, str):
            text = d
        else:
            text = (
                getattr(d, "text", None)
                or getattr(d, "snippet", None)
                or getattr(d, "page_content", None)
            )
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    context_raw = " ".join(parts)
    context = format_context(context_raw)

    # 2) DB 캐시 조회
    cached = _get_cached_response(query, context)
    if cached:
        return clean_response(cached)

    # 3) 질문 패턴 분류
    SYMPTOM_KEYWORDS = [
        "아파", "통증", "붓", "부었", "열", "두통", "복통",
        "가슴이", "숨이", "호흡", "기침", "가래", "어지럽", "쓰러질"
    ]
    EMOTION_KEYWORDS = [
        "우울", "불안", "힘들", "상실감", "지치", "불편한 마음", "멘탈",
        "죽고싶", "살기 싫", "포기하고 싶"
    ]
    TIME_KEYWORDS = [
        "시간", "몇 시", "몇시", "운영", "오픈", "마감",
        "진료시간", "진료 시간", "언제까지", "몇까지 해요"
    ]

    def detect_mode(text: str) -> str:
        if any(k in text for k in EMOTION_KEYWORDS):
            return "emotional"
        if any(k in text for k in SYMPTOM_KEYWORDS):
            return "symptom"
        if any(k in text for k in TIME_KEYWORDS):
            return "time"
        return "info"

    mode = detect_mode(query)

    # 4) 공통 스타일 규칙
    base_style = """
당신은 병원 공식 안내 챗봇입니다.
모든 답변은 존댓말로, 2~4문장 이내로 간결하게 작성합니다.
문장은 짧고 명확하게 유지하고, 과도한 감탄사나 반복 표현은 사용하지 않습니다.
"""

    if mode == "time":
        extra_rule = """
사용자의 질문은 진료시간 또는 운영 시간과 관련이 있습니다.
다음 문장을 반드시 한 번 포함합니다:
"병원 진료시간은 오전 9시부터 오후 5시 30분까지입니다."
이외의 안내는 질문 범위 안에서만 간단히 추가합니다.
"""
    elif mode == "symptom":
        extra_rule = """
사용자의 증상에 대해 일반적인 설명과, 어느 진료과에서 상담을 받을 수 있는지 중심으로 안내합니다.
응급이 의심되는 경우에만 다음 문장을 마지막에 한 번 포함할 수 있습니다:
"증상이 갑자기 심해지거나 호흡이 곤란해지는 경우 응급실 방문이 필요할 수 있습니다."
진료시간이나 불필요한 추가 정보는 언급하지 않습니다.
"""
    elif mode == "emotional":
        extra_rule = """
감정이나 심리적 어려움에 대한 질문입니다.
사용자의 감정을 간단히 인정하되, 1~2문장 이내에서 조용한 톤으로 공감 표현을 하고,
필요 시 전문 상담이나 진료를 고려할 수 있다는 정도로 안내합니다.
과도한 위로나 사적인 조언은 하지 않습니다.
"""
    else:  # info
        extra_rule = """
병원 이용 안내, 예약 방법, 위치, 일반 정보와 관련된 질문입니다.
컨텍스트에서 핵심적인 정보만 선택하여 간결하게 요약해서 답변합니다.
질문과 직접 관련 없는 정보는 포함하지 않습니다.
"""

    system_prompt = base_style + extra_rule

    user_message = f"""
[사용자 질문]
{query}

[참고용 컨텍스트]
다음 내용은 사용자의 질문에 답하기 위한 참고 자료입니다.
이 내용을 그대로 복사하지 말고, 의미를 유지하면서 다른 표현으로 정리하여 답변하십시오.

--- context start ---
{context}
--- context end ---
"""

    # 5) LLM 호출 (Gemini / Groq Failover)
    raw_reply = call_llm_with_failover(system_prompt, user_message, temperature=0.2)
    if not raw_reply:
        return "현재 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주시기 바랍니다."

    final_reply = clean_response(raw_reply)

    # 6) DB 캐시 저장
    _save_cache_response(query, context, final_reply)

    return final_reply
