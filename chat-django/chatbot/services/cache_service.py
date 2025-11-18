# chatbot/services/cache_service.py
import hashlib
from typing import Tuple, List
from chatbot.models import ChatCache

def make_query_hash(query: str, context_text: str) -> str:
    """
    질문 + 컨텍스트 문자열을 기반으로 SHA-256 해시 생성
    """
    base = (query.strip() + "||" + context_text.strip()).encode("utf-8")
    return hashlib.sha256(base).hexdigest()

def get_cached_response(query: str, context_text: str) -> ChatCache | None:
    """
    동일한 query + context 조합이 이미 있는지 조회
    """
    qh = make_query_hash(query, context_text)
    return ChatCache.objects.filter(query_hash=qh).first()

def save_cache(query: str, context_text: str, response: str) -> ChatCache:
    """
    새로 생성된 LLM 응답을 캐시에 저장
    """
    qh = make_query_hash(query, context_text)
    obj, created = ChatCache.objects.get_or_create(
        query_hash=qh,
        defaults={
            "query": query,
            "context": context_text,
            "response": response,
            "hit_count": 1,
        },
    )
    if not created:
        obj.hit_count += 1
        obj.save(update_fields=["hit_count"])
    return obj
