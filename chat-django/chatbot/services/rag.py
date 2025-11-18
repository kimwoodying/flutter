from __future__ import annotations

import numpy as np
import hashlib

from chatbot.config import get_settings
from chatbot.services.embeddings import embed_texts
from chatbot.services.vector_store import get_vector_store
from chatbot.services.gemini_client import call_gemini_with_rag
from chatbot.models import ChatCache  # 🔸 캐시 모델 추가 import


def clean_response(text: str) -> str:
    """
    LLM 응답을 프론트로 보내기 전 마지막 정리.
    - 마크다운 굵게(**), 밑줄(__) 제거
    - 문장 끝이 어색하면 마침표 추가
    """
    if not text:
        return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."

    text = text.replace("**", "").replace("__", "").strip()

    if text and text[-1] not in {".", "!", "?", "다", "~"}:
        text += "."

    return text


def run_rag(user_message: str) -> dict:
    """
    병원 안내용 RAG 파이프라인 진입점.

    1) 사용자 질문 임베딩
    2) FAISS 벡터 검색
    3) 상위 문서 메타데이터를 LLM으로 전달
    4) Gemini/Groq 기반 톤 제어된 답변 생성
    """
    try:
        settings = get_settings()

        # 1) 질문 임베딩 생성
        embeddings = embed_texts([user_message])
        if not embeddings:
            raise ValueError("임베딩을 생성할 수 없습니다.")

        query_vector = np.array(embeddings[0], dtype="float32")

        # 2) FAISS 검색
        store = get_vector_store()
        top_k = getattr(settings, "top_k", 5)
        search_results = store.search(query_vector, top_k)

        if not search_results:
            return {
                "reply": (
                    "죄송합니다. 관련된 정보를 찾지 못해 정확한 안내가 어렵습니다. "
                    "자세한 사항은 병원 대표번호(042-000-0000)로 문의해 주시기 바랍니다."
                ),
                "sources": [],
            }

        # (score, metadata) 튜플 리스트 → 유사도 threshold 적용
        min_score = 0.5
        relevant_results = [(score, meta) for score, meta in search_results if score > min_score]

        # threshold 넘는 게 없으면 최상위 하나만이라도 사용
        if not relevant_results:
            relevant_results = [search_results[0]]

        # 상위 몇 개만 컨텍스트로 사용
        max_docs = 3
        contexts = [meta for _, meta in relevant_results[:max_docs]]

        if not contexts:
            return {
                "reply": (
                    "죄송합니다. 관련된 정보를 찾지 못해 정확한 안내가 어렵습니다. "
                    "자세한 사항은 병원 대표번호(042-000-0000)로 문의해 주시기 바랍니다."
                ),
                "sources": [],
            }

        # 3) LLM 호출
        raw_reply = call_gemini_with_rag(user_message, contexts)
        if not raw_reply:
            return {
                "reply": (
                    "죄송합니다. 현재 답변을 생성하는 데 문제가 발생했습니다. "
                    "잠시 후 다시 시도해 주시거나, 병원 대표번호(042-000-0000)로 문의해 주시기 바랍니다."
                ),
                "sources": [],
            }

        reply_text = clean_response(raw_reply)

        # 4) 출처 정보 → 지금은 숨기고 빈 리스트만
        return {
            "reply": reply_text,
            "sources": [],
        }

    except Exception as e:
        import traceback

        print(f"Error in RAG pipeline: {str(e)}\n{traceback.format_exc()}")
        return {
            "reply": (
                "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다. "
                "잠시 후 다시 시도해 주시기 바랍니다."
            ),
            "sources": [],
        }


# =========================
# 🔥 여기부터 캐싱 래퍼 추가
# =========================

def _make_query_hash(user_message: str) -> str:
    """
    질문 문자열만 기준으로 SHA-256 해시 생성.
    (나중에 컨텍스트 버전까지 넣고 싶으면 여기서 섞어줘도 됨)
    """
    base = user_message.strip().lower().encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def run_rag_with_cache(user_message: str) -> dict:
    """
    1) DB(ChatCache)에서 동일 질문 캐시 조회
    2) 있으면 → 바로 리턴 (hit_count 증가)
    3) 없으면 → 기존 run_rag 실행 후 결과를 캐시에 저장
    """
    query = user_message.strip()
    if not query:
        return {
            "reply": "질문이 비어 있습니다. 다시 입력해 주세요.",
            "sources": [],
        }

    qh = _make_query_hash(query)

    # 1) 캐시 조회
    try:
        cached = ChatCache.objects.filter(query_hash=qh).first()
    except Exception as e:
        print(f"[ChatCache] 조회 오류: {e}")
        cached = None

    if cached:
        try:
            cached.hit_count += 1
            cached.save(update_fields=["hit_count"])
        except Exception as e:
            print(f"[ChatCache] hit_count 업데이트 오류: {e}")

        return {
            "reply": cached.response,
            "sources": [],   # 캐시에서도 sources는 비워서 리턴
        }

    # 2) 캐시 없으면 → 원래 RAG 실행
    result = run_rag(query)
    reply_text = result.get("reply") or ""

    # 3) 캐시 저장
    if reply_text:
        try:
            ChatCache.objects.create(
                query_hash=qh,
                query=query,
                context="",     # 나중에 컨텍스트 전문까지 저장하고 싶으면 여기 채워도 됨
                response=reply_text,
                hit_count=1,
            )
        except Exception as e:
            print(f"[ChatCache] 저장 오류: {e}")

    return result
