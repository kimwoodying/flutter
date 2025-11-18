"""
main.py
병원 모바일 앱 챗봇 FastAPI 백엔드 (RAG + DB 라우팅 예시)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, create_engine, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hospital-chatbot")

app = FastAPI(title="Hospital Chatbot Backend", version="1.0.0")

KNOWLEDGE_BASE_DIR = Path("./knowledge_base")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_DIM = 768
TOP_K = 5

DOCUMENTS: List["Document"] = []
CHUNKS: List["DocumentChunk"] = []
CHUNK_LOOKUP: Dict[str, "DocumentChunk"] = {}
VECTOR_STORE: Optional["VectorStore"] = None

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://acorn:acorn1234@34.42.223.43:3306/hospital_db",
)
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    birth_date = Column(DateTime)


class Reservation(Base):
    __tablename__ = "reservations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False)
    department = Column(String, nullable=False)
    scheduled_at = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)


class Queue(Base):
    __tablename__ = "queues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False)
    department = Column(String, nullable=False)
    ticket_number = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)


class ExamResult(Base):
    __tablename__ = "exam_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    taken_at = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)
    summary = Column(String)


@dataclass
class Document:
    doc_id: str
    metadata: Dict[str, Any]
    content: str


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="환자/사용자 식별자")
    message: str = Field(..., description="사용자 질문 텍스트")


class ChatResponse(BaseModel):
    answer: str
    source: Literal["rag", "db", "fallback", "safety"]
    references: Optional[List[str]] = None
    payload: Optional[Dict[str, Any]] = None


def get_db_session() -> Session:
    return SessionLocal()


def load_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> List[Document]:
    documents: List[Document] = []
    if not directory.exists():
        logger.warning("Knowledge base directory %s does not exist.", directory)
        return documents

    for path in sorted(directory.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        raw_text = path.read_text(encoding="utf-8").strip()
        metadata, content = parse_front_matter(raw_text)
        doc_id = path.stem
        documents.append(Document(doc_id=doc_id, metadata=metadata, content=content))
    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents


def parse_front_matter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    if not raw_text.startswith("---"):
        return {}, raw_text
    parts = raw_text.split("\n---", maxsplit=1)
    if len(parts) != 2:
        return {}, raw_text
    front = parts[0].strip("- \n")
    body = parts[1].lstrip("\n")
    metadata: Dict[str, Any] = {}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks


def get_embedding(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=EMBEDDING_DIM)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class VectorStore:
    def __init__(self, embeddings: np.ndarray, ids: List[str], use_faiss: bool = True):
        self.embeddings = embeddings.astype(np.float32)
        self.ids = ids
        if use_faiss and faiss is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            self._use_faiss = True
            logger.info("FAISS index built with %d vectors.", len(ids))
        else:
            self.index = None
            self._use_faiss = False
            logger.warning("FAISS not available. Falling back to numpy search.")

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        query = query.astype(np.float32)
        if self._use_faiss and self.index is not None:
            distances, indices = self.index.search(query[None, :], top_k)
            results: List[Tuple[str, float]] = []
            for idx, score in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                results.append((self.ids[idx], float(score)))
            return results
        sims = self.embeddings @ query
        best_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.ids[i], float(sims[i])) for i in best_indices if sims[i] > 0]


def assemble_chunks(documents: List[Document]) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for doc in documents:
        for idx, text in enumerate(chunk_text(doc.content)):
            chunk_id = f"{doc.doc_id}::chunk-{idx}"
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_index"] = idx
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc.doc_id,
                    text=text,
                    metadata=chunk_meta,
                )
            )
    logger.info("Generated %d chunks from %d documents.", len(chunks), len(documents))
    return chunks


def build_faiss_index(chunks: List[DocumentChunk]) -> VectorStore:
    if not chunks:
        raise ValueError("No document chunks available to build the index.")
    embeddings = np.vstack([get_embedding(chunk.text) for chunk in chunks])
    ids = [chunk.chunk_id for chunk in chunks]
    return VectorStore(embeddings=embeddings, ids=ids, use_faiss=True)


def build_prompt(contexts: List[str], question: str) -> str:
    system_prompt = (
        "당신은 병원 이용 안내를 돕는 챗봇입니다. 의학적 진단이나 치료 지침을 제공해서는 안 되며, "
        "응급 상황으로 보이면 119에 신고하거나 응급실 방문을 안내하세요. 답변은 반드시 한국어로 간결하고 친절하게 작성하세요."
    )
    context_block = "\n\n".join(f"[참고 자료]\n{ctx}" for ctx in contexts)
    user_block = f"[사용자 질문]\n{question}"
    return f"{system_prompt}\n\n{context_block}\n\n{user_block}"


def call_llm(prompt: str) -> str:
    logger.debug("LLM prompt preview:\n%s", prompt[:1000])
    return (
        "현재는 예시 답변입니다. 실제 운영 환경에서는 LLM API를 호출해 상황에 맞는 안내를 제공합니다."
    )


def rag_answer(question: str, k: int = TOP_K) -> Tuple[str, List[str]]:
    if VECTOR_STORE is None or not CHUNK_LOOKUP:
        raise RuntimeError("Knowledge base is not ready. Please rebuild the index.")
    query_vec = get_embedding(question)
    search_results = VECTOR_STORE.search(query_vec, k)
    if not search_results:
        return (
            "관련 안내 문서를 찾지 못했습니다. 담당 직원에게 문의하시거나 고객센터로 연락해 주세요.",
            [],
        )
    selected_chunks = [CHUNK_LOOKUP[chunk_id] for chunk_id, _ in search_results]
    context_texts = [chunk.text for chunk in selected_chunks]
    prompt = build_prompt(context_texts, question)
    answer = call_llm(prompt)
    references = [chunk.document_id for chunk in selected_chunks]
    return answer, references


def get_patient_reservations(patient_id: str) -> List[Dict[str, Any]]:
    try:
        with get_db_session() as session:
            stmt = (
                select(Reservation)
                .where(Reservation.patient_id == patient_id)
                .order_by(Reservation.scheduled_at.asc())
            )
            rows = session.scalars(stmt).all()
            return [
                {
                    "department": row.department,
                    "scheduled_at": row.scheduled_at.isoformat(),
                    "status": row.status,
                }
                for row in rows
            ]
    except (OperationalError, SQLAlchemyError) as exc:
        logger.error("Reservation query failed: %s", exc)
        return []


def get_today_queue_status(patient_id: str) -> List[Dict[str, Any]]:
    try:
        with get_db_session() as session:
            stmt = (
                select(Queue)
                .where(Queue.patient_id == patient_id)
                .order_by(Queue.created_at.desc())
            )
            rows = session.scalars(stmt).all()
            return [
                {
                    "department": row.department,
                    "ticket_number": row.ticket_number,
                    "status": row.status,
                    "issued_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
    except (OperationalError, SQLAlchemyError) as exc:
        logger.error("Queue query failed: %s", exc)
        return []


def get_latest_exam_results(patient_id: str) -> List[Dict[str, Any]]:
    try:
        with get_db_session() as session:
            stmt = (
                select(ExamResult)
                .where(ExamResult.patient_id == patient_id)
                .order_by(ExamResult.taken_at.desc())
            )
            rows = session.scalars(stmt).all()
            return [
                {
                    "exam_name": row.name,
                    "taken_at": row.taken_at.isoformat(),
                    "status": row.status,
                    "summary": row.summary or "",
                }
                for row in rows
            ]
    except (OperationalError, SQLAlchemyError) as exc:
        logger.error("Exam result query failed: %s", exc)
        return []


def db_answer(user_id: str, question: str) -> Tuple[str, List[str], Dict[str, Any]]:
    lower_q = question.lower()
    payload: Dict[str, Any] = {}

    if any(keyword in lower_q for keyword in ("예약", "스케줄", "진료 시간")):
        reservations = get_patient_reservations(user_id)
        payload["reservations"] = reservations
        if reservations:
            lines = [
                f"- {item['department']} 진료, 시간: {item['scheduled_at']}, 상태: {item['status']}"
                for item in reservations
            ]
            answer = "예약 정보를 안내드립니다:\n" + "\n".join(lines)
        else:
            answer = (
                "조회된 예약 내역이 없습니다. 자세한 확인은 병원 안내데스크나 콜센터로 문의해 주세요."
            )
        return answer, ["reservations"], payload

    if any(keyword in lower_q for keyword in ("대기", "번호표", "순번", "큐")):
        queues = get_today_queue_status(user_id)
        payload["queues"] = queues
        if queues:
            latest = queues[0]
            answer = (
                f"현재 {latest['department']}의 대기 번호는 {latest['ticket_number']}이며 상태는 '{latest['status']}'입니다."
                " 안내에 따라 진료를 준비해 주세요."
            )
        else:
            answer = "현재 접수된 대기 번호가 없습니다. 접수를 완료했는지 다시 확인해 주세요."
        return answer, ["queues"], payload

    if any(keyword in lower_q for keyword in ("검사", "결과", "리포트", "판독")):
        results = get_latest_exam_results(user_id)
        payload["exam_results"] = results
        if results:
            latest = results[0]
            answer = (
                f"{latest['exam_name']} 검사 결과는 상태 '{latest['status']}'이며 검사일은 {latest['taken_at']}입니다."
                " 자세한 내용은 담당 의료진과 상담해 주세요."
            )
        else:
            answer = "현재 조회 가능한 검사 결과가 없습니다. 검사 완료 여부를 병원에 문의해 주세요."
        return answer, ["exam_results"], payload

    answer = "개인 정보에 대한 질문으로 인식되지 않았습니다. 조금 더 구체적인 요청을 주시면 확인해 드리겠습니다."
    return answer, ["general"], payload


EMERGENCY_KEYWORDS = (
    "호흡곤란",
    "숨이",
    "가슴 통증",
    "의식",
    "피가 많이",
    "대량 출혈",
    "119",
    "응급",
    "쇼크",
    "쓰러졌",
    "경련",
    "심정지",
)


def is_emergency(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def route_query(message: str) -> Literal["rag", "db", "fallback"]:
    lowered = message.lower()
    if any(keyword in lowered for keyword in ("예약", "대기", "번호표", "나의", "내", "검사")):
        return "db"
    if any(keyword in lowered for keyword in ("안내", "위치", "병원", "주차", "보험", "앱", "사용법", "faq", "문의")):
        return "rag"
    return "rag" if message else "fallback"


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    if is_emergency(message):
        return ChatResponse(
            answer=(
                "응급 상황으로 보입니다. 즉시 119에 신고하시거나 가까운 응급실로 이동해 주세요. "
                "해당 상황에 대해서는 의료진의 직접적인 진료가 필요합니다."
            ),
            source="safety",
        )

    route = route_query(message)
    logger.info("Routing message '%s' to %s", message, route)

    if route == "db":
        answer, references, payload_data = db_answer(payload.user_id, message)
        return ChatResponse(answer=answer, source="db", references=references, payload=payload_data)

    if route == "rag":
        try:
            answer, references = rag_answer(message)
            return ChatResponse(answer=answer, source="rag", references=references)
        except RuntimeError as exc:
            logger.error("RAG pipeline error: %s", exc)
            fallback_answer = "현재 안내 시스템이 준비되지 않았습니다. 고객센터나 안내데스크로 문의해 주세요."
            return ChatResponse(answer=fallback_answer, source="fallback")

    fallback_message = "요청을 이해하지 못했습니다. 병원 이용과 관련된 질문을 다시 입력해 주세요."
    return ChatResponse(answer=fallback_message, source="fallback")


@app.on_event("startup")
def bootstrap() -> None:
    global DOCUMENTS, CHUNKS, CHUNK_LOOKUP, VECTOR_STORE
    DOCUMENTS = load_documents(KNOWLEDGE_BASE_DIR)
    CHUNKS = assemble_chunks(DOCUMENTS)
    CHUNK_LOOKUP = {chunk.chunk_id: chunk for chunk in CHUNKS}
    VECTOR_STORE = build_faiss_index(CHUNKS) if CHUNKS else None
    logger.info("Startup bootstrap completed. Documents=%d, Chunks=%d", len(DOCUMENTS), len(CHUNKS))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
