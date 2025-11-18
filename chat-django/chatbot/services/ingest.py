# chatbot/services/ingest.py
from __future__ import annotations

import json
from pathlib import Path

import faiss

from chatbot.config import get_settings
from chatbot.services.embeddings import embed_texts  # ✅ 공용 임베딩 함수 사용


def chunk_text(text: str, max_len: int = 400, overlap: int = 50) -> list[str]:
    """
    긴 텍스트를 RAG용으로 적당한 길이로 잘라주는 함수.
    너무 짧으면 검색 성능이 떨어지고, 너무 길면 토큰 낭비라서 300~500자 정도 추천.
    """
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        start = end - overlap  # 일정 부분 겹치게

    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    settings = get_settings()

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw 폴더가 없습니다: {raw_dir.resolve()}")

    texts: list[str] = []
    metadata: list[dict] = []

    print(f"📂 TXT 로딩 중... ({raw_dir.resolve()})")
    for txt_file in raw_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")

        # 지금처럼 하나의 파일 안에 --- 블록 여러 개가 있어도
        # 전체를 통으로 chunk_text에 넘기면 알아서 잘려서 들어감
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append(
                {
                    "id": len(metadata),
                    "source": txt_file.name,
                    "chunk": i,
                    "text": chunk,
                }
            )

    if not texts:
        raise ValueError("raw 폴더에 .txt 내용이 없습니다. 먼저 병원 안내 txt를 넣어주세요.")

    print(f"🧠 임베딩 생성 중... (총 {len(texts)}개 chunk)")
    # ✅ 공용 임베딩 함수 사용 (SentenceTransformer 직접 생성 X)
    vectors = embed_texts(texts)

    import numpy as np
    vectors = np.asarray(vectors, dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ FAISS 인덱스 저장 완료:", index_path.resolve())
    print("✅ 메타데이터 저장 완료:", metadata_path.resolve())


if __name__ == "__main__":
    main()
