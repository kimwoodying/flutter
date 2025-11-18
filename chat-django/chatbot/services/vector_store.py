from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from chatbot.config import get_settings

settings = get_settings()

class VectorStore:
    def __init__(self, index_path: Path, metadata_path: Path):
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS 인덱스 또는 메타데이터 파일이 존재하지 않습니다. 먼저 ingest 스크립트를 실행하세요."
            )

        # FAISS 인덱스 로드
        self._index = faiss.read_index(str(index_path))

        # 메타데이터 로드
        with metadata_path.open(encoding="utf-8") as f:
            raw_meta = json.load(f)

        # list / dict 어떤 형식이든 dict[id]로 통일
        if isinstance(raw_meta, list):
            self._metadata = {int(item["id"]): item for item in raw_meta}
        elif isinstance(raw_meta, dict):
            self._metadata = {int(key): value for key, value in raw_meta.items()}
        else:
            raise TypeError(f"지원하지 않는 metadata 형식입니다: {type(raw_meta)}")

    # 🔥 여기부터 새로 추가 🔥
    def search(self, query_vector, top_k: int = 5):
        """
        쿼리 벡터(query_vector)에 가장 가까운 top_k개의 문서를 FAISS에서 검색해
        (score, meta) 튜플 리스트로 반환.
        rag.py에서 (score, meta)로 언패킹해서 사용함.
        """

        q = np.array(query_vector, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        distances, indices = self._index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            meta = self._metadata.get(int(idx))
            if not meta:
                continue

            # id를 메타에 포함시켜두면 나중에 디버깅할 때 편함
            meta_with_id = {
                "id": int(idx),
                **meta,
            }

            # 🔥 rag.py가 기대하는 형태: (score, meta)
            results.append((float(dist), meta_with_id))

        return results
    
    
@lru_cache(maxsize=1)
def get_vector_store():
    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.metadata_path)
    return VectorStore(index_path, metadata_path)
