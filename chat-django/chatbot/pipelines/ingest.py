from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    doc_id: int
    title: str
    text: str
    snippet: str


def load_documents(source_dir: Path) -> List[Document]:
    """Load and process documents from the source directory."""
    docs: List[Document] = []
    doc_id = 0
    
    for path in sorted(source_dir.glob("*.txt")):
        try:
            if path.name == '.gitkeep':
                continue
                
            content = path.read_text(encoding="utf-8").strip()
            
            # Handle front matter if present
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    # Extract title from front matter if available
                    front_matter = parts[1].strip()
                    title = path.stem
                    # Try to get title from front matter
                    for line in front_matter.split('\n'):
                        if line.startswith('title:'):
                            title = line.split(':', 1)[1].strip()
                            break
                    # Use the part after the second '---' as the main content
                    text = parts[2].strip()
                else:
                    title = path.stem
                    text = content
            else:
                title = path.stem
                text = content
            
            # Clean up the text
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Create a snippet (first 200 chars of text, but try to end at a sentence)
            snippet = text[:200]
            if len(text) > 200:
                # Try to find the last sentence end within the snippet
                for punct in ('.', '!', '?', '다.'):
                    last_punct = snippet.rfind(punct)
                    if last_punct > 100:  # Only if we have a reasonable amount of text
                        snippet = snippet[:last_punct + 1]
                        break
            
            docs.append(Document(
                doc_id=doc_id,
                title=title,
                text=text,
                snippet=snippet
            ))
            doc_id += 1
            
        except Exception as e:
            print(f"Error loading document {path}: {str(e)}")
            continue
            
    return docs


def split_into_chunks(docs: Iterable[Document], chunk_size: int = 500, overlap: int = 100) -> List[Document]:
    chunks: List[Document] = []
    chunk_id = 0
    for doc in docs:
        text = doc.text
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            snippet = chunk_text[:200].replace("\n", " ")
            chunks.append(
                Document(
                    doc_id=chunk_id,
                    title=doc.title,
                    text=chunk_text,
                    snippet=snippet,
                )
            )
            chunk_id += 1
            start += chunk_size - overlap
    return chunks


def build_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def save_metadata(chunks: List[Document], path: Path) -> None:
    payload = {
        str(chunk.doc_id): {
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "snippet": chunk.snippet,
        }
        for chunk in chunks
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(
    raw_dir: Path,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> None:
    docs = load_documents(raw_dir)
    if not docs:
        raise RuntimeError("raw_docs 폴더에 문서가 없습니다.")

    chunks = split_into_chunks(docs, chunk_size=chunk_size, overlap=overlap)
    texts = [chunk.text for chunk in chunks]
    embeddings = build_embeddings(texts, embedding_model)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    save_index(index, index_path)
    save_metadata(chunks, metadata_path)
    print(f"FAISS 인덱스 생성 완료: {len(chunks)}개 청크")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    index_path = base_dir / "data" / "faiss.index"
    metadata_path = base_dir / "data" / "metadata.json"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    ingest(
        raw_dir=raw_dir,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )
