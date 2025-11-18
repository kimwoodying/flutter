from chatbot.services.embeddings import embed_texts
from chatbot.services.vector_store import get_vector_store
from chatbot.services.rag import _build_context_block


def main():
    query = "응급실 운영 시간 알려줘"
    embeddings = embed_texts([query])
    if not embeddings:
        print("임베딩을 생성할 수 없습니다.")
        return

    import numpy as np

    query_vec = np.array(embeddings[0], dtype="float32")
    store = get_vector_store()
    results = store.search(query_vec, top_k=5)
    print(f"검색 결과 개수: {len(results)}")
    contexts = [meta for _, meta in results]
    for i, meta in enumerate(contexts, start=1):
        print(f"-- result {i} --")
        print(f"doc_id: {meta.get('doc_id')}")
        print(f"title: {meta.get('title')}")
        print(f"snippet: {meta.get('snippet')[:120]}")

    context_block = _build_context_block(contexts)
    print('\n=== context_block ===')
    print(context_block)


if __name__ == '__main__':
    main()
