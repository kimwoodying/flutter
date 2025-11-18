from types import SimpleNamespace

from chatbot.services.gemini_client import format_context


def build_parts_from_docs(retrieved_docs):
    parts = []
    for d in retrieved_docs:
        if isinstance(d, str):
            parts.append(d)
            continue
        if isinstance(d, dict):
            text = d.get("page_content") or d.get("text") or d.get("snippet")
            if text:
                parts.append(text)
            continue
        text = getattr(d, "page_content", None) or getattr(d, "text", None) or getattr(d, "snippet", None)
        if text:
            parts.append(text)
    return parts


def main():
    docs = [
        {"page_content": "참고자료: 이 병원은 24시간 응급실을 운영합니다.\n출처: 내부 문서"},
        {"text": "## 진료과 목록\n- 내과\n- 외과"},
        SimpleNamespace(page_content="참고자료: 주차장은 지하 2층에 있습니다."),
        "추가 메모: 예약은 전화로 가능합니다."
    ]

    parts = build_parts_from_docs(docs)
    joined = "\n\n".join(parts)
    cleaned = format_context(joined)

    print("=== 원문 합친 내용 ===")
    print(joined)
    print()
    print("=== format_context 결과 ===")
    print(cleaned)


if __name__ == "__main__":
    main()
