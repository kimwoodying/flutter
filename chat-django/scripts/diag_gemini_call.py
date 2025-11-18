import os
from chatbot.services.gemini_client import call_gemini_with_rag


def main():
    query = "진료시간"
    docs = [
        {"page_content": "병원 진료시간은 평일 09:00-17:30 입니다."}
    ]
    try:
        resp = call_gemini_with_rag(query, docs)
        print("Gemini response:\n", resp)
    except Exception as e:
        print("Gemini call failed:\n", str(e))


if __name__ == '__main__':
    # Ensure DUMMY key is used if not set
    os.environ.setdefault('GEMINI_API_KEY', 'DUMMY')
    main()
