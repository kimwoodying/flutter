import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from chatbot.models import ChatMessage
from chatbot.services.rag import run_rag_with_cache


@csrf_exempt
@require_POST
def chat_view(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)

    message = payload.get("message")
    if not message:
        return JsonResponse({"error": "message 필드가 필요합니다."}, status=400)

    session_id = payload.get("session_id", "")
    metadata = payload.get("metadata")

    try:
        result = run_rag_with_cache(message)
    except FileNotFoundError as exc:
        return JsonResponse(
            {
                "error": "지식 베이스가 준비되지 않았습니다. 먼저 문서를 색인화해주세요.",
                "detail": str(exc),
            },
            status=503,
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"error": f"RAG 파이프라인 오류: {exc}"}, status=500)

    ChatMessage.objects.create(
        session_id=session_id,
        user_question=message,
        bot_answer=result.get("reply", ""),
        sources=result.get("sources"),
        metadata=metadata,
    )

    return JsonResponse(result, status=200)