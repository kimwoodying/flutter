from django.db import models


class ChatMessage(models.Model):
    session_id = models.CharField(
        "세션 ID",
        max_length=255,
        blank=True,
        help_text="사용자 구분용 세션 식별자",
    )
    user_question = models.TextField("사용자 질문")
    bot_answer = models.TextField("챗봇 응답")
    sources = models.JSONField(
        "참고 자료",
        blank=True,
        null=True,
        help_text="RAG 검색 결과(JSON)"
    )
    metadata = models.JSONField(
        "요청 메타데이터",
        blank=True,
        null=True,
        help_text="사용자 장치 또는 기타 부가 정보"
    )
    created_at = models.DateTimeField("생성 시각", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "채팅 내역"
        verbose_name_plural = "채팅 내역"

    def __str__(self) -> str:
        return f"{self.session_id or 'anonymous'}: {self.user_question[:30]}"
class ChatCache(models.Model):
    """
    질문 + 컨텍스트 조합에 대한 LLM 응답 캐시.
    같은 질문/컨텍스트 조합이면 DB에서 바로 꺼내서 LLM 호출을 생략.
    """

    query_hash = models.CharField(max_length=64, unique=True)  # SHA-256 해시
    query = models.TextField()                                # 원본 질문
    context = models.TextField(blank=True)                    # 사용된 컨텍스트 텍스트
    response = models.TextField()                             # LLM 생성 응답
    hit_count = models.PositiveIntegerField(default=1)        # 몇 번이나 재사용됐는지
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chatbot_cache"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.query[:30]}... ({self.hit_count} hits)"