from django.contrib import admin
from .models import ChatMessage

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session_id", "user_question", "bot_answer", "created_at")
    list_filter = ("created_at",)
    search_fields = ("session_id", "user_question", "bot_answer")
