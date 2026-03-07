from django.db.models import Q
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Book
from .serializers import BookSerializer
from .services.hf_ai import CandidateBook, HFAIService, HFAIServiceError


class AIRecommendationsAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        query = (request.GET.get("q") or "").strip()
        if not query:
            return Response(
                {"error": 'Необходимо передать параметр запроса "q"'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        candidates_qs = Book.objects.select_related("author").prefetch_related("categories").filter(
            Q(title__icontains=query)
            | Q(author__first_name__icontains=query)
            | Q(author__last_name__icontains=query)
            | Q(description__icontains=query)
        )[:20]

        if not candidates_qs:
            candidates_qs = Book.objects.select_related("author").prefetch_related("categories").order_by("-created_at")[:20]

        candidates = [
            CandidateBook(
                book_id=book.id,
                title=book.title,
                author=str(book.author) if book.author else "Неизвестный автор",
                categories=", ".join(book.categories.values_list("name", flat=True)),
                price=str(book.price),
                description=(book.description or "")[:280],
            )
            for book in candidates_qs
        ]

        try:
            service = HFAIService()
            ai_result = service.recommend(query=query, candidates=candidates)

            ids_from_ai = ai_result.get("book_ids", [])
            selected_books = [book for book in candidates_qs if book.id in ids_from_ai][:5]
            if not selected_books:
                selected_books = list(candidates_qs[:5])

            return Response(
                {
                    "query": query,
                    "answer": ai_result.get("answer", ""),
                    "results": BookSerializer(selected_books, many=True).data,
                    "count": len(selected_books),
                    "fallback_used": False,
                    "fallback_reason": "",
                }
            )

        except HFAIServiceError as exc:
            fallback_books = list(candidates_qs[:5])
            return Response(
                {
                    "query": query,
                    "answer": "AI-сервис временно недоступен. Показаны базовые результаты поиска.",
                    "results": BookSerializer(fallback_books, many=True).data,
                    "count": len(fallback_books),
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                }
            )