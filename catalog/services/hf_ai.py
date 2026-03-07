import json
from dataclasses import dataclass
from typing import List
import os
from django.conf import settings
from huggingface_hub import InferenceClient


class HFAIServiceError(Exception):
    pass


@dataclass
class CandidateBook:
    book_id: int
    title: str
    author: str
    categories: str
    price: str
    description: str


class HFAIService:
    def __init__(self) -> None:
        token = os.getenv("HF_TOKEN", "")
        model = os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")


        if not token or not model:
            raise HFAIServiceError("Не настроены HF_TOKEN или HF_CHAT_MODEL")

        self.client = InferenceClient(token=token)
        self.model = model

    def recommend(self, query: str, candidates: List[CandidateBook]) -> dict:
        if not candidates:
            raise HFAIServiceError("Не переданы кандидаты книг")

        books_block = "\n".join(
            [
                (
                    f"ID: {b.book_id}; title: {b.title}; author: {b.author}; "
                    f"categories: {b.categories}; price: {b.price}; description: {b.description}"
                )
                for b in candidates
            ]
        )

        prompt = (
            "Ты AI-консультант книжного магазина. "
            "На основе запроса пользователя выбери до 5 самых подходящих книг. "
            "Ответ верни только в JSON без markdown. "
            "Формат: {\"answer\": string, \"book_ids\": [int,...]}.\n\n"
            f"Запрос пользователя: {query}\n\n"
            f"Кандидаты:\n{books_block}"
        )

        try:
            completion = self.client.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Отвечай строго валидным JSON-объектом.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.2,
            )
        except Exception as exc:
            detail = str(exc).strip() or repr(exc).strip() or "без текста ошибки"
            raise HFAIServiceError(
                f"Ошибка запроса к Hugging Face ({exc.__class__.__name__}): {detail}"
            ) from exc

        content = completion.choices[0].message.content if completion.choices else ""
        if not content:
            raise HFAIServiceError("Hugging Face вернул пустой ответ")

        parsed = self._extract_json(content)
        if "answer" not in parsed or "book_ids" not in parsed:
            raise HFAIServiceError("В ответе Hugging Face нет обязательных полей answer/book_ids")

        parsed["book_ids"] = [int(i) for i in parsed.get("book_ids", []) if str(i).isdigit()]
        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise HFAIServiceError("Ответ Hugging Face не является JSON")
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise HFAIServiceError("Не удалось разобрать JSON из ответа Hugging Face") from exc