# генератор текста через DeepCode API
# модель: deepseek-ai/DeepSeek-V4-Flash

import os
import time

import requests
from dotenv import load_dotenv

from dataset_generation.config import (
    DEEPCODE_CHAT_ENDPOINT,
    MAX_TOKENS,
    TEMPERATURE,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)


load_dotenv()

DEEPCODE_API_KEY = os.getenv("DEEPCODE_API_KEY")

if not DEEPCODE_API_KEY:
    raise RuntimeError(
        "DEEPCODE_API_KEY не найден.\n"
        "Проверь файл .env в корне проекта."
    )


class DeepCodeGenerator:
    # поддерживает прямой prompt и контролируемое преобразование текста

    def __init__(self, model: str):
        self.model = model

    def generate(
        self,
        text: str | None = None,
        transformation: str | None = None,
        tone: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
    ) -> str:

        selected_model = (
            model
            if model is not None
            else self.model
        )

        if prompt is not None:

            system_prompt = (
                "Ты выполняешь задачу точно по инструкции пользователя. "
                "Отвечай только результатом задачи без дополнительных "
                "объяснений."
            )

            user_prompt = prompt

        else:

            if text is None:
                raise ValueError(
                    "Необходимо передать text или prompt."
                )

            if transformation is None:
                raise ValueError(
                    "Необходимо передать transformation "
                    "при использовании text."
                )

            system_prompt = (
                "Ты выполняешь контролируемое преобразование "
                "текста. Возвращай только итоговый текст."
            )

            user_prompt = self._build_prompt(
                text=text,
                transformation=transformation,
                tone=tone,
            )

        payload = {
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        headers = {
            "Authorization": f"Bearer {DEEPCODE_API_KEY}",
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                response = requests.post(
                    DEEPCODE_CHAT_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )

                if not response.ok:

                    error_text = response.text

                    raise RuntimeError(
                        "DeepCode API вернул HTTP "
                        f"{response.status_code}: "
                        f"{error_text}"
                    )

                try:
                    data = response.json()

                except ValueError as error:

                    raise RuntimeError(
                        "DeepCode вернул некорректный JSON:\n"
                        f"{response.text[:2000]}"
                    ) from error

                choices = data.get("choices")

                if not choices:

                    raise RuntimeError(
                        "DeepCode не вернул choices.\n"
                        f"Ответ API: {data}"
                    )

                choice = choices[0]

                message = choice.get("message")

                if not message:

                    raise RuntimeError(
                        "DeepCode не вернул message.\n"
                        f"Ответ API: {data}"
                    )

                content = message.get("content")

                reasoning = message.get("reasoning")

                finish_reason = choice.get(
                    "finish_reason"
                )

                if isinstance(content, str) and content.strip():

                    return content.strip()

                print()
                print(
                    "DeepCode вернул пустой content."
                )

                print(
                    f"finish_reason: {finish_reason}"
                )

                if reasoning:

                    print(
                        "Получен reasoning:"
                    )

                    print(
                        reasoning[:2000]
                    )

                if (
                    finish_reason == "length"
                    and reasoning
                ):

                    raise RuntimeError(
                        "DeepSeek потратил все "
                        "completion tokens на reasoning, "
                        "поэтому message.content остался пустым."
                    )

                raise RuntimeError(
                    "DeepCode вернул пустой "
                    "message.content."
                )

            except requests.exceptions.Timeout as error:

                last_error = error

                print()
                print(
                    f"Timeout DeepCode "
                    f"(попытка {attempt}/{MAX_RETRIES})"
                )

                if attempt < MAX_RETRIES:

                    print(
                        "Повторяем запрос..."
                    )

                    time.sleep(1)

            except requests.exceptions.ConnectionError as error:

                last_error = error

                print()
                print(
                    f"Ошибка соединения с DeepCode "
                    f"(попытка {attempt}/{MAX_RETRIES})"
                )

                print(error)

                if attempt < MAX_RETRIES:

                    print(
                        "Повторяем запрос..."
                    )

                    time.sleep(1)

            except Exception as error:

                last_error = error

                print()
                print(
                    f"Ошибка генерации "
                    f"(попытка {attempt}/{MAX_RETRIES}): "
                    f"{error}"
                )

                if attempt < MAX_RETRIES:

                    print(
                        "Повторяем запрос..."
                    )

                    time.sleep(1)

        raise RuntimeError(
            "Не удалось получить ответ от DeepCode "
            f"после {MAX_RETRIES} попыток. "
            f"Последняя ошибка: {last_error}"
        )

    def _build_prompt(
        self,
        text: str,
        transformation: str,
        tone: str | None = None,
    ) -> str:

        prompt = f"""
Ты выполняешь контролируемое преобразование
фрагмента электронного письма.

Тебе передан ОДИН конкретный фрагмент
существующего письма.

Ты НЕ видишь остальные части письма.

Твоя задача — изменить только этот фрагмент.

ЗАДАЧА:

Тип преобразования:
{transformation}
"""

        if tone:

            prompt += f"""
Дополнительная тональность:
{tone}
"""

        prompt += f"""

СТРОГИЕ ПРАВИЛА:

1. Работай ТОЛЬКО с текстом между
   BEGIN TEXT и END TEXT.

2. НЕ создавай новое письмо.

3. НЕ продолжай письмо.

4. НЕ добавляй приветствие, обращение или подпись,
   если их нет в исходном фрагменте.

5. НЕ добавляй информацию из своего опыта.

6. НЕ придумывай новые факты.

7. Сохраняй фактическое содержание.

8. Сохраняй даты.

9. Сохраняй числа.

10. Сохраняй имена.

11. Сохраняй названия организаций.

12. Сохраняй конкретные сведения,
    если выбранный тип преобразования
    явно не требует их удаления.

13. Не объясняй свои действия.

14. Не пиши комментарии.

15. Не используй кавычки вокруг результата.

16. Не пиши вступления вроде:
    "Вот изменённый текст"
    "Конечно"
    "Изменённый вариант"

17. Верни ТОЛЬКО изменённый фрагмент.

18. Не копируй исходный фрагмент полностью
    без изменений.

19. Результат должен соответствовать
    исходному фрагменту.

20. Не добавляй текст из других частей письма.

21. Если исходный фрагмент содержит HTML,
    сохрани HTML-разметку и измени только
    текстовое содержание внутри неё.

22. Не удаляй HTML-теги без необходимости.

23. Не изменяй URL, номера телефонов,
    даты, суммы, идентификаторы и другие
    конкретные значения без явного требования
    преобразования.

24. Для transformation="expand" действительно
    добавляй содержательные пояснения,
    но не добавляй новые факты.

25. Для transformation="shorten" действительно
    сокращай текст, сохраняя основные факты.

26. Для transformation="paraphrase" и
    transformation="rewrite" существенно
    переформулируй предложения.

27. Для transformation="simplify" упрощай
    формулировки и синтаксис, сохраняя смысл.

28. Для transformation="formalize" делай стиль
    более формальным.

29. Для transformation="casualize" делай стиль
    более неформальным.

30. Для transformation="friendly" делай тон
    более дружелюбным.

31. Для transformation="professional" делай
    формулировки более профессиональными.

32. Для transformation="neutral" и
    transformation="neutralize" убирай
    эмоциональность и субъективные формулировки.

33. Для transformation="tone_change" меняй
    тональность текста, но сохраняй факты.

34. Для transformation="grammar_improvement"
    исправляй грамматику и естественность языка,
    не меняя фактическое содержание.

35. Для transformation="rephrase_structure"
    меняй структуру и порядок формулировок,
    сохраняя смысл и факты.

ИСХОДНЫЙ ФРАГМЕНТ:

---BEGIN TEXT---

{text}

---END TEXT---

ПЕРЕД ОТВЕТОМ ПРОВЕРЬ:

- изменён ли исходный текст;
- выполнен ли именно тип преобразования;
- сохранён ли смысл;
- сохранены ли факты;
- сохранены ли даты;
- сохранены ли числа;
- сохранены ли имена;
- сохранены ли организации;
- не добавлена ли информация,
  которой не было в исходном фрагменте;
- не добавлены ли приветствие или подпись;
- не создано ли новое письмо;
- сохранена ли HTML-разметка, если она была;
- сохранены ли URL и другие конкретные значения.

Сейчас верни ТОЛЬКО итоговый
преобразованный фрагмент.
"""

        return prompt