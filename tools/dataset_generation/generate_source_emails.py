"""
generate_source_emails.py

Генерация новых исходных электронных писем
для расширения датасета.

Результат:
    generated_dataset/source_emails_new.jsonl

Особенности:
- использует DeepCode API;
- использует DeepSeek-V4-Flash;
- генерирует русские и английские письма;
- балансирует домены;
- балансирует plain_text / html;
- проверяет дубликаты;
- сохраняет результат в JSONL;
- можно безопасно запускать повторно;
- корректно обрабатывает content=None;
- учитывает DeepSeek reasoning;
- не зависает надолго на одном запросе.
"""

import json
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from dataset_generation.config import (
    DEEPCODE_CHAT_ENDPOINT,
    MODELS,
    NEW_SOURCE_EMAILS_TARGET,
    SOURCE_EMAILS_FILE,
    REQUEST_TIMEOUT,
    RANDOM_SEED,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_RETRIES,
)


# ============================================================
# SETTINGS
# ============================================================

random.seed(RANDOM_SEED)

load_dotenv()

DEEPCODE_API_KEY = os.getenv("DEEPCODE_API_KEY")

if not DEEPCODE_API_KEY:
    raise RuntimeError(
        "DEEPCODE_API_KEY не найден.\n"
        "Проверь файл .env в корне проекта."
    )


MODEL = MODELS[0]


# ============================================================
# DOMAINS
# ============================================================

DOMAIN_DESCRIPTIONS = {

    "corporate": (
        "деловая переписка между коллегами, "
        "совещания, задачи, проекты, документы, "
        "рабочие напоминания"
    ),

    "personal": (
        "личная переписка между знакомыми или друзьями, "
        "встречи, планы, бытовые вопросы"
    ),

    "advertising": (
        "рекламное письмо с предложением товара или услуги, "
        "скидка, акция, специальное предложение"
    ),

    "spam": (
        "подозрительное или нежелательное письмо, "
        "массовая рассылка, сомнительное предложение, "
        "просьба перейти по ссылке"
    ),

    "graymail": (
        "легитимная массовая рассылка, newsletter, "
        "новости, подборки материалов, уведомления "
        "о подписке"
    ),

    "transactional": (
        "транзакционное письмо: заказ, оплата, "
        "подтверждение операции, статус заказа"
    ),

    "notifications": (
        "автоматическое уведомление сервиса "
        "о событии или изменении состояния аккаунта"
    ),

    "marketing": (
        "маркетинговая коммуникация бренда, "
        "новая коллекция, акция, предложение клиенту"
    ),

    "support": (
        "письмо службы поддержки, обращение пользователя, "
        "ответ специалиста, статус обращения"
    ),

    "finance": (
        "финансовая тематика: платежи, банковские операции, "
        "счёт, перевод, финансовое уведомление"
    ),

    "education": (
        "образовательная тематика: университет, курс, "
        "задание, экзамен, учебный портал"
    ),

    "shopping": (
        "покупки в интернет-магазине: заказ, товар, "
        "доставка, возврат, корзина"
    ),

    "delivery": (
        "доставка: отправление, курьер, дата доставки, "
        "номер отслеживания"
    ),
}


# ============================================================
# PROMPT VARIANTS
# ============================================================

PROMPT_VARIANTS = [

    "Создай короткое естественное письмо.",

    "Создай реалистичное письмо средней длины.",

    "Создай письмо, похожее на настоящее пользовательское письмо.",

    "Создай письмо с несколькими независимыми смысловыми частями.",

    "Создай письмо, которое выглядит как обычная переписка реального пользователя.",

    "Создай реалистичное письмо с конкретными деталями, но без лишней информации.",

    "Создай письмо, которое могло бы встретиться в реальной электронной почте.",

    "Создай естественное письмо примерно из 3–6 предложений.",

]


# ============================================================
# HTML REQUIREMENTS
# ============================================================

HTML_REQUIREMENTS = [

    (
        "Используй простой HTML с тегами "
        "<html>, <body>, <p>."
    ),

    (
        "Используй HTML с заголовком <h2> "
        "и несколькими абзацами <p>."
    ),

    (
        "Используй HTML-структуру с <table>, "
        "<tr>, <td> и текстовыми элементами."
    ),

    (
        "Используй реалистичную HTML-разметку "
        "электронного письма."
    ),

]


# ============================================================
# SOURCE PROMPT
# ============================================================

def build_source_prompt(
    language: str,
    domain: str,
    email_format: str,
) -> str:

    domain_description = DOMAIN_DESCRIPTIONS[domain]

    variant = random.choice(PROMPT_VARIANTS)

    if language == "ru":

        language_instruction = (
            "Письмо должно быть полностью написано "
            "на русском языке."
        )

    else:

        language_instruction = (
            "The email must be written entirely "
            "in English."
        )

    if email_format == "html":

        html_instruction = random.choice(
            HTML_REQUIREMENTS
        )

        format_instruction = f"""
Формат:

Письмо должно содержать HTML-разметку.

{html_instruction}

Верни полноценный HTML-текст письма.

Не используй Markdown.
"""

    else:

        format_instruction = """
Формат:

Обычный plain text.

Не используй HTML-разметку.
Не используй Markdown.
"""

    # --------------------------------------------------------
    # ВАЖНО:
    # Prompt теперь короче.
    #
    # Это уменьшает вероятность того, что модель
    # потратит слишком много токенов на reasoning.
    # --------------------------------------------------------

    if language == "ru":

        prompt = f"""
Создай ОДНО реалистичное электронное письмо.

Домен:
{domain}

Тематика:
{domain_description}

{language_instruction}

{variant}

{format_instruction}

Правила:

- создай только одно письмо;
- верни только письмо;
- не объясняй ответ;
- не упоминай AI, ChatGPT, DeepSeek или генерацию;
- не используй placeholders вроде [Имя], [Дата], [Компания];
- используй вымышленные, но реалистичные данные;
- письмо должно быть логически согласованным;
- не делай письмо слишком длинным;
- не используй одинаковую структуру во всех письмах;
- не начинай каждый текст одинаково;
- добавь конкретные детали, которые можно будет
  использовать для дальнейшего выделения фрагментов.

Сгенерируй письмо сейчас.
"""

    else:

        prompt = f"""
Create ONE realistic email.

Domain:
{domain}

Topic:
{domain_description}

{language_instruction}

{variant}

{format_instruction}

Rules:

- create exactly one email;
- return only the email;
- do not explain the answer;
- do not mention AI, ChatGPT, DeepSeek, or generation;
- do not use placeholders such as [Name], [Date], [Company];
- use fictional but realistic details;
- keep the email logically consistent;
- do not make the email too long;
- vary the structure;
- do not always start the email the same way;
- include concrete details useful for later
  fragment extraction.

Generate the email now.
"""

    return prompt.strip()


# ============================================================
# DEBUG RESPONSE
# ============================================================

def print_api_debug(data: dict) -> None:

    print()
    print("    ---------- API DEBUG ----------")

    try:

        debug_text = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )

        # Не печатаем потенциально огромный ответ.
        if len(debug_text) > 8000:

            debug_text = (
                debug_text[:8000]
                + "\n... DEBUG OUTPUT TRUNCATED ..."
            )

        print(debug_text)

    except Exception:

        print(
            "    Не удалось вывести API response."
        )

    print(
        "    -------- END API DEBUG --------"
    )
    print()


# ============================================================
# API REQUEST
# ============================================================

def generate_email(
    language: str,
    domain: str,
    email_format: str,
) -> str:

    prompt = build_source_prompt(
        language=language,
        domain=domain,
        email_format=email_format,
    )

    payload = {
        "model": MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "Генерируй только запрошенный "
                    "текст электронного письма. "
                    "Не объясняй ответ."
                ),
            },

            {
                "role": "user",
                "content": prompt,
            },
        ],

        "temperature": TEMPERATURE,

        # Увеличили запас из-за reasoning.
        "max_tokens": MAX_TOKENS,
    }

    headers = {
        "Authorization": (
            f"Bearer {DEEPCODE_API_KEY}"
        ),

        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.post(
                DEEPCODE_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            choices = data.get("choices")

            if not choices:

                raise RuntimeError(
                    "API не вернул choices."
                )

            message = choices[0].get(
                "message"
            )

            if not message:

                raise RuntimeError(
                    "API не вернул message."
                )

            content = message.get(
                "content"
            )

            # ------------------------------------------------
            # КЛЮЧЕВАЯ ПРОВЕРКА
            # ------------------------------------------------

            if content is None:

                finish_reason = (
                    choices[0].get(
                        "finish_reason"
                    )
                )

                reasoning = (
                    message.get(
                        "reasoning"
                    )
                )

                print(
                    "    API вернул "
                    "content=None."
                )

                print(
                    f"    finish_reason: "
                    f"{finish_reason}"
                )

                if reasoning:

                    print(
                        "    Причина: модель "
                        "потратила completion "
                        "на reasoning."
                    )

                print_api_debug(data)

                # Если это последняя попытка —
                # выбрасываем ошибку.

                if attempt == MAX_RETRIES:

                    raise RuntimeError(
                        "API вернул content=None."
                    )

                print(
                    "    Повторяем один раз..."
                )

                time.sleep(1)

                continue

            content = content.strip()

            if not content:

                raise RuntimeError(
                    "API вернул пустую строку."
                )

            return content

        except requests.Timeout as error:

            last_error = error

            print(
                f"    Timeout "
                f"(попытка {attempt}/"
                f"{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                time.sleep(1)

        except requests.RequestException as error:

            last_error = error

            print(
                f"    Ошибка HTTP "
                f"(попытка {attempt}/"
                f"{MAX_RETRIES}): "
                f"{error}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(1)

        except Exception as error:

            last_error = error

            print(
                f"    Ошибка API "
                f"(попытка {attempt}/"
                f"{MAX_RETRIES}): "
                f"{error}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(1)

    raise RuntimeError(
        "Не удалось получить письмо: "
        f"{last_error}"
    )


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_generated_email(
    text: str,
    email_format: str,
) -> str:

    text = text.strip()

    # Удаляем Markdown fences.

    text = re.sub(
        r"^```(?:html)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    prefixes = [
        "Вот письмо:",
        "Вот пример письма:",
        "Готовое письмо:",
        "Here is the email:",
        "Here is an example email:",
        "Generated email:",
    ]

    for prefix in prefixes:

        if text.lower().startswith(
            prefix.lower()
        ):

            text = text[
                len(prefix):
            ].strip()

    return text


# ============================================================
# VALIDATION
# ============================================================

def validate_email(
    text: str,
    language: str,
    email_format: str,
) -> tuple[bool, str]:

    if not text:

        return (
            False,
            "пустой текст",
        )

    plain = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    plain = re.sub(
        r"\s+",
        " ",
        plain,
    ).strip()

    # --------------------------------------------------------
    # Длина
    # --------------------------------------------------------

    if len(plain) < 50:

        return (
            False,
            "слишком короткое письмо",
        )

    if len(plain) > 5000:

        return (
            False,
            "слишком длинное письмо",
        )

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    if email_format == "html":

        lower = text.lower()

        if (
            "<html" not in lower
            and "<body" not in lower
            and "<p" not in lower
        ):

            return (
                False,
                "нет HTML-разметки",
            )

    else:

        if "<html" in text.lower():

            return (
                False,
                "plain_text содержит HTML",
            )

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    cyrillic = len(
        re.findall(
            r"[А-Яа-яЁё]",
            plain,
        )
    )

    latin = len(
        re.findall(
            r"[A-Za-z]",
            plain,
        )
    )

    if language == "ru":

        if cyrillic < 10:

            return (
                False,
                "слишком мало кириллицы",
            )

    else:

        if latin < 20:

            return (
                False,
                "слишком мало латиницы",
            )

    return (
        True,
        "ok",
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_for_duplicate(
    text: str,
) -> str:

    text = text.lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE,
    )

    return text.strip()


# ============================================================
# EXISTING DATA
# ============================================================

def load_existing_emails() -> set[str]:

    existing = set()

    paths = [

        Path(
            SOURCE_EMAILS_FILE
        ),

        Path(
            "generated_dataset/"
            "dataset_deepseek.jsonl"
        ),
    ]

    for path in paths:

        if not path.exists():

            continue

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:

                        continue

                    try:

                        item = json.loads(
                            line
                        )

                        text = item.get(
                            "original_email",
                            "",
                        )

                        if text:

                            existing.add(
                                normalize_for_duplicate(
                                    text
                                )
                            )

                    except json.JSONDecodeError:

                        continue

        except OSError as error:

            print(
                f"Предупреждение: "
                f"не удалось прочитать "
                f"{path}: {error}"
            )

    return existing


# ============================================================
# GENERATION PLAN
# ============================================================

def build_generation_plan(
    total: int,
) -> list[tuple[str, str, str]]:

    combinations = []

    for language in [
        "ru",
        "en",
    ]:

        for domain in DOMAIN_DESCRIPTIONS:

            for email_format in [
                "plain_text",
                "html",
            ]:

                combinations.append(
                    (
                        language,
                        domain,
                        email_format,
                    )
                )

    random.shuffle(
        combinations
    )

    plan = []

    for index in range(total):

        plan.append(
            combinations[
                index % len(combinations)
            ]
        )

    random.shuffle(plan)

    return plan


# ============================================================
# SAVE
# ============================================================

def save_email(
    record: dict,
) -> None:

    path = Path(
        SOURCE_EMAILS_FILE
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GENERATE NEW SOURCE EMAILS"
    )
    print("=" * 70)

    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Target new emails: "
        f"{NEW_SOURCE_EMAILS_TARGET}"
    )

    print(
        f"Output: "
        f"{SOURCE_EMAILS_FILE}"
    )

    print(
        f"Max tokens: "
        f"{MAX_TOKENS}"
    )

    print(
        f"Timeout: "
        f"{REQUEST_TIMEOUT}s"
    )

    print("=" * 70)
    print()

    existing = load_existing_emails()

    print(
        f"Existing unique emails: "
        f"{len(existing)}"
    )

    print()

    plan = build_generation_plan(
        NEW_SOURCE_EMAILS_TARGET
    )

    generated = 0
    attempts = 0

    # --------------------------------------------------------
    # Защита от бесконечного цикла.
    # --------------------------------------------------------

    max_total_attempts = (
        NEW_SOURCE_EMAILS_TARGET * 5
    )

    stats = {
        "ru": 0,
        "en": 0,
        "plain_text": 0,
        "html": 0,
    }

    domain_stats = {
        domain: 0
        for domain in DOMAIN_DESCRIPTIONS
    }

    print(
        "Starting generation..."
    )

    print()

    while (
        generated
        < NEW_SOURCE_EMAILS_TARGET
    ):

        if attempts >= max_total_attempts:

            raise RuntimeError(
                "Слишком много неудачных "
                "попыток. "
                "Генерация остановлена."
            )

        language, domain, email_format = (
            plan[
                generated
                % len(plan)
            ]
        )

        attempts += 1

        print(
            f"[{generated + 1}/"
            f"{NEW_SOURCE_EMAILS_TARGET}] "
            f"{language} | "
            f"{domain} | "
            f"{email_format}"
        )

        try:

            text = generate_email(
                language=language,
                domain=domain,
                email_format=email_format,
            )

            text = clean_generated_email(
                text=text,
                email_format=email_format,
            )

        except Exception as error:

            print(
                f"    ✗ ошибка генерации: "
                f"{error}"
            )

            print()

            # ВАЖНО:
            # generated не увеличиваем.
            # Следующая попытка получает
            # тот же target.

            continue

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        valid, reason = validate_email(
            text=text,
            language=language,
            email_format=email_format,
        )

        if not valid:

            print(
                f"    ✗ rejected: "
                f"{reason}"
            )

            print()

            continue

        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        normalized = (
            normalize_for_duplicate(
                text
            )
        )

        if normalized in existing:

            print(
                "    ✗ duplicate"
            )

            print()

            continue

        existing.add(
            normalized
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        record = {

            "id": (
                f"source_new_"
                f"{generated + 1:05d}"
            ),

            "original_email": text,

            "language": language,

            "domain": domain,

            "email_format": email_format,

            "source": "deepcode",

            "model": MODEL,

            "generated_by": MODEL,
        }

        save_email(
            record
        )

        generated += 1

        stats[language] += 1

        stats[email_format] += 1

        domain_stats[domain] += 1

        print(
            "    ✓ saved"
        )

        print()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print(
        "GENERATION COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        f"Generated: "
        f"{generated}"
    )

    print(
        f"Attempts:  "
        f"{attempts}"
    )

    print()

    print(
        "Languages:"
    )

    print(
        f"  ru: "
        f"{stats['ru']}"
    )

    print(
        f"  en: "
        f"{stats['en']}"
    )

    print()

    print(
        "Formats:"
    )

    print(
        f"  plain_text: "
        f"{stats['plain_text']}"
    )

    print(
        f"  html: "
        f"{stats['html']}"
    )

    print()

    print(
        "Domains:"
    )

    for domain, count in sorted(
        domain_stats.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        print(
            f"  {domain}: "
            f"{count}"
        )

    print()

    print("=" * 70)

    print(
        "Saved to:"
    )

    print(
        f"  {SOURCE_EMAILS_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()