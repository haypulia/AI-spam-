import json
import os
import re
import glob
from email import policy
from email.parser import BytesParser
from email.header import decode_header

import requests

from config import Config
from utils import ensure_dir, get_timestamp, truncate_text, clean_string, clean_json_data, save_json


class EmailAIAnalyzer:
    def __init__(self, api_key, base_url=Config.BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"Анализатор инициализирован. API-ключ: {api_key[:8]}...")

    def decode_email_header(self, header):
        if header is None:
            return ""

        if isinstance(header, str):
            try:
                for encoding in ['utf-8', 'latin-1', 'cp1251', 'koi8-r']:
                    try:
                        return header.encode('latin-1').decode(encoding)
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        continue
                return clean_string(header)
            except Exception:
                return clean_string(header)

        try:
            decoded_parts = []
            for part, encoding in decode_header(header):
                if isinstance(part, bytes):
                    try:
                        if encoding:
                            decoded_parts.append(part.decode(encoding, errors='ignore'))
                        else:
                            for enc in ['utf-8', 'latin-1', 'cp1251', 'koi8-r']:
                                try:
                                    decoded_parts.append(part.decode(enc))
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                decoded_parts.append(part.decode('utf-8', errors='ignore'))
                    except Exception:
                        decoded_parts.append(str(part))
                else:
                    decoded_parts.append(str(part))
            return clean_string(''.join(decoded_parts))
        except Exception:
            return clean_string(str(header))

    def load_email_from_file(self, file_path):
        try:
            with open(file_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)

            subject = self.decode_email_header(msg.get("Subject", "Без темы"))
            sender = self.decode_email_header(msg.get("From", "Неизвестный отправитель"))

            subject = clean_string(subject)
            sender = clean_string(sender)

            html_body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        try:
                            html_body = part.get_content()
                            break
                        except Exception:
                            continue
            else:
                if msg.get_content_type() == "text/html":
                    try:
                        html_body = msg.get_content()
                    except Exception:
                        html_body = ""

            if not html_body:
                try:
                    html_body = msg.get_body(preferencelist=("plain",)).get_content()
                except Exception:
                    html_body = ""

            if not html_body or len(html_body.strip()) < 10:
                print(f"Письмо {file_path} не содержит HTML")
                return None

            html_body = clean_string(html_body)

            print(f"Загружено письмо: {subject}")

            return {
                "subject": subject,
                "sender": sender,
                "html": html_body
            }

        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            return None

    def analyze_email(self, email_html, model=Config.DEFAULT_MODEL):
        email_html = clean_string(email_html)

        html_preview = email_html[:Config.MAX_HTML_LENGTH]
        if len(email_html) > Config.MAX_HTML_LENGTH:
            html_preview += "\n\n... (HTML обрезан для экономии места)"

        prompt = f"""
    Ты — эксперт по обнаружению AI-сгенерированных писем.

    Проанализируй HTML-код письма.

    Проверь:
    1. Структурные аномалии: глубокая вложенность таблиц, пустые ячейки, странные inline-стили.
    2. Комментарии: HTML-комментарии, скрытые подсказки, служебные метки.
    3. Шаблонные элементы: {{name}}, {{email}}, {{variable}}, повторяющиеся конструкции.

    HTML:
    {html_preview}

    Ответь ТОЛЬКО JSON.

    Формат:
    {{
        "ai_probability": 85,
        "suspicious_elements": [
            {{
                "type": "comment",
                "content": "<!-- Generated -->",
                "reason": "Служебный комментарий"
            }}
        ],
        "summary": "Краткое описание"
    }}

    Не добавляй текст до или после JSON.
    """

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Ты эксперт по безопасности."},
                {"role": "user", "content": prompt}
            ],
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P,
            "max_tokens": 1000
        }

        try:
            print(f"Отправка запроса к модели {model}...")

            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8', errors='ignore')

            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                timeout=Config.REQUEST_TIMEOUT
            )

            if response.status_code != 200:
                error_text = response.text[:200]
                print(f"Ошибка сервера: {error_text}")
                return {"error": f"HTTP {response.status_code}: {error_text}"}

            result = response.json()

            if "choices" not in result:
                return {"error": "Нет choices в ответе API"}

            content = result["choices"][0]["message"]["content"]
            content = clean_string(content)

            try:
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    print(f"AI вероятность: {parsed.get('ai_probability', '?')}%")
                    return parsed
            except Exception:
                pass

            return {"raw_response": content}

        except Exception as e:
            return {"error": clean_string(str(e))}

    def save_results(self, email_data, analysis_result):
        ensure_dir(Config.ANALYSIS_RESULTS_DIR)

        filename = (
            f"{Config.ANALYSIS_RESULTS_DIR}/"
            f"analysis_{get_timestamp()}.json"
        )

        result = {
            "timestamp": get_timestamp(),
            "email": {
                "subject": clean_string(email_data.get("subject")),
                "sender": clean_string(email_data.get("sender")),
                "html_preview": clean_string(truncate_text(email_data.get("html", ""), 500))
            },
            "analysis": clean_json_data(analysis_result)  # Используем рекурсивную очистку
        }

        save_json(result, filename)
        print(f"Сохранено: {filename}")
        return filename

    def process_directory(self, dir_path, model=Config.DEFAULT_MODEL):
        eml_files = glob.glob(os.path.join(dir_path, "*.eml"))

        if not eml_files:
            print(f"В папке {dir_path} нет .eml файлов")
            return []

        print(f"\nНайдено {len(eml_files)} писем")

        all_results = []

        for idx, file_path in enumerate(eml_files, 1):
            print(f"\n[{idx}/{len(eml_files)}] {os.path.basename(file_path)}")

            email_data = self.load_email_from_file(file_path)
            if not email_data:
                continue

            analysis = self.analyze_email(email_data["html"], model)

            entry = {
                "file": clean_string(os.path.basename(file_path)),
                "email": {
                    "subject": clean_string(email_data.get("subject")),
                    "sender": clean_string(email_data.get("sender"))
                },
                "analysis": analysis
            }

            all_results.append(entry)

            if "ai_probability" in analysis:
                print(f"AI: {analysis['ai_probability']}%")
                print(f"{analysis.get('summary', '')}")

        save_json(all_results, Config.FULL_ANALYSIS_PATH)
        print(f"\nПолный отчет: {Config.FULL_ANALYSIS_PATH}")

        return all_results