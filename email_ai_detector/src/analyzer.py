import json
import os
import re
import glob
from email import policy
from email.parser import BytesParser
import requests
from config import Config
from utils import ensure_dir, get_timestamp, truncate_text


class EmailAIAnalyzer:
    def __init__(self, api_key, base_url=Config.BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"Анализатор инициализирован. API-ключ: {api_key[:8]}...")

    def load_email_from_file(self, file_path):
        try:
            with open(file_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)

            subject_raw = msg.get("Subject", "Без темы")
            sender_raw = msg.get("From", "Неизвестный отправитель")

            try:
                subject = subject_raw.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                subject = subject_raw.encode('utf-8', errors='ignore').decode('utf-8')

            try:
                sender = sender_raw.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                sender = sender_raw.encode('utf-8', errors='ignore').decode('utf-8')

            html_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        html_body = part.get_content()
                        break
            else:
                if msg.get_content_type() == "text/html":
                    html_body = msg.get_content()

            if not html_body:
                try:
                    html_body = msg.get_body(preferencelist=("plain",)).get_content()
                except Exception:
                    html_body = ""

            if not html_body or len(html_body.strip()) < 10:
                print(f"Письмо {file_path} не содержит HTML")
                return None

            print(f"Загружено письмо: {subject}")
            return {
                "subject": subject,
                "sender": sender,
                "html": html_body
            }

        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            return None

    def split_html_into_blocks(self, html):
        if len(html) > 10000:
            html = html[:10000] + "..."
            print(f" HTML обрезан до 10000 символов")

        blocks = []

        table_pattern = r'(<table[^>]*>.*?</table>)'
        tables = re.findall(table_pattern, html, re.DOTALL | re.IGNORECASE)
        for i, table in enumerate(tables):
            blocks.append({
                'type': 'table',
                'content': table,
                'position': f'table_{i}'
            })

        div_pattern = r'(<div[^>]*class=["\'](?:section|content|main|header|footer)[^"\']*["\'][^>]*>.*?</div>)'
        divs = re.findall(div_pattern, html, re.DOTALL | re.IGNORECASE)
        for i, div in enumerate(divs):
            blocks.append({
                'type': 'div_section',
                'content': div,
                'position': f'section_{i}'
            })

        list_pattern = r'(<ul[^>]*>.*?</ul>|<ol[^>]*>.*?</ol>)'
        lists = re.findall(list_pattern, html, re.DOTALL | re.IGNORECASE)
        for i, lst in enumerate(lists):
            blocks.append({
                'type': 'list',
                'content': lst,
                'position': f'list_{i}'
            })

        if not blocks:
            blocks.append({
                'type': 'text_block',
                'content': html,
                'position': 'full_text'
            })

        return blocks

    def analyze_block(self, block_html, model=Config.DEFAULT_MODEL):
        clean_html = block_html.encode('utf-8', errors='ignore').decode('utf-8')
        if len(clean_html) > 2000:
            clean_html = clean_html[:2000] + "..."

        prompt = f"""Analyze this HTML fragment for AI-generated patterns. 
    Return ONLY JSON.

    Flags to detect:
    - TOO_DEEP_NESTING (nested tables > 5 levels)
    - EMPTY_CELL (empty table cells)
    - SUSPICIOUS_COMMENT (<!-- Generated -->)
    - TEMPLATE_VARIABLE ({{name}})
    - DUPLICATE_STYLES (repeated inline styles)

    HTML:
    {clean_html}

    JSON:
    {{"ai_probability": 0-100, "flags": [], "summary": ""}}"""

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an HTML security expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.8
        }

        try:
            print(f"  → Анализ блока ({len(clean_html)} символов)...")

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=Config.REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result:
                    content = result["choices"][0]["message"]["content"]
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        return {
                            "ai_probability": parsed.get("ai_probability", 0),
                            "flags": parsed.get("flags", []),
                            "summary": parsed.get("summary", "")
                        }
                return {"ai_probability": 0, "flags": [], "summary": "Ошибка парсинга"}

            print(f"HTTP {response.status_code}")
            return {"ai_probability": 0, "flags": [], "summary": f"HTTP {response.status_code}"}

        except Exception as e:
            print(f"Ошибка: {str(e)[:50]}")
            return {"ai_probability": 0, "flags": [], "summary": str(e)[:50]}

    def analyze_email_blocks(self, email_html, model=Config.DEFAULT_MODEL):
        blocks = self.split_html_into_blocks(email_html)
        results = []

        for block in blocks:
            block_result = self.analyze_block(block['content'], model)
            results.append({
                'block_type': block['type'],
                'position': block['position'],
                'ai_score': block_result.get('ai_probability', 0),
                'flags': block_result.get('flags', []),
                'summary': block_result.get('summary', '')
            })

        global_score = self.calculate_global_score(results)
        global_flags = self.extract_global_flags(results)
        high_risk = [b for b in results if b['ai_score'] > 70]

        return {
            'blocks': results,
            'global_ai_score': global_score,
            'global_flags': global_flags,
            'high_risk_blocks': high_risk
        }

    def calculate_global_score(self, block_results):
        if not block_results:
            return 0
        scores = [b.get('ai_score', 0) for b in block_results if b.get('ai_score', 0) > 0]
        if not scores:
            return 0
        return round(sum(scores) / len(scores))

    def extract_global_flags(self, block_results):
        all_flags = []
        for block in block_results:
            for flag in block.get('flags', []):
                if flag not in all_flags:
                    all_flags.append(flag)
        return all_flags

    def save_results(self, email_data, analysis_result):
        ensure_dir(Config.ANALYSIS_RESULTS_DIR)
        filename = f"{Config.ANALYSIS_RESULTS_DIR}/analysis_{get_timestamp()}.json"

        result = {
            "timestamp": get_timestamp(),
            "email": {
                "subject": email_data.get("subject"),
                "sender": email_data.get("sender"),
                "html_preview": truncate_text(email_data.get("html", ""), 500)
            },
            "analysis": {
                "global_ai_score": analysis_result.get("global_ai_score", 0),
                "global_flags": analysis_result.get("global_flags", []),
                "high_risk_blocks": analysis_result.get("high_risk_blocks", []),
                "blocks": analysis_result.get("blocks", [])
            }
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

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

            print(f"Анализ письма: {email_data.get('subject', '')[:30]}...")
            print(f"HTML длина: {len(email_data['html'])} символов")
            analysis = self.analyze_email_blocks(email_data["html"], model)

            entry = {
                "file": os.path.basename(file_path),
                "email": {
                    "subject": email_data.get("subject"),
                    "sender": email_data.get("sender")
                },
                "analysis": analysis
            }
            all_results.append(entry)

            if "global_ai_score" in analysis:
                print(f"Глобальный AI: {analysis['global_ai_score']}%")
                print(f"Найдено блоков: {len(analysis.get('blocks', []))}")
                print(f"Флаги: {', '.join(analysis.get('global_flags', []))}")

        with open(Config.FULL_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nПолный отчет: {Config.FULL_ANALYSIS_PATH}")
        return all_results
