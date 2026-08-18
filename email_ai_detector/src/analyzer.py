import json
import os
import re
import glob
import time
from email import policy
from email.parser import BytesParser
import requests
from config import Config
from utils import ensure_dir, get_timestamp, truncate_text
from ocr import ImageOCR
from engine import analyze_chunk_vector, analyze_email_vector

class EmailAIAnalyzer:
    def __init__(self, api_key, base_url=Config.BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.ocr = ImageOCR(languages="rus+eng")

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

            # Извлекаем изображения из письма
            images = self.extract_email_images(
                msg,
                html_body)

            # Распознаём текст
            ocr_results = self.ocr.analyze_images(images)

            # Печатаем информацию об OCR
            for result in ocr_results:
                if result["has_text"]:
                    print(
                        f"OCR: {result['source']} -> "
                        f"{len(result['text'])} символов"
                    )

            # Не отбрасываем письмо, если оно состоит только из картинки
            if not html_body or len(html_body.strip()) < 10:
                if not ocr_results:
                    print(f"Письмо {file_path} не содержит HTML и изображений")
                    return None

                html_body = ""

            print(f"Загружено письмо: {subject}")

            return {
                "subject": subject,
                "sender": sender,
                "html": html_body,
                "images": images,
                "ocr_results": ocr_results
            }

        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            return None

    def extract_email_images(self, msg, html_body=""):
        """
        Извлекает изображения из email.

        Поддерживает:
        - MIME image/* части;
        - inline изображения через Content-ID;
        - data:image/...;base64,... внутри HTML.
        """

        images = []

        # Изображения внутри MIME-письма
        for part in msg.walk():
            content_type = part.get_content_type()

            if not content_type.startswith("image/"):
                continue

            try:
                image_bytes = part.get_payload(decode=True)

                if not image_bytes:
                    continue

                filename = part.get_filename()
                content_id = part.get("Content-ID")

                source = filename or content_id or content_type

                images.append({
                    "source": source,
                    "bytes": image_bytes,
                    "content_id": content_id
                })

            except Exception as e:
                print(
                    f"Ошибка извлечения изображения "
                    f"{content_type}: {e}"
                )

        # data:image/...;base64,... внутри HTML
        data_uri_images = self.ocr.extract_data_uri_images(
            html_body
        )

        for index, image_bytes in enumerate(data_uri_images):
            images.append({
                "source": f"data_uri_{index}",
                "bytes": image_bytes,
                "content_id": None
            })

        return images

    def split_html_into_blocks(self, html):
        if len(html) > 25000:
            html = html[:25000] + "..."
            print(f" HTML обрезан до 25000 символов")

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
        if len(clean_html) > 10000:
            clean_html = clean_html[:10000] + "..."

        prompt = f"""Analyze this HTML fragment for AI-generated patterns. 
    Return ONLY JSON.

    Flags to detect (check ALL of them):
    - TOO_DEEP_NESTING (nested tables > 5 levels)
    - EMPTY_CELL (empty table cells with &nbsp;)
    - DUPLICATE_STYLES (repeated inline styles in many elements)
    - SUSPICIOUS_COMMENT (<!-- Generated -->, <!-- Template -->, etc.)
    - TEMPLATE_VARIABLE ({{{{name}}}}, {{{{email}}}}, {{{{variable}}}})
    - REPETITIVE_PATTERN (same block repeated 3+ times)
    - INLINE_STYLE_OVERUSE (more than 3 inline styles per element)
    - SUSPICIOUS_LINK (links with redirects, unusual domains)
    - MISSING_ALT (images without alt text)
    - STRUCTURAL_REDUNDANCY (unnecessary nested divs or tables)

    HTML:
    {clean_html}

    JSON:
    {{"ai_probability": 0-100, "flags": [], "summary": ""}}"""

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an HTML security expert. Check ALL flags."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.8
        }

        try:
            print(f"    → Анализ блока ({len(clean_html)} символов)...")

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
                        print(f"AI: {parsed.get('ai_probability', 0)}%")
                        return {
                            "ai_probability": parsed.get("ai_probability", 0),
                            "flags": parsed.get("flags", []),
                            "summary": parsed.get("summary", "")
                        }
                return {"ai_probability": 0, "flags": [], "summary": "Ошибка парсинга"}

            print(f"HTTP {response.status_code}")
            return {"ai_probability": 0, "flags": [], "summary": f"HTTP {response.status_code}"}

        except Exception as e:
            print(f"Ошибка: {str(e)[:100]}")
            return {"ai_probability": 0, "flags": [], "summary": str(e)[:50]}

    def analyze_ocr_text(self, ocr_results, model=Config.DEFAULT_MODEL):
        """
        Анализ текста, распознанного с изображений,
        исключительно на предмет AI-generated текста.
        """

        if not ocr_results:
            return {
                "ai_probability": 0,
                "flags": [],
                "summary": ""
            }

        texts = []

        for result in ocr_results:
            text = result.get("text", "").strip()

            if text:
                texts.append(
                    f"IMAGE: {result.get('source', 'unknown')}\n{text}"
                )

        if not texts:
            return {
                "ai_probability": 0,
                "flags": [],
                "summary": ""
            }

        ocr_text = "\n\n".join(texts)

        if len(ocr_text) > 10000:
            ocr_text = ocr_text[:10000] + "..."

        prompt = f"""
Analyze the following text extracted from an image in an email.

Your ONLY task is to estimate whether the text appears
to be generated or strongly assisted by an AI language model.

Do NOT classify:
- spam
- phishing
- scams
- malicious intent
- security threats
- advertising risk

Focus ONLY on AI-generated writing.

IMPORTANT:
- Short text is NOT automatically human-written.
- A short slogan, heading, banner, product description,
  or marketing phrase can still be fully AI-generated.
- Do NOT use text length alone as evidence.
- Judge the wording, structure, semantics, and phrasing.

Evaluate these AI-generation indicators:

1. GENERIC_FORMULATION
   Generic statements that could apply to almost any
   product, company, or situation.

2. FORMULAIC_STRUCTURE
   Highly predictable headline/subheadline/body structure.

3. SEMANTIC_VAGUENESS
   Abstract claims without concrete information,
   examples, facts, or specific details.

4. OVERLY_POLISHED_LANGUAGE
   Unusually polished, neutral, smooth, or
   corporate wording with little personal voice.

5. TEMPLATE_LIKE_LANGUAGE
   Wording that looks reusable across many different
   companies, products, or campaigns.

6. REPETITIVE_SYNTAX
   Repeated grammatical or semantic patterns.

7. AI_STYLE_MARKERS
   Phrases and constructions commonly produced
   by large language models.

8. MARKETING_ABSTRACTION
   Generic phrases such as:
   "innovative approach",
   "effective solutions",
   "unlock your potential",
   "drive sustainable growth",
   "optimize processes",
   when they are used without concrete supporting details.

9. HUMAN_SPECIFICITY
   Specific personal details, concrete actions,
   dates, names, references, informal wording,
   mistakes, abbreviations, or other details
   that make the text look naturally human-written.

IMPORTANT:
Do NOT assume that marketing language is AI-generated.
Do NOT assume that a short text is human-written.
Use all available linguistic evidence.

Return ONLY valid JSON.

OCR TEXT:

{ocr_text}

JSON format:

{{
    "ai_probability": 0-100,
    "flags": [],
    "summary": ""
}}

The flags must describe ONLY AI-generation evidence.

Examples of valid flags:
- "generic_formulation"
- "formulaic_structure"
- "semantic_vagueness"
- "overly_polished_language"
- "template_like_language"
- "repetitive_syntax"
- "ai_style_markers"
- "marketing_abstraction"
- "human_specificity"
"""

        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert forensic analyst "
                        "specializing in AI-generated text detection. "
                        "Your task is to estimate whether the provided "
                        "text was generated or strongly assisted by an "
                        "AI language model. "
                        "Do not classify spam, phishing, scams, or "
                        "malicious intent. "
                        "Do not use text length alone as evidence."
                        )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.8
        }

        try:
            print(
                f"    → Анализ OCR "
                f"({len(ocr_text)} символов)..."
            )

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

                    json_match = re.search(
                        r'\{.*\}',
                        content,
                        re.DOTALL
                    )

                    if json_match:
                        parsed = json.loads(
                            json_match.group()
                        )

                        probability = parsed.get(
                            "ai_probability",
                            0
                        )

                        try:
                            probability = float(
                                probability
                            )
                        except (TypeError, ValueError):
                            probability = 0

                        # Нормализуем в 0..100
                        if probability <= 1:
                            probability *= 100

                        probability = max(
                            0,
                            min(100, probability)
                        )

                        print(
                            f"    OCR AI: "
                            f"{probability:.0f}%"
                        )

                        return {
                            "ai_probability": probability,
                            "flags": parsed.get(
                                "flags",
                                []
                            ),
                            "summary": parsed.get(
                                "summary",
                                ""
                            )
                        }

                return {
                    "ai_probability": 0,
                    "flags": [],
                    "summary": "Ошибка парсинга"
                }

            print(
                f"OCR HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            return {
                "ai_probability": 0,
                "flags": [],
                "summary": (
                    f"HTTP {response.status_code}"
                )
            }

        except Exception as e:
            print(
                f"OCR ошибка: {str(e)[:100]}"
            )

            return {
                "ai_probability": 0,
                "flags": [],
                "summary": str(e)[:100]
            }

    def calculate_ocr_risk(self, ocr_analysis):
        """
        Итоговый OCR risk score 0-100.

        Использует:
        1. вероятность от LLM;
        2. семантические флаги LLM.

        Это защищает от ситуации, когда модель говорит
        probability=0, но одновременно выставляет phishing/scam.
        """

        score = float(
            ocr_analysis.get("ai_probability", 0)
        )

        flags = [
            str(flag).lower().strip()
            for flag in ocr_analysis.get("flags", [])
        ]

        flag_scores = []

        # Критические признаки
        critical_flags = {
            "phishing",
            "scam",
            "spam",
        }

        # Очень сильные рекламные признаки
        strong_flags = {
            "aggressive advertising",
            "aggressive_advertising",
            "suspicious call to action",
            "suspicious calls to action",
            "suspicious_call_to_action",
        }

        # Средние признаки
        medium_flags = {
            "urgent language",
            "urgent_language",
        }

        if any(flag in critical_flags for flag in flags):
            flag_scores.append(90)

        if any(flag in strong_flags for flag in flags):
            flag_scores.append(70)

        if any(flag in medium_flags for flag in flags):
            flag_scores.append(55)

        if flag_scores:
            score = max(score, max(flag_scores))

        return max(0, min(100, score))

    def build_chunk_vector_data(self, block_result):
        """
        Преобразует результат LLM анализа HTML-блока
        в формат, который ожидает engine.analyze_chunk_vector().
        """

        probability = block_result.get(
            "ai_probability",
            0
        )

        try:
            probability = float(probability)
        except (TypeError, ValueError):
            probability = 0.0

        # analyze_block() возвращает 0..100
        if probability > 1:
            probability /= 100

        probability = max(
            0.0,
            min(1.0, probability)
        )

        flags = {
            str(flag).upper().strip()
            for flag in block_result.get("flags", [])
        }

        return {
            "llm_probability": probability,

            "template_variable": (
                "TEMPLATE_VARIABLE" in flags
            ),

            "repeating_pattern": (
                "REPETITIVE_PATTERN" in flags
            ),

            "suspicious_comments": (
                "SUSPICIOUS_COMMENT" in flags
            ),

            "deep_nesting": (
                "TOO_DEEP_NESTING" in flags
            ),

            "duplicated_styles": (
                "DUPLICATE_STYLES" in flags
            ),

            "empty_cell": (
                "EMPTY_CELL" in flags
            )
        }

    def analyze_email_blocks(self, email_html, ocr_results=None, model=Config.DEFAULT_MODEL):
        """
        Анализ всего письма:

        1. Разбивает HTML на блоки.
        2. Анализирует каждый HTML-блок.
        3. Прогоняет каждый блок через engine scoring.
        4. Анализирует OCR-текст с изображений.
        5. Объединяет HTML + OCR в итоговый AI score.
        """

        blocks = self.split_html_into_blocks(
            email_html
        )

        results = []

        ocr_results = ocr_results or []

        total_blocks = len(blocks)

        # 1. HTML BLOCKS

        for idx, block in enumerate(blocks):

            print(
                f"  → Анализ блока "
                f"{idx + 1}/{total_blocks} "
                f"({len(block['content'])} символов)..."
            )

            # LLM анализ HTML
            block_result = self.analyze_block(block["content"], model)

            # данные для engine
            vector_data = self.build_chunk_vector_data(
                block_result
            )

            # AI score блока через engine
            vector_result = analyze_chunk_vector(
                vector_data
            )

            results.append(
                {
                "block_type": block["type"],
                "position": block["position"],

                # Сырой ответ модели
                "llm_probability": block_result.get("ai_probability", 0),

                # Итог engine
                "ai_score": round(vector_result["AI_Score"] * 100, 1),

                "verdict": vector_result["Verdict"],

                "confidence": vector_result["Confidence"],

                "flags": block_result.get("flags", []),

                "summary": block_result.get("summary", ""),
                "engine_explanation": vector_result.get("Explanation",""),
                "signals": vector_result.get("Signals", {})
            }
            )

            if idx < total_blocks - 1:
                time.sleep(2)
        
        # 2. HTML GLOBAL SCORE

        html_scores = [
            block["ai_score"]
            for block in results
        ]

        if html_scores:
            html_global_score = (sum(html_scores) / len(html_scores))
        else:
            html_global_score = 0

        # 0..100 → 0..1
        html_score_normalized = (html_global_score / 100)

        # 3. OCR

        ocr_analysis = self.analyze_ocr_text(ocr_results, model)

        ocr_probability = ocr_analysis.get("ai_probability", 0)

        try:
            ocr_probability = float(ocr_probability)

        except (TypeError, ValueError):
            ocr_probability = 0

        has_ocr_text = any(result.get("has_text", False) and result.get("text", "").strip()
            for result in ocr_results
        )

        # analyze_ocr_text() возвращает 0..100
        ocr_score_normalized = (
            ocr_probability / 100
        )

        # 4. FINAL EMAIL SCORE

        final_result = analyze_email_vector(html_score=html_score_normalized, 
                                            ocr_score=ocr_score_normalized,
                                            has_ocr=has_ocr_text)

        # 5. GLOBAL FLAGS HTML

        global_flags = self.extract_global_flags(results)

        # 6. HIGH RISK BLOCKS

        high_risk = [block for block in results if block["ai_score"] >= 70]

        # 7. CONSOLE OUTPUT

        print(f"    HTML AI: " f"{html_global_score:.1f}%")

        print(f"    OCR AI:  " f"{ocr_probability:.1f}%")

        print(f"    FINAL AI: " f"{final_result['AI_Score_Percent']:.1f}%")

        print(f"    Verdict: " f"{final_result['Verdict']}")

        # 8. RETURN

        return {
            "blocks": results, "html_ai_score": round(html_global_score, 1),
            "ocr_ai_score": round(ocr_probability, 1), 
            "global_ai_score": final_result["AI_Score_Percent"],
            "verdict": final_result["Verdict"],
            "global_flags": global_flags,
            "high_risk_blocks": high_risk,
            "engine": {"final_score": final_result["AI_Score"], 
            "signals": final_result["Signals"],
            "explanation": final_result["Explanation"]},
            "ocr": { "images_count": len(ocr_results),
            "text": ocr_results, "ai_score": round(ocr_probability,1),
            "flags": ocr_analysis.get("flags", []),
            "summary": ocr_analysis.get("summary", "")}
        }


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

        blocks_summary = []
        for block in analysis_result.get("blocks", []):
            blocks_summary.append({
                "position": block.get("position"),
                "block_type": block.get("block_type"),
                "ai_score": block.get("ai_score"),
                "flags": block.get("flags", [])
            })

        result = {
            "timestamp": get_timestamp(),
            "email": {
                "subject": email_data.get("subject"),
                "sender": email_data.get("sender"),
                "html_preview": truncate_text(email_data.get("html", ""), 500)
            },
            "analysis": {
                "html_ai_score": analysis_result.get("html_ai_score", 0),

                "ocr_ai_score": analysis_result.get("ocr_ai_score", 0),

                "global_ai_score": analysis_result.get("global_ai_score", 0),

                "verdict": analysis_result.get("verdict", ""),

                "engine": analysis_result.get("engine", {}),

                "blocks_summary": blocks_summary,

                "global_flags": analysis_result.get("global_flags", []),

                "high_risk_blocks": analysis_result.get("high_risk_blocks",[]),

                "blocks_details": analysis_result.get("blocks", []),

                "ocr": analysis_result.get("ocr", {})
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
            analysis = self.analyze_email_blocks(email_data["html"], email_data.get("ocr_results", []), model)

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
