import json
import os
from datetime import datetime


def ensure_dir(path):  # создает папку, если её нет
    os.makedirs(path, exist_ok=True)


def save_json(data, path):  # сохраняет данные в JSON файл
    cleaned_data = clean_json_data(data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)


def load_json(path):  # загружает данные из JSON файла
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp():  # возвращает текущую метку времени
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def truncate_text(text, max_len=500):  # обрезает текст до максимальной длины символов
    if not text:
        return ""

    text = str(text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def clean_string(s): # очищает строку от проблемных символов
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode('utf-8', errors='ignore').decode('utf-8')


def clean_json_data(obj): # рекурсивно очищает все строки в JSON-подобном объекте
    if isinstance(obj, str):
        return clean_string(obj)
    elif isinstance(obj, dict):
        return {clean_string(k): clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    else:
        return obj