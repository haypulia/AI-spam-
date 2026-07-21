import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API настройки
    API_KEY = os.getenv("DEEPCODE_API_KEY", "sk-...")
    BASE_URL = "http://deepcode.ci.nsu.ru/api/chat/completions"
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Gemma-4-31B")

    # пути к данным
    DRWEB_DATA_DIR = os.getenv("DRWEB_DATA_DIR", "./data/raw/drweb")
    ANALYSIS_RESULTS_DIR = "./data/processed/analysis_results"
    FULL_ANALYSIS_PATH = "./data/processed/full_analysis.json"
    DATASETS_DIR = "./data/datasets"

    # параметры модели
    TEMPERATURE = 0.2
    TOP_P = 0.8
    MAX_HTML_LENGTH = 8000
    REQUEST_TIMEOUT = 90