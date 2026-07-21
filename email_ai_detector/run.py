import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src")) # добавление папку src в путь

from analyzer import EmailAIAnalyzer
from config import Config
from utils import ensure_dir

def main():
    print("="*60)
    print("ЗАПУСК EMAIL AI АНАЛИЗАТОРА")
    print(f"Папка с письмами: {Config.DRWEB_DATA_DIR}")
    print(f"Модель: {Config.DEFAULT_MODEL}")
    print("="*60)

    ensure_dir(Config.ANALYSIS_RESULTS_DIR)     # создание необходимых папок
    ensure_dir(Config.DATASETS_DIR)

    analyzer = EmailAIAnalyzer(
        api_key=Config.API_KEY,
        base_url=Config.BASE_URL
    )

    results = analyzer.process_directory(
        dir_path=Config.DRWEB_DATA_DIR,
        model=Config.DEFAULT_MODEL
    )

    print(f"\nГотово! Обработано писем: {len(results)}")
    print(f"Результаты в папке: {Config.ANALYSIS_RESULTS_DIR}")

if __name__ == "__main__":
    main()