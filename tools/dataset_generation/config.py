# основные настройки генератора датасета.

DEEPCODE_BASE_URL = "https://deepcode.ci.nsu.ru"

DEEPCODE_CHAT_ENDPOINT = (
    f"{DEEPCODE_BASE_URL}/api/chat/completions"
)

MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash",
]

MODEL_TARGETS = {
    "deepseek-ai/DeepSeek-V4-Flash": 80,
}

MAX_TOKENS = 4096

TEMPERATURE = 0.8

REQUEST_TIMEOUT = 120

MAX_RETRIES = 3

MIN_FRAGMENT_LENGTH = 20

MAX_FRAGMENT_LENGTH = 500

MAX_SENTENCES_IN_FRAGMENT = 3

COSINE_SIMILARITY_THRESHOLD = 0.98

NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.20

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

LANGUAGES = [
    "ru",
    "en",
]

DOMAINS = [
    "corporate",
    "personal",
    "advertising",
    "spam",
    "graymail",
    "transactional",
    "notifications",
    "marketing",
    "support",
    "finance",
    "education",
    "shopping",
    "delivery",
]

EMAIL_FORMATS = [
    "plain_text",
    "html",
]

OUTPUT_DIR = "generated_dataset"

OUTPUT_FILE = "dataset.jsonl"

SOURCE_EMAILS_FILE = (
    "generated_dataset/source_emails_new.jsonl"
)

# общее количество исходных писем
NEW_SOURCE_EMAILS_TARGET = 80

RANDOM_SEED = 42